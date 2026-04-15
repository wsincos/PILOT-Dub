# src/data


## Dubbing Dataset

这是一个用于多模态配音任务的 PyTorch `Dataset` 实现。它能够同时加载音素序列、Encodec 音频码本以及对应的视频帧，并支持动态 Batching、Padding 和数据增强（视频裁剪）。

### 核心功能
-   **多模态加载**：同时读取文本（Phonemes）、音频（Encodec Codes）和视频（Video Frames）。
-   **数据预处理**：
    -   **视频**：自动加载 `.mp4`，转为灰度图，进行中心裁剪（CenterCrop）和归一化（Normalize）。
    -   **音频**：加载多码本（n_codebooks）的 Encodec 数据，支持长音频截断和短音频 Padding。
    -   **文本**：将音素映射为 ID，并根据音频长度进行对齐或截断。
-   **特殊逻辑**：支持读取 `split_len.txt`，用于处理拼接数据的原始长度信息。

### 目录结构要求
该 Dataset 期望的数据集根目录结构如下：

```text
dataset_dir/
├── manifest/
│   ├── train.txt       # 训练集列表 (格式: 0 \t filename \t length)
│   ├── validation.txt  # 验证集列表
│   └── test.txt        # 测试集列表
├── phonemes/           # 音素文本文件 (*.txt)
├── encodec_16khz_4codebooks/ # Encodec 音频码本文件 (*.txt)
├── video/              # 视频文件 (*.mp4)
├── vocab.txt           # 音素词表 (格式: id token)
└── split_len.txt       # 记录原始音频长度的文件 (格式: filename,orig_len)
```

### 返回数据格式

`__getitem__` 返回一个字典，包含单个样本的数据；经过 `collate_fn` 处理后的 Batch 字典包含以下 Key：

| Key | 类型 | 形状 (Batch First) | 说明 |
| :--- | :--- | :--- | :--- |
| **`x`** | LongTensor | `[B, T_text]` | Padding 后的音素序列 ID |
| **`x_lens`** | LongTensor | `[B]` | 原始音素序列长度 |
| **`y`** | LongTensor | `[B, K, T_audio]` | Padding 后的 Encodec 码本 (K=码本数) |
| **`y_lens`** | LongTensor | `[B]` | 原始音频序列长度 |
| **`v`** | FloatTensor | `[B, T_video, H, W]` | 预处理后的视频帧序列 (灰度) |
| **`v_lens`** | LongTensor | `[B]` | 视频帧数 |
| **`split_length`** | Int | - | 原始音频片段的长度（用于拼接数据） |
| **`text_padding_mask`** | BoolTensor | `[B, T_text]` | 文本 Padding Mask |
| **`v_padding_mask`** | BoolTensor | `[B, T_video]` | 视频 Padding Mask |

### 使用示例

```python
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.data.dubbing_dataset import dataset

# 1. 配置参数
cfg = OmegaConf.create({
    "dataset_dir": "/path/to/data",
    "exp_dir": "/path/to/exp",
    "encodec_sr": 50,       # Encodec 采样率
    "n_codebooks": 4,       # 码本数量
    "special_first": False,
    "n_special": 0,
    "text_pad_token": 0,
    "audio_pad_token": 0,
})

# 2. 实例化 Dataset
train_dataset = dataset(
    split="train",
    cfg=cfg,
    audio_max_length=20.0, # 最大音频长度(秒)
    image_crop_size=88     # 视频裁剪尺寸 (代码中硬编码为88，可修改)
)

# 3. 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=train_dataset.collate  # 必须使用自定义的 collate 函数
)

# 4. 迭代数据
for batch in train_loader:
    phonemes = batch['x']
    audio_codes = batch['y']
    video_frames = batch['v']
    print(f"Audio shape: {audio_codes.shape}") # [B, 4, T]
    print(f"Video shape: {video_frames.shape}") # [B, T, 88, 88]
    break
```

### 注意事项

1.  **视频处理**：代码默认将视频转换为**灰度图** (`cv2.COLOR_BGR2GRAY`)。
2.  **异常处理**：如果在加载某个样本时发生错误（如文件缺失或长度不符），Dataset 会随机重试加载另一个样本，直到成功。
3.  **Split Length**：`split_len.txt` 是必须的，用于确定拼接数据中原始部分的边界。

## Custom Samplers for Distributed Training `sampler.py`

This module (`src/data/sampler.py`) provides custom PyTorch `Sampler` implementations designed for efficient distributed training, specifically focusing on **Dynamic Batching** to minimize padding in sequence-based tasks (like audio or NLP).

### Overview

The file contains the following key classes:

1.  **`DistributedDynamicBatchSampler`**: The core component. It groups examples of similar lengths into the same batch to minimize padding, significantly speeding up training on variable-length data. It supports Distributed Data Parallel (DDP).
2.  **`StatefulDistributedSampler`**: A standard distributed sampler that supports state recovery (resuming from a specific step within an epoch).
3.  **`StatefulSampler`**: A single-process version of the stateful sampler.

---

### Classes Detail

#### 1. `DistributedDynamicBatchSampler`

This sampler is modified from *SpeechBrain*. It buckets examples by length and creates batches where all examples have approximately the same length.

*   **Key Features:**
    *   **Dynamic Batching:** Reduces padding overhead by grouping similar-length items.
    *   **Distributed Support:** Automatically handles data splitting across multiple GPUs (`rank` and `num_replicas`).
    *   **Bucketing:** Uses quantiles (log-normal distribution) or fixed boundaries to define length buckets.
    *   **Max Token Limit:** Batches are constructed based on a `max_batch_length` (total tokens in a batch) rather than a fixed batch size, optimizing GPU memory usage.

*   **Usage in Lightning DataModule:**
    ```python
    from src.data.sampler import DistributedDynamicBatchSampler

    # In your train_dataloader
    sampler = DistributedDynamicBatchSampler(
        dataset=train_dataset,
        args=self.cfg,                  # Config containing max_num_tokens, num_buckets, etc.
        num_replicas=self.trainer.world_size,
        rank=self.trainer.global_rank,
        shuffle=True,
        lengths_list=train_dataset.lengths_list, # List of lengths for all items
        epoch=self.trainer.current_epoch
    )

    dataloader = DataLoader(dataset, batch_sampler=sampler, ...)
    ```

*   **Required Config Arguments (`args`):**
    *   `max_num_tokens` (or `val_max_num_tokens`): Maximum sum of lengths allowed in one batch.
    *   `num_buckets`: Number of length buckets to create.
    *   `audio_max_length`: Max allowed audio length (for clipping).
    *   `encodec_sr`: Sampling rate (used to calculate max length in frames).

#### 2. `StatefulDistributedSampler`

A wrapper around the standard `DistributedSampler` that adds the ability to resume training from a specific step within an epoch.

*   **Key Methods:**
    *   `set_epoch_resume(epoch, cur_step)`: Skips the first `cur_step` batches when iterating.

#### 3. `StatefulSampler`

Similar to `StatefulDistributedSampler` but for single-GPU/CPU training.

---

### Helper Functions & Classes

*   **`AverageMeter`**: A simple utility class to compute and store the average and current value of a metric.
*   **`print_model_info`**: A utility to log model architecture and parameter counts.

### How Dynamic Batching Works

1.  **Bucketing**: The dataset is analyzed to create `num_buckets` based on sequence lengths.
2.  **Assignment**: Each example is assigned to a bucket.
3.  **Batching**: Examples within a bucket are grouped until they reach `max_batch_length` (total tokens) or `max_batch_ex` (max examples).
4.  **Shuffling**:
    *   If `shuffle=True`, the order of batches is randomized deterministically based on the `epoch` and `seed`.
    *   This ensures that while batches contain similar-length items, the order in which the model sees them varies every epoch.

### Notes

*   **Determinism**: All samplers use `torch.Generator` with a seed derived from `seed + epoch`. This ensures that in a DDP setting, all processes# Custom Samplers for Distributed Training

This module (`src/data/sampler.py`) provides custom PyTorch `Sampler` implementations designed for efficient distributed training, specifically focusing on **Dynamic Batching** to minimize padding in sequence-based tasks (like audio or NLP).

（
## 用于分布式训练的自定义采样器 `sampler.py`

本模块 (sampler.py) 提供了自定义的 PyTorch `Sampler` 实现，专为高效的分布式训练设计，特别关注通过 **动态批处理 (Dynamic Batching)** 来最小化序列任务（如音频或 NLP）中的 Padding。

### 概览

该文件包含以下核心类：

1.  **`DistributedDynamicBatchSampler`**：核心组件。它将长度相近的样本分到同一个 Batch 中以最小化 Padding，从而显著加快变长数据的训练速度。支持分布式数据并行 (DDP)。
2.  **`StatefulDistributedSampler`**：标准分布式采样器，支持状态恢复（从 Epoch 内的特定 Step 恢复）。
3.  **`StatefulSampler`**：状态采样器的单进程版本。

---

### 类详细说明

#### 1. `DistributedDynamicBatchSampler`

此采样器修改自 *SpeechBrain*。它根据长度对样本进行分桶，并创建每个样本长度大致相同的 Batch。

*   **核心特性：**
    *   **动态批处理**：通过对相似长度的项目进行分组来减少 Padding 开销。
    *   **分布式支持**：自动处理多 GPU 之间的数据切分 (`rank` 和 `num_replicas`)。
    *   **分桶**：使用分位数（对数正态分布）或固定边界来定义长度桶。
    *   **最大 Token 限制**：基于 `max_batch_length`（Batch 中的总 Token 数）而非固定 Batch Size 构建 Batch，优化 GPU 显存利用率。

*   **在 Lightning DataModule 中的用法：**
    ```python
    from src.data.sampler import DistributedDynamicBatchSampler

    # 在你的 train_dataloader 中
    sampler = DistributedDynamicBatchSampler(
        dataset=train_dataset,
        args=self.cfg,                  # 包含 max_num_tokens, num_buckets 等的配置
        num_replicas=self.trainer.world_size,
        rank=self.trainer.global_rank,
        shuffle=True,
        lengths_list=train_dataset.lengths_list, # 所有项目的长度列表
        epoch=self.trainer.current_epoch
    )

    dataloader = DataLoader(dataset, batch_sampler=sampler, ...)
    ```

*   **必需的配置参数 (`args`)：**
    *   `max_num_tokens` (或 `val_max_num_tokens`)：一个 Batch 中允许的最大长度总和。
    *   `num_buckets`：要创建的长度桶数量。
    *   `audio_max_length`：允许的最大音频长度（用于裁剪）。
    *   `encodec_sr`：采样率（用于计算以帧为单位的最大长度）。

#### 2. `StatefulDistributedSampler`

标准 `DistributedSampler` 的包装器，增加了从 Epoch 内特定 Step 恢复训练的功能。

*   **核心方法：**
    *   `set_epoch_resume(epoch, cur_step)`：迭代时跳过前 `cur_step` 个 Batch。

#### 3. `StatefulSampler`

类似于 `StatefulDistributedSampler`，但用于单 GPU/CPU 训练。

---

### 辅助函数与类

*   **`AverageMeter`**：一个简单的工具类，用于计算和存储指标的当前值及平均值。
*   **`print_model_info`**：用于记录模型架构和参数计数的工具。

### 动态批处理的工作原理

1.  **分桶 (Bucketing)**：分析数据集，根据序列长度创建 `num_buckets`。
2.  **分配 (Assignment)**：每个样本被分配到一个桶中。
3.  **批处理 (Batching)**：桶内的样本被分组，直到达到 `max_batch_length`（总 Token 数）或 `max_batch_ex`（最大样本数）。
4.  **打乱 (Shuffling)**：
    *   如果 `shuffle=True`，Batch 的顺序会根据 `epoch` 和 `seed` 进行确定性随机打乱。
    *   这确保了虽然 Batch 包含长度相似的项目，但模型每轮看到的顺序是不同的。

### 注意事项

*   **确定性**：所有采样器都使用基于 `seed + epoch` 生成种子的 `torch.Generator`。这确保了在 DDP 环境下，所有进程在切分数据前对打乱顺序达成一致。

）

## `src/data/preprocess_data.sh`

save_wav.py
inputs: 
- root_dir, 
- split_name
outputs: 
- 生成一个文件列表: {root_dir}/file.list, 每一行是一个视频文件的路径，格式为：split/子文件夹/文件名。
- 从视频文件中提取出音频，重采样到16kHZ并做裁剪，保存到{root_dir}/{split_name}/{子文件夹}/{文件名}.wav


phonemize_lrs.py
inputs:
- root_dir: 数据集的根目录，包含音频和文本文件。
- split_name: 数据集的子目录名称（如 train、test）。
- save_dir: 预处理后的数据保存路径。
- encodec_model_path: Encodec 模型的权重路径，用于音频编码。
outputs:
- 音素序列文件:
    - 路径: {save_dir}/phonemes_/{segment_id}.txt
    - 内容: 每段**文本对应的音素序列**，经过音素化处理后保存为 .txt 文件。
- Encodec 编码文件:
    - 路径: {save_dir}/encodec_16khz_4codebooks_/{segment_id}.txt
    - 内容: 每段音频对应的离散化编码，使用 Encodec 模型生成。
- 音素词汇表:
    - 路径: {save_dir}/vocab.txt
    - 内容: 所有音素及其对应的 ID，生成音素词汇表以供后续模型使用。
- 日志信息:
    - 包括音频长度统计、过滤的样本数量等，输出到终端。

> NOTE: 我们这里并没有直接使用它生成的vocab.txt，而是用`voicecraft_dub`模型中对应的phn2num.txt进行了替换，以便后续pretrained模型训练能够使用相同的embedding。


detect_landmark.py
inputs
- `--root`: 视频文件的根目录。
- `--landmark`: 保存关键点的目标目录。
- `--manifest`: 包含视频文件名列表的清单文件，每一行是一个视频文件的相对路径。
- `--face_preprocess_dir`: 存放 dlib 模型文件的目录，包含以下文件：
  - `shape_predictor_68_face_landmarks.dat`: 用于提取 68 个面部关键点。
  - `mmod_human_face_detector.dat`: CNN 人脸检测器，用于检测人脸区域。
outputs
- **关键点文件**：
  - 路径：`{landmark_dir}/{视频文件名}.pkl` (`{landmark_dir} = {save_dir}/landmark`)
  - 内容：每一帧的 68 个面部关键点，存储为 NumPy 数组的列表。
    - 示例：
      ```python
      [
          [[x1, y1], [x2, y2], ..., [x68, y68]],  # 第 1 帧
          [[x1, y1], [x2, y2], ..., [x68, y68]],  # 第 2 帧
          ...
      ]
      ```

align_mouth.py
inputs
- `--video-direc`: 原始视频文件的目录。
- `--landmark-direc`: 面部关键点文件的目录。
- `--filename-path`: 包含视频文件名列表的清单文件，每一行是一个视频文件的相对路径。
- `--lip_video_dir`: 裁剪后嘴部视频的保存目录。
- `--face_preprocess_dir`: 存放面部对齐所需的文件目录，包含以下文件：
  - `20words_mean_face.npy`: 标准化面部关键点文件。
outputs
- **嘴部视频文件**：
  - 路径：`{lip_video_dir}/{视频文件名}.mp4` (`{lip_video_dir} = {save_dir}/video`)
  - 内容：裁剪后的嘴部区域视频。


construct_dataset.py
inputs
- `--root_dir`: 数据集的根目录，包含以下文件夹：
  - `encodec_16khz_4codebooks_`: 存放音频的 Encodec 编码。
  - `phonemes_`: 存放音频对应的音素序列。
outputs
1. **合并后的数据**：
   - **Encodec 编码文件**：
     - 路径：`{root_dir}/encodec_16khz_4codebooks/{新文件名}.txt`
     - 内容：合并后的 Encodec 编码。
   - **音素序列文件**：
     - 路径：`{root_dir}/phonemes/{新文件名}.txt`
     - 内容：合并后的音素序列。
2. **清单文件**：
   - 路径：`{root_dir}/manifest/train.txt`
   - 内容：描述数据集结构的清单文件。
3. **样本长度文件**：
   - 路径：`{root_dir}/split_len.txt`
   - 内容：记录每个样本的**音频**长度信息。


总结：
将数据重构成如下形式：
- phonemes: 保存当前训练文件的text/phoneme信息（reference + target）
- encodec_16khz_4codebooks: 保存当前训练文件的音频codebooks（reference + target）
- video: 保存所有提取到的lip的视频信息(没有concat，所有视频按照原本的名字存储)

- manifest/train.txt: 训练数据的list和实验对应的长度, e.g. `0	CGksAjzI0go__50011__50012	444`
- split_len:  当前实验文件的名字以及 audio split对应的音频位置. e.g. `00j9bKdiOjk__50001__50002.txt,292`

> 注意：为了得到validation和test数据集，我们将train.txt改名为train_origin.txt,然后用`src/data/utils/split_data.py`按照90%, 5%, 5%的比例进行划分，得到train， test， valid
> 得到的结果如下
Total samples: 530686
Saved train.txt: 477617 samples
Saved validation.txt: 26534 samples
Saved test.txt: 26535 samples

## Tokenizer (tokenizer.py)
该文件存放了模型所需的 Tokenizer 定义及工具接口：

- Audio Tokenizer:封装了 Encodec 模型，用于将音频波形转换为离散的 Codebook 索引，或将索引解码回波形。
- Text Tokenizer: 处理文本到音素的转换，以及音素到 ID 的映射（基于 vocab.txt）。