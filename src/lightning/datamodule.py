import lightning as L
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Optional

from src.data.dubbing_dataset import dataset as DubbingDataset
from src.data.sampler import DistributedDynamicBatchSampler 

class StatefulDataLoader(DataLoader):
    def state_dict(self):
        sampler = getattr(self, "batch_sampler", None) or getattr(self, "sampler", None)
        if hasattr(sampler, "state_dict"):
            return sampler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        sampler = getattr(self, "batch_sampler", None) or getattr(self, "sampler", None)
        if hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state_dict)

class VoiceCraftDubDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Called before training starts, used for initializate dataset
        """
        if stage == "fit" or stage is None:
            self.train_dataset = DubbingDataset(split="train", cfg=self.cfg, **self.cfg.dataset)
            self.val_dataset = DubbingDataset(split="validation", cfg=self.cfg, **self.cfg.dataset)
        # TODO: test stage if needed

    def train_dataloader(self):
        """
        return train_dataset DataLoader
        """
        # 检查是否启用了 dynamic_batching (假设在 cfg 根目录下，或者 cfg.dataloader 下，请根据实际 yaml 调整)
        use_dynamic_batching = self.cfg.dataset.get("dynamic_batching", False)

        if use_dynamic_batching:
            # --- mode A: use Dynamic Batch Sampler ---
            # Lightning 的 Trainer 会自动注入 self.trainer，我们可以从中获取 world_size 和 rank
            sampler = DistributedDynamicBatchSampler(
                self.train_dataset,
                self.cfg.dataloader,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
                seed=self.cfg.seed,
                drop_last=True,
                lengths_list=self.train_dataset.lengths_list, # 确保 dataset 有这个属性
                verbose=True,
                epoch=self.trainer.current_epoch # 传入当前 epoch 以保证随机种子随 epoch 变化
            )
            return StatefulDataLoader(
                self.train_dataset,
                batch_sampler=sampler, # 注意：这里使用 batch_sampler
                num_workers=self.cfg.dataloader.num_workers, 
                collate_fn=self.train_dataset.collate,
                pin_memory=True,
                persistent_workers=True if self.cfg.dataloader.num_workers > 0 else False
            )
        else:
            # --- mode B: Standard Sampler ---
            # Lightning 会自动将 shuffle=True 转换为 DistributedSampler
            return StatefulDataLoader(
                self.train_dataset,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=True,
                num_workers=self.cfg.dataloader.num_workers, 
                collate_fn=self.train_dataset.collate,
                pin_memory=True,
                persistent_workers=True if self.cfg.dataloader.num_workers > 0 else False,
                drop_last=True
            )

    def val_dataloader(self):
        """
        return validation_dataset DataLoader
        """
        use_dynamic_batching = self.cfg.dataset.get("dynamic_batching", False)

        if use_dynamic_batching:
            sampler = DistributedDynamicBatchSampler(
                self.val_dataset,
                self.cfg.dataloader,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True, # shuffle for bucketing
                seed=self.cfg.seed,
                drop_last=True,
                lengths_list=self.val_dataset.lengths_list,
                verbose=True,
                epoch=self.trainer.current_epoch
            )
            return StatefulDataLoader(
                self.val_dataset,
                batch_sampler=sampler,
                num_workers=self.cfg.dataloader.num_workers,
                collate_fn=self.val_dataset.collate,
                pin_memory=True,
                persistent_workers=True if self.cfg.dataloader.num_workers > 0 else False
            )
        else:
            return StatefulDataLoader(
                self.val_dataset,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=False,
                num_workers=self.cfg.dataloader.num_workers,
                collate_fn=self.val_dataset.collate,
                pin_memory=True,
                persistent_workers=True if self.cfg.dataloader.num_workers > 0 else False,
                drop_last=False
            )
