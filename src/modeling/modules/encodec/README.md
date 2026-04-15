# Standalone EnCodec Module

A lightweight, self-contained EnCodec implementation extracted from Meta's audiocraft library.

## Features

- ✅ **No audiocraft dependency**: Completely standalone module
- ✅ **Minimal dependencies**: Only requires PyTorch, omegaconf, einops, numpy
- ✅ **Full functionality**: Supports model loading, encoding, and decoding
- ✅ **Compatible with VoiceCraft checkpoints**: Works with existing pretrained models

## Structure

```
src/encodec/
├── __init__.py           # Main exports
├── loader.py             # Checkpoint loading utilities
├── model.py              # CompressionModel and EncodecModel
├── builders.py           # Model builder functions
├── quantization/         # Quantization modules (RVQ, Dummy)
│   ├── __init__.py
│   ├── base.py
│   ├── vq.py
│   └── core_vq.py
└── modules/              # Neural network modules
    ├── __init__.py
    ├── conv.py           # Streamable convolutions
    ├── lstm.py           # Streamable LSTM
    ├── seanet.py         # SEANet encoder/decoder
    └── streaming.py      # Streaming utilities
```

## Usage

### Loading a Model

```python
from encodec import model_from_checkpoint

# Load model from checkpoint
model = model_from_checkpoint(
    'pretrained_models/encodec_4cb2048_giga.th',
    device='cpu'  # or 'cuda'
)

# Model properties
print(f"Sample rate: {model.sample_rate} Hz")
print(f"Channels: {model.channels}")
print(f"Num codebooks: {model.total_codebooks}")
```

### Encoding Audio

```python
import torch

# Input audio shape: (batch, channels, samples)
audio = torch.randn(1, 1, 16000)

# Encode to discrete codes
codes, scale = model.encode(audio)
# codes shape: (batch, num_codebooks, frames)
# scale: optional normalization scale
```

### Decoding Audio

```python
# Decode back to audio
reconstructed = model.decode(codes, scale)
# reconstructed shape: (batch, channels, samples)
```

## Integration with VoiceCraft

The tokenizer in `data/tokenizer.py` has been updated to use this module:

```python
from src.encodec import model_from_checkpoint

class AudioTokenizer:
    def __init__(self, device=None, signature=None):
        model = model_from_checkpoint(signature, device="cpu")
        self.sample_rate = model.sample_rate
        self.channels = model.channels
        # ...
```

## Testing

Run the test script to verify everything works:

```bash
python test_encodec.py
```

Expected output:
```
============================================================
✓✓✓ ALL TESTS PASSED ✓✓✓
============================================================
```

## Extracted Components

This module includes the following components from audiocraft:

- **Model**: `EncodecModel` (main compression model)
- **Quantizers**: `ResidualVectorQuantizer`, `DummyQuantizer`
- **Modules**: `SEANetEncoder`, `SEANetDecoder`, streamable convolutions and LSTM
- **Utilities**: Checkpoint loading and model building

## Dependencies

- `torch` - PyTorch framework
- `omegaconf` - Configuration management
- `einops` - Tensor operations
- `numpy` - Numerical operations

## License

This code is derived from Meta's audiocraft library. See original license in audiocraft source.
