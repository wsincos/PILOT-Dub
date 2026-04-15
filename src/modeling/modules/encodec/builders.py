# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal builders for EnCodec model.
Simplified from audiocraft to remove unnecessary dependencies.
"""

import typing as tp
import omegaconf
import torch

from .model import CompressionModel, EncodecModel
from . import quantization as qt
from . import modules


def dict_from_config(cfg: tp.Any) -> dict:
    """Convert omegaconf config to dict."""
    if isinstance(cfg, omegaconf.DictConfig):
        return omegaconf.OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def get_quantizer(quantizer: str, cfg: omegaconf.DictConfig, dimension: int) -> qt.BaseQuantizer:
    """Build quantizer from config."""
    klass = {
        'no_quant': qt.DummyQuantizer,
        'rvq': qt.ResidualVectorQuantizer
    }[quantizer]
    kwargs = dict_from_config(getattr(cfg, quantizer))
    if quantizer != 'no_quant':
        kwargs['dimension'] = dimension
    return klass(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    """Build encoder and decoder."""
    if encoder_name == 'seanet':
        kwargs = dict_from_config(getattr(cfg, 'seanet'))
        encoder_override_kwargs = kwargs.pop('encoder')
        decoder_override_kwargs = kwargs.pop('decoder')
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        encoder = modules.SEANetEncoder(**encoder_kwargs)
        decoder = modules.SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected encoder {encoder_name}")


def get_compression_model(cfg: omegaconf.DictConfig) -> CompressionModel:
    """Build compression model from config."""
    if cfg.compression_model == 'encodec':
        kwargs = dict_from_config(getattr(cfg, 'encodec'))
        encoder_name = kwargs.pop('autoencoder')
        quantizer_name = kwargs.pop('quantizer')
        encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
        quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
        frame_rate = kwargs['sample_rate'] // encoder.hop_length
        renormalize = kwargs.pop('renormalize', False)
        kwargs.pop('renorm', None)
        return EncodecModel(
            encoder, decoder, quantizer,
            frame_rate=frame_rate, 
            renormalize=renormalize, 
            **kwargs
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")
