# src/models/builders.py
import torch
import torch.nn as nn
import omegaconf

from .voicecraft import VoiceCraftModel
from ..modules.transformer.transformer import LayerNorm, TransformerEncoder, TransformerEncoderLayer
from ..modules.embeddings import (
    LearnedPositionalEmbedding, 
    RoPEPositionalEmbedding,
    SinePositionalEmbedding, 
    TokenEmbedding)
from ..modules.codebooks_patterns import (
    CodebooksPatternProvider,
    DelayedPatternProvider,
    MusicLMPattern,
    ParallelPatternProvider,
    UnrolledPatternProvider,
    VALLEPattern,
)
from ..modules.data_processors import VoiceCraftDataProcessor
from ..utils.utils import dict_from_config


def build_text_embedding(text_embedding_name, cfg: omegaconf.DictConfig) -> nn.Module:
    """Instantiate token embedding layers for codebooks."""
    if text_embedding_name == "text_token_embedding":
        text_embedding_cfg = dict_from_config(getattr(cfg, "text_token_embedding"))
        text_embedding = TokenEmbedding(**text_embedding_cfg)
        return text_embedding
    else:
        raise ValueError(f"Unexpected text embedding: {text_embedding_name}")
    
def build_audio_embedding(audio_embedding_name, cfg: omegaconf.DictConfig) -> nn.ModuleList:
    """Instantiate audio embedding layers for codebooks."""
    if audio_embedding_name == "audio_token_embeddings":
        audio_embedding_cfg = dict_from_config(getattr(cfg, "audio_token_embedding"))

        n_codebooks = audio_embedding_cfg.pop("n_codebooks")
        vocab_size = cfg.audio_vocab_size + cfg.n_special
        n_audio_tokens = [vocab_size] * n_codebooks # codebooks sizes

        audio_embedding = nn.ModuleList(
            TokenEmbedding(
                vocab_size=n_audio_tokens[k],
                **audio_embedding_cfg
            ) for k in range(n_codebooks)
        )
        return audio_embedding, n_audio_tokens
    else:
        raise ValueError(f"Unexpected audio embeddings: {audio_embedding_name}")
    
def build_positional_embeddings(positional_embedding_name, cfg: omegaconf.DictConfig) -> nn.Module:
    """Instantiate positional embedding layers."""
    if hasattr(cfg, positional_embedding_name):
        positional_embedding_cfg = dict_from_config(getattr(cfg, positional_embedding_name))
        klass = {
            "sine": SinePositionalEmbedding,
            "learned": LearnedPositionalEmbedding,
            "rope": RoPEPositionalEmbedding
        }[positional_embedding_cfg.pop("type")]

        return klass(**positional_embedding_cfg)
    else:
        raise ValueError(f"Unexpected positional embedding: {positional_embedding_name}")

def build_codebooks_pattern_provider(pattern_provider_name, cfg: omegaconf.DictConfig) -> CodebooksPatternProvider:
    """Instantiate a codebooks pattern provider object."""
    if hasattr(cfg, pattern_provider_name):
        pattern_provider_cfg = dict_from_config(getattr(cfg, pattern_provider_name))
        klass = {
            'parallel': ParallelPatternProvider,
            'delay': DelayedPatternProvider,
            'unroll': UnrolledPatternProvider,
            'valle': VALLEPattern,
            'musiclm': MusicLMPattern,
        }[pattern_provider_cfg.pop("type")]
        return klass(**pattern_provider_cfg)
    else:
        raise ValueError(f"Unexpected pattern provider: {pattern_provider_name}")

def build_decoder(decoder_name, cfg: omegaconf.DictConfig) -> TransformerEncoder:
    """Instantiate a transformer encoder."""
    if decoder_name == "voicecraft_decoder":
        transformer_cfg = dict_from_config(getattr(cfg, decoder_name))

        layer_name = transformer_cfg.pop("layer_cls")
        assert hasattr(cfg, layer_name), f"Layer config {layer_name} not found in cfg."
        layer_cfg = dict_from_config(getattr(cfg, layer_name))
        if layer_cfg.pop("layer_norm_type") == "layer_norm":
            layer_cfg["layer_norm_cls"] = LayerNorm
        else:
            raise ValueError(f"Unexpected layer norm type in {layer_name}")
        layer = TransformerEncoderLayer(**layer_cfg)

        if transformer_cfg.pop("transformer_norm_type") == "layer_norm":
            transformer_cfg["norm"] = LayerNorm(transformer_cfg.pop("norm_dim"))
        return TransformerEncoder(layer, **transformer_cfg)
    else:
        raise ValueError(f"Unexpected transformer: {decoder_name}")
    
def build_predict_layer(predict_layer_name, n_audio_tokens, cfg: omegaconf.DictConfig) -> nn.ModuleList:
    """Instantiate prediction heads for the model."""
    if predict_layer_name == "voicecraft_predict_layer":
        predict_layer = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(cfg.d_model, cfg.audio_vocab_size//2), 
                              nn.GELU(), 
                              nn.Linear(cfg.audio_vocab_size//2, n_audio_tokens[k])
                             ) for k in range(cfg.n_codebooks)
            ]
        )
        return predict_layer
    else:
        raise ValueError(f"Unexpected predict heads config.")

def build_data_processor(processor_name, cfg: omegaconf.DictConfig):
    if processor_name == "voicecraft_data_processor":
        processor_cfg = dict_from_config(getattr(cfg, processor_name))
        pattern_provider_name = processor_cfg.pop("pattern_provider")
        processor_cfg["pattern_provider"] = build_codebooks_pattern_provider(pattern_provider_name, cfg)
        return VoiceCraftDataProcessor(**processor_cfg)
    else:
        raise ValueError(f"Unexpected data processor: {processor_name}")

def get_voicecraft_model(cfg: omegaconf.DictConfig) -> VoiceCraftModel:
    """Instantiate VoiceCraftModel from config."""
    if cfg.tts_model == "voicecraft":
        kwargs = dict_from_config(getattr(cfg, "voicecraft"))
        text_embedding_name = kwargs.pop("text_embedding")
        audio_embedding_name = kwargs.pop("audio_embedding")
        text_positional_embedding_name = kwargs.pop("text_positional_embedding")
        audio_positional_embedding_name = kwargs.pop("audio_positional_embedding")
        decoder_name = kwargs.pop("decoder")
        predict_layer_name = kwargs.pop("predict_layer")
        processor_name = kwargs.pop("processor")

        text_embedding = build_text_embedding(text_embedding_name, cfg)
        audio_embedding, n_audio_tokens = build_audio_embedding(audio_embedding_name, cfg)
        text_positional_embedding = build_positional_embeddings(text_positional_embedding_name, cfg)
        audio_positional_embedding = build_positional_embeddings(audio_positional_embedding_name, cfg)
        mask_embedding = nn.Parameter(torch.randn(cfg.max_n_spans, cfg.d_model), requires_grad=True)
        decoder = build_decoder(decoder_name, cfg)
        predict_layer = build_predict_layer(predict_layer_name, n_audio_tokens, cfg)
        processor = build_data_processor(processor_name, cfg)

        return VoiceCraftModel(
            text_embedding=text_embedding,
            audio_embedding=audio_embedding,
            text_positional_embedding=text_positional_embedding,
            audio_positional_embedding=audio_positional_embedding,
            mask_embedding=mask_embedding,
            decoder=decoder,
            predict_layer=predict_layer,
            processor=processor,
        )
        
    else:
        raise ValueError(f"Unexpected TTS model: {cfg.tts_model}")
