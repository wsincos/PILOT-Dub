import copy
import numbers
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .activation import MultiheadAttention
from .scaling import ActivationBalancer, BalancedDoubleSwish
from .scaling import BasicNorm as _BasicNorm

_shape_t = Union[int, List[int], torch.Size]


class RMSNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            output = self._rms_norm(input)
            return (output, embedding)

        assert embedding is None
        return self._rms_norm(input)

    def _rms_norm(self, x: Tensor) -> Tensor:
        rms = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


class BasicNorm(_BasicNorm):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super(BasicNorm, self).__init__(d_model, eps=eps)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                super(BasicNorm, self).forward(input),
                embedding,
            )

        assert embedding is None
        return super(BasicNorm, self).forward(input)


class BalancedBasicNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super(BalancedBasicNorm, self).__init__()
        self.balancer = ActivationBalancer(
            d_model,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.norm = BasicNorm(d_model, eps, device=device, dtype=dtype)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return self.norm((self.balancer(input), embedding))

        assert embedding is None
        return self.norm(self.balancer(input))


class IdentityNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super(IdentityNorm, self).__init__()

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            return input

        assert embedding is None
        return input


def _normalize_segment_boundaries(
    segment_boundaries: Optional[Any],
    batch_size: int,
    seq_len: int,
) -> List[List[Tuple[int, int]]]:
    if segment_boundaries is None:
        return [[(0, seq_len)] for _ in range(batch_size)]

    if isinstance(segment_boundaries, torch.Tensor):
        if segment_boundaries.dim() != 3 or segment_boundaries.size(-1) != 2:
            raise ValueError(
                "segment_boundaries must be [B, S, 2] when provided as Tensor"
            )
        boundaries = segment_boundaries.detach().cpu().tolist()
    else:
        boundaries = segment_boundaries

    result: List[List[Tuple[int, int]]] = []
    for b in range(batch_size):
        segs = []
        for start, end in boundaries[b]:
            start_i = max(0, min(seq_len, int(start)))
            end_i = max(start_i, min(seq_len, int(end)))
            if end_i > start_i:
                segs.append((start_i, end_i))
        if not segs:
            segs = [(0, seq_len)]
        result.append(segs)
    return result


def _unpack_src_mask(
    src_mask: Optional[Any],
) -> Tuple[Optional[Tensor], Optional[Any]]:
    if src_mask is None:
        return None, None
    if isinstance(src_mask, tuple) and len(src_mask) == 2:
        return src_mask[0], src_mask[1]
    if isinstance(src_mask, dict):
        return src_mask.get("attn_mask"), src_mask.get("segment_boundaries")
    return src_mask, None


class CausalDepthwiseConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        groups: Optional[int] = None,
        bias: bool = True,
        pointwise: bool = True,
        pointwise_groups: int = 1,
    ) -> None:
        super(CausalDepthwiseConv1d, self).__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.kernel_size = kernel_size
        self.groups = channels if groups is None else groups
        if pointwise_groups < 1:
            raise ValueError("pointwise_groups must be >= 1")
        if pointwise and channels % pointwise_groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by pointwise_groups ({pointwise_groups})"
            )
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=self.groups,
            bias=bias,
        )
        self.pointwise = (
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                bias=bias,
                groups=pointwise_groups,
            )
            if pointwise
            else None
        )

    def forward(
        self,
        x: Tensor,
        segment_boundaries: Optional[Any] = None,
    ) -> Tensor:
        # x: [B, T, C]
        bsz, seq_len, channels = x.shape
        x_t = x.transpose(1, 2)  # [B, C, T]
        out = x.new_zeros(bsz, seq_len, channels)
        boundaries = _normalize_segment_boundaries(segment_boundaries, bsz, seq_len)

        for b in range(bsz):
            for start, end in boundaries[b]:
                if end <= start:
                    continue
                seg = x_t[b : b + 1, :, start:end]
                seg = F.pad(seg, (self.kernel_size - 1, 0))
                seg = self.depthwise(seg)
                if self.pointwise is not None:
                    seg = self.pointwise(seg)
                out[b, start:end, :] = seg.transpose(1, 2)[0]

        return out


class ConvSwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        kernel_size: int = 5,
        conv_groups: Optional[int] = None,
        conv_bias: bool = True,
        conv_pointwise: bool = True,
        conv_pointwise_groups: int = 1,
        linear1_cls: nn.Module = nn.Linear,
        linear2_cls: nn.Module = nn.Linear,
    ) -> None:
        super(ConvSwiGLU, self).__init__()
        self.w_gate = linear1_cls(d_model, d_ff)
        self.w_value = linear1_cls(d_model, d_ff)
        self.conv = CausalDepthwiseConv1d(
            d_ff,
            kernel_size=kernel_size,
            groups=conv_groups,
            bias=conv_bias,
            pointwise=conv_pointwise,
            pointwise_groups=conv_pointwise_groups,
        )
        self.w_out = linear2_cls(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        segment_boundaries: Optional[Any] = None,
    ) -> Tensor:
        gate = self.w_gate(x)
        gate = self.conv(gate, segment_boundaries)
        gate = F.silu(gate)
        value = self.w_value(x)
        out = gate * value
        out = self.dropout(out)
        out = self.w_out(out)
        return out


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        linear1_self_attention_cls: nn.Module = nn.Linear,
        linear2_self_attention_cls: nn.Module = nn.Linear,
        linear1_feedforward_cls: nn.Module = nn.Linear,
        linear2_feedforward_cls: nn.Module = nn.Linear,
        layer_norm_cls: nn.Module = RMSNorm,
        layer_norm_eps: float = 1e-5,
        adaptive_layer_norm: bool = False,
        conv_kernel_size: int = 5,
        conv_groups: Optional[int] = None,
        conv_bias: bool = True,
        conv_pointwise: bool = True,
        conv_pointwise_groups: int = 1,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )

        # Standard Transformer FFN (no convolution)
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)
        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if layer_norm_cls == IdentityNorm:
            norm2 = BalancedBasicNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
        else:
            norm2 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Any] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        need_weights: Optional[bool] = False,
        past: Optional[Tensor] = None,
    ) -> Tensor:
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        src_mask, segment_boundaries = _unpack_src_mask(src_mask)

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if need_weights:
            if self.norm_first:
                out, attn = self._sa_block_attn(
                    self.norm1(x, stage_embedding),
                    src_mask,
                    src_key_padding_mask,
                    past,
                )
                out, present = out
                x = x + out
                x = x + self._ff_block(self.norm2(x, stage_embedding))
            else:
                out, attn = self._sa_block_attn(
                    x, src_mask, src_key_padding_mask, past
                )
                out, present = out
                x = self.norm1(x + out, stage_embedding)
                x = self.norm2(x + self._ff_block(x), stage_embedding)
            assert not is_src_tuple
            return (x, attn)
        else:
            if self.norm_first:
                out = self._sa_block(
                    self.norm1(x, stage_embedding),
                    src_mask,
                    src_key_padding_mask,
                    past,
                )
                out, present = out
                x = x + out
                x = x + self._ff_block(self.norm2(x, stage_embedding))
            else:
                out = self._sa_block(x, src_mask, src_key_padding_mask)
                out, present = out
                x = self.norm1(x + out, stage_embedding)
                x = self.norm2(x + self._ff_block(x), stage_embedding)

            if is_src_tuple:
                x = (x, stage_embedding)
            if present is not None:
                x = [x, present]
            return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        past: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            past=past,
        )
        x, present = x
        return self.dropout1(x), present

    def _sa_block_attn(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        past: Optional[Tensor] = None,
    ) -> Tensor:
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            past=past,
        )
        x, present = x
        return (self.dropout1(x), present), attn

    def _ff_block(
        self, x: Tensor
    ) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Any] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_layer_states: bool = False,
        need_weights: Optional[bool] = False,
        past: Optional[Tensor] = None,
    ) -> Tensor:
        if return_layer_states:
            assert not need_weights
            layer_states = []
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    past=past,
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output

        if need_weights:
            assert not return_layer_states
            layer_attn = []
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    past=past,
                )
                layer_attn.append(output[1])

            if self.norm is not None:
                output = self.norm(output)

            return layer_attn, output

        output = src
        all_present = []
        for n_layer, mod in enumerate(self.layers):
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                past=None if past is None else past[n_layer],
            )
            if isinstance(output, list):
                output, present = output
                all_present.append(present)

        if self.norm is not None:
            output = self.norm(output)
        if all_present != []:
            all_present = torch.stack(all_present, dim=0)
            output = [output, all_present]
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )
