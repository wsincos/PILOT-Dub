# src/utils/utils.py
from __future__ import annotations

import ast
import operator as op
from typing import Any, Dict

import omegaconf
from omegaconf import OmegaConf

import torch

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval(expr: str) -> Any:
    """
    Safely evaluate a simple arithmetic expression.
    Supports numbers, + - * / // % **, and parentheses.
    Disallows names, attribute access, function calls, imports, etc.
    """
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        # numbers (py3.9)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value

        # unary ops: +x, -x
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(n.op)](_eval(n.operand))

        # binary ops: x + y, x * y, ...
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(n.op)](_eval(n.left), _eval(n.right))

        # parentheses are represented by AST structure already

        raise ValueError(f"Unsupported expression for eval resolver: {expr!r}")

    return _eval(node)


def _ensure_eval_resolver(enable_eval: bool = True, safe: bool = True) -> None:
    """
    Ensure OmegaConf has an 'eval' resolver registered.
    Should be called before OmegaConf.to_container(..., resolve=True).
    """
    if not enable_eval:
        return

    if OmegaConf.has_resolver("eval"):
        return

    if safe:
        OmegaConf.register_new_resolver("eval", lambda s: _safe_eval(str(s)))
    else:
        # 不推荐：真正的 Python eval（有安全风险）
        OmegaConf.register_new_resolver("eval", lambda s: eval(str(s)))

def dict_from_config(
    cfg: omegaconf.DictConfig,
    *,
    resolve: bool = True,
    enable_eval: bool = True,
    safe_eval: bool = True,
) -> Dict[str, Any]:
    """Convenience function to map an omegaconf configuration to a dictionary.

    Args:
        cfg (omegaconf.DictConfig): Original configuration to map to dict.
    Returns:
        dict: Config as dictionary object.
    """
    _ensure_eval_resolver(enable_eval=enable_eval, safe=safe_eval)
    return omegaconf.OmegaConf.to_container(cfg, resolve=True)

def select_cfg(cfg, key: str, default=None):
    """Safe select for nested keys: 'a.b.c'"""
    return omegaconf.OmegaConf.select(cfg, key, default=default)


def move_to_cuda(sample, device=None):
    """
    Recursively moves all Tensors in the sample to the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _move(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        elif isinstance(x, dict):
            return type(x)((k, _move(v)) for k, v in x.items())
        elif isinstance(x, (list, tuple, set)):
            return type(x)(_move(item) for item in x)
        else:
            return x

    return _move(sample)