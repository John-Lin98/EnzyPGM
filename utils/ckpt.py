# utils/ckpt.py
import sys
sys.path.append("..")
import os
import re
import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer


_EPOCH_RE = re.compile(r".*?(?:epoch)?(\d+)\.(?:pt|pth|ckpt)$", re.IGNORECASE)


def _unwrap_model(model: Module) -> Module:
    """拿到真实 nn.Module（兼容 DDP / DataParallel）"""
    return getattr(model, "module", model)


def _normalize_ckpt_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    if p.is_dir():
        return p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")
    return p


def _parse_epoch_from_name(name: str) -> Optional[int]:
    m = _EPOCH_RE.match(name)
    return int(m.group(1)) if m else None


def save_checkpoint(
    ckpt_dir: Union[str, Path],
    model: Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    保存 checkpoint。
    - ckpt_dir: 保存目录
    - filename: 自定义文件名；若不提供，则使用 "epoch{epoch:05d}.pt" 或 "latest.pt"
    - extra: 额外信息（如 config、metrics）
    返回：实际保存的路径
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = _unwrap_model(model)
    state = {
        "model": model_to_save.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state["extra"] = extra

    if filename is None:
        if epoch is not None:
            filename = f"epoch{epoch:05d}.pt"
        else:
            filename = "latest.pt"

    path = ckpt_dir / filename
    torch.save(state, path)
    return path


def find_latest_checkpoint(
    ckpt_dir: Union[str, Path],
    pattern: str = "*.pt",
    prefer_epoch_number: bool = True,
) -> Optional[Path]:
    """
    在目录中寻找“最新”的 ckpt。
    - prefer_epoch_number=True：优先用文件名中的 epoch 数字最大者；找不到就按 mtime。
    - 返回 Path 或 None（若目录为空）
    """
    ckpt_dir = Path(ckpt_dir)
    files = sorted(ckpt_dir.glob(pattern))
    if not files:
        return None

    if prefer_epoch_number:
        parsed = [(f, _parse_epoch_from_name(f.name)) for f in files]
        with_num = [p for p in parsed if p[1] is not None]
        if with_num:
            with_num.sort(key=lambda x: x[1])  # 按 epoch 升序
            return with_num[-1][0]

    # 兜底：按修改时间
    files.sort(key=lambda f: f.stat().st_mtime)
    return files[-1]


def load_for_resume(
    model: Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[Any],
    ckpt_dir_or_file: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[Optional[int], Optional[int], Optional[Path]]:
    """
    恢复训练：
    - 若传入目录：自动寻找最新 ckpt
    - 若传入具体文件：直接加载
    - 返回: (epoch, step, used_path)
      * epoch/step 可能为 None（取决于保存时是否写入）
    """
    p = _normalize_ckpt_path(ckpt_dir_or_file)
    if p.is_dir():
        ckpt_path = find_latest_checkpoint(p)
        if ckpt_path is None:
            return None, None, None
    else:
        ckpt_path = p

    ckpt = torch.load(ckpt_path, map_location=map_location)

    # 模型
    _unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)

    # 优化器/调度器
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            # 优化器不匹配时可忽略，让用户决定是否继续
            print(f"[ckpt] optimizer state load failed: {e}")

    if scheduler is not None and "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[ckpt] scheduler state load failed: {e}")

    epoch = ckpt.get("epoch", None)
    step = ckpt.get("step", None)
    return epoch, step, Path(ckpt_path)


def load_weights_for_inference(
    model: Module,
    ckpt_file: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Path:
    """
    推理/生成阶段加载权重（只加载 model.state_dict）。
    返回：实际加载的 ckpt 路径
    """
    ckpt_file = Path(ckpt_file)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location=map_location)
    _unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)
    return ckpt_file
