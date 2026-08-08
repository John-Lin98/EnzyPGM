import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from pathlib import Path
from typing import Union

import esm
import logging

logger = logging.getLogger(__name__)


def _as_dict_like(cfg: Union[dict, object]):
    """支持 dict / argparse.Namespace / 任意属性对象的统一取值。"""
    if isinstance(cfg, dict):
        return cfg
    class _DictLike:
        def __init__(self, o): self.o = o
        def get(self, k, default=None): return getattr(self.o, k, default)
    return _DictLike(cfg)

# RFFLayer.py
import os, re, torch, esm
from types import SimpleNamespace

def _infer_arch_from_any(ckpt_path: str, md: dict, fallback: str) -> str:
    # 1) ckpt 内部
    if isinstance(md, dict):
        args = md.get("args", None)
        if isinstance(args, dict) and args.get("arch"):
            return args["arch"]
        if hasattr(args, "arch") and args.arch:
            return args.arch
        cfg = md.get("cfg", None)
        if isinstance(cfg, dict) and cfg.get("arch"):
            return cfg["arch"]
        if hasattr(cfg, "arch") and cfg.arch:
            return cfg.arch
    # 2) 文件名
    m = re.search(r"(esm2_[tT]\d+_[^_/]+_UR50[DS])", os.path.basename(ckpt_path))
    if m:
        return m.group(1)
    # 3) 兜底
    return fallback

def load_esm_from_local(cfg, device, eval_mode=True):
    ckpt_path = cfg.get("esm_ckpt") or cfg.get("esm_path")
    fallback  = cfg.get("esm_model", "esm2_t33_650M_UR50D")
    assert ckpt_path and os.path.isfile(ckpt_path), f"ckpt 不存在: {ckpt_path}"

    # 只从本地加载，不联网
    md = torch.load(ckpt_path, map_location="cpu")
    arch = _infer_arch_from_any(ckpt_path, md, fallback)

    # 关键一步：确保 model_data['args'].arch 存在（供 v1 loader 使用）
    args = md.get("args", None)
    if isinstance(args, dict):
        if not args.get("arch"):
            args["arch"] = arch
        # 转为支持属性访问的对象
        md["args"] = SimpleNamespace(**args)
    elif hasattr(args, "arch"):
        if not getattr(args, "arch", None):
            args.arch = arch
    else:
        md["args"] = SimpleNamespace(arch=arch)

    # 调官方核心加载（不会下载，因为我们提供了 model_data）
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        model_name=arch, model_data=md
    )

    model = model.to(device)
    if eval_mode:
        model.eval()
    return model, alphabet

class ResidueFunctionFusionLayer(nn.Module):
    """
    ResidueFunctionFusionLayer (RFFL, 分层前向版)
    - 仅负责：ESM2 的逐层前向（不再计算/持有 EC 的 embedding；EC 条件请在 NAELayer 中计算好后，通过 cond_add 传入）
    - forward 可指定 layer_idx，只执行该层一次前向；若是最后一层，会自动做 emb_layer_norm_after 与 lm_head 得到 logits/prob

    输入：
      src_tokens : LongTensor [B, L]
      src_lengths: LongTensor [B]              （仅用于 token_dropout 比例估计）
      coors      : FloatTensor [B, L, 3]       （接口占位，本层不使用）
      mask       : Bool/LongTensor [B, L]      （1 表示该位点被 mask，替换为 [MASK] token）
      layer_idx  : int                         （执行 encoder.layers[layer_idx] 这一层）
      x_in       : Optional[Tensor]            （[B, L, E]，上一层输出，若提供则跳过 embedding 与 token_dropout）
      cond_add   : Optional[Tensor]            （[B,1,E] 或 [B,L,E]，外部条件，如 NAELayer 里算好的 EC 条件向量）

    输出（与单层推进相匹配）：
      {
        "hidden"           : FloatTensor [B, L, E]     # 本层输出（若为最后一层则已过 emb_layer_norm_after）
        "logits"           : Optional[FloatTensor]     # 仅当 layer_idx 为最后一层提供 [B, L, V]
        "prob"             : Optional[FloatTensor]     # 仅当 layer_idx 为最后一层提供 [B, L, V]
        "padding_mask"     : BoolTensor  [B, L]        # True 为 padding
        "encoder_embedding": FloatTensor [B, L, E]     # 送入该层前的输入（embedding/或 x_in 加上 cond_add 与 dropout 后）
        "layer_idx"        : int
      }
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.device = torch.device(cfg.get('device', 'cpu'))

        # 使用官方 esm 包从本地权重加载
        self.encoder, self.alphabet = load_esm_from_local(cfg, self.device, eval_mode=False)
        self.encoder.to(self.device)

        # print('alphabet', self.alphabet)

        # print('layer num. : ', len(self.encoder.layers))

        # 缓存常用属性
        self.mask_idx     = self.encoder.mask_idx
        self.padding_idx  = self.encoder.padding_idx
        self.num_layers   = len(self.encoder.layers)
        self.embed_dim    = self.encoder.embed_tokens.embedding_dim
        self.vocab_size   = self.encoder.lm_head.weight.size(0) if hasattr(self.encoder, 'lm_head') else None

    @torch.no_grad()
    def _apply_token_dropout_rescale(self, x, tokens, padding_mask):
        """
        与 ESM 中 token dropout 的缩放保持一致：
        (1 - 0.15*0.8) / (1 - mask_ratio_observed)
        仅在我们自己做了 embedding（即 x_in 为 None）时应用。
        """
        mask_token = (tokens == self.encoder.mask_idx).unsqueeze(-1)
        x.masked_fill_(mask_token, 0.0)
        src_lengths = (~padding_mask).sum(-1)
        mask_ratio_observed = (tokens == self.encoder.mask_idx).sum(-1).to(x.dtype) / src_lengths.clamp_min(1)
        factor = (1 - 0.15 * 0.8) / (1 - mask_ratio_observed).clamp(min=1e-6)
        x.mul_(factor[:, None, None])
        return x

    def forward(
        self,
        src_tokens: torch.LongTensor,
        src_lengths: torch.LongTensor,
        coors: torch.Tensor,                  # 未使用，占位保持签名一致
        mask: torch.Tensor,
        *,
        layer_idx: int,                       # 指定仅执行的编码层索引
        x_in: Optional[torch.Tensor] = None,  # 若提供，则视为上一层输出，跳过 embedding 与 token_dropout
        cond_add: Optional[torch.Tensor] = None,  # 外部条件（如 NAELayer 中的 EC 条件），会加到送入该层前的 x 上
        padding_mask = None
    ) -> Dict[str, torch.Tensor]:

        src_tokens = src_tokens.to(self.device)
        mask       = mask.to(self.device)

        # padding mask：True 表示 padding
        # padding_mask = src_tokens.eq(self.padding_idx)  # [B, L]

        # 若未提供上一层输出，则从 tokens 走一次 embedding（并做 mask 替换/重标定）
        if x_in is None:
            # 将掩码位点替换为 [MASK] token（1=mask）
            tokens = (mask * self.mask_idx + (mask != 1) * src_tokens).long()  # [B, L]
            x = self.encoder.embed_scale * self.encoder.embed_tokens(tokens)   # [B, L, E]

            # token dropout（若启用）
            if getattr(self.encoder, "token_dropout", False):
                x = self._apply_token_dropout_rescale(x, tokens, padding_mask)
        else:
            # 使用外部传入的上一层输出
            x = x_in.to(self.device)  # [B, L, E]

        # 叠加外部条件（例如 NAELayer 中求好的 EC 条件）：
        # 支持 [B,1,E] 或 [B,L,E]，广播到序列维
        if cond_add is not None:
            if cond_add.dim() == 3 and cond_add.size(1) in (1, x.size(1)):
                x = x + cond_add.to(x.dtype).to(self.device)
            else:
                raise ValueError(f"cond_add shape must be [B,1,E] or [B,L,E], got {tuple(cond_add.shape)}")

        # 将 padding 位置置零（与 ESM 习惯一致）
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))  # [B, L, E]

        # 进入指定的 transformer 层：ESM 期望 (T,B,E)
        x_layer_in = x.transpose(0, 1)                            # (B,L,E)->(L,B,E)
        attn_mask  = None if not padding_mask.any() else padding_mask  # [B, L]

        # 边界检查
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"layer_idx out of range: {layer_idx} (num_layers={self.num_layers})")

        # 仅执行这一层
        x_layer_out, _ = self.encoder.layers[layer_idx](
            x_layer_in,
            self_attn_padding_mask=attn_mask,
            need_head_weights=False,
        )  # [L,B,E]

        # 若该层是最后一层，则追加 emb_layer_norm_after 并计算 logits/prob
        # is_last = (layer_idx == self.num_layers - 1)
        # if is_last:
        #     x_norm = self.encoder.emb_layer_norm_after(x_layer_out)  # [L,B,E]
        #     x_out  = x_norm.transpose(0, 1)                          # -> [B,L,E]
        #     logits = self.encoder.lm_head(x_out)                     # [B,L,V]
        #     prob   = F.softmax(logits, dim=-1)                       # [B,L,V]
        # else:
        x_out  = x_layer_out.transpose(0, 1)                     # -> [B,L,E]
        logits = None
        prob   = None

        # encoder_embedding：指送入本层前的表示（包含 cond_add 与 dropout 后），便于下游融合
        encoder_embedding = x                                            # [B,L,E]

        out: Dict[str, torch.Tensor] = {
            "hidden": x_out,                          # 本层输出 [B,L,E]
            "logits": logits,                         # 若为最后一层则给出 [B,L,V]，否则为 None
            "prob": prob,                             # 若为最后一层则给出 [B,L,V]，否则为 None
            "padding_mask": padding_mask,             # [B,L] True 为 padding
            "encoder_embedding": encoder_embedding,   # 送入该层前的输入表示
            "layer_idx": torch.tensor(layer_idx, device=x_out.device),
        }
        return out
    def emb_layer_norm_after_lm_head(self, x_layer_out):

        # print('x_layer_out : ' , x_layer_out.shape)
        x_norm = self.encoder.emb_layer_norm_after(x_layer_out.transpose(0, 1))  # [L,B,E]
        # print('x_norm : ' , x_norm.shape)
        x_out  = x_norm.transpose(0, 1)                        # -> [B,L,E]
        # print('x_out : ' , x_out.shape)
        logits = self.encoder.lm_head(x_out)                    # [B,L,V]
        prob   = F.softmax(logits, dim=-1)                       # [B,L,V]

        return x_out, logits, prob







# class ResidueFunctionFusionLayer(nn.Module):
#     """
#     ResidueFunctionFusionLayer
#     --------------------------
#     这是把 ESM2 编码器（语言侧）与四级 EC 嵌入做“逐残基融合”的一层封装。
#     - 输入：
#         src_tokens : LongTensor [B, L]
#         src_lengths: LongTensor [B]（可不严格依赖，仅与 token_dropout 的比例计算有关）
#         coors      : FloatTensor [B, L, 3]（接口占位，本层不使用）
#         mask       : LongTensor/BoolTensor [B, L]，1 表示该位点被 mask（将替换为 [MASK] token）
#         ec1..ec4   : LongTensor [B]，四级 EC 标签（样本级，全局条件，会广播到序列各位点）
#     - 输出：
#         {
#           "hidden"          : FloatTensor [B, L, E]      # 最终层（经 emb_layer_norm_after）的残基表示
#           "logits"          : FloatTensor [B, L, V]      # 经过 lm_head 的词表 logits
#           "prob"            : FloatTensor [B, L, V]      # softmax 后的氨基酸分布
#           "padding_mask"    : BoolTensor  [B, L]         # True 为 padding
#           "encoder_embedding": FloatTensor [B, L, E]     # 进入 transformer 前的嵌入（含 EC 融合）
#         }
#     备注：
#       - EC 融合方式为“逐位点相加”（将样本级 EC 嵌入广播到序列长度维）。
#       - token_dropout 与 ESM 原实现保持一致。
#     """

#     def __init__(
#         self,
#         cfg: dict,
#     ):
#         super().__init__()
#         self.device = cfg.get('device', 'cpu')

#         # 加载 ESM2 及其 Alphabet/权重（复用你工程里的函数）
#         self.encoder, self.alphabet = load_esm_from_local(cfg, self.device)
#         self.encoder.to(self.device)

#         # 从 encoder 拿 embedding 维度（与 ESM2 配置一致）
#         embed_dim = self.encoder.embed_tokens.embedding_dim

#         # 四级 EC 嵌入（样本级，全局条件）
#         ec_vocab_sizes = cfg['ec_vocab_sizes']

#         ec1_size, ec2_size, ec3_size, ec4_size = ec_vocab_sizes
#         self.ec1_embeddings = nn.Embedding(ec1_size, embed_dim)
#         self.ec2_embeddings = nn.Embedding(ec2_size, embed_dim)
#         self.ec3_embeddings = nn.Embedding(ec3_size, embed_dim)
#         self.ec4_embeddings = nn.Embedding(ec4_size, embed_dim)
#         nn.init.normal_(self.ec1_embeddings.weight, mean=0.0, std=embed_dim ** -0.5)
#         nn.init.normal_(self.ec2_embeddings.weight, mean=0.0, std=embed_dim ** -0.5)
#         nn.init.normal_(self.ec3_embeddings.weight, mean=0.0, std=embed_dim ** -0.5)
#         nn.init.normal_(self.ec4_embeddings.weight, mean=0.0, std=embed_dim ** -0.5)

#         # 方便外部使用
#         self.mask_idx = self.encoder.mask_idx
#         self.padding_idx = self.encoder.padding_idx
#         self.num_layers = len(self.encoder.layers)

#     @torch.no_grad()
#     def _apply_token_dropout_rescale(self, x, tokens, padding_mask):
#         """
#         与 ESM 中 token dropout 的缩放保持一致：
#         (1 - 0.15*0.8) / (1 - mask_ratio_observed)
#         """
#         mask_token = (tokens == self.encoder.mask_idx).unsqueeze(-1)
#         x.masked_fill_(mask_token, 0.0)

#         # src_lengths：非 padding 的计数
#         src_lengths = (~padding_mask).sum(-1)
#         # 实际 mask 比例
#         mask_ratio_observed = (tokens == self.encoder.mask_idx).sum(-1).to(x.dtype) / src_lengths.clamp_min(1)
#         factor = (1 - 0.15 * 0.8) / (1 - mask_ratio_observed).clamp(min=1e-6)
#         # 广播到 [B, 1, 1]
#         x.mul_(factor[:, None, None])
#         return x

#     def forward(
#         self,
#         src_tokens: torch.LongTensor,
#         src_lengths: torch.LongTensor,
#         coors: torch.Tensor,  # 未使用，占位确保接口一致
#         mask: torch.Tensor,
#         ec1: torch.LongTensor,
#         ec2: torch.LongTensor,
#         ec3: torch.LongTensor,
#         ec4: torch.LongTensor,
#     ) -> Dict[str, torch.Tensor]:

#         # [B, L]
#         src_tokens = src_tokens.to(self.device)
#         mask = mask.to(self.device)
#         ec1, ec2, ec3, ec4 = ec1.to(self.device), ec2.to(self.device), ec3.to(self.device), ec4.to(self.device)

#         # padding mask：True 表示 padding
#         padding_mask = src_tokens.eq(self.padding_idx)

#         # 将掩码位点替换为 [MASK] token
#         tokens = mask * self.mask_idx + (mask != 1) * src_tokens
#         tokens = tokens.long()

#         # 词嵌入（含 scale）
#         x = self.encoder.embed_scale * self.encoder.embed_tokens(tokens)  # [B, L, E]
#         embed = x

#         # 样本级 EC 嵌入（[B,1,E]）广播到序列长度维
#         B, L, E = x.size()
#         ec1_emb = self.ec1_embeddings(ec1.view(-1, 1)).view(B, 1, E)
#         ec2_emb = self.ec2_embeddings(ec2.view(-1, 1)).view(B, 1, E)
#         ec3_emb = self.ec3_embeddings(ec3.view(-1, 1)).view(B, 1, E)
#         ec4_emb = self.ec4_embeddings(ec4.view(-1, 1)).view(B, 1, E)
#         x = x + ec1_emb + ec2_emb + ec3_emb + ec4_emb

#         # token dropout（若开启）
#         if getattr(self.encoder, "token_dropout", False):
#             x = self._apply_token_dropout_rescale(x, tokens, padding_mask)

#         # 将 padding 位点置零（与 ESM 一致地做 mask）
#         if padding_mask is not None:
#             x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

#         # 进入 Transformer 层堆叠：ESM 期望 (T,B,E)
#         x = x.transpose(0, 1)  # (B,L,E) -> (L,B,E)
#         attn_mask = None if not padding_mask.any() else padding_mask  # [B, L]

#         for layer in self.encoder.layers:
#             x, _ = layer(
#                 x,  # (L,B,E)
#                 self_attn_padding_mask=attn_mask,
#                 need_head_weights=False,
#             )

#         # 最后一层 LayerNorm
#         x = self.encoder.emb_layer_norm_after(x)  # (L,B,E)

#         # 回到 (B,L,E)
#         x = x.transpose(0, 1)

#         # 语言建模头（逐位点 logits/prob）
#         logits = self.encoder.lm_head(x)         # [B, L, V]
#         prob = F.softmax(logits, dim=-1)         # [B, L, V]

#         return {
#             "hidden": x,                         # [B, L, E]
#             "logits": logits,                    # [B, L, V]
#             "prob": prob,                        # [B, L, V]
#             "padding_mask": padding_mask,        # [B, L]
#             "encoder_embedding": embed,          # [B, L, E]
#         }
