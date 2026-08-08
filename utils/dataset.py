# dataset.py
# -*- coding: utf-8 -*-
"""
JSONL → variable-length batches (token-budget packing) for PEGM.init_data(...)
---------------------------------------------------------------------------
Reads a JSON-lines file where each line is a single expanded sample dict
(like the example you gave). Packs samples into batches such that the sum
of sequence lengths in a batch is ≤ cfg["max_tokens"] (batch size varies).
Each yielded batch is a List[dict] ready to pass into PEGM.init_data(...).

Expected per-line JSON fields (min set):
{
  "seqs": ["..."],                   # protein sequence (string)   [REQUIRED]
  "coords": [[[x,y,z], ...]],        # protein coords  (L x 3)     [REQUIRED]
  "ec4": ["2.5.1.18"],               # EC label (string)           [REQUIRED]
  "ligand_coords": [[[x,y,z], ...]], # ligand coords (Ll x 3)      [REQUIRED]
  "ligand_feats": [[[f1..f5], ...]], # ligand 5D features (Ll x 5) [REQUIRED]
  "motifs": [idx0, idx1, ...]        # any mask/list your init_data expects
  # optional (ignored by init_data): pdbs, ligands, pocket_* ...
}

Notes:
- In the JSON example you posted, most fields are wrapped in lists of length 1.
  This loader unwraps singletons so that each sample uses raw tensors/lists.
- Packing budget defaults to protein length only; to budget on protein+ligand,
  set cfg["budget_mode"] = "protein+ligand".
"""
from __future__ import annotations
import sys
sys.path.append("..")
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union, Any


def _as_list(x: Any) -> List[Any]:
    """Ensure x is a list; unwrap singleton lists when appropriate."""
    if isinstance(x, list):
        # unwrap 1-length nested containers like [[...]] for coords
        if len(x) == 1 and isinstance(x[0], (list, str)):
            return x[0] if not (isinstance(x[0], list) and len(x[0]) == 0) else x
        return x
    return [x]


def _unwrap_sample_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw JSON dict into the minimal fields PEGM.init_data needs.
    We only keep the keys init_data uses: 'seqs','coords','ec4','ligand_coords',
    'ligand_feats','motifs'. 另外把 ligand_seq/ligand_seqs 透传，方便推理时写回。
    """
    out: Dict[str, Any] = {}

    # Required
    out["seqs"] = _as_list(d.get("seqs", []))
    out["coords"] = _as_list(d.get("coords", []))
    out["ec4"] = _as_list(d.get("ec4", []))
    out["ligand_coords"] = _as_list(d.get("ligand_coords", []))
    out["ligand_feats"] = _as_list(d.get("ligand_feats", []))
    out["pocket_idxs"] = _as_list(d.get("pocket_idxs", []))
    # optional: ligand sequence(s)
    if "ligand_seq" in d:
        out["ligand_seq"] = d.get("ligand_seq")
    if "ligand_seqs" in d:
        out["ligand_seqs"] = d.get("ligand_seqs")


    # Motifs: some datasets store as list of ints directly; ensure present
    motifs = d.get("motifs", None)
    if motifs is None:
        # fallback: empty list (your init_data should handle/construct masks)
        motifs = []
    out["motifs"] = motifs

    # Light validation
    if not out["seqs"] or not isinstance(out["seqs"][0], str):
        raise ValueError("Missing/invalid 'seqs' field (expect list with one string).")
    if not out["coords"] or not isinstance(out["coords"][0], list):
        raise ValueError("Missing/invalid 'coords' field (expect [L,3] list).")
    if not out["ec4"] or not isinstance(out["ec4"][0], str):
        raise ValueError("Missing/invalid 'ec4' field (expect list with one EC string).")
    if not out["ligand_coords"] or not isinstance(out["ligand_coords"][0], list):
        raise ValueError("Missing/invalid 'ligand_coords' field.")
    if not out["ligand_feats"] or not isinstance(out["ligand_feats"][0], list):
        raise ValueError("Missing/invalid 'ligand_feats' field.")

    return out


class EnzymeLigandDataset:
    """
    Streaming JSONL dataset with token-budget batch packing.

    cfg keys:
      - data_path: str | List[str]   path(s) to JSONL file(s)
      - max_tokens: int              batch token budget (>=1) [REQUIRED]
      - shuffle: bool                shuffle samples each epoch (default True)
      - seed: int                    RNG seed for shuffling (default 42)
      - budget_mode: "protein" | "protein+ligand"   (default "protein")
      - drop_remainder: bool         drop last incomplete batch (default False)
      - max_samples: Optional[int]   limit #samples for debugging (default None)
    """

    def __init__(self, cfg: dict, is_valid_set=False):
        self.cfg = cfg
        paths = cfg.get("valid_data_path") if is_valid_set else cfg.get("data_path")

        if paths is None:
            raise ValueError("cfg['data_path'] must be provided (str or list of str).")
        if isinstance(paths, (str, Path)):
            self.paths: List[Path] = [Path(paths)]
        else:
            self.paths = [Path(p) for p in paths]


        # print(f'path : {paths}')

        self.max_tokens: int = int(cfg.get("max_tokens", 0))
        if self.max_tokens <= 0:
            raise ValueError("cfg['max_tokens'] must be a positive integer.")

        self.shuffle: bool = bool(cfg.get("shuffle", True))
        self.seed: int = int(cfg.get("seed", 42))
        self.budget_mode: str = str(cfg.get("budget_mode", "protein")).lower()
        assert self.budget_mode in ("protein", "protein+ligand")
        self.drop_remainder: bool = bool(cfg.get("drop_remainder", False))
        self.max_samples: Optional[int] = cfg.get("max_samples", None)
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)

        self.len_ = None

        # Load all lines lazily per epoch (can be large); keep file refs only.
        # If you prefer to preload, you can read once into memory here.

    def _iter_samples(self) -> Iterator[dict]:
        """Yield normalized samples (dicts) one by one from all files."""
        count = 0
        for p in self.paths:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # print('line in dataset : ', line)
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Skip malformed lines
                        continue
                    try:
                        sample = _unwrap_sample_fields(raw)
                    except Exception:
                        # Skip samples missing required fields
                        continue
                    yield sample
                    count += 1
                    if self.max_samples is not None and count >= self.max_samples:
                        return


    def _sample_length(self, s: Dict[str, Any]) -> int:
        """Compute token cost for packing using sequence lengths only."""
        # protein sequence length
        seq = s.get("seqs", "")
        Lp = len(seq[0]) if isinstance(seq, (list, tuple)) else len(seq)

        # ligand sequence length
        lig_seq = s.get("ligand_seqs", "")
        Ll = len(lig_seq[0]) if isinstance(lig_seq, (list, tuple)) else len(lig_seq)

        if self.budget_mode == "protein+ligand":
            return max(1, Lp + Ll)
        return max(1, Lp)

    def _maybe_shuffle(self, items: List[Dict[str, Any]]) -> None:
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(items)
    def __len__(self):
        if self.len_ is not None:
            return self.len_
        else:
            self.len_ = len(list(self._iter_samples()))
            return self.len_
    def get_epoch_iterator(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Yield batches: each batch is a List[dict] compatible with PEGM.init_data(...).
        """
        # Materialize one epoch worth into memory for shuffling (optional).
        samples = list(self._iter_samples())
        if len(samples) == 0:
            return iter([])

        self._maybe_shuffle(samples)

        batch: List[Dict[str, Any]] = []
        tokens_in_batch = 0

        for s in samples:
            cost = self._sample_length(s)
            # If a single sample exceeds max_tokens, still put it alone.
            if cost > self.max_tokens:
                if batch and not self.drop_remainder:
                    yield batch
                    batch = []
                    tokens_in_batch = 0
                yield [s]
                continue

            # If adding this sample would exceed the budget, flush current batch.
            if tokens_in_batch + cost > self.max_tokens and batch:
                yield batch
                batch = []
                tokens_in_batch = 0

            batch.append(s)
            tokens_in_batch += cost

        if batch and not self.drop_remainder:
            yield batch


# ----------------------------- Usage Example -----------------------------
# cfg = {
#   "data_path": "/path/to/train.jsonl",
#   "max_tokens": 4096,
#   "shuffle": True,
#   "seed": 1234,
#   "budget_mode": "protein+ligand",
#   "drop_remainder": False,
# }
# ds = EnzymeLigandDataset(cfg)
# for batch in ds.get_epoch_iterator():
#     # batch is List[dict], pass directly to PEGM.init_data(batch)
#     data = PEGM.init_data(batch)
#     ...
