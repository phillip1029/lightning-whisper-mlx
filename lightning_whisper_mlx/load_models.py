import json
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file as safe_load_file
from mlx.utils import tree_unflatten

from . import whisper


def _to_mx_tree(tree, dtype: mx.Dtype):
    """
    Recursively convert numpy arrays in a nested structure to mx.array(dtype).
    Leaves that are already mx.array are returned as-is.
    """
    if isinstance(tree, mx.array):
        return tree
    if isinstance(tree, np.ndarray):
        return mx.array(tree, dtype=dtype)
    if isinstance(tree, dict):
        return {k: _to_mx_tree(v, dtype) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        out = [_to_mx_tree(v, dtype) for v in tree]
        return type(tree)(out)  # preserve list/tuple
    return tree


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    # ---- load config ----
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    # ---- load weights (safetensors preferred; npz fallback) ----
    safetensors_candidates = [
        model_path / "weights.safetensors",
        model_path / "model.safetensors",
    ]
    npz_path = model_path / "weights.npz"

    weights = None

    # Try safetensors first
    for st_path in safetensors_candidates:
        if st_path.exists():
            st_dict = safe_load_file(str(st_path))              # dict[str, np.ndarray]
            weights = tree_unflatten(list(st_dict.items()))     # nested python structure (still numpy leaves)
            weights = _to_mx_tree(weights, dtype=dtype)         # convert numpy -> mx.array
            break

    # Fallback to npz
    if weights is None:
        if not npz_path.exists():
            raise FileNotFoundError(
                f"No weights file found in {model_path}. "
                f"Expected one of: {[p.name for p in safetensors_candidates]} or {npz_path.name}"
            )
        npz_dict = mx.load(str(npz_path))                      # dict-like, leaves are mx.array
        weights = tree_unflatten(list(npz_dict.items()))

    model = whisper.Whisper(model_args, dtype)

    # ---- optional quantization (unchanged) ----
    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    # Now safe: weights leaves are mx.array
    model.update(weights)
    mx.eval(model.parameters())
    return model