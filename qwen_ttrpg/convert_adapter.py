"""Convert Qwen adapters, preserving low-rank factors through head permutations.

The pinned llama.cpp converter cannot reshape the input dimension of a LoRA
tensor. A column permutation of B@A is B@permute_columns(A); a row permutation
is permute_rows(B)@A. Apply that identity without materializing dense weights.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys

from .util import digest, now, file_digest


def factor_reorder(original, tensor, dim, num_k_heads, num_v_per_k, head_dim):
    if not hasattr(tensor, "_lora_A"):
        return original(tensor, dim, num_k_heads, num_v_per_k, head_dim)
    if len(tensor.shape) != 2 or dim not in {0, 1, -1, -2}:
        raise ValueError(
            "Unsupported low-rank head permutation; refusing a lossy conversion."
        )
    a, b = tensor._lora_A, tensor._lora_B
    if dim % 2 == 0:
        b = original(b, 0, num_k_heads, num_v_per_k, head_dim)
    else:
        a = original(a, 1, num_k_heads, num_v_per_k, head_dim)
    return type(tensor)(a, b)


def verify_factor_permutation():
    """Numerically check both row and column identities before conversion."""
    import torch

    class Factors:
        def __init__(self, a, b):
            self._lora_A, self._lora_B = a, b
            self.shape = (b.shape[0], a.shape[1])

    def reorder(tensor, dim, k, v, width):
        shape = list(tensor.shape)
        expanded = shape[:dim] + [k, v, width] + shape[dim + 1 :]
        dims = list(range(len(expanded)))
        dims[dim], dims[dim + 1] = dims[dim + 1], dims[dim]
        return tensor.reshape(expanded).permute(dims).contiguous().reshape(shape)

    gen = torch.Generator().manual_seed(71)
    errors = []
    for dim in [0, 1]:
        a = torch.randn(4, 24, generator=gen, dtype=torch.float64)
        b = torch.randn(24, 4, generator=gen, dtype=torch.float64)
        actual = factor_reorder(reorder, Factors(a, b), dim, 2, 3, 4)
        expected = reorder(b @ a, dim, 2, 3, 4)
        err = float((expected - actual._lora_B @ actual._lora_A).abs().max())
        if err > 1e-10:
            raise ValueError("Low-rank head permutation failed its numerical check.")
        errors.append(err)
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    runtime = Path(args.runtime).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Choose a new converted adapter path.")
    sys.path.insert(0, str(runtime))
    from conversion import qwen

    cls = qwen._LinearAttentionVReorderBase
    original = cls._reorder_v_heads
    errors = verify_factor_permutation()
    cls._reorder_v_heads = staticmethod(
        lambda tensor, dim, k, v, width: factor_reorder(
            original, tensor, dim, k, v, width
        )
    )
    converter = runtime / "convert_lora_to_gguf.py"
    sys.argv = [
        str(converter),
        "--base",
        args.base,
        "--outfile",
        str(output),
        "--outtype",
        "f16",
        args.adapter,
    ]
    runpy.run_path(str(converter), run_name="__main__")
    report = {
        "created": now(),
        "converter_sha256": file_digest(converter),
        "qwen_conversion_sha256": file_digest(runtime / "conversion/qwen.py"),
        "factor_permutation_max_errors": errors,
        "adapter_sha256": file_digest(Path(args.adapter) / "adapter_model.safetensors"),
        "gguf_sha256": file_digest(output),
        "bytes": output.stat().st_size,
    }
    output.with_suffix(".conversion.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
