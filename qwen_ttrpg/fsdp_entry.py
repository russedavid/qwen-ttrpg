"""Initialize both CPU and CUDA collectives before the training framework loads.

CPU-offloaded FSDP2 gradients need a CPU backend for global gradient norms, while
model computation uses NCCL. Explicit initialization avoids backend selection
depending on the order in which a framework constructs its shared state.
"""

import os
import runpy


def main():
    import torch
    import torch.distributed as dist

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    print(
        f"Qwen training rank {dist.get_rank()}: backend={dist.get_backend()}",
        flush=True,
    )
    try:
        runpy.run_module("axolotl.cli.train", run_name="__main__")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
