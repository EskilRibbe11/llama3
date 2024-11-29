import torch



if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl", init_method="env://?use_libuv=False")