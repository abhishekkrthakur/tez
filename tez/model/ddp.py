import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def cleanup():
    dist.destroy_process_group()


def init_process(rank, size, fn, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def create_ddp_model(model, rank, world_size):
    setup(rank, world_size)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    




def run_model(model_fn, world_size):
    mp.spawn(model_fn, args=(world_size,), nprocs=world_size, join=True)
