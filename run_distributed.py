import argparse
import os

import torch
from args import get_training_args
from run_training import train


def run_distributed(local_rank: int, args: argparse.Namespace):
    """Run in distributed setting.

    Args:
        local_rank (int): The local rank.
        args (argparse.Namespace): Arguments passed.
    """
    global_rank = args.node_rank * args.num_gpus_node + local_rank
    world_size = args.num_gpus_node * args.num_nodes

    torch.cuda.set_device(local_rank)
    if torch.distributed.is_available() is False:
        exit()

    torch.distributed.init_process_group(
        "nccl",
        init_method="".join(["tcp://", args.hostname, ":", str(args.port_id)]),
        rank=global_rank,
        world_size=world_size,
    )

    if local_rank == 0:
        print("Starting Distributed training...")

    if torch.distributed.is_initialized() is False:
        exit()

    distributed = True
    train(args, distributed)


if __name__ == "__main__":
    parser = get_training_args()
    parser.add_argument(
        '--num_nodes', 
        default=1, 
        type=int, 
        help='total number of nodes')

    parser.add_argument(
        '--num_gpus_node',
        default=1, 
        type=int, 
        help='number of gpus per node')

    parser.add_argument(
        '--node_rank', 
        default=0, 
        type=int, 
        help='number of the node used')

    parser.add_argument(
        '--hostname', 
        default="example-machine", 
        type=str, 
        help='local host name for distributed training')

    parser.add_argument(
        '--port_id', 
        default=23400, 
        type=int, 
        help='port id for distributed training')
        
    args = parser.parse_args()
    torch.multiprocessing.spawn(run_distributed, nprocs=args.num_gpus_node, args=(args,))

