# test.py

import argparse

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam
from transformers import get_scheduler
from transformers.deepspeed import HfDeepSpeedConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.SELU(),
        nn.Linear(1024, 1024),
        nn.SELU(),
        nn.Linear(1024, 1),
        nn.Sigmoid(),
    )

    deepspeed.init_distributed()

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    args.device = device
    args.global_rank = dist.get_rank()

    dist.barrier()

    ds_config = {
        'train_batch_size': None,
        'train_micro_batch_size_per_gpu': 8,
        'gradient_accumulation_steps': 1,
        'steps_per_print': 10,
        'zero_optimization': {
            'stage': 3,
            'offload_param': {
                'device': 'none',
            },
            'offload_optimizer': {
                'device': 'none',
            },
            'param_persistence_threshold': 1e4,
            'max_live_parameters': 3e7,
            'prefetch_bucket_size': 3e7,
            'memory_efficient_linear': False,
            'gather_16bit_weights_on_model_save': True,
        },
        'gradient_clipping': 1.0,
        'prescale_gradients': False,
        'wall_clock_breakdown': False,
    }

    _dstchf = HfDeepSpeedConfig(ds_config)

    optimizer = FusedAdam(
        [{'params': list(model.parameters()), 'weight_decay': 0.0}],
        lr=1e-3,
        betas=(0.9, 0.95),
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=100,
    )

    model, *_ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )


if __name__ == '__main__':
    main()
