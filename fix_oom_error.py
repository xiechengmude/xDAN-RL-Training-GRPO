#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix Out of Memory (OOM) errors in DeepSpeed-based RL training
Specifically targeting issues in PPO training with Ray
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, Optional

def parse_args():
    parser = argparse.ArgumentParser(description='Fix OOM errors in DeepSpeed RL training')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the training config file')
    parser.add_argument('--ds_config', type=str, required=True,
                        help='Path to the DeepSpeed config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the optimized configs (default: overwrite input)')
    parser.add_argument('--reduce_batch_size', type=float, default=0.5,
                        help='Factor to reduce batch size by (default: 0.5)')
    parser.add_argument('--increase_grad_accum', type=int, default=2,
                        help='Multiply gradient accumulation steps by this factor (default: 2)')
    return parser.parse_args()

def load_json_config(config_path: str) -> Dict[str, Any]:
    """Load JSON config from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def save_json_config(config: Dict[str, Any], output_path: str) -> None:
    """Save JSON config to file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Optimized config saved to: {output_path}")
    except Exception as e:
        print(f"Error saving config file: {e}")
        sys.exit(1)

def optimize_deepspeed_config(ds_config: Dict[str, Any], reduce_factor: float, 
                             increase_grad_accum: int) -> Dict[str, Any]:
    """Optimize DeepSpeed config to fix OOM errors"""
    # Create a copy to avoid modifying the original
    optimized = ds_config.copy()
    changes = []
    
    # 1. Adjust batch size if present
    if 'train_batch_size' in optimized:
        old_batch_size = optimized['train_batch_size']
        new_batch_size = max(1, int(old_batch_size * reduce_factor))
        optimized['train_batch_size'] = new_batch_size
        changes.append(f"Reduced batch size from {old_batch_size} to {new_batch_size}")
    
    # 2. Increase gradient accumulation steps
    old_grad_accum = optimized.get('gradient_accumulation_steps', 1)
    new_grad_accum = old_grad_accum * increase_grad_accum
    optimized['gradient_accumulation_steps'] = new_grad_accum
    changes.append(f"Increased gradient accumulation steps from {old_grad_accum} to {new_grad_accum}")
    
    # 3. Ensure ZeRO optimization is properly configured
    if 'zero_optimization' not in optimized:
        optimized['zero_optimization'] = {}
    
    # Set ZeRO stage to 3 (most memory efficient)
    old_stage = optimized['zero_optimization'].get('stage', 0)
    optimized['zero_optimization']['stage'] = 3
    if old_stage != 3:
        changes.append(f"Changed ZeRO stage from {old_stage} to 3")
    
    # 4. Enable CPU offloading for optimizer states
    if 'offload_optimizer' not in optimized['zero_optimization']:
        optimized['zero_optimization']['offload_optimizer'] = {
            'device': 'cpu',
            'pin_memory': True
        }
        changes.append("Enabled optimizer state offloading to CPU")
    
    # 5. Enable CPU offloading for parameters (ZeRO-3 only)
    if 'offload_param' not in optimized['zero_optimization']:
        optimized['zero_optimization']['offload_param'] = {
            'device': 'cpu',
            'pin_memory': True
        }
        changes.append("Enabled parameter offloading to CPU")
    
    # 6. Reduce communication buffer sizes
    optimized['zero_optimization']['reduce_bucket_size'] = 5e7  # 50MB
    optimized['zero_optimization']['allgather_bucket_size'] = 5e7  # 50MB
    changes.append("Reduced communication buffer sizes to 50MB")
    
    # 7. Enable activation checkpointing
    optimized['activation_checkpointing'] = {
        'partition_activations': True,
        'cpu_checkpointing': True,
        'contiguous_memory_optimization': True,
        'number_checkpoints': 1,
        'synchronize_checkpoint_boundary': False,
        'profile': False
    }
    changes.append("Enabled activation checkpointing")
    
    # 8. Set PyTorch memory allocation config
    optimized['torch_cuda_alloc_conf'] = 'expandable_segments:True'
    changes.append("Set PyTorch CUDA memory allocator to use expandable segments")
    
    return optimized, changes

def optimize_training_config(train_config: Dict[str, Any], reduce_factor: float) -> Dict[str, Any]:
    """Optimize training config to fix OOM errors"""
    # Create a copy to avoid modifying the original
    optimized = train_config.copy()
    changes = []
    
    # Adjust PPO-specific parameters if present
    if 'ppo' in optimized:
        ppo_config = optimized['ppo']
        
        # Reduce chunk size if present
        if 'chunk_size' in ppo_config:
            old_chunk_size = ppo_config['chunk_size']
            new_chunk_size = max(1, int(old_chunk_size * reduce_factor))
            ppo_config['chunk_size'] = new_chunk_size
            changes.append(f"Reduced PPO chunk size from {old_chunk_size} to {new_chunk_size}")
        
        # Reduce mini batch size if present
        if 'mini_batch_size' in ppo_config:
            old_mini_batch = ppo_config['mini_batch_size']
            new_mini_batch = max(1, int(old_mini_batch * reduce_factor))
            ppo_config['mini_batch_size'] = new_mini_batch
            changes.append(f"Reduced PPO mini batch size from {old_mini_batch} to {new_mini_batch}")
    
    # Adjust Ray-specific parameters if present
    if 'ray' in optimized:
        ray_config = optimized['ray']
        
        # Adjust resources per actor if present
        if 'resources_per_actor' in ray_config:
            resources = ray_config['resources_per_actor']
            
            # Reduce GPU memory if specified
            if 'GPU' in resources:
                old_gpu = resources['GPU']
                # Keep the same number of GPUs but add a note about memory
                changes.append(f"Note: Consider reducing GPU memory per actor in Ray configuration")
            
            # Increase CPU resources if needed for offloading
            if 'CPU' in resources:
                old_cpu = resources['CPU']
                new_cpu = max(old_cpu, 4)  # Ensure at least 4 CPUs for offloading
                resources['CPU'] = new_cpu
                if new_cpu > old_cpu:
                    changes.append(f"Increased CPU resources per actor from {old_cpu} to {new_cpu}")
    
    return optimized, changes

def main():
    args = parse_args()
    
    # Load configs
    ds_config = load_json_config(args.ds_config)
    train_config = load_json_config(args.config)
    
    # Optimize configs
    opt_ds_config, ds_changes = optimize_deepspeed_config(
        ds_config, args.reduce_batch_size, args.increase_grad_accum)
    
    opt_train_config, train_changes = optimize_training_config(
        train_config, args.reduce_batch_size)
    
    # Print summary of changes
    print("\n===== Memory Optimization Summary =====")
    print("DeepSpeed config changes:")
    for i, change in enumerate(ds_changes, 1):
        print(f"  {i}. {change}")
    
    print("\nTraining config changes:")
    if train_changes:
        for i, change in enumerate(train_changes, 1):
            print(f"  {i}. {change}")
    else:
        print("  No changes were made to the training configuration.")
    print("=======================================\n")
    
    # Save optimized configs
    ds_output = args.output if args.output else args.ds_config
    train_output = args.output.replace('.json', '_train.json') if args.output else args.config
    
    save_json_config(opt_ds_config, ds_output)
    save_json_config(opt_train_config, train_output)
    
    # Print additional recommendations
    print("\nAdditional recommendations to fix OOM errors:")
    print("1. Set these environment variables before training:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("   export NCCL_ASYNC_ERROR_HANDLING=1")
    print("   export NCCL_P2P_LEVEL=NVL")
    print("   export NCCL_IB_TIMEOUT=22")
    print("   export NCCL_DEBUG=INFO")
    print("   export NCCL_SOCKET_IFNAME=^lo,docker")
    print("2. Consider using a smaller model or reducing model precision (fp16/bf16)")
    print("3. If using Ray, try increasing the number of workers while reducing per-worker batch size")
    print("4. Monitor GPU memory usage with 'nvidia-smi' during training")
    print("5. If still encountering OOM errors, try:")
    print("   - Further reducing batch size")
    print("   - Increasing gradient accumulation steps")
    print("   - Using model parallelism in addition to data parallelism")
    print("   - Applying tensor parallelism for very large models")

if __name__ == "__main__":
    main()
