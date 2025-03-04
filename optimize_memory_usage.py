#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU Memory Optimization Script for DeepSpeed-based RL Training
This script helps optimize DeepSpeed configuration to reduce memory usage
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize DeepSpeed config for memory usage')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the DeepSpeed config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the optimized config (default: overwrite input)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='New batch size to use (smaller to reduce memory)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--offload', action='store_true',
                        help='Enable CPU offloading for optimizer states and parameters')
    parser.add_argument('--zero_stage', type=int, choices=[1, 2, 3], default=None,
                        help='ZeRO optimization stage (1-3)')
    parser.add_argument('--reduce_bucket_size', type=int, default=None,
                        help='Reduce bucket size in bytes for ZeRO optimization')
    parser.add_argument('--allgather_bucket_size', type=int, default=None,
                        help='Allgather bucket size in bytes for ZeRO optimization')
    parser.add_argument('--memory_efficient_attention', action='store_true',
                        help='Enable memory efficient attention')
    parser.add_argument('--activation_checkpointing', action='store_true',
                        help='Enable activation checkpointing')
    parser.add_argument('--estimate', action='store_true',
                        help='Estimate memory usage without changing the config')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load DeepSpeed config from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save DeepSpeed config to file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Optimized config saved to: {output_path}")
    except Exception as e:
        print(f"Error saving config file: {e}")
        sys.exit(1)

def optimize_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply memory optimization techniques to DeepSpeed config"""
    # Create a copy to avoid modifying the original
    optimized = config.copy()
    
    changes_made = []
    
    # Adjust batch size if specified
    if args.batch_size is not None:
        if 'train_batch_size' in optimized:
            old_batch_size = optimized['train_batch_size']
            optimized['train_batch_size'] = args.batch_size
            changes_made.append(f"Reduced batch size from {old_batch_size} to {args.batch_size}")
    
    # Adjust gradient accumulation steps if specified
    if args.gradient_accumulation_steps is not None:
        old_gas = optimized.get('gradient_accumulation_steps', 1)
        optimized['gradient_accumulation_steps'] = args.gradient_accumulation_steps
        changes_made.append(f"Changed gradient accumulation steps from {old_gas} to {args.gradient_accumulation_steps}")
    
    # Configure ZeRO optimization
    if 'zero_optimization' not in optimized:
        optimized['zero_optimization'] = {}
    
    if args.zero_stage is not None:
        old_stage = optimized['zero_optimization'].get('stage', 0)
        optimized['zero_optimization']['stage'] = args.zero_stage
        changes_made.append(f"Changed ZeRO stage from {old_stage} to {args.zero_stage}")
    
    # Enable CPU offloading if requested
    if args.offload:
        # Configure offloading for optimizer states
        if 'offload_optimizer' not in optimized['zero_optimization']:
            optimized['zero_optimization']['offload_optimizer'] = {
                'device': 'cpu',
                'pin_memory': True
            }
            changes_made.append("Enabled optimizer state offloading to CPU")
        
        # Configure offloading for parameters (only for ZeRO-3)
        if optimized['zero_optimization'].get('stage', 0) == 3:
            if 'offload_param' not in optimized['zero_optimization']:
                optimized['zero_optimization']['offload_param'] = {
                    'device': 'cpu',
                    'pin_memory': True
                }
                changes_made.append("Enabled parameter offloading to CPU (ZeRO-3)")
    
    # Adjust communication buffer sizes
    if args.reduce_bucket_size is not None:
        optimized['zero_optimization']['reduce_bucket_size'] = args.reduce_bucket_size
        changes_made.append(f"Set reduce bucket size to {args.reduce_bucket_size}")
    
    if args.allgather_bucket_size is not None:
        optimized['zero_optimization']['allgather_bucket_size'] = args.allgather_bucket_size
        changes_made.append(f"Set allgather bucket size to {args.allgather_bucket_size}")
    
    # Enable memory efficient attention if requested
    if args.memory_efficient_attention:
        if 'fp16' not in optimized:
            optimized['fp16'] = {'enabled': True}
        
        if 'bf16' not in optimized:
            optimized['bf16'] = {'enabled': False}
        
        # Add memory efficient attention config
        if 'memory_efficient_attention' not in optimized:
            optimized['memory_efficient_attention'] = True
            changes_made.append("Enabled memory efficient attention")
    
    # Enable activation checkpointing if requested
    if args.activation_checkpointing:
        if 'activation_checkpointing' not in optimized:
            optimized['activation_checkpointing'] = {
                'partition_activations': True,
                'cpu_checkpointing': True,
                'contiguous_memory_optimization': True,
                'number_checkpoints': 1,
                'synchronize_checkpoint_boundary': False,
                'profile': False
            }
            changes_made.append("Enabled activation checkpointing")
    
    # Set PyTorch memory allocation config
    optimized['torch_cuda_alloc_conf'] = 'expandable_segments:True'
    changes_made.append("Set PyTorch CUDA memory allocator to use expandable segments")
    
    return optimized, changes_made

def estimate_memory_usage(config: Dict[str, Any]) -> None:
    """Provide a rough estimate of memory usage based on config"""
    # This is a simplified estimation
    batch_size = config.get('train_batch_size', 1)
    zero_stage = config.get('zero_optimization', {}).get('stage', 0)
    offload_optimizer = 'offload_optimizer' in config.get('zero_optimization', {})
    offload_param = 'offload_param' in config.get('zero_optimization', {})
    grad_accum = config.get('gradient_accumulation_steps', 1)
    
    print("\n===== Memory Usage Estimation =====")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {grad_accum}")
    print(f"Effective batch size: {batch_size * grad_accum}")
    print(f"ZeRO stage: {zero_stage}")
    print(f"Optimizer state offloading: {'Enabled' if offload_optimizer else 'Disabled'}")
    print(f"Parameter offloading: {'Enabled' if offload_param else 'Disabled'}")
    
    # Memory reduction factors (approximate)
    if zero_stage == 1:
        print("ZeRO-1: Optimizer states partitioned across GPUs (reduces ~33% memory)")
    elif zero_stage == 2:
        print("ZeRO-2: Optimizer states + gradients partitioned across GPUs (reduces ~66% memory)")
    elif zero_stage == 3:
        print("ZeRO-3: Optimizer states + gradients + parameters partitioned across GPUs (reduces ~90% memory)")
    
    if offload_optimizer:
        print("Optimizer offloading: Further reduces GPU memory by moving optimizer states to CPU")
    
    if offload_param:
        print("Parameter offloading: Further reduces GPU memory by moving parameters to CPU when not needed")
    
    print("\nRecommendations:")
    if batch_size > 1 and not offload_optimizer and zero_stage < 2:
        print("- Consider reducing batch size")
    
    if zero_stage < 3:
        print(f"- Consider increasing ZeRO stage to {min(zero_stage + 1, 3)}")
    
    if zero_stage >= 2 and not offload_optimizer:
        print("- Consider enabling optimizer state offloading")
    
    if zero_stage == 3 and not offload_param and not offload_optimizer:
        print("- Consider enabling parameter offloading")
    
    if grad_accum == 1:
        print("- Consider using gradient accumulation to reduce memory usage while maintaining effective batch size")
    
    print("=====================================")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.estimate:
        estimate_memory_usage(config)
        return
    
    optimized_config, changes = optimize_config(config, args)
    
    # Print summary of changes
    print("\n===== Memory Optimization Summary =====")
    if changes:
        for i, change in enumerate(changes, 1):
            print(f"{i}. {change}")
    else:
        print("No changes were made to the configuration.")
    print("=======================================\n")
    
    # Save optimized config
    output_path = args.output if args.output else args.config
    save_config(optimized_config, output_path)
    
    # Print usage instructions
    print("\nTo apply these changes to your training:")
    print(f"1. Use the optimized config file: {output_path}")
    print("2. Set the following environment variable to avoid memory fragmentation:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("3. Consider adding gradient checkpointing in your model configuration")
    print("4. For ZeRO-3, consider using smaller communication buffers if you encounter OOM during communication")
    print("\nIf you still encounter OOM errors, try:")
    print("- Further reducing batch size")
    print("- Increasing gradient accumulation steps")
    print("- Using a smaller model or reducing model precision (fp16/bf16)")
    print("- Using model parallelism in addition to data parallelism")

if __name__ == "__main__":
    main()
