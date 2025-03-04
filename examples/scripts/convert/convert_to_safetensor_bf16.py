import os
import torch
from safetensors.torch import save_file
import glob

def load_zero_checkpoint(checkpoint_dir):
    """Load and merge Zero checkpoint files directly"""
    print(f"Loading checkpoint from {checkpoint_dir}")
    
    # Find all model state files
    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*_model_states.pt")))
    if not model_files:
        raise ValueError(f"No model state files found in {checkpoint_dir}")
    
    # Load and merge all model states
    state_dict = {}
    for model_file in model_files:
        print(f"Loading {model_file}")
        checkpoint = torch.load(model_file, map_location='cpu')
        if 'module' in checkpoint:
            state_dict.update(checkpoint['module'])
        else:
            state_dict.update(checkpoint)
    
    return state_dict

def convert_to_bf16(state_dict):
    """Convert state dict to bf16 format"""
    bf16_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float16]:
            bf16_state_dict[key] = tensor.to(torch.bfloat16)
        else:
            bf16_state_dict[key] = tensor
    return bf16_state_dict

def convert_zero_checkpoint_to_safetensor(checkpoint_dir, output_file=None, tag=None, exclude_frozen_parameters=False, use_bf16=False):
    """
    Convert a DeepSpeed Zero checkpoint to SafeTensor format
    
    Args:
        checkpoint_dir: Directory containing the Zero checkpoint files
        output_file: Path to save the SafeTensor file. If None, will use checkpoint folder name
        tag: Optional tag to identify the checkpoint
        exclude_frozen_parameters: Whether to exclude frozen parameters
        use_bf16: Whether to save in bfloat16 format
    """
    print(f"Converting checkpoint from {checkpoint_dir}")
    
    # Load the checkpoint
    state_dict = load_zero_checkpoint(checkpoint_dir)
    
    if use_bf16:
        print("Converting to BF16 format...")
        state_dict = convert_to_bf16(state_dict)
    
    # If no output file specified, use the checkpoint directory name
    if output_file is None:
        # Get the checkpoint directory name
        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        # Create output directory if it doesn't exist
        os.makedirs(checkpoint_name, exist_ok=True)
        # Create output path with precision suffix
        suffix = "_bf16" if use_bf16 else "_fp32"
        output_file = os.path.join(checkpoint_name, f"model{suffix}.safetensors")
    else:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to safetensors format
    print(f"Saving to SafeTensor format at {output_file}")
    save_file(state_dict, output_file)
    print("Conversion completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="Directory containing the DeepSpeed Zero checkpoint files")
    parser.add_argument("--output_file",
                        type=str,
                        default=None,
                        help="Path to save the converted SafeTensor file. If not specified, will use checkpoint folder name")
    parser.add_argument("-t",
                        "--tag",
                        type=str,
                        default=None,
                        help="Checkpoint tag used as a unique identifier")
    parser.add_argument("--exclude_frozen_parameters",
                        action='store_true',
                        help="Exclude frozen parameters")
    parser.add_argument("--bf16",
                        action='store_true',
                        help="Save model in bfloat16 format")
    parser.add_argument("-d",
                        "--debug",
                        action='store_true',
                        help="Enable debug mode")
    
    args = parser.parse_args()
    
    convert_zero_checkpoint_to_safetensor(
        args.checkpoint_dir,
        args.output_file,
        tag=args.tag,
        exclude_frozen_parameters=args.exclude_frozen_parameters,
        use_bf16=args.bf16
    )
