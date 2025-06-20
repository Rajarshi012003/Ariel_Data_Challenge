#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import subprocess
from datetime import datetime


def check_data(data_dir):
    """
    Check if data directory exists and has required structure
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        bool: True if data directory is valid
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist")
        return False
        
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Error: Train directory '{train_dir}' does not exist")
        return False
        
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory '{test_dir}' does not exist, will skip inference")
        
    train_labels_path = os.path.join(data_dir, 'train_labels.csv')
    if not os.path.exists(train_labels_path):
        print(f"Error: Train labels file '{train_labels_path}' does not exist")
        return False
        
    wavelengths_path = os.path.join(data_dir, 'wavelengths.csv')
    if not os.path.exists(wavelengths_path):
        print(f"Error: Wavelengths file '{wavelengths_path}' does not exist")
        return False
        
    return True


def run_command(command, description):
    """
    Run a command and print output
    
    Args:
        command (list): Command to run
        description (str): Description of command
        
    Returns:
        int: Return code
    """
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Ariel Data Challenge 2024 Pipeline')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to output directory')
    parser.add_argument('--train_config', type=str, default='configs/train_config.yaml', help='Path to training configuration file')
    parser.add_argument('--infer_config', type=str, default='configs/inference_config.yaml', help='Path to inference configuration file')
    parser.add_argument('--skip_train', action='store_true', help='Skip training')
    parser.add_argument('--skip_infer', action='store_true', help='Skip inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Check data directory
    if not check_data(args.data_dir):
        return 1
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'submissions'), exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_name = f"submission_{timestamp}"
    
    # Train models
    if not args.skip_train:
        train_cmd = [
            "python", "scripts/train.py",
            "--config", args.train_config,
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--seed", str(args.seed),
            "--device", args.device
        ]
        
        ret_code = run_command(train_cmd, "Training Models")
        if ret_code != 0:
            print("Error: Training failed")
            return ret_code
    else:
        print("\n=== Skipping Training ===")
        
    # Run inference
    if not args.skip_infer:
        # Validation inference first
        val_infer_cmd = [
            "python", "scripts/infer.py",
            "--config", args.infer_config,
            "--data_dir", args.data_dir,
            "--models_dir", os.path.join(args.output_dir, "models"),
            "--output_dir", os.path.join(args.output_dir, "submissions"),
            "--device", args.device,
            "--submission_name", f"{submission_name}_val"
        ]
        
        # Modify config for validation
        val_config_path = os.path.join(args.output_dir, "val_inference_config.yaml")
        with open(args.infer_config, 'r') as f:
            val_config = yaml.safe_load(f)
            
        val_config['inference']['mode'] = 'val'
        
        with open(val_config_path, 'w') as f:
            yaml.dump(val_config, f)
            
        val_infer_cmd[2] = val_config_path
        
        ret_code = run_command(val_infer_cmd, "Validation Inference")
        if ret_code != 0:
            print("Warning: Validation inference failed")
            
        # Test inference
        if os.path.exists(os.path.join(args.data_dir, 'test')):
            test_infer_cmd = [
                "python", "scripts/infer.py",
                "--config", args.infer_config,
                "--data_dir", args.data_dir,
                "--models_dir", os.path.join(args.output_dir, "models"),
                "--output_dir", os.path.join(args.output_dir, "submissions"),
                "--device", args.device,
                "--submission_name", submission_name
            ]
            
            ret_code = run_command(test_infer_cmd, "Test Inference")
            if ret_code != 0:
                print("Error: Test inference failed")
                return ret_code
        else:
            print("\n=== Skipping Test Inference (No test data found) ===")
    else:
        print("\n=== Skipping Inference ===")
        
    print("\n=== Pipeline Completed Successfully ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Submission files: {os.path.join(args.output_dir, 'submissions', submission_name + '.csv')}")
    
    return 0


if __name__ == '__main__':
    exit(main()) 