#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import time

from src.utils.data_loader import ArielDataset, create_dataloader
from src.pipeline import ArielPipeline


def main():
    """
    Main function for inference
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Inference for Ariel Data Challenge')
    parser.add_argument('--config', type=str, default='configs/inference_config.yaml', help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--models_dir', type=str, default='output/models', help='Path to models directory')
    parser.add_argument('--output_dir', type=str, default='submissions', help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--wavelengths_path', type=str, default='data/wavelengths.csv', help='Path to wavelengths file')
    parser.add_argument('--submission_name', type=str, default=f'submission_{time.strftime("%Y%m%d_%H%M%S")}', help='Name of submission file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load wavelengths
    wavelengths = pd.read_csv(args.wavelengths_path)
    n_wavelengths = len(wavelengths)
    print(f"Number of wavelengths: {n_wavelengths}")
    
    # Set model paths
    jnet_model_path = os.path.join(args.models_dir, config['models']['jnet'])
    physnet_model_path = os.path.join(args.models_dir, config['models']['physnet'])
    spec_model_path = os.path.join(args.models_dir, config['models']['bayesian_resnet'])
    
    # Check if models exist
    if not os.path.exists(jnet_model_path):
        print(f"Warning: JitterNet model not found at {jnet_model_path}")
        jnet_model_path = None
    if not os.path.exists(physnet_model_path):
        print(f"Warning: PhySNet model not found at {physnet_model_path}")
        physnet_model_path = None
    if not os.path.exists(spec_model_path):
        print(f"Warning: Spectral extraction model not found at {spec_model_path}")
        spec_model_path = None
        
    # Create pipeline
    pipeline = ArielPipeline(
        device=device,
        jnet_model_path=jnet_model_path,
        physnet_model_path=physnet_model_path,
        spec_model_path=spec_model_path,
        n_wavelengths=n_wavelengths,
        mc_samples=config['inference']['mc_samples']
    )
    
    # Get test planet IDs
    if config['inference']['mode'] == 'test':
        data_subdir = 'test'
        planet_ids = [d for d in os.listdir(os.path.join(args.data_dir, data_subdir)) 
                     if os.path.isdir(os.path.join(args.data_dir, data_subdir, d))]
        labels_path = None
    else:  # 'val' mode for validation
        data_subdir = 'train'
        # Use a small subset of training data
        planet_ids = [d for d in os.listdir(os.path.join(args.data_dir, data_subdir)) 
                     if os.path.isdir(os.path.join(args.data_dir, data_subdir, d))][:config['inference']['val_size']]
        labels_path = os.path.join(args.data_dir, 'train_labels.csv')
        
    print(f"Found {len(planet_ids)} planets in the {data_subdir} set")
    
    # Create dataloader
    dataloader = create_dataloader(
        data_dir=os.path.join(args.data_dir, data_subdir),
        planet_ids=planet_ids,
        labels_path=labels_path,
        wavelengths_path=args.wavelengths_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=None,  # We'll apply preprocessing in the pipeline
        mode='test',
        shuffle=False
    )
    
    # Process each planet and collect results
    all_planet_ids = []
    all_spectra = []
    all_uncertainties = []
    
    print(f"\nProcessing {len(planet_ids)} planets...")
    
    for batch_idx, (data, _) in enumerate(tqdm(dataloader)):
        # Process planet
        spectrum, uncertainty = pipeline.process_planet(
            data, 
            mc_dropout=config['inference']['mc_dropout'],
            apply_prior=config['inference']['apply_spectral_prior']
        )
        
        # Store results
        all_planet_ids.append(int(data['planet_id']))
        all_spectra.append(spectrum)
        all_uncertainties.append(uncertainty)
        
        # For debugging, print progress every 10 planets
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(dataloader)} planets")
    
    # Stack results
    all_spectra = np.stack(all_spectra)
    all_uncertainties = np.stack(all_uncertainties)
    
    # Apply post-processing if configured
    if config['inference']['scale_uncertainties']:
        scale_factor = config['inference']['uncertainty_scale_factor']
        print(f"Scaling uncertainties by factor {scale_factor}")
        all_uncertainties *= scale_factor
    
    # Create submission file
    output_path = os.path.join(args.output_dir, f"{args.submission_name}.csv")
    submission_df = pipeline.create_submission(all_planet_ids, all_spectra, all_uncertainties, output_path)
    
    print(f"\nSubmission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    
    # Evaluate if validation mode and ground truth is available
    if config['inference']['mode'] == 'val' and labels_path is not None:
        from src.pipeline import compute_gll_metric
        
        # Load ground truth
        ground_truth = pd.read_csv(labels_path)
        
        # Filter to selected planets
        ground_truth = ground_truth[ground_truth['planet_id'].isin(all_planet_ids)]
        
        # Prepare for evaluation
        solution = ground_truth.copy()
        submission = submission_df.copy()
        
        # Compute metric
        naive_mean = config['evaluation']['naive_mean']
        naive_sigma = config['evaluation']['naive_sigma']
        sigma_true = config['evaluation']['sigma_true']
        
        score = compute_gll_metric(
            solution, 
            submission, 
            'planet_id', 
            naive_mean, 
            naive_sigma, 
            sigma_true
        )
        
        print(f"\nValidation Score: {score:.4f}")
        
        # Save score to file
        with open(os.path.join(args.output_dir, f"{args.submission_name}_score.txt"), 'w') as f:
            f.write(f"GLL Score: {score:.6f}\n")
            f.write(f"Parameters:\n")
            f.write(f"  MC Samples: {config['inference']['mc_samples']}\n")
            f.write(f"  MC Dropout: {config['inference']['mc_dropout']}\n")
            f.write(f"  Apply Prior: {config['inference']['apply_spectral_prior']}\n")
            f.write(f"  Scale Uncertainties: {config['inference']['scale_uncertainties']}\n")
            if config['inference']['scale_uncertainties']:
                f.write(f"  Scale Factor: {config['inference']['uncertainty_scale_factor']}\n")


if __name__ == '__main__':
    main() 