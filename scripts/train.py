#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import time
import random
from datetime import datetime

from src.utils.data_loader import ArielDataset, create_dataloader, get_train_val_split
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.jitter_correction.tcn_jnet import TCNJNet
from src.denoising.physnet import PhySNet, MaskedAutoencoderViT
from src.spectral_extraction.bayesian_resnet import BayesianResNet1D, combined_loss
from src.uncertainty.uncertainty_quantification import UncertaintyQuantification


def set_seed(seed):
    """
    Set seeds for reproducibility
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_fn(data):
    """
    Preprocessing function for dataloaders
    
    Args:
        data (dict): Raw data dictionary
        
    Returns:
        dict: Preprocessed data
    """
    preprocessor = PreprocessingPipeline(
        dark_subtraction=True,
        flat_correction=True,
        bad_pixel_interp=True,
        temporal_norm=True
    )
    return preprocessor.process(data)


def train_tcn_jnet(config, train_loader, val_loader, device, output_dir):
    """
    Train TCN-JNet for jitter correction
    
    Args:
        config (dict): Configuration dictionary
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved model
    """
    print("\n=== Training TCN-JNet ===")
    
    # Create model
    model = TCNJNet(
        fgs_spatial_features=config['jnet']['spatial_features'],
        airs_spatial_features=config['jnet']['spatial_features'],
        hidden_channels=config['jnet']['hidden_channels'],
        max_displacement=config['jnet']['max_displacement']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['jnet']['learning_rate'],
        weight_decay=config['jnet']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Create loss function
    mse_loss = nn.MSELoss()
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs', 'tcn_jnet'))
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'models', 'tcn_jnet_best.pth')
    
    for epoch in range(config['jnet']['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['jnet']['epochs']}")):
            # Extract FGS1 and AIRS-CH0 signals
            fgs_signal = torch.from_numpy(data['FGS1']['signal']).float().to(device)
            airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
            
            # Add batch dimension if needed
            if len(fgs_signal.shape) == 3:
                fgs_signal = fgs_signal.unsqueeze(0)
            if len(airs_signal.shape) == 3:
                airs_signal = airs_signal.unsqueeze(0)
                
            # Forward pass
            jitter, _ = model(fgs_signal, airs_signal)
            
            # Create synthetic target (zero jitter) for simplicity
            # In a real scenario, you'd have ground truth jitter values
            target_jitter = torch.zeros_like(jitter)
            
            # Compute loss
            loss = mse_loss(jitter, target_jitter)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(val_loader, desc="Validation")):
                # Extract FGS1 and AIRS-CH0 signals
                fgs_signal = torch.from_numpy(data['FGS1']['signal']).float().to(device)
                airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
                
                # Add batch dimension if needed
                if len(fgs_signal.shape) == 3:
                    fgs_signal = fgs_signal.unsqueeze(0)
                if len(airs_signal.shape) == 3:
                    airs_signal = airs_signal.unsqueeze(0)
                    
                # Forward pass
                jitter, _ = model(fgs_signal, airs_signal)
                
                # Create synthetic target (zero jitter) for simplicity
                target_jitter = torch.zeros_like(jitter)
                
                # Compute loss
                loss = mse_loss(jitter, target_jitter)
                
                val_loss += loss.item()
                
        # Compute average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['jnet']['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.6f}")
            
    # Save final model
    final_model_path = os.path.join(output_dir, 'models', 'tcn_jnet_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Close tensorboard writer
    writer.close()
    
    return best_model_path


def train_physnet_mae(config, train_loader, val_loader, device, output_dir):
    """
    Pretrain PhySNet with Masked Autoencoding (Ti-MAE)
    
    Args:
        config (dict): Configuration dictionary
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved model
    """
    print("\n=== Pretraining PhySNet with MAE ===")
    
    # Create model
    model = MaskedAutoencoderViT(
        img_size=config['physnet_mae']['img_size'],
        patch_size=config['physnet_mae']['patch_size'],
        in_channels=1,
        embed_dim=config['physnet_mae']['embed_dim'],
        depth=config['physnet_mae']['depth'],
        num_heads=config['physnet_mae']['num_heads'],
        decoder_embed_dim=config['physnet_mae']['decoder_embed_dim'],
        decoder_depth=config['physnet_mae']['decoder_depth'],
        decoder_num_heads=config['physnet_mae']['decoder_num_heads'],
        mask_ratio=config['physnet_mae']['mask_ratio']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['physnet_mae']['learning_rate'],
        weight_decay=config['physnet_mae']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['physnet_mae']['epochs'],
        eta_min=1e-6
    )
    
    # Create loss function (MSE for reconstruction)
    mse_loss = nn.MSELoss()
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs', 'physnet_mae'))
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'models', 'physnet_mae_best.pth')
    
    for epoch in range(config['physnet_mae']['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['physnet_mae']['epochs']}")):
            # Extract AIRS-CH0 signal
            airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
            
            # For MAE pretraining, we need [B, C, H, W] format
            # Select random frames from temporal dimension
            batch_size, time_steps, height, width = airs_signal.shape
            random_frames = torch.randint(0, time_steps, (batch_size,))
            frames = airs_signal[torch.arange(batch_size), random_frames].unsqueeze(1)  # [B, 1, H, W]
            
            # Forward pass
            output = model(frames)
            
            # Get predictions and targets
            pred = output['pred']  # [B, L, patch_size^2*C]
            
            # Convert to patches
            target = model.patchify(frames)  # [B, L, patch_size^2*C]
            
            # Compute loss only on masked patches
            mask = output['mask']  # [B, L], 1 is masked, 0 is kept
            loss = mse_loss(pred[mask.bool()], target[mask.bool()])
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tqdm(val_loader, desc="Validation")):
                # Extract AIRS-CH0 signal
                airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
                
                # For MAE pretraining, we need [B, C, H, W] format
                # Select random frames from temporal dimension
                batch_size, time_steps, height, width = airs_signal.shape
                random_frames = torch.randint(0, time_steps, (batch_size,))
                frames = airs_signal[torch.arange(batch_size), random_frames].unsqueeze(1)  # [B, 1, H, W]
                
                # Forward pass
                output = model(frames)
                
                # Get predictions and targets
                pred = output['pred']  # [B, L, patch_size^2*C]
                
                # Convert to patches
                target = model.patchify(frames)  # [B, L, patch_size^2*C]
                
                # Compute loss only on masked patches
                mask = output['mask']  # [B, L], 1 is masked, 0 is kept
                loss = mse_loss(pred[mask.bool()], target[mask.bool()])
                
                val_loss += loss.item()
                
        # Compute average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['physnet_mae']['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.6f}")
            
    # Save final model
    final_model_path = os.path.join(output_dir, 'models', 'physnet_mae_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Close tensorboard writer
    writer.close()
    
    return best_model_path


def train_physnet(config, train_loader, val_loader, device, output_dir, pretrained_path=None):
    """
    Train PhySNet for denoising
    
    Args:
        config (dict): Configuration dictionary
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device
        output_dir (str): Output directory
        pretrained_path (str, optional): Path to pretrained model
        
    Returns:
        str: Path to saved model
    """
    print("\n=== Training PhySNet ===")
    
    # Create model
    model = PhySNet(
        in_channels=1,
        out_channels=1,
        temporal=config['physnet']['temporal'],
        use_spectral_conv=config['physnet']['use_spectral_conv'],
        use_physics_attn=config['physnet']['use_physics_attn']
    ).to(device)
    
    # Load pretrained weights if available
    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['physnet']['learning_rate'],
        weight_decay=config['physnet']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Create loss function
    mse_loss = nn.MSELoss()
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs', 'physnet'))
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'models', 'physnet_best.pth')
    
    for epoch in range(config['physnet']['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['physnet']['epochs']}")):
            # Extract AIRS-CH0 signal
            # For synthetic data, use clean signal as target and add noise to input
            airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
            
            # For denoising, we need [B, C, T, H, W] format
            if len(airs_signal.shape) == 3:  # [T, H, W]
                airs_signal = airs_signal.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                
            # Create clean target for denoising
            # In a real scenario, you'd have ground truth clean signals
            # Here we use a simple approach: assume current signal is clean and add synthetic noise
            clean_signal = airs_signal.clone()
            noisy_signal = clean_signal + torch.randn_like(clean_signal) * config['physnet']['noise_level']
            
            # Forward pass
            output = model(noisy_signal)
            
            # Compute loss
            loss = mse_loss(output, clean_signal)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc="Validation")):
                # Extract AIRS-CH0 signal
                airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
                
                # For denoising, we need [B, C, T, H, W] format
                if len(airs_signal.shape) == 3:  # [T, H, W]
                    airs_signal = airs_signal.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                    
                # Create clean target for denoising
                clean_signal = airs_signal.clone()
                noisy_signal = clean_signal + torch.randn_like(clean_signal) * config['physnet']['noise_level']
                
                # Forward pass
                output = model(noisy_signal)
                
                # Compute loss
                loss = mse_loss(output, clean_signal)
                
                val_loss += loss.item()
                
        # Compute average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['physnet']['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.6f}")
            
    # Save final model
    final_model_path = os.path.join(output_dir, 'models', 'physnet_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Close tensorboard writer
    writer.close()
    
    return best_model_path


def train_bayesian_resnet(config, train_loader, val_loader, device, output_dir):
    """
    Train Bayesian ResNet for spectral extraction
    
    Args:
        config (dict): Configuration dictionary
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved model
    """
    print("\n=== Training Bayesian ResNet ===")
    
    # Create model
    model = BayesianResNet1D(
        in_channels=config['bayesian_resnet']['in_channels'],
        n_wavelengths=config['bayesian_resnet']['n_wavelengths'],
        hidden_channels=config['bayesian_resnet']['hidden_channels'],
        dropout_p=config['bayesian_resnet']['dropout_p'],
        use_template_attention=config['bayesian_resnet']['use_template_attention']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['bayesian_resnet']['learning_rate'],
        weight_decay=config['bayesian_resnet']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs', 'bayesian_resnet'))
    
    # Create uncertainty quantification for calibration
    uq = UncertaintyQuantification()
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'models', 'bayesian_resnet_best.pth')
    
    for epoch in range(config['bayesian_resnet']['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['bayesian_resnet']['epochs']}")):
            # Extract AIRS-CH0 signal
            airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
            
            # Process input signal
            if len(airs_signal.shape) == 3:  # [T, H, W]
                airs_signal = airs_signal.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                
            # Process target spectra
            if targets is None:
                # If no targets are available, use synthetic targets
                # In a real scenario, you'd have ground truth spectra
                targets = torch.rand(airs_signal.shape[0], config['bayesian_resnet']['n_wavelengths']).to(device)
            else:
                targets = torch.from_numpy(targets).float().to(device)
                
            # Forward pass
            mean, log_var = model(airs_signal)
            
            # Compute combined loss (heteroscedastic + continuity)
            loss = combined_loss(
                targets, 
                mean, 
                log_var, 
                continuity_weight=config['bayesian_resnet']['continuity_weight']
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc="Validation")):
                # Extract AIRS-CH0 signal
                airs_signal = torch.from_numpy(data['AIRS-CH0']['signal']).float().to(device)
                
                # Process input signal
                if len(airs_signal.shape) == 3:  # [T, H, W]
                    airs_signal = airs_signal.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                    
                # Process target spectra
                if targets is None:
                    # If no targets are available, use synthetic targets
                    targets = torch.rand(airs_signal.shape[0], config['bayesian_resnet']['n_wavelengths']).to(device)
                else:
                    targets = torch.from_numpy(targets).float().to(device)
                    
                # Forward pass
                mean, log_var = model(airs_signal)
                
                # Compute combined loss
                loss = combined_loss(
                    targets, 
                    mean, 
                    log_var, 
                    continuity_weight=config['bayesian_resnet']['continuity_weight']
                )
                
                val_loss += loss.item()
                
        # Compute average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['bayesian_resnet']['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.6f}")
            
    # Calibrate uncertainties with validation data
    if config['bayesian_resnet']['calibrate_uncertainties']:
        print("Calibrating uncertainties...")
        optimal_temperature = uq.calibrate_with_validation_data(model, val_loader, device)
        print(f"Optimal temperature: {optimal_temperature:.4f}")
        
        # Save temperature to a file
        with open(os.path.join(output_dir, 'models', 'temperature.txt'), 'w') as f:
            f.write(str(optimal_temperature))
            
    # Save final model
    final_model_path = os.path.join(output_dir, 'models', 'bayesian_resnet_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Close tensorboard writer
    writer.close()
    
    return best_model_path


def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Ariel Data Challenge models')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Save configuration to output directory
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    # Get device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Get planet IDs from data directory
    planet_ids = [d for d in os.listdir(os.path.join(args.data_dir, 'train')) if os.path.isdir(os.path.join(args.data_dir, 'train', d))]
    print(f"Found {len(planet_ids)} planets in the training set")
    
    # Split into training and validation sets
    train_ids, val_ids = get_train_val_split(planet_ids, val_ratio=config['data']['val_ratio'], random_state=args.seed)
    print(f"Training set: {len(train_ids)} planets, Validation set: {len(val_ids)} planets")
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_dir=os.path.join(args.data_dir, 'train'),
        planet_ids=train_ids,
        labels_path=os.path.join(args.data_dir, 'train_labels.csv'),
        wavelengths_path=os.path.join(args.data_dir, 'wavelengths.csv'),
        adc_info_path=os.path.join(args.data_dir, 'train_adc_info.csv'),
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        transform=preprocess_fn if config['data']['apply_preprocessing'] else None,
        mode='train',
        shuffle=True
    )
    
    val_loader = create_dataloader(
        data_dir=os.path.join(args.data_dir, 'train'),
        planet_ids=val_ids,
        labels_path=os.path.join(args.data_dir, 'train_labels.csv'),
        wavelengths_path=os.path.join(args.data_dir, 'wavelengths.csv'),
        adc_info_path=os.path.join(args.data_dir, 'train_adc_info.csv'),
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        transform=preprocess_fn if config['data']['apply_preprocessing'] else None,
        mode='val',
        shuffle=False
    )
    
    # Train models based on configuration
    if config['jnet']['train']:
        jnet_model_path = train_tcn_jnet(config, train_loader, val_loader, device, args.output_dir)
    else:
        jnet_model_path = None
        
    if config['physnet_mae']['train']:
        physnet_mae_model_path = train_physnet_mae(config, train_loader, val_loader, device, args.output_dir)
    else:
        physnet_mae_model_path = None
        
    if config['physnet']['train']:
        physnet_model_path = train_physnet(
            config, 
            train_loader, 
            val_loader, 
            device, 
            args.output_dir, 
            pretrained_path=physnet_mae_model_path if config['physnet']['use_pretrained'] else None
        )
    else:
        physnet_model_path = None
        
    if config['bayesian_resnet']['train']:
        bayesian_resnet_model_path = train_bayesian_resnet(config, train_loader, val_loader, device, args.output_dir)
    else:
        bayesian_resnet_model_path = None
        
    # Print final paths
    print("\n=== Training Complete ===")
    if jnet_model_path:
        print(f"TCN-JNet model path: {jnet_model_path}")
    if physnet_mae_model_path:
        print(f"PhySNet MAE model path: {physnet_mae_model_path}")
    if physnet_model_path:
        print(f"PhySNet model path: {physnet_model_path}")
    if bayesian_resnet_model_path:
        print(f"Bayesian ResNet model path: {bayesian_resnet_model_path}")


if __name__ == '__main__':
    main()