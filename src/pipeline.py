import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.preprocessing.preprocessing import PreprocessingPipeline
from src.jitter_correction.tcn_jnet import JitterCorrection
from src.denoising.physnet import PhySNet
from src.spectral_extraction.bayesian_resnet import SpectralExtractionModel
from src.uncertainty.uncertainty_quantification import UncertaintyQuantification


class ArielPipeline:
    """
    Complete pipeline for Ariel Data Challenge 2024
    """
    def __init__(self, 
                device=None,
                jnet_model_path=None,
                physnet_model_path=None,
                spec_model_path=None,
                n_wavelengths=283,
                mc_samples=5):
        """
        Initialize pipeline
        
        Args:
            device (torch.device): Device to use
            jnet_model_path (str): Path to TCN-JNet model weights
            physnet_model_path (str): Path to PhySNet model weights
            spec_model_path (str): Path to spectral extraction model weights
            n_wavelengths (int): Number of wavelengths
            mc_samples (int): Number of MC dropout samples
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_wavelengths = n_wavelengths
        
        # Initialize components
        self.preprocessing = PreprocessingPipeline(
            dark_subtraction=True,
            flat_correction=True,
            bad_pixel_interp=True,
            temporal_norm=True,
            interp_method='median',
            norm_method='robust_zscore'
        )
        
        self.jitter_correction = JitterCorrection(
            model_path=jnet_model_path,
            device=self.device
        )
        
        # Initialize PhySNet denoising model
        self.physnet = PhySNet(
            in_channels=1,
            out_channels=1,
            temporal=True,
            use_spectral_conv=True,
            use_physics_attn=True
        ).to(self.device)
        
        # Load PhySNet weights if provided
        if physnet_model_path:
            self.physnet.load_state_dict(torch.load(physnet_model_path, map_location=self.device))
        
        # Initialize spectral extraction model
        self.spectral_model = SpectralExtractionModel(
            model_path=spec_model_path,
            device=self.device,
            n_wavelengths=n_wavelengths,
            mc_samples=mc_samples
        )
        
        # Initialize uncertainty quantification
        self.uncertainty = UncertaintyQuantification(
            n_samples=mc_samples,
            temperature=1.0
        )
        
    def preprocess_data(self, data_dict):
        """
        Preprocess raw data
        
        Args:
            data_dict (dict): Dictionary with raw signals and calibration data
            
        Returns:
            dict: Dictionary with preprocessed data
        """
        # Apply preprocessing pipeline
        return self.preprocessing.process(data_dict)
    
    def correct_jitter(self, data_dict):
        """
        Apply jitter correction
        
        Args:
            data_dict (dict): Dictionary with preprocessed data
            
        Returns:
            dict: Dictionary with jitter-corrected data
        """
        # Create a copy to avoid modifying the original
        corrected_dict = {
            'planet_id': data_dict['planet_id'],
            'AIRS-CH0': {
                'signal': data_dict['AIRS-CH0']['signal'].copy(),
                'calibration': data_dict['AIRS-CH0']['calibration']
            },
            'FGS1': {
                'signal': data_dict['FGS1']['signal'].copy(),
                'calibration': data_dict['FGS1']['calibration']
            }
        }
        
        # Apply jitter correction
        corrected_dict['AIRS-CH0']['signal'] = self.jitter_correction.correct_jitter(
            data_dict['AIRS-CH0']['signal'],
            data_dict['FGS1']['signal']
        )
        
        return corrected_dict
    
    def denoise_signal(self, data_dict):
        """
        Apply denoising
        
        Args:
            data_dict (dict): Dictionary with jitter-corrected data
            
        Returns:
            dict: Dictionary with denoised data
        """
        # Create a copy to avoid modifying the original
        denoised_dict = {
            'planet_id': data_dict['planet_id'],
            'AIRS-CH0': {
                'signal': data_dict['AIRS-CH0']['signal'].copy(),
                'calibration': data_dict['AIRS-CH0']['calibration']
            },
            'FGS1': {
                'signal': data_dict['FGS1']['signal'].copy(),
                'calibration': data_dict['FGS1']['calibration']
            }
        }
        
        # Convert signal to tensor
        airs_signal = torch.from_numpy(data_dict['AIRS-CH0']['signal']).float().to(self.device)
        
        # Add batch and channel dimensions [B, C, T, H, W]
        airs_signal = airs_signal.unsqueeze(0).unsqueeze(0)
        
        # Apply denoising
        self.physnet.eval()
        with torch.no_grad():
            denoised_signal = self.physnet(airs_signal)
            
        # Remove batch and channel dimensions
        denoised_signal = denoised_signal.squeeze(0).squeeze(0).cpu().numpy()
        
        # Update signal
        denoised_dict['AIRS-CH0']['signal'] = denoised_signal
        
        return denoised_dict
    
    def extract_spectrum(self, data_dict, mc_dropout=True, apply_prior=True):
        """
        Extract spectrum and uncertainties from signals
        
        Args:
            data_dict (dict): Dictionary with processed signals
            mc_dropout (bool): Whether to use MC dropout
            apply_prior (bool): Whether to apply spectral smoothness prior
            
        Returns:
            tuple: Spectrum and uncertainties
        """
        # Extract AIRS-CH0 signal
        airs_signal = data_dict['AIRS-CH0']['signal']
        
        # Convert to tensor and add batch and channel dimensions if needed
        if isinstance(airs_signal, np.ndarray):
            airs_signal = torch.from_numpy(airs_signal).float().to(self.device)
            
        if len(airs_signal.shape) == 3:  # [T, H, W]
            airs_signal = airs_signal.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
            
        # Extract spectrum using the spectral model
        spectrum, uncertainty = self.spectral_model.extract_spectrum(airs_signal, mc_dropout=mc_dropout)
        
        # Apply spectral smoothness prior if requested
        if apply_prior:
            uncertainty = self.spectral_model.apply_spectral_smoothness_prior(
                spectrum, uncertainty, smoothness_weight=0.1
            )
            
        # Return first spectrum and uncertainty (if batch size is 1)
        if spectrum.shape[0] == 1:
            return spectrum[0], uncertainty[0]
        
        return spectrum, uncertainty
    
    def process_planet(self, data_dict, mc_dropout=True, apply_prior=True):
        """
        Process single planet data through the entire pipeline
        
        Args:
            data_dict (dict): Dictionary with raw signals and calibration data
            mc_dropout (bool): Whether to use MC dropout for uncertainty estimation
            apply_prior (bool): Whether to apply spectral smoothness prior
            
        Returns:
            tuple: Spectrum and uncertainties
        """
        # Preprocess data
        preprocessed_dict = self.preprocess_data(data_dict)
        
        # Apply jitter correction
        corrected_dict = self.correct_jitter(preprocessed_dict)
        
        # Apply denoising
        denoised_dict = self.denoise_signal(corrected_dict)
        
        # Extract spectrum
        spectrum, uncertainty = self.extract_spectrum(denoised_dict, mc_dropout, apply_prior)
        
        return spectrum, uncertainty
    
    def create_submission(self, planet_ids, spectra, uncertainties, output_path):
        """
        Create submission file
        
        Args:
            planet_ids (list): List of planet IDs
            spectra (np.ndarray): Extracted spectra
            uncertainties (np.ndarray): Uncertainty estimates
            output_path (str): Path to save submission file
            
        Returns:
            pd.DataFrame: Submission dataframe
        """
        # Convert to numpy arrays if not already
        if torch.is_tensor(spectra):
            spectra = spectra.cpu().numpy()
        if torch.is_tensor(uncertainties):
            uncertainties = uncertainties.cpu().numpy()
            
        # Create wavelength column names
        wavelength_cols = [f'wl_{i+1}' for i in range(self.n_wavelengths)]
        sigma_cols = [f'sigma_{i+1}' for i in range(self.n_wavelengths)]
        
        # Create dataframe
        df = pd.DataFrame()
        df['planet_id'] = planet_ids
        
        # Add spectra and uncertainties
        for i, wl_col in enumerate(wavelength_cols):
            df[wl_col] = spectra[:, i]
            
        for i, sigma_col in enumerate(sigma_cols):
            df[sigma_col] = uncertainties[:, i]
            
        # Save to file
        df.to_csv(output_path, index=False)
        
        return df


def compute_gll_metric(solution, submission, row_id_column_name, naive_mean, naive_sigma, sigma_true):
    """
    Compute Gaussian Log-Likelihood metric for evaluation
    
    Args:
        solution (pd.DataFrame): Ground truth spectra
        submission (pd.DataFrame): Predicted spectra and uncertainties
        row_id_column_name (str): Name of ID column
        naive_mean (float): Mean from train set
        naive_sigma (float): Standard deviation from train set
        sigma_true (float): True standard deviation
        
    Returns:
        float: Score between 0 and 1
    """
    # Remove ID columns
    solution_values = solution.drop(columns=[row_id_column_name])
    submission_values = submission.drop(columns=[row_id_column_name])
    
    # Check for negative values
    if submission_values.min().min() < 0:
        raise ValueError('Negative values in the submission')
    
    # Get dimensions
    n_wavelengths = len(solution_values.columns)
    
    # Split submission into means and sigmas
    y_pred = submission_values.iloc[:, :n_wavelengths].values
    sigma_pred = np.clip(submission_values.iloc[:, n_wavelengths:].values, a_min=1e-15, a_max=None)
    y_true = solution_values.values
    
    # Compute Gaussian log-likelihood
    def compute_gll(y_true, y_pred, sigma):
        return -0.5 * (np.log(2 * np.pi) + np.log(sigma**2) + (y_true - y_pred)**2 / (sigma**2))
    
    # Compute GLL for prediction, ideal case, and naive case
    gll_pred = np.sum(compute_gll(y_true, y_pred, sigma_pred))
    gll_true = np.sum(compute_gll(y_true, y_true, sigma_true * np.ones_like(y_true)))
    gll_mean = np.sum(compute_gll(y_true, naive_mean * np.ones_like(y_true), naive_sigma * np.ones_like(y_true)))
    
    # Compute final score
    score = (gll_pred - gll_mean) / (gll_true - gll_mean)
    
    # Clip score to [0, 1]
    return float(np.clip(score, 0.0, 1.0)) 