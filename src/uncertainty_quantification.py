import torch
import numpy as np
from scipy.stats import norm


class UncertaintyQuantification:
    """
    Uncertainty quantification module combining heteroscedastic outputs and MC dropout
    """
    def __init__(self, n_samples=5, temperature=1.0):
        """
        Initialize uncertainty quantification
        
        Args:
            n_samples (int): Number of MC dropout samples
            temperature (float): Temperature scaling parameter
        """
        self.n_samples = n_samples
        self.temperature = temperature
        
    def calibrate_uncertainties(self, spectra, uncertainties, targets=None, scaling_factor=None):
        """
        Calibrate uncertainty estimates
        
        Args:
            spectra (np.ndarray): Predicted spectra [B, L]
            uncertainties (np.ndarray): Uncertainty estimates [B, L]
            targets (np.ndarray, optional): Ground truth targets for calibration
            scaling_factor (float, optional): Manual scaling factor for uncertainties
            
        Returns:
            np.ndarray: Calibrated uncertainties
        """
        if scaling_factor is not None:
            # Apply manual scaling factor
            calibrated_uncertainties = uncertainties * scaling_factor
        elif targets is not None:
            # Compute appropriate scaling using validation data
            errors = np.abs(spectra - targets)
            mean_error = np.mean(errors)
            mean_uncertainty = np.mean(uncertainties)
            
            # Scaling factor based on error-to-uncertainty ratio
            scaling_factor = mean_error / max(mean_uncertainty, 1e-6)
            
            # Apply scaling
            calibrated_uncertainties = uncertainties * scaling_factor
        else:
            # No calibration if neither is provided
            calibrated_uncertainties = uncertainties
            
        return calibrated_uncertainties
    
    def apply_temperature_scaling(self, log_vars, temperature=None):
        """
        Apply temperature scaling to log variances
        
        Args:
            log_vars (torch.Tensor): Log variances [B, L]
            temperature (float, optional): Temperature (if None, use self.temperature)
            
        Returns:
            torch.Tensor: Scaled log variances
        """
        if temperature is None:
            temperature = self.temperature
            
        # Temperature scaling
        scaled_log_vars = log_vars - torch.log(torch.tensor(temperature))
        
        return scaled_log_vars
    
    def combine_mc_samples(self, means, log_vars):
        """
        Combine multiple MC dropout samples
        
        Args:
            means (torch.Tensor): Mean predictions [S, B, L]
            log_vars (torch.Tensor): Log variance predictions [S, B, L]
            
        Returns:
            tuple: Combined mean and variance
        """
        # Convert log_vars to variances
        vars_aleatoric = torch.exp(log_vars)  # [S, B, L]
        
        # Compute statistics across samples
        mean_prediction = torch.mean(means, dim=0)  # [B, L]
        var_epistemic = torch.var(means, dim=0)  # [B, L]
        var_aleatoric = torch.mean(vars_aleatoric, dim=0)  # [B, L]
        
        # Total variance is sum of epistemic and aleatoric components
        var_total = var_epistemic + var_aleatoric
        
        return mean_prediction, var_total
    
    def apply_smoothness_prior(self, spectra, uncertainties, smoothness_weight=0.1, window_size=5):
        """
        Apply smoothness prior to uncertainties
        
        Args:
            spectra (np.ndarray): Predicted spectra [B, L]
            uncertainties (np.ndarray): Uncertainty estimates [B, L]
            smoothness_weight (float): Weight for smoothness regularization
            window_size (int): Window size for local smoothness
            
        Returns:
            np.ndarray: Regularized uncertainties
        """
        batch_size, n_wavelengths = spectra.shape
        regularized = np.copy(uncertainties)
        
        for b in range(batch_size):
            # Calculate absolute gradients in spectrum
            gradients = np.abs(np.diff(spectra[b]))
            gradients = np.pad(gradients, (0, 1), mode='edge')
            
            # Normalize gradients to [0, 1]
            if np.max(gradients) > 0:
                gradients = gradients / np.max(gradients)
                
            # Compute smoothness factor (inverse of gradients)
            smoothness = 1.0 - gradients
            
            # Apply smoothness prior
            regularized[b] = uncertainties[b] * (1.0 - smoothness_weight * smoothness)
            
            # Apply local consistency with a rolling window
            if window_size > 1:
                for i in range(n_wavelengths):
                    start = max(0, i - window_size // 2)
                    end = min(n_wavelengths, i + window_size // 2 + 1)
                    local_mean = np.mean(regularized[b, start:end])
                    regularized[b, i] = 0.7 * regularized[b, i] + 0.3 * local_mean
        
        return regularized
    
    def compute_gaussian_log_likelihood(self, y_true, y_pred, y_uncertainty):
        """
        Compute Gaussian log-likelihood for model evaluation
        
        Args:
            y_true (np.ndarray): Ground truth values [B, L]
            y_pred (np.ndarray): Predicted values [B, L]
            y_uncertainty (np.ndarray): Uncertainty values [B, L]
            
        Returns:
            float: Mean log-likelihood
        """
        # Avoid division by zero
        y_uncertainty = np.maximum(y_uncertainty, 1e-6)
        
        # Compute log-likelihood for each point
        log_likelihood = norm.logpdf(y_true, loc=y_pred, scale=y_uncertainty)
        
        # Return mean log-likelihood
        return np.mean(log_likelihood)
    
    def compute_calibration_metrics(self, y_true, y_pred, y_uncertainty, bins=10):
        """
        Compute uncertainty calibration metrics
        
        Args:
            y_true (np.ndarray): Ground truth values [B, L]
            y_pred (np.ndarray): Predicted values [B, L]
            y_uncertainty (np.ndarray): Uncertainty values [B, L]
            bins (int): Number of bins for binning
            
        Returns:
            dict: Dictionary with calibration metrics
        """
        # Compute standardized errors
        standardized_errors = np.abs(y_true - y_pred) / np.maximum(y_uncertainty, 1e-6)
        
        # Reshape for easier processing
        standardized_errors = standardized_errors.flatten()
        
        # Create confidence level bins
        confidence_levels = np.linspace(0, 1, bins + 1)[1:]
        
        # For a well-calibrated model, standardized_error should be <= z_alpha
        # with probability alpha, where z_alpha is the alpha quantile of standard normal
        expected_props = confidence_levels
        observed_props = np.zeros_like(expected_props)
        
        for i, alpha in enumerate(confidence_levels):
            z_alpha = norm.ppf(alpha)
            observed_props[i] = np.mean(standardized_errors <= z_alpha)
            
        # Compute calibration error
        calibration_error = np.mean(np.abs(observed_props - expected_props))
        
        # Compute sharpness (mean uncertainty)
        sharpness = np.mean(y_uncertainty)
        
        return {
            'confidence_levels': confidence_levels,
            'expected_props': expected_props,
            'observed_props': observed_props,
            'calibration_error': calibration_error,
            'sharpness': sharpness
        }
    
    def compute_prediction_intervals(self, y_pred, y_uncertainty, confidence=0.95):
        """
        Compute prediction intervals
        
        Args:
            y_pred (np.ndarray): Predicted values [B, L]
            y_uncertainty (np.ndarray): Uncertainty values [B, L]
            confidence (float): Confidence level
            
        Returns:
            tuple: Lower and upper bounds
        """
        # Compute z-score for the given confidence level
        z = norm.ppf(0.5 + confidence / 2)
        
        # Compute bounds
        lower_bound = y_pred - z * y_uncertainty
        upper_bound = y_pred + z * y_uncertainty
        
        return lower_bound, upper_bound
    
    def compute_ensemble_predictions(self, model, inputs, n_samples=None):
        """
        Compute ensemble predictions using MC dropout
        
        Args:
            model (torch.nn.Module): Model with dropout
            inputs (torch.Tensor): Input tensor
            n_samples (int, optional): Number of samples (if None, use self.n_samples)
            
        Returns:
            tuple: Mean and variance
        """
        if n_samples is None:
            n_samples = self.n_samples
            
        # Enable dropout
        model.train()
        
        # Storage for predictions
        means = []
        log_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, log_var = model(inputs)
                means.append(mean)
                log_vars.append(log_var)
                
        # Stack predictions
        means = torch.stack(means, dim=0)  # [S, B, L]
        log_vars = torch.stack(log_vars, dim=0)  # [S, B, L]
        
        # Apply temperature scaling to log_vars
        scaled_log_vars = self.apply_temperature_scaling(log_vars)
        
        # Combine predictions
        mean_pred, var_pred = self.combine_mc_samples(means, scaled_log_vars)
        
        return mean_pred, torch.sqrt(var_pred)  # Return mean and std
    
    def calibrate_with_validation_data(self, model, val_loader, device):
        """
        Calibrate model uncertainties using validation data
        
        Args:
            model (torch.nn.Module): Model
            val_loader (torch.utils.data.DataLoader): Validation data loader
            device (torch.device): Device
            
        Returns:
            float: Optimal temperature value
        """
        # Lists to store predictions and targets
        all_means = []
        all_log_vars = []
        all_targets = []
        
        # Get predictions for validation set
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                mean, log_var = model(inputs)
                
                all_means.append(mean.cpu())
                all_log_vars.append(log_var.cpu())
                all_targets.append(targets.cpu())
                
        # Concatenate results
        all_means = torch.cat(all_means, dim=0)
        all_log_vars = torch.cat(all_log_vars, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Define loss function for optimization
        def nll_loss(temperature):
            # Apply temperature scaling
            scaled_log_vars = all_log_vars - torch.log(torch.tensor(temperature))
            variances = torch.exp(scaled_log_vars)
            
            # Compute negative log-likelihood
            nll = 0.5 * torch.mean(
                torch.log(variances) + (all_targets - all_means)**2 / variances
            )
            
            return nll.item()
        
        # Simple grid search for optimal temperature
        temperatures = np.logspace(-1, 1, 20)
        losses = [nll_loss(t) for t in temperatures]
        
        # Find optimal temperature
        optimal_idx = np.argmin(losses)
        optimal_temperature = temperatures[optimal_idx]
        
        # Update temperature
        self.temperature = optimal_temperature
        
        return optimal_temperature 