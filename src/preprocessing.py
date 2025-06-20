import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DarkCurrentSubtraction:
    """
    Class for subtracting dark current from signal data
    """
    def __init__(self, polynomial_degree=3):
        """
        Initialize dark current subtraction
        
        Args:
            polynomial_degree (int): Degree of polynomial for fitting
        """
        self.polynomial_degree = polynomial_degree
        
    def __call__(self, signal, dark_frame):
        """
        Subtract dark current from signal
        
        Args:
            signal (np.ndarray): Signal data
            dark_frame (pd.DataFrame or np.ndarray): Dark frame data
            
        Returns:
            np.ndarray: Signal with dark current subtracted
        """
        # Convert dark frame to numpy array and reshape if it's a DataFrame
        if hasattr(dark_frame, 'values'):
            dark_values = dark_frame.values
        else:
            dark_values = dark_frame
        
        if len(signal.shape) == 3:  # (t, h, w)
            _, h, w = signal.shape
            dark_reshaped = dark_values.reshape(h, w)
            
            # Subtract dark current from each frame
            corrected_signal = signal - dark_reshaped[np.newaxis, :, :]
        else:
            raise ValueError("Signal must have shape (t, h, w)")
            
        return corrected_signal


class FlatFieldCorrection:
    """
    Class for flat field correction
    """
    def __init__(self):
        """
        Initialize flat field correction
        """
        pass
        
    def __call__(self, signal, flat_frame):
        """
        Apply flat field correction to signal
        
        Args:
            signal (np.ndarray): Signal data
            flat_frame (pd.DataFrame or np.ndarray): Flat field frame data
            
        Returns:
            np.ndarray: Signal with flat field correction applied
        """
        # Convert flat field to numpy array and reshape if it's a DataFrame
        if hasattr(flat_frame, 'values'):
            flat_values = flat_frame.values
        else:
            flat_values = flat_frame
        
        if len(signal.shape) == 3:  # (t, h, w)
            _, h, w = signal.shape
            flat_reshaped = flat_values.reshape(h, w)
            
            # Avoid division by zero
            flat_reshaped = np.maximum(flat_reshaped, 1e-6)
            
            # Normalize flat field (optional)
            flat_normalized = flat_reshaped / np.mean(flat_reshaped)
            
            # Apply flat field correction
            corrected_signal = signal / flat_normalized[np.newaxis, :, :]
        else:
            raise ValueError("Signal must have shape (t, h, w)")
            
        return corrected_signal


class BadPixelInterpolation:
    """
    Class for interpolating bad pixels
    """
    def __init__(self, method='median', kernel_size=3):
        """
        Initialize bad pixel interpolation
        
        Args:
            method (str): Interpolation method ('median', 'mean', or 'unet')
            kernel_size (int): Size of kernel for neighborhood interpolation
        """
        self.method = method
        self.kernel_size = kernel_size
        self.unet_model = None
        
        # If using UNet, initialize model
        if method == 'unet':
            self.unet_model = self._init_unet()
            
    def _init_unet(self):
        """
        Initialize U-Net model for bad pixel inpainting
        
        Returns:
            nn.Module: UNet model
        """
        # Simple UNet for inpainting
        class UNetInpaint(nn.Module):
            def __init__(self):
                super(UNetInpaint, self).__init__()
                # Encoder
                self.enc1 = nn.Conv2d(2, 16, 3, padding=1)  # Input: image + mask
                self.enc2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
                self.enc3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
                
                # Decoder
                self.dec3 = nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2)
                self.dec2 = nn.ConvTranspose2d(32 + 32, 16, 4, padding=1, stride=2)
                self.dec1 = nn.Conv2d(16 + 16, 1, 3, padding=1)
                
            def forward(self, x, mask):
                # x: (b, 1, h, w), mask: (b, 1, h, w)
                x_in = torch.cat([x, mask], dim=1)  # (b, 2, h, w)
                
                # Encoding
                e1 = F.relu(self.enc1(x_in))  # (b, 16, h, w)
                e2 = F.relu(self.enc2(e1))  # (b, 32, h/2, w/2)
                e3 = F.relu(self.enc3(e2))  # (b, 64, h/4, w/4)
                
                # Decoding with skip connections
                d3 = F.relu(self.dec3(e3))  # (b, 32, h/2, w/2)
                d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))  # (b, 16, h, w)
                d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (b, 1, h, w)
                
                # Replace only masked pixels
                return x * (1 - mask) + d1 * mask
        
        return UNetInpaint()
        
    def _median_interpolation(self, frame, bad_pixel_mask):
        """
        Interpolate bad pixels using median of neighborhood
        
        Args:
            frame (np.ndarray): Single frame
            bad_pixel_mask (np.ndarray): Mask indicating bad pixels
            
        Returns:
            np.ndarray: Frame with bad pixels interpolated
        """
        h, w = frame.shape
        corrected_frame = frame.copy()
        
        # Get indices of bad pixels
        bad_indices = np.where(bad_pixel_mask)
        
        half_k = self.kernel_size // 2
        
        # For each bad pixel
        for y, x in zip(bad_indices[0], bad_indices[1]):
            # Define neighborhood bounds
            y_min = max(0, y - half_k)
            y_max = min(h, y + half_k + 1)
            x_min = max(0, x - half_k)
            x_max = min(w, x + half_k + 1)
            
            # Extract neighborhood excluding bad pixels
            neighborhood = frame[y_min:y_max, x_min:x_max]
            neighborhood_mask = bad_pixel_mask[y_min:y_max, x_min:x_max]
            good_values = neighborhood[~neighborhood_mask]
            
            # If there are good pixels in the neighborhood
            if len(good_values) > 0:
                corrected_frame[y, x] = np.median(good_values)
            else:
                # If no good pixels in neighborhood, use global median
                corrected_frame[y, x] = np.median(frame[~bad_pixel_mask])
                
        return corrected_frame
    
    def _mean_interpolation(self, frame, bad_pixel_mask):
        """
        Interpolate bad pixels using mean of neighborhood
        
        Args:
            frame (np.ndarray): Single frame
            bad_pixel_mask (np.ndarray): Mask indicating bad pixels
            
        Returns:
            np.ndarray: Frame with bad pixels interpolated
        """
        h, w = frame.shape
        corrected_frame = frame.copy()
        
        # Get indices of bad pixels
        bad_indices = np.where(bad_pixel_mask)
        
        half_k = self.kernel_size // 2
        
        # For each bad pixel
        for y, x in zip(bad_indices[0], bad_indices[1]):
            # Define neighborhood bounds
            y_min = max(0, y - half_k)
            y_max = min(h, y + half_k + 1)
            x_min = max(0, x - half_k)
            x_max = min(w, x + half_k + 1)
            
            # Extract neighborhood excluding bad pixels
            neighborhood = frame[y_min:y_max, x_min:x_max]
            neighborhood_mask = bad_pixel_mask[y_min:y_max, x_min:x_max]
            good_values = neighborhood[~neighborhood_mask]
            
            # If there are good pixels in the neighborhood
            if len(good_values) > 0:
                corrected_frame[y, x] = np.mean(good_values)
            else:
                # If no good pixels in neighborhood, use global mean
                corrected_frame[y, x] = np.mean(frame[~bad_pixel_mask])
                
        return corrected_frame
    
    def _unet_interpolation(self, signal, bad_pixel_mask):
        """
        Interpolate bad pixels using UNet
        
        Args:
            signal (np.ndarray): Signal data (t, h, w)
            bad_pixel_mask (np.ndarray): Mask indicating bad pixels (h, w)
            
        Returns:
            np.ndarray: Signal with bad pixels interpolated
        """
        if self.unet_model is None:
            raise ValueError("UNet model not initialized")
        
        device = next(self.unet_model.parameters()).device
        
        # Convert to torch tensors
        signal_tensor = torch.from_numpy(signal).float().to(device)
        mask_tensor = torch.from_numpy(bad_pixel_mask).float().to(device)
        
        # Process each frame
        corrected_signal = []
        for i in range(signal.shape[0]):
            frame = signal_tensor[i].unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
            mask = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
            
            with torch.no_grad():
                corrected_frame = self.unet_model(frame, mask)
                
            corrected_signal.append(corrected_frame.squeeze().cpu().numpy())
            
        return np.stack(corrected_signal)
        
    def __call__(self, signal, dead_pixel_frame):
        """
        Interpolate bad pixels in signal
        
        Args:
            signal (np.ndarray): Signal data
            dead_pixel_frame (pd.DataFrame or np.ndarray): Dead pixel data
            
        Returns:
            np.ndarray: Signal with bad pixels interpolated
        """
        # Convert dead pixel frame to numpy array and reshape if it's a DataFrame
        if hasattr(dead_pixel_frame, 'values'):
            dead_values = dead_pixel_frame.values
        else:
            dead_values = dead_pixel_frame
        
        if len(signal.shape) == 3:  # (t, h, w)
            t, h, w = signal.shape
            dead_pixel_mask = dead_values.reshape(h, w).astype(bool)
            
            # Skip if no bad pixels
            if not np.any(dead_pixel_mask):
                return signal
            
            # Apply interpolation based on method
            if self.method == 'median':
                corrected_signal = np.zeros_like(signal)
                for i in range(t):
                    corrected_signal[i] = self._median_interpolation(signal[i], dead_pixel_mask)
                    
            elif self.method == 'mean':
                corrected_signal = np.zeros_like(signal)
                for i in range(t):
                    corrected_signal[i] = self._mean_interpolation(signal[i], dead_pixel_mask)
                    
            elif self.method == 'unet':
                corrected_signal = self._unet_interpolation(signal, dead_pixel_mask)
                
            else:
                raise ValueError(f"Unknown interpolation method: {self.method}")
                
            return corrected_signal
        else:
            raise ValueError("Signal must have shape (t, h, w)")


class TemporalNormalization:
    """
    Class for temporal normalization of signal data
    """
    def __init__(self, method='robust_zscore', clip_quantile=0.05):
        """
        Initialize temporal normalization
        
        Args:
            method (str): Normalization method ('zscore', 'robust_zscore', 'minmax')
            clip_quantile (float): Quantile for clipping outliers
        """
        self.method = method
        self.clip_quantile = clip_quantile
        
    def _zscore_normalization(self, signal):
        """
        Apply Z-score normalization
        
        Args:
            signal (np.ndarray): Signal data
            
        Returns:
            np.ndarray: Normalized signal
        """
        mean = np.mean(signal)
        std = np.std(signal)
        
        # Avoid division by zero
        if std < 1e-6:
            std = 1.0
            
        return (signal - mean) / std
    
    def _robust_zscore_normalization(self, signal):
        """
        Apply robust Z-score normalization with outlier clipping
        
        Args:
            signal (np.ndarray): Signal data
            
        Returns:
            np.ndarray: Normalized signal
        """
        # Flatten signal for computing statistics
        flattened = signal.flatten()
        
        # Compute quantiles for clipping
        lower_quantile = np.quantile(flattened, self.clip_quantile)
        upper_quantile = np.quantile(flattened, 1 - self.clip_quantile)
        
        # Clip outliers
        clipped = np.clip(signal, lower_quantile, upper_quantile)
        
        # Compute median and MAD
        median = np.median(clipped)
        mad = np.median(np.abs(clipped - median))
        
        # Avoid division by zero (MAD can be 0 for constant signals)
        if mad < 1e-6:
            mad = 1.0
            
        # Scale factor to make MAD estimate consistent with std for normal distribution
        scale_factor = 1.4826
        
        return (signal - median) / (scale_factor * mad)
    
    def _minmax_normalization(self, signal):
        """
        Apply Min-Max normalization
        
        Args:
            signal (np.ndarray): Signal data
            
        Returns:
            np.ndarray: Normalized signal
        """
        # Flatten signal for computing statistics
        flattened = signal.flatten()
        
        # Compute quantiles for robust min and max
        min_val = np.quantile(flattened, self.clip_quantile)
        max_val = np.quantile(flattened, 1 - self.clip_quantile)
        
        # Avoid division by zero
        if max_val - min_val < 1e-6:
            return np.zeros_like(signal)
            
        return (signal - min_val) / (max_val - min_val)
        
    def __call__(self, signal):
        """
        Apply temporal normalization to signal
        
        Args:
            signal (np.ndarray): Signal data
            
        Returns:
            np.ndarray: Normalized signal
        """
        if self.method == 'zscore':
            return self._zscore_normalization(signal)
        elif self.method == 'robust_zscore':
            return self._robust_zscore_normalization(signal)
        elif self.method == 'minmax':
            return self._minmax_normalization(signal)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline combining all operations
    """
    def __init__(self, 
                dark_subtraction=True,
                flat_correction=True,
                bad_pixel_interp=True,
                temporal_norm=True,
                interp_method='median',
                norm_method='robust_zscore'):
        """
        Initialize preprocessing pipeline
        
        Args:
            dark_subtraction (bool): Whether to apply dark current subtraction
            flat_correction (bool): Whether to apply flat field correction
            bad_pixel_interp (bool): Whether to apply bad pixel interpolation
            temporal_norm (bool): Whether to apply temporal normalization
            interp_method (str): Bad pixel interpolation method
            norm_method (str): Temporal normalization method
        """
        self.dark_subtraction = dark_subtraction
        self.flat_correction = flat_correction
        self.bad_pixel_interp = bad_pixel_interp
        self.temporal_norm = temporal_norm
        
        # Initialize components
        self.dark_subtractor = DarkCurrentSubtraction() if dark_subtraction else None
        self.flat_corrector = FlatFieldCorrection() if flat_correction else None
        self.bad_pixel_interpolator = BadPixelInterpolation(method=interp_method) if bad_pixel_interp else None
        self.temporal_normalizer = TemporalNormalization(method=norm_method) if temporal_norm else None
        
    def process(self, data_dict):
        """
        Process data through the pipeline
        
        Args:
            data_dict (dict): Dictionary containing signal and calibration data
            
        Returns:
            dict: Dictionary with processed data
        """
        # Create a copy to avoid modifying the original
        processed_dict = {
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
        
        # Handle batched data - we need to process each item in the batch separately
        is_batched = len(processed_dict['AIRS-CH0']['signal'].shape) == 4
        
        # Process AIRS-CH0 data
        if is_batched:
            # For batched data, process each batch item separately
            batch_size = processed_dict['AIRS-CH0']['signal'].shape[0]
            processed_signals = []
            
            for b in range(batch_size):
                signal = processed_dict['AIRS-CH0']['signal'][b]
                calib = {k: v[b] if isinstance(v, np.ndarray) and len(v.shape) > 2 else v 
                        for k, v in processed_dict['AIRS-CH0']['calibration'].items()}
                
                # Apply processing steps
                if self.dark_subtraction:
                    signal = self.dark_subtractor(signal, calib['dark'])
                if self.flat_correction:
                    signal = self.flat_corrector(signal, calib['flat'])
                if self.bad_pixel_interp:
                    signal = self.bad_pixel_interpolator(signal, calib['dead'])
                if self.temporal_norm:
                    signal = self.temporal_normalizer(signal)
                
                processed_signals.append(signal)
            
            # Combine processed signals back into batch
            processed_dict['AIRS-CH0']['signal'] = np.stack(processed_signals)
        else:
            # Process non-batched data
            if self.dark_subtraction:
                processed_dict['AIRS-CH0']['signal'] = self.dark_subtractor(
                    processed_dict['AIRS-CH0']['signal'],
                    processed_dict['AIRS-CH0']['calibration']['dark']
                )
            if self.flat_correction:
                processed_dict['AIRS-CH0']['signal'] = self.flat_corrector(
                    processed_dict['AIRS-CH0']['signal'],
                    processed_dict['AIRS-CH0']['calibration']['flat']
                )
            if self.bad_pixel_interp:
                processed_dict['AIRS-CH0']['signal'] = self.bad_pixel_interpolator(
                    processed_dict['AIRS-CH0']['signal'],
                    processed_dict['AIRS-CH0']['calibration']['dead']
                )
            if self.temporal_norm:
                processed_dict['AIRS-CH0']['signal'] = self.temporal_normalizer(
                    processed_dict['AIRS-CH0']['signal']
                )
        
        # Process FGS1 data
        if is_batched:
            # For batched data, process each batch item separately
            batch_size = processed_dict['FGS1']['signal'].shape[0]
            processed_signals = []
            
            for b in range(batch_size):
                signal = processed_dict['FGS1']['signal'][b]
                calib = {k: v[b] if isinstance(v, np.ndarray) and len(v.shape) > 2 else v 
                        for k, v in processed_dict['FGS1']['calibration'].items()}
                
                # Apply processing steps
                if self.dark_subtraction:
                    signal = self.dark_subtractor(signal, calib['dark'])
                if self.flat_correction:
                    signal = self.flat_corrector(signal, calib['flat'])
                if self.bad_pixel_interp:
                    signal = self.bad_pixel_interpolator(signal, calib['dead'])
                if self.temporal_norm:
                    signal = self.temporal_normalizer(signal)
                
                processed_signals.append(signal)
            
            # Combine processed signals back into batch
            processed_dict['FGS1']['signal'] = np.stack(processed_signals)
        else:
            # Process non-batched data
            if self.dark_subtraction:
                processed_dict['FGS1']['signal'] = self.dark_subtractor(
                    processed_dict['FGS1']['signal'],
                    processed_dict['FGS1']['calibration']['dark']
                )
            if self.flat_correction:
                processed_dict['FGS1']['signal'] = self.flat_corrector(
                    processed_dict['FGS1']['signal'],
                    processed_dict['FGS1']['calibration']['flat']
                )
            if self.bad_pixel_interp:
                processed_dict['FGS1']['signal'] = self.bad_pixel_interpolator(
                    processed_dict['FGS1']['signal'],
                    processed_dict['FGS1']['calibration']['dead']
                )
            if self.temporal_norm:
                processed_dict['FGS1']['signal'] = self.temporal_normalizer(
                    processed_dict['FGS1']['signal']
                )
            
        return processed_dict 