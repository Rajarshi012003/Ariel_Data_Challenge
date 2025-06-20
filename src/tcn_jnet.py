import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with dilation
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Initialize temporal block
        
        Args:
            n_inputs (int): Number of input channels
            n_outputs (int): Number of output channels
            kernel_size (int): Size of the convolutional kernel
            stride (int): Stride of the convolution
            dilation (int): Dilation rate
            padding (int): Padding size
            dropout (float): Dropout rate
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.dropout1,
            self.conv2, self.bn2, nn.ReLU(), self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, T]
            
        Returns:
            torch.Tensor: Output tensor [B, C, T]
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with multiple layers of dilated convolutions
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        Initialize TCN
        
        Args:
            num_inputs (int): Number of input channels
            num_channels (list): Number of channels in each layer
            kernel_size (int): Kernel size
            dropout (float): Dropout rate
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, 
                             stride=1, dilation=dilation_size, 
                             padding=padding, dropout=dropout)
            )
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, T]
            
        Returns:
            torch.Tensor: Output tensor [B, C, T]
        """
        return self.network(x)


class SpatialFeatureExtractor(nn.Module):
    """
    Extract spatial features from image frames
    """
    def __init__(self, in_channels=1, out_channels=32):
        """
        Initialize spatial feature extractor
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(SpatialFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Output tensor [B, C_out, H/2, W/2]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class TemporalFeatureExtractor(nn.Module):
    """
    Extract temporal features from sequences
    """
    def __init__(self, in_channels, hidden_channels=[32, 64, 128]):
        """
        Initialize temporal feature extractor
        
        Args:
            in_channels (int): Number of input channels
            hidden_channels (list): Number of hidden channels
        """
        super(TemporalFeatureExtractor, self).__init__()
        
        self.tcn = TemporalConvNet(in_channels, hidden_channels)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, T]
            
        Returns:
            torch.Tensor: Output tensor [B, C_out, T]
        """
        return self.tcn(x)


class CrossCorrelationLayer(nn.Module):
    """
    Cross-correlation layer to match FGS1 jitter patterns with AIRS-CH0 spatial features
    """
    def __init__(self, max_displacement=10):
        """
        Initialize cross-correlation layer
        
        Args:
            max_displacement (int): Maximum displacement for correlation
        """
        super(CrossCorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        
    def forward(self, x1, x2):
        """
        Forward pass
        
        Args:
            x1 (torch.Tensor): First input tensor [B, C, T]
            x2 (torch.Tensor): Second input tensor [B, C, T]
            
        Returns:
            torch.Tensor: Cross-correlation tensor
        """
        batch_size, channels, length = x1.size()
        
        # Initialize output
        out = torch.zeros(batch_size, 2*self.max_displacement+1, length, device=x1.device)
        
        # Compute cross-correlation for different displacements
        for d in range(-self.max_displacement, self.max_displacement+1):
            if d > 0:
                # Shift x2 right
                x2_shifted = F.pad(x2[:, :, :-d], (d, 0), "constant", 0)
            elif d < 0:
                # Shift x2 left
                x2_shifted = F.pad(x2[:, :, -d:], (0, -d), "constant", 0)
            else:
                # No shift
                x2_shifted = x2
                
            # Compute correlation
            corr = torch.sum(x1 * x2_shifted, dim=1)
            out[:, d+self.max_displacement, :] = corr
            
        return out


class PSFConstrainedAlignment(nn.Module):
    """
    PSF-constrained alignment layer
    """
    def __init__(self, psf_template, groups=32):
        """
        Initialize PSF-constrained alignment
        
        Args:
            psf_template (torch.Tensor): PSF template
            groups (int): Number of groups for convolution
        """
        super(PSFConstrainedAlignment, self).__init__()
        self.register_buffer('psf_template', psf_template)
        self.groups = groups
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, T]
            
        Returns:
            torch.Tensor: Aligned tensor
        """
        # Apply PSF-constrained alignment using grouped convolution
        return F.conv1d(x, self.psf_template.unsqueeze(1).unsqueeze(0), 
                      groups=self.groups, padding=self.psf_template.size(0)//2)


class TCNJNet(nn.Module):
    """
    TCN-JNet for jitter correction
    """
    def __init__(self, 
                fgs_spatial_features=32, 
                airs_spatial_features=32,
                hidden_channels=[32, 64, 128, 256],
                max_displacement=10,
                psf_template=None):
        """
        Initialize TCN-JNet
        
        Args:
            fgs_spatial_features (int): Number of spatial features for FGS1
            airs_spatial_features (int): Number of spatial features for AIRS-CH0
            hidden_channels (list): Number of hidden channels for TCN
            max_displacement (int): Maximum displacement for cross-correlation
            psf_template (torch.Tensor): PSF template
        """
        super(TCNJNet, self).__init__()
        
        # Spatial feature extractors
        self.fgs_spatial = SpatialFeatureExtractor(1, fgs_spatial_features)
        self.airs_spatial = SpatialFeatureExtractor(1, airs_spatial_features)
        
        # Temporal feature extractors
        self.fgs_temporal = TemporalFeatureExtractor(fgs_spatial_features, hidden_channels)
        self.airs_temporal = TemporalFeatureExtractor(airs_spatial_features, hidden_channels)
        
        # Cross-correlation layer
        self.cross_corr = CrossCorrelationLayer(max_displacement)
        
        # PSF-constrained alignment
        if psf_template is None:
            # Generate a default Gaussian PSF template
            x = torch.arange(-10, 11).float()
            psf_template = torch.exp(-0.5 * (x / 2.0)**2)
            psf_template = psf_template / psf_template.sum()
            
        self.psf_align = PSFConstrainedAlignment(psf_template, groups=hidden_channels[-1])
        
        # Final layers
        self.conv1 = nn.Conv1d(hidden_channels[-1], hidden_channels[-1], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels[-1])
        self.conv2 = nn.Conv1d(hidden_channels[-1], 2, kernel_size=1)  # x, y jitter
        
    def forward(self, fgs_frames, airs_frames):
        """
        Forward pass
        
        Args:
            fgs_frames (torch.Tensor): FGS1 frames [B, T, H, W]
            airs_frames (torch.Tensor): AIRS-CH0 frames [B, T, H, W]
            
        Returns:
            tuple: Jitter correction values and aligned AIRS-CH0 frames
        """
        batch_size, time_steps, height, width = fgs_frames.shape
        
        # Process each frame
        fgs_features = []
        airs_features = []
        
        for t in range(time_steps):
            # Extract spatial features
            fgs_feat = self.fgs_spatial(fgs_frames[:, t, :, :].unsqueeze(1))
            airs_feat = self.airs_spatial(airs_frames[:, t, :, :].unsqueeze(1))
            
            # Flatten spatial dimensions
            fgs_feat = fgs_feat.view(batch_size, -1)
            airs_feat = airs_feat.view(batch_size, -1)
            
            fgs_features.append(fgs_feat)
            airs_features.append(airs_feat)
            
        # Stack features along time dimension
        fgs_features = torch.stack(fgs_features, dim=2)  # [B, C, T]
        airs_features = torch.stack(airs_features, dim=2)  # [B, C, T]
        
        # Extract temporal features
        fgs_temporal = self.fgs_temporal(fgs_features)
        airs_temporal = self.airs_temporal(airs_features)
        
        # Cross-correlate
        corr_features = self.cross_corr(fgs_temporal, airs_temporal)
        
        # Apply PSF constraint
        aligned_features = self.psf_align(corr_features)
        
        # Predict jitter
        x = F.relu(self.bn1(self.conv1(aligned_features)))
        jitter = self.conv2(x)  # [B, 2, T]
        
        # Transpose to [B, T, 2] for easier interpretation
        jitter = jitter.permute(0, 2, 1)  # [B, T, 2]
        
        return jitter, aligned_features


class JitterCorrection:
    """
    Jitter correction using TCN-JNet
    """
    def __init__(self, model_path=None, device=None):
        """
        Initialize jitter correction
        
        Args:
            model_path (str): Path to model weights
            device (torch.device): Device to use
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = TCNJNet().to(self.device)
        
        # Load model weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
    def correct_jitter(self, airs_signal, fgs_signal):
        """
        Correct jitter in AIRS-CH0 signal using FGS1 signal
        
        Args:
            airs_signal (np.ndarray): AIRS-CH0 signal [T, H, W]
            fgs_signal (np.ndarray): FGS1 signal [T, H, W]
            
        Returns:
            np.ndarray: Jitter-corrected AIRS-CH0 signal
        """
        # Convert to torch tensors
        airs_tensor = torch.from_numpy(airs_signal).float().to(self.device)
        fgs_tensor = torch.from_numpy(fgs_signal).float().to(self.device)
        
        # Downsample FGS1 to match AIRS-CH0 time steps if needed
        t_airs = airs_tensor.shape[0]
        t_fgs = fgs_tensor.shape[0]
        
        if t_fgs > t_airs:
            # Take every (t_fgs / t_airs) frames
            step = t_fgs // t_airs
            fgs_tensor = fgs_tensor[::step, :, :]
            
        # Add batch dimension
        airs_tensor = airs_tensor.unsqueeze(0)  # [1, T, H, W]
        fgs_tensor = fgs_tensor.unsqueeze(0)  # [1, T, H, W]
        
        # Get jitter predictions
        with torch.no_grad():
            jitter, _ = self.model(fgs_tensor, airs_tensor)
            
        # Apply jitter correction
        corrected_signal = self._apply_jitter_correction(airs_tensor, jitter)
        
        return corrected_signal.squeeze(0).cpu().numpy()
    
    def _apply_jitter_correction(self, signal, jitter):
        """
        Apply jitter correction to signal
        
        Args:
            signal (torch.Tensor): Signal [B, T, H, W]
            jitter (torch.Tensor): Jitter values [B, T, 2]
            
        Returns:
            torch.Tensor: Jitter-corrected signal
        """
        batch_size, time_steps, height, width = signal.shape
        device = signal.device
        
        # Create sampling grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device).float(),
            torch.arange(width, device=device).float()
        )
        
        grid = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, time_steps, 1, 1, 1)  # [B, T, H, W, 2]
        
        # Apply jitter correction
        jitter_expanded = jitter.unsqueeze(2).unsqueeze(3).repeat(1, 1, height, width, 1)  # [B, T, H, W, 2]
        corrected_grid = grid - jitter_expanded
        
        # Normalize grid coordinates to [-1, 1]
        corrected_grid[:, :, :, :, 0] = 2.0 * corrected_grid[:, :, :, :, 0] / (width - 1) - 1.0
        corrected_grid[:, :, :, :, 1] = 2.0 * corrected_grid[:, :, :, :, 1] / (height - 1) - 1.0
        
        # Reshape signal and grid for grid_sample
        signal_reshaped = signal.view(-1, 1, height, width)  # [B*T, 1, H, W]
        grid_reshaped = corrected_grid.view(-1, height, width, 2)  # [B*T, H, W, 2]
        
        # Apply sampling
        corrected_signal = F.grid_sample(
            signal_reshaped, grid_reshaped, 
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Reshape back
        corrected_signal = corrected_signal.view(batch_size, time_steps, height, width)
        
        return corrected_signal 