import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralDropout(nn.Module):
    """
    Dropout layer for spectral data
    """
    def __init__(self, p=0.2):
        """
        Initialize spectral dropout
        
        Args:
            p (float): Dropout probability
        """
        super().__init__()
        self.p = p
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, L]
            
        Returns:
            torch.Tensor: Output tensor [B, C, L]
        """
        if not self.training or self.p == 0:
            return x
            
        # Apply dropout to entire spectral channels (features)
        dropout_mask = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], 1, device=x.device) * (1 - self.p)) / (1 - self.p)
        return x * dropout_mask
        

class ResidualBlock1D(nn.Module):
    """
    1D residual block with spectral dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.2):
        """
        Initialize residual block
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            kernel_size (int): Kernel size
            stride (int): Stride
            dropout_p (float): Dropout probability
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = SpectralDropout(dropout_p)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                             stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = SpectralDropout(dropout_p)
        
        # Skip connection (if dimensions change)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, L]
            
        Returns:
            torch.Tensor: Output tensor [B, C, L]
        """
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        skip = self.skip(x)
        
        # Combine and activate
        out = F.relu(out + skip)
        out = self.dropout2(out)
        
        return out


class TemplateAttention(nn.Module):
    """
    Attention mechanism using spectral templates
    """
    def __init__(self, embed_dim, num_templates=10):
        """
        Initialize template attention
        
        Args:
            embed_dim (int): Embedding dimension
            num_templates (int): Number of spectral templates
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_templates = num_templates
        
        # Learnable spectral templates
        self.templates = nn.Parameter(torch.randn(num_templates, embed_dim))
        
        # Attention projection
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, L]
            
        Returns:
            torch.Tensor: Output tensor [B, C, L]
        """
        batch_size, channels, seq_len = x.shape
        
        # Transpose to [B, L, C]
        x_trans = x.transpose(1, 2)
        
        # Compute queries from input
        queries = self.query_proj(x_trans)  # [B, L, C]
        
        # Compute keys and values from templates
        keys = self.key_proj(self.templates)  # [T, C]
        values = self.value_proj(self.templates)  # [T, C]
        
        # Attention scores [B, L, T]
        attn_scores = torch.matmul(queries, keys.transpose(0, 1)) / np.sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum of values [B, L, C]
        context = torch.matmul(attn_weights, values)
        
        # Apply output projection
        output = self.output_proj(context)
        
        # Add residual connection
        output = output + x_trans
        
        # Transpose back to [B, C, L]
        output = output.transpose(1, 2)
        
        return output


class BayesianResNet1D(nn.Module):
    """
    Bayesian ResNet-1D for spectral extraction with uncertainty estimation
    """
    def __init__(self, 
                in_channels=32, 
                n_wavelengths=283,
                hidden_channels=[64, 128, 256, 128, 64],
                dropout_p=0.2,
                use_template_attention=True,
                num_templates=10):
        """
        Initialize Bayesian ResNet-1D
        
        Args:
            in_channels (int): Input channels
            n_wavelengths (int): Number of wavelengths in output spectrum
            hidden_channels (list): Hidden channel dimensions
            dropout_p (float): Dropout probability
            use_template_attention (bool): Whether to use template attention
            num_templates (int): Number of spectral templates
        """
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, hidden_channels[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])
        self.dropout = SpectralDropout(dropout_p)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        
        for i in range(len(hidden_channels)-1):
            self.res_blocks.append(
                ResidualBlock1D(
                    hidden_channels[i], 
                    hidden_channels[i+1], 
                    kernel_size=3, 
                    stride=2 if i < len(hidden_channels)//2 else 1,
                    dropout_p=dropout_p
                )
            )
            
        # Template attention
        self.use_template_attention = use_template_attention
        if use_template_attention:
            self.template_attn = TemplateAttention(hidden_channels[-1], num_templates)
            
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Output layers (mean and log_var)
        self.fc_mean = nn.Linear(hidden_channels[-1], n_wavelengths)
        self.fc_log_var = nn.Linear(hidden_channels[-1], n_wavelengths)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, T, H, W] or [B, C, H, W]
            
        Returns:
            tuple: Mean and log variance of spectrum
        """
        # Reshape input if needed (from 3D or 4D to 3D)
        if len(x.shape) == 5:  # [B, C, T, H, W]
            # Average over time and spatial dimensions
            x = x.mean(dim=2)  # [B, C, H, W]
            
        if len(x.shape) == 4:  # [B, C, H, W]
            # Average or max pool over one spatial dimension
            x = x.mean(dim=2)  # [B, C, W]
            
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Apply template attention if enabled
        if self.use_template_attention:
            x = self.template_attn(x)
            
        # Global average pooling
        x = self.gap(x).squeeze(-1)  # [B, C]
        
        # Output mean and log_var
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        # Ensure positive spectrum with sigmoid
        mean = torch.sigmoid(mean)
        
        # Ensure reasonable variance range
        log_var = torch.clamp(log_var, min=-10, max=2)
        
        return mean, log_var
    

class SpectralExtractionModel:
    """
    Wrapper class for spectral extraction with uncertainty quantification
    """
    def __init__(self, model_path=None, device=None, n_wavelengths=283, mc_samples=5):
        """
        Initialize spectral extraction model
        
        Args:
            model_path (str): Path to model weights
            device (torch.device): Device to use
            n_wavelengths (int): Number of wavelengths in output spectrum
            mc_samples (int): Number of MC dropout forward passes
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_wavelengths = n_wavelengths
        self.mc_samples = mc_samples
        
        # Create model
        self.model = BayesianResNet1D(
            in_channels=32,
            n_wavelengths=n_wavelengths,
            hidden_channels=[64, 128, 256, 128, 64],
            dropout_p=0.2,
            use_template_attention=True
        ).to(self.device)
        
        # Load model weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
    def extract_spectrum(self, features, mc_dropout=True):
        """
        Extract spectrum from input features
        
        Args:
            features (torch.Tensor): Input features
            mc_dropout (bool): Whether to use MC dropout for uncertainty estimation
            
        Returns:
            tuple: Mean spectrum and uncertainty
        """
        # Convert to tensor if numpy
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float().to(self.device)
            
        # Ensure batch dimension
        if len(features.shape) == 3:  # [C, H, W]
            features = features.unsqueeze(0)
            
        if not mc_dropout:
            # Single forward pass (evaluation mode)
            self.model.eval()
            with torch.no_grad():
                mean, log_var = self.model(features)
                
            # Convert to variance
            var = torch.exp(log_var)
            std = torch.sqrt(var)
            
            return mean.cpu().numpy(), std.cpu().numpy()
        else:
            # Multiple forward passes with dropout enabled
            self.model.train()  # Enable dropout
            
            # Storage for samples
            means = []
            log_vars = []
            
            with torch.no_grad():
                for _ in range(self.mc_samples):
                    mean, log_var = self.model(features)
                    means.append(mean)
                    log_vars.append(log_var)
                    
            # Stack samples
            means = torch.stack(means, dim=0)  # [S, B, L]
            log_vars = torch.stack(log_vars, dim=0)  # [S, B, L]
            
            # Convert log_vars to vars
            vars_epistemic = torch.exp(log_vars)
            
            # Total uncertainty combines epistemic (from MC samples) and aleatoric (from model prediction)
            mean_epistemic = torch.mean(means, dim=0)  # [B, L]
            var_epistemic = torch.var(means, dim=0)  # [B, L]
            var_aleatoric = torch.mean(vars_epistemic, dim=0)  # [B, L]
            
            # Total variance
            var_total = var_epistemic + var_aleatoric
            std_total = torch.sqrt(var_total)
            
            return mean_epistemic.cpu().numpy(), std_total.cpu().numpy()
    
    def apply_spectral_smoothness_prior(self, spectrum, uncertainty, smoothness_weight=0.1):
        """
        Apply spectral smoothness prior to improve uncertainty estimates
        
        Args:
            spectrum (np.ndarray): Predicted spectrum
            uncertainty (np.ndarray): Uncertainty values
            smoothness_weight (float): Weight for smoothness prior
            
        Returns:
            np.ndarray: Updated uncertainty values
        """
        # Compute spectral gradients
        gradients = np.abs(np.diff(spectrum, axis=-1))
        
        # Smoothness factor (inverse of gradients)
        smoothness = 1.0 / (1.0 + gradients)
        
        # Pad to match original size
        smoothness = np.pad(smoothness, ((0, 0), (0, 1)), mode='edge')
        
        # Scale smoothness to [0, 1]
        smoothness = smoothness / np.max(smoothness)
        
        # Apply prior to uncertainty (reduce uncertainty in smooth regions)
        updated_uncertainty = uncertainty * (1.0 - smoothness_weight * smoothness)
        
        return updated_uncertainty


def spectral_continuity_loss(mean):
    """
    Compute spectral continuity loss to enforce smooth spectra
    
    Args:
        mean (torch.Tensor): Mean prediction
        
    Returns:
        torch.Tensor: Loss value
    """
    # Compute absolute differences between adjacent wavelengths
    continuity_loss = torch.mean(torch.abs(mean[:, 1:] - mean[:, :-1]))
    
    return continuity_loss


def heteroscedastic_loss(y_true, y_pred, log_var):
    """
    Compute heteroscedastic loss for uncertainty estimation
    
    Args:
        y_true (torch.Tensor): Ground truth
        y_pred (torch.Tensor): Predicted mean
        log_var (torch.Tensor): Predicted log variance
        
    Returns:
        torch.Tensor: Loss value
    """
    # Convert log_var to precision (1/variance)
    precision = torch.exp(-log_var)
    
    # Compute loss
    loss = 0.5 * torch.mean(
        precision * (y_true - y_pred) ** 2 + log_var
    )
    
    return loss


def combined_loss(y_true, y_pred, log_var, continuity_weight=0.001):
    """
    Combine heteroscedastic loss with continuity loss
    
    Args:
        y_true (torch.Tensor): Ground truth
        y_pred (torch.Tensor): Predicted mean
        log_var (torch.Tensor): Predicted log variance
        continuity_weight (float): Weight for continuity loss
        
    Returns:
        torch.Tensor: Loss value
    """
    # Heteroscedastic loss
    h_loss = heteroscedastic_loss(y_true, y_pred, log_var)
    
    # Continuity loss
    c_loss = spectral_continuity_loss(y_pred)
    
    # Combined loss
    total_loss = h_loss + continuity_weight * c_loss
    
    return total_loss 