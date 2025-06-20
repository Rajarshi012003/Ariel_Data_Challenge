import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXtV2 block with temporal support
    """
    def __init__(self, dim, dim_out, temporal=False):
        """
        Initialize ConvNeXt block
        
        Args:
            dim (int): Input dimension
            dim_out (int): Output dimension
            temporal (bool): Whether to use temporal convolutions
        """
        super().__init__()
        self.temporal = temporal
        
        # Depthwise convolution
        if temporal:
            self.dwconv = nn.Conv3d(dim, dim, kernel_size=(3, 7, 7), padding=(1, 3, 3), groups=dim)
        else:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim_out)
        
        # Skip connection
        self.skip = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        input = x
        
        # Apply depthwise convolution
        if self.temporal:
            # x: [B, C, T, H, W]
            x = self.dwconv(x)
            x = rearrange(x, 'b c t h w -> b t h w c')
        else:
            # x: [B, C, H, W]
            x = self.dwconv(x)
            x = rearrange(x, 'b c h w -> b h w c')
        
        # Channel-wise normalization and projection
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Reshape back
        if self.temporal:
            x = rearrange(x, 'b t h w c -> b c t h w')
            input = rearrange(input, 'b c t h w -> b t h w c')
            input = self.skip(input)
            input = rearrange(input, 'b t h w c -> b c t h w')
        else:
            x = rearrange(x, 'b h w c -> b c h w')
            input = rearrange(input, 'b c h w -> b h w c')
            input = self.skip(input)
            input = rearrange(input, 'b h w c -> b c h w')
            
        return x + input


class SpectralConv1d(nn.Module):
    """
    Spectral convolution layer for wavelength-aware features
    """
    def __init__(self, in_channels, out_channels, modes=16):
        """
        Initialize spectral convolution
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            modes (int): Number of Fourier modes to keep
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep
        
        # Complex weights for Fourier space convolution
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, 2, dtype=torch.float32)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, L]
            
        Returns:
            torch.Tensor: Output tensor [B, C_out, L]
        """
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=2)
        
        # Limit to modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1)//2 + 1, 
                           device=x.device, dtype=torch.cfloat)
        
        # Apply convolution in Fourier space
        modes = min(self.modes, x_ft.size(-1))
        weights_complex = torch.complex(self.weights[..., 0], self.weights[..., 1])
        
        # Matrix multiplication in complex space
        out_ft[:, :, :modes] = torch.einsum("bix,iox->box", x_ft[:, :, :modes], weights_complex[:, :, :modes])
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=2)
        
        return x


class PhysicsAttention(nn.Module):
    """
    Physics-aware attention mechanism
    """
    def __init__(self, psf_template=None, embed_dim=32):
        """
        Initialize physics attention
        
        Args:
            psf_template (torch.Tensor): PSF template
            embed_dim (int): Embedding dimension
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Generate PSF mask if not provided
        if psf_template is None:
            # Generate a default Gaussian PSF
            x = torch.arange(-10, 11).float()
            psf_template = torch.exp(-0.5 * (x / 2.0)**2)
            psf_template = psf_template / psf_template.sum()
            
        self.register_buffer('psf_mask', self._create_psf_mask(psf_template))
        
        # Attention layers
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def _create_psf_mask(self, psf_template):
        """
        Create PSF attention mask from template
        
        Args:
            psf_template (torch.Tensor): 1D PSF template
            
        Returns:
            torch.Tensor: 2D PSF mask
        """
        # Create 2D PSF mask from 1D template
        length = psf_template.size(0)
        mask = torch.zeros(length, length)
        
        for i in range(length):
            for j in range(length):
                dist = abs(i - j)
                if dist < length:
                    mask[i, j] = psf_template[dist]
                    
        return mask
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, L]
            
        Returns:
            torch.Tensor: Output tensor [B, C, L]
        """
        batch_size, channels, seq_len = x.shape
        
        # Reshape for attention
        x_reshaped = x.permute(0, 2, 1)  # [B, L, C]
        
        # Project to query, key, value
        qkv = self.qkv_proj(x_reshaped)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.embed_dim)
        qkv = qkv.permute(2, 0, 1, 3)  # [3, B, L, C]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        
        # Apply PSF mask
        psf_mask = self.psf_mask
        if psf_mask.size(0) != seq_len:
            # Resize PSF mask if needed
            psf_mask = F.interpolate(
                psf_mask.unsqueeze(0).unsqueeze(0),
                size=(seq_len, seq_len),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
        # Apply physics constraint
        attn = attn * psf_mask.to(attn.device)
        
        # Softmax and apply attention
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        # Project back
        out = self.output_proj(out)
        
        # Reshape back
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        return out


class GatedLinearUnits(nn.Module):
    """
    Gated Linear Units for feature modulation
    """
    def __init__(self, dim):
        """
        Initialize GLU
        
        Args:
            dim (int): Input dimension
        """
        super().__init__()
        self.proj = nn.Conv1d(dim, 2 * dim, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, C, L]
            
        Returns:
            torch.Tensor: Output tensor [B, C, L]
        """
        x_proj = self.proj(x)
        x_proj = x_proj.chunk(2, dim=1)
        
        return x_proj[0] * torch.sigmoid(x_proj[1])


class ConvNeXtV2Encoder(nn.Module):
    """
    ConvNeXtV2-based encoder for PhySNet
    """
    def __init__(self, in_channels=1, channels=[32, 64, 128, 256], temporal=True):
        """
        Initialize ConvNeXtV2 encoder
        
        Args:
            in_channels (int): Input channels
            channels (list): Channel dimensions
            temporal (bool): Whether to use temporal convolutions
        """
        super().__init__()
        
        # Initial convolution
        if temporal:
            self.stem = nn.Conv3d(in_channels, channels[0], kernel_size=(1, 4, 4), stride=(1, 4, 4))
        else:
            self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=4, stride=4)
            
        # ConvNeXt blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            self.blocks.append(ConvNeXtBlock(channels[i], channels[i+1], temporal=temporal))
            
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            list: List of feature maps
        """
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
            
        return features
    

class PhySNet(nn.Module):
    """
    Physics-Guided U-Net for denoising
    """
    def __init__(self, in_channels=1, out_channels=1, temporal=True, use_spectral_conv=True, use_physics_attn=True):
        """
        Initialize PhySNet
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            temporal (bool): Whether to use temporal dimensions
            use_spectral_conv (bool): Whether to use spectral convolutions
            use_physics_attn (bool): Whether to use physics attention
        """
        super().__init__()
        
        self.temporal = temporal
        self.use_spectral_conv = use_spectral_conv
        self.use_physics_attn = use_physics_attn
        
        # Encoder
        self.encoder = ConvNeXtV2Encoder(
            in_channels=in_channels,
            channels=[32, 64, 128, 256],
            temporal=temporal
        )
        
        # Decoder with skip connections
        self.decoders = nn.ModuleList()
        
        # Decoder blocks (upsampling + conv)
        if temporal:
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.GroupNorm(8, 128),
                nn.ReLU()
            ))
            
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose3d(128 + 128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.GroupNorm(8, 64),
                nn.ReLU()
            ))
            
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose3d(64 + 64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.GroupNorm(8, 32),
                nn.ReLU()
            ))
            
            self.final_conv = nn.Conv3d(32 + 32, out_channels, kernel_size=1)
        else:
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.GroupNorm(8, 128),
                nn.ReLU()
            ))
            
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2),
                nn.GroupNorm(8, 64),
                nn.ReLU()
            ))
            
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(64 + 64, 32, kernel_size=2, stride=2),
                nn.GroupNorm(8, 32),
                nn.ReLU()
            ))
            
            self.final_conv = nn.Conv2d(32 + 32, out_channels, kernel_size=1)
            
        # Spectral convolution for wavelength-aware features
        if use_spectral_conv:
            self.spectral_conv = SpectralConv1d(256, 256)
            
        # Physics attention
        if use_physics_attn:
            self.physics_attn = PhysicsAttention(embed_dim=256)
            
        # Gated linear units
        self.glu = GatedLinearUnits(32)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Denoised output
        """
        # Encode
        features = self.encoder(x)
        
        # Apply spectral convolution and physics attention on bottleneck
        bottleneck = features[-1]
        
        if self.temporal:
            b, c, t, h, w = bottleneck.shape
            
            if self.use_spectral_conv:
                # Reshape for spectral convolution
                bottleneck_spectral = bottleneck.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
                bottleneck_spectral = bottleneck_spectral.reshape(b*h*w, c, t)
                bottleneck_spectral = self.spectral_conv(bottleneck_spectral)
                bottleneck_spectral = bottleneck_spectral.reshape(b, h, w, c, t)
                bottleneck_spectral = bottleneck_spectral.permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]
                bottleneck = bottleneck + bottleneck_spectral
                
            if self.use_physics_attn:
                # Reshape for physics attention
                bottleneck_attn = bottleneck.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
                bottleneck_attn = bottleneck_attn.reshape(b*h*w, c, t)
                bottleneck_attn = self.physics_attn(bottleneck_attn)
                bottleneck_attn = bottleneck_attn.reshape(b, h, w, c, t)
                bottleneck_attn = bottleneck_attn.permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]
                bottleneck = bottleneck + bottleneck_attn
        else:
            b, c, h, w = bottleneck.shape
            
            if self.use_spectral_conv:
                # Reshape for spectral convolution along width dimension (spectral dimension)
                bottleneck_spectral = bottleneck.permute(0, 2, 1, 3)  # [B, H, C, W]
                bottleneck_spectral = bottleneck_spectral.reshape(b*h, c, w)
                bottleneck_spectral = self.spectral_conv(bottleneck_spectral)
                bottleneck_spectral = bottleneck_spectral.reshape(b, h, c, w)
                bottleneck_spectral = bottleneck_spectral.permute(0, 2, 1, 3)  # [B, C, H, W]
                bottleneck = bottleneck + bottleneck_spectral
                
            if self.use_physics_attn:
                # Reshape for physics attention along width dimension (spectral dimension)
                bottleneck_attn = bottleneck.permute(0, 2, 1, 3)  # [B, H, C, W]
                bottleneck_attn = bottleneck_attn.reshape(b*h, c, w)
                bottleneck_attn = self.physics_attn(bottleneck_attn)
                bottleneck_attn = bottleneck_attn.reshape(b, h, c, w)
                bottleneck_attn = bottleneck_attn.permute(0, 2, 1, 3)  # [B, C, H, W]
                bottleneck = bottleneck + bottleneck_attn
        
        # Replace bottleneck with processed version
        features[-1] = bottleneck
        
        # Decode with skip connections
        x = features[-1]
        
        for i, decoder in enumerate(self.decoders):
            # Get corresponding encoder features for skip connection
            skip_features = features[-(i+2)]
            
            # Decode
            x = decoder(x)
            
            # Concatenate with skip connection
            if self.temporal:
                x = torch.cat([x, skip_features], dim=1)
            else:
                x = torch.cat([x, skip_features], dim=1)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Apply GLU to final features if not temporal
        if not self.temporal:
            b, c, h, w = x.shape
            # Apply GLU along spectral dimension
            x_glu = x.permute(0, 2, 1, 3)  # [B, H, C, W]
            x_glu = x_glu.reshape(b*h, c, w)
            x_glu = self.glu(x_glu)
            x_glu = x_glu.reshape(b, h, c, w)
            x_glu = x_glu.permute(0, 2, 1, 3)  # [B, C, H, W]
            x = x + x_glu
        
        return x


class MaskedAutoencoderViT(nn.Module):
    """
    Vision Transformer with Masked Autoencoding (Ti-MAE) pretraining
    """
    def __init__(self, 
                img_size=32, 
                patch_size=4, 
                in_channels=1,
                embed_dim=192, 
                depth=12,
                num_heads=3,
                decoder_embed_dim=128,
                decoder_depth=4,
                decoder_num_heads=4,
                mask_ratio=0.75):
        """
        Initialize Ti-MAE
        
        Args:
            img_size (int): Input image size
            patch_size (int): Patch size
            in_channels (int): Input channels
            embed_dim (int): Embedding dimension
            depth (int): Transformer depth
            num_heads (int): Number of attention heads
            decoder_embed_dim (int): Decoder embedding dimension
            decoder_depth (int): Decoder depth
            decoder_num_heads (int): Decoder attention heads
            mask_ratio (float): Ratio of masked patches
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                      dim_feedforward=4*embed_dim, dropout=0.0, 
                                      activation='gelu', batch_first=True,
                                      norm_first=True)
            for _ in range(depth)
        ])
        
        # Encoder normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder embedding
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=decoder_num_heads,
                                      dim_feedforward=4*decoder_embed_dim, dropout=0.0,
                                      activation='gelu', batch_first=True,
                                      norm_first=True)
            for _ in range(decoder_depth)
        ])
        
        # Decoder normalization
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Decoder prediction
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights
        """
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize decoder
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def random_masking(self, x, mask_ratio):
        """
        Perform random masking
        
        Args:
            x (torch.Tensor): Input tokens [B, L, D]
            mask_ratio (float): Ratio of tokens to mask
            
        Returns:
            tuple: Masked tokens, mask, ids_restore
        """
        batch_size, seq_len, dim = x.shape
        
        # Calculate number of tokens to keep
        len_keep = int(seq_len * (1 - mask_ratio))
        
        # Generate random noise for masking
        noise = torch.rand(batch_size, seq_len, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_len], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through encoder with masking
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            mask_ratio (float): Masking ratio
            
        Returns:
            tuple: Encoded features, mask, ids_restore
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, L, D]
        
        # Add positions
        x = x + self.pos_embed[:, 1:, :]
        
        # Apply masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        for blk in self.encoder_blocks:
            x = blk(x)
            
        # Apply normalization
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through decoder
        
        Args:
            x (torch.Tensor): Encoded features [B, L, D]
            ids_restore (torch.Tensor): Token restore indices
            
        Returns:
            torch.Tensor: Reconstructed patches
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add positional embeddings
        x = x + self.decoder_pos_embed
        
        # Apply transformer decoder
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # Apply normalization
        x = self.decoder_norm(x)
        
        # Predict patches
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            
        Returns:
            dict: Model outputs
        """
        # Calculate latent and mask
        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        
        # Decode latent
        pred = self.forward_decoder(latent, ids_restore)
        
        return {
            'latent': latent,
            'pred': pred,
            'mask': mask,
        }
    
    def patchify(self, imgs):
        """
        Convert images to patches
        
        Args:
            imgs (torch.Tensor): Images [B, C, H, W]
            
        Returns:
            torch.Tensor: Patches [B, L, N]
        """
        p = self.patch_size
        b, c, h, w = imgs.shape
        x = imgs.reshape(b, c, h//p, p, w//p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, H/p, W/p, p, p, C]
        patches = x.reshape(b, (h//p) * (w//p), p**2 * c)
        return patches
    
    def unpatchify(self, patches):
        """
        Convert patches back to images
        
        Args:
            patches (torch.Tensor): Patches [B, L, N]
            
        Returns:
            torch.Tensor: Images [B, C, H, W]
        """
        p = self.patch_size
        h = w = self.img_size // p
        c = self.in_channels
        
        x = patches.reshape(patches.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, H/p, p, W/p, p]
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs 