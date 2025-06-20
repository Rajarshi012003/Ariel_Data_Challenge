import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ArielDataset(Dataset):
    """
    Dataset class for loading Ariel exoplanet data
    """
    def __init__(self, 
                 data_dir, 
                 planet_ids, 
                 labels_path=None, 
                 wavelengths_path=None,
                 adc_info_path=None,
                 transform=None,
                 target_transform=None,
                 mode='train'):
        """
        Initialize the Ariel dataset
        
        Args:
            data_dir (str): Directory containing the data
            planet_ids (list): List of planet IDs to include
            labels_path (str): Path to the labels file (train_labels.csv)
            wavelengths_path (str): Path to the wavelengths file
            adc_info_path (str): Path to the ADC info file
            transform (callable): Transform to apply to the data
            target_transform (callable): Transform to apply to the targets
            mode (str): 'train' or 'test'
        """
        self.data_dir = data_dir
        self.planet_ids = planet_ids
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        
        # Load ADC info for gain and offset
        if adc_info_path:
            self.adc_info = pd.read_csv(adc_info_path)
        else:
            self.adc_info = None
            
        # Load wavelengths
        if wavelengths_path:
            self.wavelengths = pd.read_csv(wavelengths_path)
        else:
            self.wavelengths = None
            
        # Load labels for training
        if labels_path and mode == 'train':
            self.labels = pd.read_csv(labels_path)
        else:
            self.labels = None
            
        # Load axis info
        self.axis_info = pd.read_parquet(os.path.join(data_dir, '..', 'axis_info.parquet'))
            
    def __len__(self):
        return len(self.planet_ids)
    
    def load_calibration_data(self, planet_id, instrument):
        """
        Load calibration data for a specific planet and instrument
        
        Args:
            planet_id (str): Planet ID
            instrument (str): Instrument name (AIRS-CH0 or FGS1)
            
        Returns:
            dict: Dictionary containing calibration data
        """
        calib_dir = os.path.join(self.data_dir, str(planet_id), f"{instrument}_calibration")
        
        calibration = {
            'dark': pd.read_parquet(os.path.join(calib_dir, 'dark.parquet')),
            'dead': pd.read_parquet(os.path.join(calib_dir, 'dead.parquet')),
            'flat': pd.read_parquet(os.path.join(calib_dir, 'flat.parquet')),
            'linear_corr': pd.read_parquet(os.path.join(calib_dir, 'linear_corr.parquet')),
            'read': pd.read_parquet(os.path.join(calib_dir, 'read.parquet'))
        }
        
        return calibration
    
    def load_signal_data(self, planet_id, instrument):
        """
        Load signal data for a specific planet and instrument
        
        Args:
            planet_id (str): Planet ID
            instrument (str): Instrument name (AIRS-CH0 or FGS1)
            
        Returns:
            np.ndarray: Signal data
        """
        signal_path = os.path.join(self.data_dir, str(planet_id), f"{instrument}_signal.parquet")
        signal_df = pd.read_parquet(signal_path)
        
        # Apply ADC correction if available
        if self.adc_info is not None:
            planet_info = self.adc_info[self.adc_info['planet_id'] == int(planet_id)]
            if not planet_info.empty:
                gain = planet_info[f"{instrument}_adc_gain"].values[0]
                offset = planet_info[f"{instrument}_adc_offset"].values[0]
                signal_df = signal_df * gain + offset
        
        # Reshape based on instrument
        if instrument == 'AIRS-CH0':
            # 11,250 rows, each a 32x356 image
            signal = signal_df.values.reshape(-1, 32, 356)
        elif instrument == 'FGS1':
            # 135,000 rows, each a 32x32 image
            signal = signal_df.values.reshape(-1, 32, 32)
        
        return signal
    
    def get_target(self, planet_id):
        """
        Get the target spectrum for a planet
        
        Args:
            planet_id (str): Planet ID
            
        Returns:
            np.ndarray: Target spectrum (if available)
        """
        if self.labels is not None:
            planet_row = self.labels[self.labels['planet_id'] == int(planet_id)]
            if not planet_row.empty:
                # Get all columns except 'planet_id'
                spectrum = planet_row.drop('planet_id', axis=1).values[0]
                return spectrum
        return None
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (data_dict, target)
        """
        planet_id = self.planet_ids[idx]
        
        # Load signals
        airs_signal = self.load_signal_data(planet_id, 'AIRS-CH0')
        fgs_signal = self.load_signal_data(planet_id, 'FGS1')
        
        # Load calibration data
        airs_calib = self.load_calibration_data(planet_id, 'AIRS-CH0')
        fgs_calib = self.load_calibration_data(planet_id, 'FGS1')
        
        # Prepare data dictionary
        data = {
            'planet_id': planet_id,
            'AIRS-CH0': {
                'signal': airs_signal,
                'calibration': airs_calib
            },
            'FGS1': {
                'signal': fgs_signal,
                'calibration': fgs_calib
            }
        }
        
        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)
            
        # Get target
        target = self.get_target(planet_id)
        
        # Apply target transforms if provided
        if self.target_transform and target is not None:
            target = self.target_transform(target)
            
        return data, target


def custom_collate_fn(batch):
    """
    Custom collate function to handle pandas DataFrames and other non-tensor objects in the batch
    
    Args:
        batch (list): List of (data, target) pairs
        
    Returns:
        tuple: (data_batch, target_batch)
    """
    data_batch = {}
    target_batch = []
    
    # Extract all data and targets
    for data, target in batch:
        # Handle planet_id
        if 'planet_id' not in data_batch:
            data_batch['planet_id'] = []
        data_batch['planet_id'].append(data['planet_id'])
        
        # Handle instruments
        for instrument in ['AIRS-CH0', 'FGS1']:
            if instrument not in data_batch:
                data_batch[instrument] = {'signal': [], 'calibration': {}}
                
            # Handle signal
            data_batch[instrument]['signal'].append(data[instrument]['signal'])
            
            # Handle calibration data - first sample only for initialization
            if not data_batch[instrument]['calibration']:
                for calib_key in data[instrument]['calibration']:
                    data_batch[instrument]['calibration'][calib_key] = []
                    
            # Add calibration data to batch
            for calib_key in data[instrument]['calibration']:
                calib_data = data[instrument]['calibration'][calib_key]
                # Convert DataFrames to numpy arrays for batch processing
                if isinstance(calib_data, pd.DataFrame):
                    calib_data = calib_data.values
                data_batch[instrument]['calibration'][calib_key].append(calib_data)
        
        # Handle targets
        if target is not None:
            target_batch.append(target)
    
    # Convert lists to numpy arrays or tensors where appropriate
    for instrument in ['AIRS-CH0', 'FGS1']:
        # Convert signal lists to numpy arrays
        data_batch[instrument]['signal'] = np.array(data_batch[instrument]['signal'])
        
        # Convert calibration lists to numpy arrays
        for calib_key in data_batch[instrument]['calibration']:
            try:
                data_batch[instrument]['calibration'][calib_key] = np.array(
                    data_batch[instrument]['calibration'][calib_key]
                )
            except ValueError:
                # Keep as list if conversion fails (e.g., different shapes)
                pass
    
    # Convert targets to numpy array if available
    if target_batch:
        target_batch = np.array(target_batch)
    
    return data_batch, target_batch


def create_dataloader(data_dir, 
                     planet_ids, 
                     labels_path=None, 
                     wavelengths_path=None,
                     adc_info_path=None,
                     batch_size=1,
                     num_workers=4,
                     transform=None,
                     target_transform=None,
                     mode='train',
                     shuffle=True):
    """
    Create a dataloader for the Ariel dataset
    
    Args:
        data_dir (str): Directory containing the data
        planet_ids (list): List of planet IDs to include
        labels_path (str): Path to the labels file
        wavelengths_path (str): Path to the wavelengths file
        adc_info_path (str): Path to the ADC info file
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        transform (callable): Transform to apply to the data
        target_transform (callable): Transform to apply to the targets
        mode (str): 'train' or 'test'
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    dataset = ArielDataset(
        data_dir=data_dir,
        planet_ids=planet_ids,
        labels_path=labels_path,
        wavelengths_path=wavelengths_path,
        adc_info_path=adc_info_path,
        transform=transform,
        target_transform=target_transform,
        mode=mode
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return dataloader


def get_train_val_split(planet_ids, val_ratio=0.2, random_state=42):
    """
    Split the list of planet IDs into training and validation sets
    
    Args:
        planet_ids (list): List of planet IDs
        val_ratio (float): Ratio of validation set size
        random_state (int): Random seed
        
    Returns:
        tuple: (train_ids, val_ids)
    """
    np.random.seed(random_state)
    np.random.shuffle(planet_ids)
    
    val_size = int(len(planet_ids) * val_ratio)
    train_ids = planet_ids[val_size:]
    val_ids = planet_ids[:val_size]
    
    return train_ids, val_ids


class PatchSampler:
    """
    Class for sampling patches from full signals for memory efficiency
    """
    def __init__(self, patch_size=(32, 32, 64), stride=None):
        """
        Initialize the patch sampler
        
        Args:
            patch_size (tuple): Size of the patches to extract (h, w, t)
            stride (tuple): Stride for extraction (if None, use patch size)
        """
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        
    def extract_patches(self, signal):
        """
        Extract patches from a signal
        
        Args:
            signal (np.ndarray): Signal with shape (t, h, w)
            
        Returns:
            list: List of patches with shape (n_patches, t, h, w)
        """
        t, h, w = signal.shape
        ph, pw, pt = self.patch_size
        sh, sw, st = self.stride
        
        patches = []
        
        for i in range(0, t - pt + 1, st):
            for j in range(0, h - ph + 1, sh):
                for k in range(0, w - pw + 1, sw):
                    patch = signal[i:i+pt, j:j+ph, k:k+pw]
                    patches.append(patch)
                    
        return patches
    
    def reconstruct_from_patches(self, patches, original_shape):
        """
        Reconstruct signal from patches
        
        Args:
            patches (list): List of patches
            original_shape (tuple): Shape of the original signal (t, h, w)
            
        Returns:
            np.ndarray: Reconstructed signal
        """
        t, h, w = original_shape
        ph, pw, pt = self.patch_size
        sh, sw, st = self.stride
        
        # Initialize output with zeros
        output = np.zeros(original_shape)
        counts = np.zeros(original_shape)
        
        patch_idx = 0
        for i in range(0, t - pt + 1, st):
            for j in range(0, h - ph + 1, sh):
                for k in range(0, w - pw + 1, sw):
                    output[i:i+pt, j:j+ph, k:k+pw] += patches[patch_idx]
                    counts[i:i+pt, j:j+ph, k:k+pw] += 1
                    patch_idx += 1
                    
        # Average overlapping regions
        output /= np.maximum(counts, 1)
        
        return output 


class DataVisualizer:
    """
    Class for visualizing data from the ArielDataset
    """
    def __init__(self, save_dir=None):
        """
        Initialize the data visualizer
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def plot_signal(self, signal, instrument, planet_id, save=False):
        """
        Plot a signal from an instrument
        
        Args:
            signal (np.ndarray): Signal data with shape (t, h, w)
            instrument (str): Instrument name
            planet_id (str): Planet ID
            save (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        
        t, h, w = signal.shape
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot first frame
        axes[0].imshow(signal[0], norm=LogNorm())
        axes[0].set_title(f'First Frame (t=0)')
        
        # Plot middle frame
        mid_t = t // 2
        axes[1].imshow(signal[mid_t], norm=LogNorm())
        axes[1].set_title(f'Middle Frame (t={mid_t})')
        
        # Plot last frame
        axes[2].imshow(signal[-1], norm=LogNorm())
        axes[2].set_title(f'Last Frame (t={t-1})')
        
        # Set figure title
        fig.suptitle(f'Planet {planet_id} - {instrument} Signal')
        
        # Save if requested
        if save and self.save_dir:
            save_path = os.path.join(self.save_dir, f'planet_{planet_id}_{instrument}_signal.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_spectrum(self, spectrum, wavelengths=None, planet_id=None, save=False):
        """
        Plot a spectrum
        
        Args:
            spectrum (np.ndarray): Spectrum data
            wavelengths (np.ndarray): Wavelength values
            planet_id (str): Planet ID
            save (bool): Whether to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot spectrum
        if wavelengths is not None:
            ax.plot(wavelengths, spectrum)
            ax.set_xlabel('Wavelength (μm)')
        else:
            ax.plot(spectrum)
            ax.set_xlabel('Wavelength Index')
            
        ax.set_ylabel('Transit Depth')
        
        # Set title
        if planet_id:
            ax.set_title(f'Planet {planet_id} Spectrum')
        else:
            ax.set_title('Spectrum')
            
        # Add grid
        ax.grid(alpha=0.3)
        
        # Save if requested
        if save and self.save_dir and planet_id:
            save_path = os.path.join(self.save_dir, f'planet_{planet_id}_spectrum.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_dataset_sample(self, dataset, idx, save=False):
        """
        Visualize a sample from the dataset
        
        Args:
            dataset (ArielDataset): Dataset
            idx (int): Index of sample
            save (bool): Whether to save the plot
            
        Returns:
            dict: Dictionary of figure objects
        """
        data, target = dataset[idx]
        planet_id = data['planet_id']
        
        figures = {}
        
        # Plot AIRS signal
        airs_signal = data['AIRS-CH0']['signal']
        figures['airs'] = self.plot_signal(airs_signal, 'AIRS-CH0', planet_id, save=save)
        
        # Plot FGS signal
        fgs_signal = data['FGS1']['signal']
        figures['fgs'] = self.plot_signal(fgs_signal, 'FGS1', planet_id, save=save)
        
        # Plot spectrum if available
        if target is not None and dataset.wavelengths is not None:
            wavelengths = dataset.wavelengths.iloc[0].values
            figures['spectrum'] = self.plot_spectrum(target, wavelengths, planet_id, save=save)
        elif target is not None:
            figures['spectrum'] = self.plot_spectrum(target, planet_id=planet_id, save=save)
            
        return figures


class DataAugmentation:
    """
    Class for data augmentation
    """
    def __init__(self, 
                 add_noise=False, 
                 noise_level=0.05,
                 random_flip=False,
                 random_crop=False,
                 crop_size=None):
        """
        Initialize data augmentation
        
        Args:
            add_noise (bool): Whether to add Gaussian noise
            noise_level (float): Standard deviation of noise
            random_flip (bool): Whether to randomly flip images
            random_crop (bool): Whether to randomly crop images
            crop_size (tuple): Size of crop (h, w)
        """
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_size = crop_size
        
    def __call__(self, data):
        """
        Apply augmentations to data
        
        Args:
            data (dict): Data dictionary
            
        Returns:
            dict: Augmented data
        """
        # Make a copy to avoid modifying the original
        data_aug = {
            'planet_id': data['planet_id'],
            'AIRS-CH0': {
                'signal': data['AIRS-CH0']['signal'].copy(),
                'calibration': data['AIRS-CH0']['calibration']
            },
            'FGS1': {
                'signal': data['FGS1']['signal'].copy(),
                'calibration': data['FGS1']['calibration']
            }
        }
        
        # Add Gaussian noise
        if self.add_noise:
            for instrument in ['AIRS-CH0', 'FGS1']:
                signal = data_aug[instrument]['signal']
                noise = np.random.normal(0, self.noise_level * np.mean(signal), signal.shape)
                data_aug[instrument]['signal'] = signal + noise
                
        # Random horizontal flip
        if self.random_flip and np.random.rand() > 0.5:
            for instrument in ['AIRS-CH0', 'FGS1']:
                data_aug[instrument]['signal'] = data_aug[instrument]['signal'][:, :, ::-1]
                
        # Random crop
        if self.random_crop and self.crop_size:
            crop_h, crop_w = self.crop_size
            
            for instrument in ['AIRS-CH0', 'FGS1']:
                signal = data_aug[instrument]['signal']
                t, h, w = signal.shape
                
                if h > crop_h and w > crop_w:
                    start_h = np.random.randint(0, h - crop_h + 1)
                    start_w = np.random.randint(0, w - crop_w + 1)
                    
                    data_aug[instrument]['signal'] = signal[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        return data_aug


def create_transforms(config):
    """
    Create transforms based on config
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        callable: Transform function
    """
    # Create preprocessing pipeline
    preprocessor = PreprocessingPipeline(
        dark_subtraction=config.get('dark_subtraction', True),
        flat_correction=config.get('flat_correction', True),
        bad_pixel_interp=config.get('bad_pixel_interp', True),
        temporal_norm=config.get('temporal_norm', True)
    )
    
    # Create data augmentation
    augmenter = None
    if config.get('augmentation', False):
        augmenter = DataAugmentation(
            add_noise=config.get('add_noise', False),
            noise_level=config.get('noise_level', 0.05),
            random_flip=config.get('random_flip', False),
            random_crop=config.get('random_crop', False),
            crop_size=config.get('crop_size', None)
        )
    
    # Define transform function
    def transform(data):
        # Apply preprocessing
        data = preprocessor.process(data)
        
        # Apply augmentation if enabled and in training mode
        if augmenter is not None:
            data = augmenter(data)
            
        return data
    
    return transform

