import os
import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from scipy.io import loadmat
import warnings

class BCIDataset(Dataset):
    """
    PyTorch Dataset wrapper for BCI Competition IV Dataset 2a.
    Handles loading, filtering, epoching, and normalization.
    """

    def __init__(self, data_dir: str, subject_id: int, is_train: bool = True, transform=None):
        """
        Initializes the Dataset.
        Note: The evaluation 'E' files from BCIC-IV-2a do not contain class labels 
        in the downloaded GDF. A separate labels file is required for testing.
        
        Args:
            data_dir (str): Path to the raw .gdf files.
            subject_id (int): Subject ID from 1 to 9.
            is_train (bool): True to load 'T' files (train), False to load 'E' files (test). Default is True.
            transform (Callable, optional): Optional transform to apply to the data.
        """
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.is_train = is_train
        self.transform = transform
        
        self.file_suffix = 'T' if is_train else 'E'
        self.file_name = f"A{subject_id:02d}{self.file_suffix}.gdf"
        self.file_path = os.path.join(self.data_dir, self.file_name)
        
        # Load and preprocess the data
        self.data, self.labels = self._load_and_preprocess()

    def _load_and_preprocess(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads GDF file, applies bandpass filter 4-40Hz, extracts motor imagery 
        epochs (0 to 4s post-cue), normalizes per-channel, and returns tensors.
        
        For evaluation files (is_train=False), labels are loaded from a separate
        true_labels file since BCIC-IV-2a evaluation GDFs contain no class labels.
        """
        if not os.path.exists(self.file_path):
            print(f"Warning: File {self.file_path} not found. Returning dummy data.")
            return torch.randn(288, 1, 22, 1000), torch.randint(0, 4, (288,))

        # 1. Load raw GDF — suppress MNE verbose output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(
                self.file_path, 
                preload=True, 
                eog=['EOG-left', 'EOG-central', 'EOG-right'],
                verbose=False
            )

        # 2. Bandpass filter 4-40 Hz before epoching
        raw.filter(l_freq=4.0, h_freq=40.0, fir_design='firwin', verbose=False)

        # 3. Pick only EEG channels (exclude EOG, STI)
        eeg_picks = mne.pick_types(
            raw.info, meg=False, eeg=True, 
            eog=False, stim=False, exclude='bads'
        )

        # 4. Extract events and build motor imagery event array
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # BCIC-IV-2a event codes (may appear as integers or strings in annotations)
        # 769=left hand, 770=right hand, 771=foot, 772=tongue
        # We search by both string and integer representation for robustness
        mi_event_codes = {}
        for desc, code in event_id.items():
            desc_str = str(desc)
            if '769' in desc_str: mi_event_codes[code] = 0
            elif '770' in desc_str: mi_event_codes[code] = 1
            elif '771' in desc_str: mi_event_codes[code] = 2
            elif '772' in desc_str: mi_event_codes[code] = 3

        # If annotation-based mapping fails, fall back to raw event codes
        if not mi_event_codes:
            fallback_map = {769: 0, 770: 1, 771: 2, 772: 3}
            mi_event_codes = {
                k: v for k, v in fallback_map.items() 
                if k in events[:, 2]
            }

        if not mi_event_codes:
            raise ValueError(
                f"Could not find motor imagery events in {self.file_path}. "
                f"Available event_id: {event_id}"
            )

        # Filter events to motor imagery only and remap to 0-3
        mi_events = []
        for ev in events:
            if ev[2] in mi_event_codes:
                new_ev = ev.copy()
                new_ev[2] = mi_event_codes[ev[2]]
                mi_events.append(new_ev)

        mi_events = np.array(mi_events)

        if len(mi_events) == 0:
            raise ValueError(
                f"Zero motor imagery trials found in {self.file_path}. "
                f"Check event codes. Available: {event_id}"
            )

        # 5. Epoch extraction: 0 to 4 seconds post-cue
        # baseline=None because tmin=0.0 (no pre-stimulus data available)
        # This is correct for motor imagery — no baseline correction needed
        sfreq = raw.info['sfreq']
        epochs = mne.Epochs(
            raw, mi_events,
            event_id=None,
            tmin=0.0,
            tmax=4.0 - 1.0/sfreq,
            picks=eeg_picks,
            baseline=None,
            preload=True,
            verbose=False
        )

        data = epochs.get_data()   # shape: (trials, channels, samples)
        
        # 6. Handle labels
        if self.is_train:
            # Training files: labels are inside the GDF epochs
            labels = epochs.events[:, 2]  # already remapped to 0-3
        else:
            # Evaluation files: labels NOT in GDF — load from true_labels file
            # Expected file: true_labels/A0{subject_id}E_labels.npy
            # OR: user must provide labels separately
            labels_path = os.path.join(
                self.data_dir, 
                f"A{self.subject_id:02d}E_labels.npy"
            )
            if os.path.exists(labels_path):
                labels = np.load(labels_path)
                # Convert from 1-indexed to 0-indexed if needed
                if labels.min() == 1:
                    labels = labels - 1
            else:
                # If no separate label file found, use epoch event codes
                # (works if GDF was annotated — may be zeros for eval files)
                print(
                    f"Warning: No evaluation labels file found at {labels_path}. "
                    f"Using event codes from GDF (may be incorrect for eval files). "
                    f"For accurate evaluation, provide A{self.subject_id:02d}E_labels.npy"
                )
                labels = epochs.events[:, 2]

        # 7. Per-channel z-score normalization
        # Mean and std computed across time dimension for each trial and channel
        # Shape: (trials, channels, samples)
        mean = data.mean(axis=2, keepdims=True)
        std = data.std(axis=2, keepdims=True)
        data = (data - mean) / (std + 1e-6)

        # 8. Reshape for EEGNet input: (trials, 1, channels, samples)
        data = np.expand_dims(data, axis=1)

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def get_dataloaders(
    data_dir: str, 
    subject_id: int, 
    batch_size: int = 32,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders from the training GDF file only.
    
    Uses an 80/20 train/val split from the T (training) file.
    This avoids the evaluation label issue with BCIC-IV-2a E files,
    and is standard practice for within-session evaluation.
    
    Args:
        data_dir (str): Path to directory containing .gdf files.
        subject_id (int): Subject ID from 1 to 9.
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of training data to use for validation.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    from torch.utils.data import random_split

    # Load only the training file — contains both data and labels
    full_dataset = BCIDataset(
        data_dir=data_dir, 
        subject_id=subject_id, 
        is_train=True
    )
    
    # Reproducible split
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,        # Windows compatibility
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,        # Windows compatibility
        pin_memory=False
    )
    
    return train_loader, val_loader
