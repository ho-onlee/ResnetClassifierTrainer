
from dynaconf import Dynaconf

from src.final_trainer import indiv_trainer
from src.utils import saveDatasetEntry, makeDatasetEntry
from src.sound_to_tensor import extract_enhanced_features
import tqdm
import os
import random
import dill
import torch, logging
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import numpy as np
import src.args as args

logger = logging.getLogger(__name__)
settings = Dynaconf(settings_files=["../settings.toml"])


class EntriesTorchDataset(Dataset):
    """Convert entries list to PyTorch Dataset"""
    def __init__(self, entries):
        self.labels = []
        self.features_list = []
        self.label_to_idx = {}
        

        unique_labels = sorted(set(entry['label'] for entry in entries))
        for idx, label in enumerate(unique_labels):
            self.label_to_idx[label] = idx
        
        # Extract features for each entry
        for entry in entries:
            # Use enhanced features if available, otherwise use MFCC
            if entry.get('features') is not None:
                features = entry['features']
            else:
                continue  # Skip entries without features
            
            self.features_list.append(torch.tensor(features, dtype=torch.float32))
            self.labels.append(self.label_to_idx[entry['label']])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features_list[idx]
        label = self.labels[idx]
        return features, torch.tensor(label, dtype=torch.long)
    
    def get_label_mapping(self):
        """Return mapping of label names to indices"""
        return self.label_to_idx
    
    def get_num_classes(self):
        """Return number of classes"""
        return len(self.label_to_idx)

def prepare_data(new:bool=False):
    trainer = indiv_trainer()
    trainer.setParameters(sr=settings.audio.sr, n_mfcc=settings.audio.n_mfcc, n_fft=settings.audio.n_fft , project_id=1)
    dataset = trainer.prepare_data(new=new)
    dataset = trainer.triage_dataset(dataset)
    entries = makeDatasetEntry(dataset)
    folder = os.path.join(settings.folders.data_root , 'pre_processed_entries')

    for entry in tqdm(entries):
        saveDatasetEntry(entry, folder=folder)
    del entries
    del dataset


    mods= [lambda x:args.add_noise(x, noise_factor=0.05),
        lambda x:args.add_stretch(x, rate=0.9),
        lambda x:args.add_stretch(x, rate=1.10),
        lambda x:args.add_pitch_shift(x, n_steps=1),
        lambda x:args.add_pitch_shift(x, n_steps=-1),
        lambda x:args.random_gain(torch.from_numpy(x)),
        lambda x:args.random_noise(torch.from_numpy(x)),
        lambda x:args.random_lowpass(x),
        lambda x:args.random_highpass(x),
        lambda x:args.random_pitch(x)]

    datas = os.listdir(folder)
    tq = tqdm(mods, total=len(mods))
    for mod in tq:
        for sample in random.sample(datas, len(datas)//4):
            with open(f'{folder}/{sample}', 'rb') as f:
                new_sample = dill.load(f)
            new_sample['mod'] = mod
            saveDatasetEntry(new_sample, folder=folder)


def process_data():
    files = os.listdir(os.path.join(settings.folders.data_root , 'pre_processed_entries'))
    tq = tqdm(files, total=len(files))
    skipped_count = 0
    signal = []
    sr = settings.audio.sr
    for f in tq:
        with open(os.path.join(settings.folders.data_root, 'pre_processed_entries', f), 'rb') as entry_data_file_in_json:
            entry = dill.load(entry_data_file_in_json)
        audio_data_path = os.path.join(settings.folders.data_root, 'audio_data', entry['filename'])
        speech_data_path = os.path.join(settings.folders.data_root, 'speech_dataset', entry['filename'])
        offset_seconds = entry['start'] / entry['samplerate']
        duration_seconds = (entry['end'] - entry['start']) / entry['samplerate']

        if os.path.exists(audio_data_path):
            signal, sr = librosa.load(
                audio_data_path,
                sr=entry['samplerate'],
                offset=offset_seconds,
                duration=duration_seconds,
            )
        elif os.path.exists(speech_data_path):
            signal, sr = librosa.load(
                speech_data_path,
                sr=entry['samplerate'],
                offset=offset_seconds,
                duration=duration_seconds,
            )
        else:
            skipped_count += 1
            tq.set_postfix({"skipped": skipped_count, "missing": entry['filename']})
            continue
        
        # Skip empty audio signals
        if len(signal) == 0:
            skipped_count += 1
            tq.set_postfix({"skipped": skipped_count, "file": entry['filename']})
            continue
        
        # Convert to float32 numpy array to avoid dtype issues with augmentation functions
        signal = np.array(signal, dtype=np.float32)
        
        if entry['mod'] is not None:
            signal = entry['mod'](signal)
            signal = np.array(signal, dtype=np.float32)
        
        features, mfcc = extract_enhanced_features(signal, sr=int(sr))
        entry['features'] = features
        with open(os.path.join(settings.folders.data_root, 'pre_processed_entries', f), 'wb') as file:
            dill.dump(entry, file)
    print(f"\nProcessed {len(files) - skipped_count} files, skipped {skipped_count} files")


def load_entries(folder=settings.folders.data_root + '/pre_processed_entries') -> EntriesTorchDataset:
    """Load serialized feature entries and return a PyTorch-compatible dataset."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Entries folder does not exist: {folder}. "
            "Run prepare_data() and process_data() first."
        )

    entries = []
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    for f in tqdm(files, desc="Loading entries"):
        with open(os.path.join(folder, f), 'rb') as file:
            entry = dill.load(file)
            entries.append(entry)

    torch_dataset = EntriesTorchDataset(entries)
    if len(torch_dataset) == 0:
        raise ValueError(
            "Loaded 0 usable feature entries from pre_processed_entries. "
            "Run process_data() to populate `entry['features']` before training."
        )
    logger.info(f"Dataset size: {len(torch_dataset)}")
    logger.info(f"Number of classes: {torch_dataset.get_num_classes()}")
    logger.info(f"Label mapping: {torch_dataset.get_label_mapping()}")
    return torch_dataset


def load_torch_dataset(folder=settings.folders.data_root + '/pre_processed_entries') -> EntriesTorchDataset:
    """Explicit alias for loading the torch dataset used in training."""
    return load_entries(folder=folder)


def load_torch_dataloader(
    folder=settings.folders.data_root + '/pre_processed_entries',
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Load dataset entries and wrap them in a torch DataLoader."""
    dataset = load_torch_dataset(folder=folder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    