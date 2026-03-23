from math import ceil
import dill
import os
from pathlib import Path
import secrets
from dynaconf import Dynaconf
from torch.utils.data import Dataset, DataLoader
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
settings = Dynaconf(settings_files=["../settings.toml"])


def buildDirectoryStructure():
    """Creates necessary directories for data storage and processing"""
    folders = [settings.folders.data_root,
               os.path.join(settings.folders.data_root, 'audio_data'),
               os.path.join(settings.folders.data_root, 'pre_processed_entries')
               ]
    for folder in folders:
        if not os.path.exists(folder): 
            os.mkdir(folder)
            logger.info(f"Created directory: {folder}")


def split_audio(signal, window_size):
    N = len(signal)

    min_windows = ceil(N / window_size)

    for K in range(min_windows, N):
        if K == 1:
            hop = 0
        else:
            hop = (N - window_size) / (K - 1)

        if hop.is_integer() and 0 < hop <= window_size:
            hop = int(hop)
            overlap = window_size - hop
            break

    windows = []
    pos = 0
    while pos + window_size <= N:
        windows.append((pos, pos + window_size))
        pos += hop

    return windows

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)

        optimizer.zero_grad()
        out = model(mel)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def makeDatasetEntry(dataset):
    entries = []
    for data in dataset:
        for entry in data['dataset']:
            entries.append({
                'filename': data['filename'],
                'samplerate': data['samplerate'],
                'label': entry['label'],
                'start': entry['start'],
                'end': entry['end'],
                'mod': None,
                'features': None
            })
    return entries

def saveDatasetEntry(entry, folder=settings.folders.data_root + '/pre_processed_entries'):
    Path(folder).mkdir(parents=True, exist_ok=True)
    while True:
        hash = secrets.token_hex(16)
        if not os.path.exists(os.path.join(folder, f'E_{hash}.pkl')):
            break    
    with open(os.path.join(folder, f'E_{hash}.pkl'), 'wb') as f:
        dill.dump(entry, f)



    
