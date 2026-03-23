import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from dynaconf import Dynaconf
import tensorflow as tf
import numpy as np
import torchaudio 
import pandas as pd

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU detected, using CPU")
import librosa, random, torch
import json
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
import matplotlib.pyplot as plt
from urllib.parse import unquote
from tqdm import tqdm
import h5py, requests
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, cast, Optional, Tuple, Union
import numpy.typing as npt
import logging


settings = Dynaconf(settings_files=["../settings.toml"])
logger = logging.getLogger(__name__)

class indiv_trainer:    
    def __init__(self) -> None:
        self.dataset: Optional[List[Dict[str, Any]]] = None
        self.model: Optional[tf.keras.Model] = None
        self.labels: Optional[List[str]] = None
        pass

    def setParameters(self, sr: int, n_mfcc: int, n_fft: int, project_id:int=1) -> None:
        """
        Set audio processing parameters.
        
        Args:
            sr (int): Sample rate for audio processing
            n_mfcc (int): Number of MFCC coefficients to extract
            n_fft (int): Number of FFT components for spectral analysis
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.project_id = project_id
        
    @staticmethod
    def grabAPI(url: str=f"{settings.labelstudio.url}:{settings.labelstudio.port}", project_id:int=1) -> bool:
        """
        Download labeled audio data from Label Studio API server.
        
        This method connects to a Label Studio instance to export project data containing
        audio annotations and labels in JSON format.
        
        Input:
            None (uses hardcoded token and project_id)
            
        Output:
            bool: True if data export was successful, False otherwise
            
        Side Effects:
            - Creates 'exported_data.json' file in current directory with annotation data
            - Prints success/failure messages to console
        """
        url = settings.labelstudio.url+":"+str(settings.labelstudio.port)
        url = f'http://{url}/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true'
        headers = {
            'Authorization': f'Token {settings.labelstudio.token}'
        }
        response = requests.get(url, headers =headers, verify=True)  # Set verify=False to ignore SSL warnings

        if response.status_code == 200:
            with open(os.path.join(settings.folders.data_root,'exported_data.json'), 'w', encoding='utf-8') as json_file:
                json_file.write(response.text)
            logger.info("Data exported successfully.")
            return True
        else:
            logger.error(f"Failed to export data. Status code: {response.status_code}")
            return False

    @staticmethod
    def grabAudio(filename: str, project_id:int=1) -> str:
        """
        Download audio file from remote server via SSH/SCP.
        
        This method attempts to download an audio file from a remote Label Studio server
        using SSH connection and SCP file transfer. It first checks if the file already
        exists locally to avoid unnecessary downloads.
        
        Input:
            filename (str): Name of the audio file to download
            
        Output:
            str: Local file path to the downloaded audio file
            
        Raises:
            RuntimeError: If SSH transport fails or file download fails
        """
        decoded_filename = unquote(filename)
        origin_d = f'/home/worker/.local/share/label-studio/media/upload/{project_id}/' + decoded_filename
        target_d = os.path.join(settings.folders.data_root, 'audio_data')

        # Recover from a common path corruption case: a file created where a directory is expected.
        if os.path.exists(target_d) and not os.path.isdir(target_d):
            backup_path = f"{target_d}.broken_file.bak"
            if os.path.exists(backup_path):
                raise RuntimeError(
                    f"Expected directory at '{target_d}' but found a file, and backup path "
                    f"'{backup_path}' already exists. Please resolve manually."
                )
            os.replace(target_d, backup_path)
            logger.warning(
                "Moved unexpected file '%s' to '%s' so audio_data directory can be recreated.",
                target_d,
                backup_path,
            )
        os.makedirs(target_d, exist_ok=True)

        filename_candidates: List[str] = []
        for candidate in (filename, decoded_filename, os.path.basename(filename), os.path.basename(decoded_filename)):
            if candidate and candidate not in filename_candidates:
                filename_candidates.append(candidate)

        for candidate in filename_candidates:
            target_file = os.path.join(target_d, candidate)
            if os.path.exists(target_file):
                return target_file
        
        client = SSHClient()
        client.set_missing_host_key_policy(AutoAddPolicy())
        client.load_system_host_keys()
        client.connect(settings.labelstudio.url, username=settings.labelstudio.username, password=settings.labelstudio.password)

        transport = client.get_transport()
        if transport is None:
            raise RuntimeError("Failed to get SSH transport")

        scp: Optional[SCPClient] = None
        try:
            scp = SCPClient(transport)
            scp.get(origin_d, target_d)
        finally:
            if scp is not None:
                scp.close()
            client.close()

        for candidate in filename_candidates:
            target_file = os.path.join(target_d, candidate)
            if os.path.exists(target_file):
                return target_file

        raise RuntimeError(
            f"Failed to download file: {filename}. "
            f"Checked destination directory: {target_d}"
        )

    def check_file(self, filename: str) -> bool:
        """
        Check if an audio file exists in the local dataset directory.
        
        Input:
            filename (str): Name of the file to check
            
        Output:
            bool: True if file exists, False otherwise
        """
        target_d = os.path.join(settings.folders.data_root, 'audio_data')
        decoded_filename = unquote(filename)
        candidates = [filename, decoded_filename, os.path.basename(filename), os.path.basename(decoded_filename)]
        return any(os.path.exists(os.path.join(target_d, candidate)) for candidate in candidates if candidate)

    def extract_mfcc(self, segment, sr: Optional[int] = None, 
                     n_mfcc: Optional[int] = None, n_fft: Optional[int] = None) -> npt.NDArray[np.float32]:
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features from audio segment.
        
        This method converts raw audio data into MFCC features which are commonly used
        for audio classification tasks. It uses mel-scale spectrogram as intermediate
        representation and computes mean MFCC coefficients across time frames.
        
        Input:
            segment (array-like): Raw audio waveform data
            sr (int, optional): Sample rate. Uses self.sr if None
            n_mfcc (int, optional): Number of MFCC coefficients. Uses self.n_mfcc if None
            n_fft (int, optional): Number of FFT components. Uses self.n_fft if None
            
        Output:
            np.ndarray: 1D array of MFCC features (shape: n_mfcc,)
        """
        if sr is None:
            sr = self.sr
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        if n_fft is None:
            n_fft = self.n_fft
            
        segment = segment / np.max(np.abs(segment) + 1e-8)
        
        # Apply pre-emphasis filter (common in speech processing)
        pre_emphasis = 0.97
        segment = np.append(segment[0], segment[1:] - pre_emphasis * segment[:-1])
        
        mfcc = librosa.feature.mfcc(
            y=segment, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=512, 
            hop_length=int(sr * 0.010),  # 10ms hop
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        return mfcc.mean(axis=1).T


    def extractAudio(self, file_path: str, sr: Optional[int] = None) -> npt.NDArray[np.float32]:
        """
        Load and resample audio file to specified sample rate.
        
        Input:
            file_path (str): Path to the audio file
            sr (int, optional): Target sample rate. Uses self.sr if None
            
        Output:
            np.ndarray: 1D array of audio waveform data
        """
        if sr is None:
            sr = self.sr
        audio, _ = librosa.load(file_path, sr=sr)
        return audio

    def prepare_data(self,new: bool = False) -> List[Dict[str, Any]]:
        """
        Prepare complete dataset from Label Studio annotations and speech files.
        
        This method creates a comprehensive dataset by:
        1. Loading cached data if available (prepared_dataset.json)
        2. Fetching annotation data from Label Studio API
        3. Downloading and processing annotated audio files
        4. Adding speech samples from local speech_dataset directory
        5. Caching the processed dataset for future use
        
        Input:
            None
            
        Output:
            list: Dataset containing dictionaries with structure:
                {
                    'filename': str - original filename
                    'samplerate': int - audio sample rate
                    'dataset': list - list of labeled audio segments:
                        [{
                            'label': str - audio class label
                            'start': int - start sample index
                            'end': int - end sample index
                            'audio': list - raw audio waveform data
                        }]
                }
                
        Side Effects:
            - Creates 'dataset' directory if it doesn't exist
            - Downloads audio files from remote server
            - Caches processed data in 'prepared_dataset.json'
            - Sets self.dataset to the prepared dataset
        """
        if not new:
            prepared_dataset_path = os.path.join(settings.folders.data_root, 'prepared_dataset.json')
            if os.path.exists(prepared_dataset_path):
                print("Loading cached dataset from prepared_dataset.json")
                with open(prepared_dataset_path, 'r', encoding='utf-8') as f_prepared:
                    loaded_data = json.load(f_prepared)
                # Return the dataset list, not the wrapper dict
                cached_dataset = loaded_data['dataset'] if 'dataset' in loaded_data else loaded_data
                self.dataset = cast(List[Dict[str, Any]], cached_dataset)
                return self.dataset
        if not self.grabAPI(project_id=self.project_id): raise RuntimeError("Not able to retrieve JSON")
        with open(os.path.join(settings.folders.data_root,'exported_data.json'), 'r', encoding='utf-8') as json_file:  # Specify encoding
            js = json.load(json_file)  # Load the JSON data
        dataset = []
        tq = tqdm(js, total=len(js))
        for entry in tq:
            try:
                filename = entry["file_upload"]
                tq.set_description(f"Processing {filename}")
                r_annotations = entry['annotations'][0]['result'] 
                annotations = []
                dt = []
                path = self.grabAudio(filename, project_id=self.project_id)
                audio_waveform, sample_rate = librosa.load(path, sr=self.sr)
                for e in r_annotations:
                    try:
                        start = int(float(e['value']['start'])*sample_rate)
                        end = int(float(e['value']['end'])*sample_rate)
                        dt.append(dict(label=e['value']['labels'][0], start=start, end=end, audio=audio_waveform[start:end].tolist()))
                    except Exception as a:
                        print(f"Error processing annotations for {filename}: this one does not have {a}")
                        continue
                f = dict(filename=filename, samplerate=sample_rate, dataset=dt)
                    
            except Exception as e:
                exc_type, _, exc_tb = sys.exc_info()
                filename_for_error = locals().get('filename', 'unknown file')
                if exc_tb is not None and exc_type is not None:
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(
                        f"Error processing {filename_for_error}: "
                        f"{exc_type.__name__} ({e}) in {fname}:{exc_tb.tb_lineno}"
                    )
                else:
                    logger.error(f"Error processing {filename_for_error}: {str(e)}")
                continue
            dataset.append(f)
        # print(dataset
        speech_dataset_path = os.path.join(settings.folders.data_root, 'speech_dataset')
        print("Adding speech dataset from:", speech_dataset_path)
        for root, _, files in os.walk(speech_dataset_path):
            tq = tqdm(files, total=len(files))
            for file in tq: #random.sample(files, k=75):
                if file.endswith('.flac'):  # Assuming the audio files are in .wav format
                    label = 'Speech'  # Label for the audio files
                    path = os.path.join(root, file)
                    audio_waveform, sample_rate = librosa.load(path, sr=self.sr)
                    f = dict(filename=file, samplerate=sample_rate, dataset=[dict(label=label, start=0, end=len(audio_waveform), audio=audio_waveform.tolist())])
                    dataset.append(f)
        with open(os.path.join(settings.folders.data_root,'prepared_dataset.json'), 'w', encoding='utf-8') as f_json:
            json.dump(dict(dataset=dataset), f_json)
        self.dataset = dataset
        os.remove(os.path.join(settings.folders.data_root,'exported_data.json'))
        return dataset

    def triage_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:   
        """
        Clean and standardize dataset labels for better classification.
        
        This method performs label consolidation and removal of unwanted classes:
        - Maps similar medical device labels to common categories
        - Standardizes alarm-related labels
        - Removes noisy or problematic label categories
        
        Input:
            dataset (list): Dataset structure from prepare_data()
            
        Output:
            list: Cleaned dataset with standardized labels
            
        Side Effects:
            - Modifies labels in-place within the dataset structure
            - Removes entries with specified unwanted labels
        """
        self.replaceLabel('ICU Medical', 'Hospital Devices', dataset)
        self.replaceLabel('Alaris', 'Hospital Devices', dataset)
        self.replaceLabel('Baxter', 'Hospital Devices', dataset)
        self.replaceLabel('SpaceLabs', 'Hospital Devices', dataset)
        self.replaceLabel('SpaceLAbs', 'Hospital Devices', dataset)
        self.replaceLabel('Room Call', 'Alarm', dataset)
        self.replaceLabel('alarm', 'Alarm', dataset)
        self.replaceLabel('siren', 'Alarm', dataset)
        self.replaceLabel('Alarm', 'Hospital Devices', dataset) #<-- Be careful with this
        self.replaceLabel('siren', 'HVAC', dataset)
        self.replaceLabel('Pneumatic Tube', 'objects clanging (non-metal)', dataset)
        self.replaceLabel('medical air valve', 'HVAC', dataset)
        self.replaceLabel('Manual Resuscitation Bag', 'HVAC', dataset)
        self.replaceLabel('curtains', 'Roling Carts', dataset)
        self.replaceLabel('Roling Carts', 'Rolling Carts', dataset)
        self.replaceLabel('drawers', 'HVAC', dataset)
        self.dropItemsWithLabel('Lifts', dataset)
        self.dropItemsWithLabel('Cabinet', dataset)
        self.dropItemsWithLabel('Baby Crying', dataset)
        self.dropItemsWithLabel('Sink/Water', dataset)
        self.dropItemsWithLabel('velcro (BP cuff)', dataset)
        self.dropItemsWithLabel('Composition', dataset)
        return dataset


    def plot_label_distribution(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Visualize the distribution of labels in the dataset.
        
        Input:
            dataset (list): Dataset structure from prepare_data()
            
        Output:
            None (displays matplotlib bar chart)
            
        Side Effects:
            - Shows interactive matplotlib plot with label counts
        """
        import matplotlib.pyplot as plt

        labels = []
        for entry in dataset:
            for item in entry['dataset']:
                labels.append(item['label'])
        label_counts = Counter(labels)
        plt.figure(figsize=(8, 5))
        plt.bar(list(label_counts.keys()), list(label_counts.values()), color='skyblue')
        plt.xlabel('Labels')
        plt.ylabel('Counts')
        plt.title('Label Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def replaceLabel(self, old_label: str, new_label: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replace all instances of old_label with new_label in dataset.
        
        Input:
            old_label (str): Label to be replaced
            new_label (str): New label to use as replacement
            dataset (list): Dataset structure to modify
            
        Output:
            list: Modified dataset with replaced labels
        """
        for entry in dataset:
            for item in entry['dataset']:
                if item['label'] == old_label:
                    item['label'] = new_label
        return dataset

    def dropItemsWithLabel(self, label_to_drop: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove all audio segments with specified label from dataset.
        
        Input:
            label_to_drop (str): Label to remove from dataset
            dataset (list): Dataset structure to modify
            
        Output:
            list: Modified dataset with specified label removed
        """
        for entry in dataset:
            entry['dataset'] = [item for item in entry['dataset'] if item['label'] != label_to_drop]
        return dataset
    
    # def extract_enhanced_features(self, segment, sr=16000):
    #     """Extract multiple audio features for better classification"""
        
    #     # Original MFCC features
    #     mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=512, hop_length=int(sr * 0.010))
    #     mfcc_mean = mfcc.mean(axis=1)
    #     mfcc_std = mfcc.std(axis=1)
        
    #     # Spectral features
    #     spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
    #     spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
    #     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
        
    #     # Zero crossing rate (good for speech vs non-speech)
    #     zcr = librosa.feature.zero_crossing_rate(segment)[0]
        
    #     # Chroma features (good for harmonic content)
    #     chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    #     chroma_mean = chroma.mean(axis=1)
        
    #     # Tempo and rhythm (good for footsteps, rolling carts)
    #     tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
        
    #     # Combine all features
    #     features = np.concatenate([
    #         mfcc_mean, mfcc_std,
    #         [spectral_centroids.mean(), spectral_centroids.std()],
    #         [spectral_rolloff.mean(), spectral_rolloff.std()],
    #         [spectral_bandwidth.mean(), spectral_bandwidth.std()],
    #         [zcr.mean(), zcr.std()],
    #         chroma_mean,
    #         [tempo, len(beats)]
    #     ])
    #     return features


    def concatenateSound(self, dataset: List[Dict[str, Any]], sr: Optional[int] = None) -> Dict[str, npt.NDArray[np.float32]]:
        """
        Concatenate all audio segments for each unique label.
        
        This method groups audio segments by their labels and concatenates
        all segments of the same class into a single long audio clip per class.
        Useful for analyzing class distributions and creating continuous audio streams.
        
        Input:
            dataset (list): Dataset structure from prepare_data()
            sr (int, optional): Sample rate. Uses self.sr if None
            
        Output:
            dict: Dictionary mapping label names to concatenated audio arrays
                {'label1': np.ndarray, 'label2': np.ndarray, ...}
                
        Side Effects:
            - Shows progress bars during concatenation process
        """
        if sr is None:
            sr = self.sr
        labels = []
        for entry in dataset:
            for item in entry['dataset']:
                labels.append(item['label'])
        concat = dict()
        tq = tqdm(labels, total=len(labels), position=0)
        for key in tq:
            tq.set_description(f"Concatenating for label: {key}")
            concatenated_audio = np.array([], dtype=np.float32)
            tq1 = tqdm(dataset, total=len(dataset), position=1, leave=False)
            for entry in tq1:
                for item in entry['dataset']:
                    if item['label'] == key:
                        audio_segment = np.array(item['audio'], dtype=np.float32)
                        concatenated_audio = np.concatenate((concatenated_audio, audio_segment))
            concat[key] = concatenated_audio
        return concat

        

    def chop_audio(self, audio, segment_duration: float = 3.0, 
                   hop_rate: float = 1.5, sr: Optional[int] = None) -> List[npt.NDArray[np.float32]]:
        """
        Split long audio into overlapping segments of fixed duration.
        
        Input:
            audio (np.ndarray): Input audio waveform
            segment_duration (float): Duration of each segment in seconds
            hop_rate (float): Hop rate multiplier (1.5 means 50% overlap)
            sr (int, optional): Sample rate. Uses self.sr if None
            
        Output:
            list: List of audio segments, each of length segment_duration * sr
        """
        if sr is None:
            sr = self.sr
        hop_samples = int(segment_duration * hop_rate * sr)
        chopped = []
        for start in range(0, len(audio), hop_samples):
            end = start + int(segment_duration * sr)
            segment = audio[start:end]
            if len(segment) == int(segment_duration * sr):
                if len(segment) == 0:
                    print("Warning: Zero-length segment encountered.")
                chopped.append(segment)

        return chopped

    




    def merge_datasets(self, dataset):
        """
        Merge multiple dataset entries into a single dataset by concatenating audio segments of the same label. 
        Input:
            dataset (list): List of dataset entries, each containing 'dataset' key with audio segments
        Output:
            dict: Merged dataset with labels as keys and concatenated audio segments as values
        """
        datasets = [d['dataset'] for d in dataset if len(d['dataset']) > 0]
        db = dict()
        for dt in datasets:
            d = dt[0]
            if d['label'] not in db.keys():
                db[d['label']] = dict(label=d['label'], audio=d['audio'][d['start']:d['end']])
            else:
                db[d['label']]['audio'] += d['audio'][d['start']:d['end']]
        return db
    
    def split_data(self, dataset: Dict[str, Any]) -> Tuple[List[npt.NDArray[np.float32]], List[str]]:
        """
        Split dataset with MFCC features into train/test sets.
        
        Input:
            dataset (dict): Dataset dictionary with 'dataset' key containing MFCC features
            
        Output:
            tuple: ([x_train, x_test, y_train, y_test], labels) where arrays contain splits and labels list
        """
        x = []
        y = []
        labels = []
        for item in dataset['dataset']:
            x.append(item['mfcc'])
            lab = item['label']
            if lab not in labels:
                labels.append(lab)
            y.append(labels.index(lab))
        
        x = np.array(x)
        y = to_categorical(np.array(y))
        self.labels = labels
        
        # Optionally, you can split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        return [x_train, x_test, y_train, y_test], labels

    def evaluate(self, model: tf.keras.Model, x_test, 
                 y_test, label: List[str]) -> None:
        """
        Evaluate trained model performance on test data.
        
        This method computes test accuracy, generates predictions, and provides
        detailed classification metrics including precision, recall, and F1-score
        for each class.
        
        Input:
            model (tf.keras.Model): Trained model to evaluate
            x_test (np.ndarray): Test features (MFCC coefficients)
            y_test (np.ndarray): Test labels (one-hot encoded)
            label (list): List of class label names
            
        Output:
            None (prints evaluation metrics and shows visualization)
            
        Side Effects:
            - Prints test loss and accuracy
            - Displays detailed classification report
            - Shows bar chart visualization of predictions vs ground truth
        """
        # Evaluate the model on the test set
        model.evaluate(x_test, y_test, verbose=1)
        # Make predictions on the test set
        predictions = model.predict(x_test)

        # Convert predictions from probabilities to class labels
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        # Generate a classification report
        report = classification_report(true_classes, predicted_classes, target_names=[label[i] for i in np.unique(true_classes)])
        print(report)

        # Plot a few test samples and their predicted labels
        num_samples = len(true_classes)
        # plt.figure(figsize=(15, 5))
        # # for i in range(num_samples):
        # #     plt.subplot(1, num_samples, i + 1)
        # #     plt.plot(x_test[i])
        # plt.bar(range(num_samples), true_classes, color='blue', alpha=0.5)
        # # plt.title(f'Predicted: {label[predicted_classes[i]]}')
        # plt.axis('off')
        # plt.show()


    def build_model(self, num_classes: int = 9) -> tf.keras.Model:
        """
        Build and compile a CNN model for audio classification.
        
        Creates a sequential model with Conv1D layers for feature extraction,
        dense layers for classification, and dropout for regularization.
        Architecture: Conv1D -> MaxPool -> Dense -> Dropout -> Conv1D -> MaxPool -> 
                     Dense -> Dropout -> MaxPool -> Flatten -> Dense -> Dropout -> Dense
        
        Input:
            num_classes (int): Number of output classes for classification
            
        Output:
            tf.keras.Model: Compiled TensorFlow/Keras model ready for training
            
        Side Effects:
            - Sets self.model to the created model
            - Prints model summary and device information
            - Uses GPU if available, otherwise CPU
        """
        # Use GPU if available for model creation
        device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Building model on: {device_name}")

        with tf.device(device_name):
            input_shape = (self.n_mfcc, 1)
            model = tf.keras.Sequential()# pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Input(shape=input_shape))# pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Conv1D(24, 3, padding='same', activation='relu')) # pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2)) # pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Dense(48, activation='relu'))# pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Dropout(0.25)) # pyright: ignore[reportAttributeAccessIssue]
            
            # model.add(tf.keras.layers.Conv1D(48, 3, padding='same', activation='relu')) # pyright: ignore[reportAttributeAccessIssue]
            # model.add(tf.keras.layers.MaxPooling1D(pool_size=2)) # pyright: ignore[reportAttributeAccessIssue]
            # model.add(tf.keras.layers.Dense(64, activation='relu')) # pyright: ignore[reportAttributeAccessIssue]
            # model.add(tf.keras.layers.Dropout(0.25)) # pyright: ignore[reportAttributeAccessIssue]
            # model.add(tf.keras.layers.MaxPooling1D(pool_size=2)) # pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Flatten()) # pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Dense(64, activation='relu')) # pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Dropout(0.5)) # pyright: ignore[reportAttributeAccessIssue]
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # pyright: ignore[reportAttributeAccessIssue]
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy()])
            model.summary()
            # plot_model(model, to_file="wfmodel.png", show_shapes=True, show_layer_activations=True)
        self.model = model
        return model

    def train(self, model: tf.keras.Model, x_train, 
              x_test, y_train, 
              y_test) -> tf.keras.callbacks.History:
        """
        Train the neural network model with early stopping and checkpointing.
        
        This method trains the provided model using the training data with validation
        on test data. Includes early stopping to prevent overfitting and model
        checkpointing to save best weights during training.
        
        Input:
            model (tf.keras.Model): The model to train
            x_train (np.ndarray): Training features (MFCC coefficients)
            x_test (np.ndarray): Test features for validation
            y_train (np.ndarray): Training labels (one-hot encoded)
            y_test (np.ndarray): Test labels for validation
            
        Output:
            tf.keras.History: Training history containing loss and metrics
            
        Side Effects:
            - Saves model checkpoints during training
            - Displays training progress and GPU information
            - Shows loss plot after training completion
        """
        # Check GPU availability and display info
        print("GPU Available: ", tf.config.list_physical_devices('GPU'))
        print("Built with CUDA: ", tf.test.is_built_with_cuda())
        
        # Use GPU if available, otherwise fallback to CPU
        device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Training on: {device_name}")
        
        with tf.device(device_name):
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]

            # model.save_weights(os.path.join(pathname, "/cp-{epoch:04d}.weights.h5").format(epoch=0))
            
            # Convert data to tensors and place them on the selected device
            # x_train_tensor = tf.convert_to_tensor(x_train)
            # y_train_tensor = tf.convert_to_tensor(y_train)
            # x_test_tensor = tf.convert_to_tensor(x_test)
            # y_test_tensor = tf.convert_to_tensor(y_test)
            
            history = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
        
        # fig=plt.figure(figsize=(12,4))
        # plt.plot(history1.epoch, history1.history['loss'], label="Dense")
        # # plt.plot(history1.epoch, history1.history['accuracy'], label="Dense")
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        return history

    def save_model_ext(self, model: tf.keras.Model, filepath: str, overwrite: bool = True, 
                       meta_data: Optional[Any] = None) -> None:
        """
        Save trained model with optional metadata to HDF5 format.
        
        This method saves the complete model architecture, weights, and training
        configuration. Optionally includes custom metadata (like label information)
        as HDF5 attributes.
        
        Input:
            model (tf.keras.Model): Trained model to save
            filepath (str): Path where to save the model file
            overwrite (bool): Whether to overwrite existing file
            meta_data (any, optional): Additional metadata to store with model
            
        Output:
            None
            
        Side Effects:
            - Creates model file at specified filepath
            - Adds metadata as HDF5 attributes if provided
        """
        tf.keras.models.save_model(model, filepath, overwrite)
        if meta_data is not None:
            f = h5py.File(filepath, mode='a')
            f.attrs['label_data'] = meta_data
            f.close()

    def process_mfcc(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process dataset to extract MFCC features for each audio segment.
        
        This method iterates through the dataset, extracts MFCC features from
        each audio segment using the specified parameters, and adds the features
        back into the dataset structure.
        
        Input:
            dataset (list): Dataset structure from prepare_data()

        Output:
            list: Modified dataset with 'mfcc' key added to each audio segment
        """
        tq = tqdm(dataset, total=len(dataset))
        for entry in tq:
            tq.set_description(f"Extracting MFCC for {entry['filename']}")
            for item in entry['dataset']:
                segment = np.array(item['audio'], dtype=np.float32)
                mfcc = self.extract_mfcc(segment, sr=entry['samplerate'], n_mfcc=self.n_mfcc, n_fft=self.n_fft)
                item['mfcc'] = mfcc
        return dataset

    def save_model_tflite(self, model: tf.keras.Model, tflite_path: str) -> None:   
        """
        Convert and save the trained model to TensorFlow Lite format.

        Args:
            model (tf.keras.Model): Trained Keras model to convert.
            tflite_path (str): Path to save the .tflite file.

        Output:
            None

        Side Effects:
            - Creates a .tflite file at the specified path.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

if __name__ == "__main__":
    # Example usage:
    trainer = indiv_trainer()
    trainer.setParameters(sr=16000, n_mfcc=40, n_fft=512)
    
    # Example workflow:
    dataset = trainer.prepare_data(new=False)
    dataset = trainer.triage_dataset(dataset)

    with open('triaged_dataset.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps({'dataset': dataset}, cls=NumpyEncoder))
    print("Triaged dataset saved to triaged_dataset.json")
    # trainer.plot_label_distribution(dataset)
    print(dataset[0]['dataset'][0].keys())
    dataset = trainer.process_mfcc(dataset)
    print(dataset[0]['dataset'][0].keys())

    unique_labels = []
    for d in dataset:
        for item in d['dataset']:
            if item['label'] not in unique_labels:
                unique_labels.append(item['label'])
    print(f"Unique labels: {unique_labels}")
    anal = []
    for label in unique_labels:
        label_dataset = []
        other_dataset = []
        for d in dataset:
            for item in d['dataset']:
                if item['label'] == label:
                    label_dataset.append(item)
                else:
                    other_dataset.append(item)
        # print(len(label_dataset), len(other_dataset))
        sampled_other = random.sample(other_dataset, k=len(label_dataset)*2 if len(other_dataset) > len(label_dataset)*2 else len(other_dataset))
        combined = label_dataset + sampled_other
        random.shuffle(combined)

        X = [item['mfcc'] for item in combined]
        y = [1 if item['label'] == label else 0 for item in combined]

        X = np.array(X)
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
        model = trainer.build_model(num_classes=2)
        history = trainer.train(model, x_train, x_test, y_train, y_test)
        
        eval = trainer.evaluate(model, x_test, y_test, ["Other", label])
        anal.append((label, eval, history))
        trainer.save_model_tflite(model, f"models/{label}_model.tflite")

    import matplotlib.pyplot as plt

    # Plot and analyze all labels using anal
    label_names = [x[0] for x in anal]
    accuracies = []
    for label, eval_result, history in anal:
        # Extract final validation accuracy from training history
        val_acc = history.history.get('val_accuracy', [0])[-1]
        accuracies.append(val_acc)

    plt.figure(figsize=(10, 6))
    plt.bar(label_names, accuracies, color='skyblue')
    plt.xlabel('Label')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('label_accuracies.png')
    plt.show()