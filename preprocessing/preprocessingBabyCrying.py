"""preprocessingBabyCrying.py

Audio preprocessing pipeline for the baby‑crying sound‑classification project.

The script converts every *.wav* recording referenced in the metadata CSV into
fixed‑length mel‑spectrogram numpy files and generates filelists for the
training, validation and evaluation splits.
"""

import os
import torch
import librosa
import argparse
import pandas as pd
import numpy as np
import torchaudio
from pathlib import Path

def extract_spectrogram(audio, sr, window_size, hop_size, n_fft, n_mels):
    """Convert a waveform array to a fixed‑length mel‑spectrogram.

    Parameters
    ----------
    audio
        1‑D numpy array containing the waveform (floating‑point, mono).
    sr
        Target sample rate in Hz (e.g. ``16000``).
    window_size
        Window length in milliseconds.
    hop_size
        Hop size in milliseconds.
    n_fft
        FFT size for STFT.
    n_mels
        Number of mel bins.

    Returns
    -------
    np.ndarray
        2‑D array with shape ``(n_mels, 250)``.
    """

    window_length = int(round(window_size*sr/1000))
    hop_length = int(round(hop_size*sr/1000))
    audio_tensor = torch.Tensor(audio)

    # Compute mel‑spectrogram → shape: (n_mels, num_frames)
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=window_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )(audio_tensor)

    feat = spectrogram.cpu().numpy()

     # Ensure every example has exactly 250 frames
    num_frame = feat.shape[1]
    if num_frame < 250:
        # Zero‑pad on the right (time axis) to reach 250 frames.
        feat = np.pad(feat, [(0,0),(0,250-num_frame)], mode='constant')
    else:
        # Truncate to the first 250 frames.
        feat = feat[:,:250]
    
    return np.array(feat)

def extract_features(audio_path, sr, save_dir, window_size, hop_size, n_fft, n_mels):
    """Load a single waveform, extract its mel‑spectrogram and persist to *.npy*.

    The output filename mirrors the input, replacing the *.wav* extension with
    *.npy* so that data loaders can resolve audio/feature pairs easily.
    """

    filename = os.path.basename(audio_path)[:-3]+"npy"
    save_path = os.path.join(save_dir, filename)
    audio, sr = librosa.load(audio_path, sr=sr)
    specs = extract_spectrogram(audio, sr, window_size, hop_size, n_fft, n_mels)
    np.save(save_path, specs)


def make_splits(df, audio_root, n_folds=10, seed=42):
    """Create stratified train/validation/test splits.

    Files missing on disk are filtered out up‑front to avoid silent failures
    during training.
    """

    exists_mask = df["slice_file_name"].apply(
        lambda fname: (Path(audio_root) / fname).is_file()
    )
    missing_count = (~exists_mask).sum()

    if missing_count:
        print(f"[INFO] 發現 {missing_count} 筆檔案不存在，已排除。")

    df = df[exists_mask].reset_index(drop=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df["fold"] = (
        df.groupby("classID")
          .cumcount()
          .mod(n_folds)
    )

    train_set = df[df["fold"] < 8].reset_index(drop=True)
    valid_set = df[df["fold"] == 8].reset_index(drop=True)
    test_set  = df[df["fold"] == 9].reset_index(drop=True)

    return train_set, valid_set, test_set

def write_filename(df, mode, save_path):
    with open(os.path.join(save_path, f"{mode}"), "w") as f:
        for index, data in df.iterrows():
            feat = data["slice_file_name"][:-3]+"npy"
            label_id = data["classID"]
            label = data["class"]
            f.write(f"{feat} {label_id} {label}\n")

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.csv_file, skipinitialspace=True)

    for index, data in df.iterrows():
        filename = data["slice_file_name"]
        audio_path = os.path.join(args.data_dir, filename)

        if os.path.exists(audio_path):
            extract_features(audio_path, args.samplerate, args.save_dir, 
                             args.window_size, args.hop_size,
                             args.n_fft, args.n_mels)

    train_df, valid_df, test_df = make_splits(df, args.data_dir, seed=args.seed)

    save_list = os.path.join(args.save_dir, "filenames")
    os.makedirs(save_list, exist_ok=True)
    write_filename(train_df, "train.txt", save_list)
    write_filename(valid_df, "validation.txt", save_list)
    write_filename(test_df, "evaluation.txt", save_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre‑process baby‑crying audio dataset.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to metadata CSV (contains slice_file_name, classID, class …)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the raw *.wav* files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save *.npy* features and filelists.")
    parser.add_argument("--samplerate", default=16000, type=int, help="Target sample rate [Hz].")
    parser.add_argument("--window_size", default=25, type=int, help="STFT window size [ms].")
    parser.add_argument("--hop_size", default=10, type=int, help="STFT hop size [ms].")
    parser.add_argument("--n_fft", default=1024, type=int, help="FFT size.")
    parser.add_argument("--n_mels", default=128, type=int, help="Number of mel filter‑bank channels.")
    parser.add_argument("--seed", default=42, type=int, help="RNG seed for shuffling the dataset.")

    main(args)
