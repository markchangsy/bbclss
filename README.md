# Audio Classification Training Pipeline

## Overview

This repository provides an end‑to‑end **PyTorch** pipeline for **supervised sound‑event classification**. It consists of three standalone scripts that cover feature extraction, model training, and final evaluation. Although it was originally built for a baby‑cry detection task, the code is generic—changing a few paths in a JSON config lets you reuse it on any multi‑class audio dataset stored as WAV files.

```text
.
├── preprocessingBabyCrying.py   # Feature extraction & dataset split
├── train.py                     # Model training & validation
├── evaluation.py                # Final test‑set evaluation
├── models/                      # CNN backbones (DenseNet, ResNet, Inception, EfficientNet…)
├── dataloaders/
│   └── dataloader.py            # Loads *.npy features + labels (uses list files)
├── utils.py                     # I/O helpers, running‑average meter, Params wrapper
├── features/
│   └── filenames/               # Metadata & split lists
│       ├── UrbanSound8K_babycrying.csv
│       ├── train.txt
│       ├── validation.txt
│       └── evaluation.txt
└── configs/
    └── example.json             # Hyper‑parameter & path template
```

---

## 1  Installation (Docker)

Run everything in an isolated Docker container so you do **not** have to worry about local CUDA/tool‑chain mismatches.

```bash
# 1. Pull the official PyTorch runtime image (CUDA 11.3 + cuDNN 8)
docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# 2. Launch an interactive container (mount the repo at /workspace)
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  --name audiocls \
  pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime /bin/bash
```

Inside the container:

```bash
# 3. System dependencies (for audio I/O)
apt-get update && apt-get upgrade -y \
  && apt-get install -y libsndfile1-dev

# 4. Core Python libraries
pip install librosa pandas torchsummary tensorboardX

# 5. (Optional) pin exact PyTorch wheels shipped with CUDA 11.3
pip install torch==1.12.1+cu113 \
            torchvision==0.13.1+cu113 \
            torchaudio==0.12.1 \
            --extra-index-url https://download.pytorch.org/whl/cu113
```

The base image already contains PyTorch 1.12.1 with matching CUDA libraries, but step 5 pins the versions explicitly should you mutate the environment later.

When you exit (`Ctrl‑D`), the container is removed (`--rm`); all files in the mounted volume persist on your host.

---

## 2  Data preparation

### 2.1  Datasets

| Dataset                   | Description                                                                                                                                                                                                                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **UrbanSound8K**          | 8 732 urban audio clips across 10 sound classes (e.g., engine idling, dog bark, siren). In this template we extract only the *baby‑cry* class. Dataset homepage: [https://urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html). |
| **ESC-50**          |2 000 environmental audio clips (5 s each) covering 50 balanced classes. It includes a baby cry category (≈ 40 examples) but the sample count is too small for robust training, so we do not use it in this pipeline. Repo: https://github.com/karolpiczak/ESC-50 |
| **BabyCrying (AudioSet)** | Additional baby‑cry segments crawled from Google’s AudioSet                                                                                                                                                                                   |

### 2.2  Required files and directories

1. **`metadata.csv`** – a CSV with at least the following columns:

   | column            | description                                |
   | ----------------- | ------------------------------------------ |
   | `slice_file_name` | Relative path or filename of each WAV file |
   | `classID`         | Integer label starting at 0                |
   | `class`           | Human‑readable label                       |
2. **Raw audio directory** – all WAV files referenced in `metadata.csv`.

### 2.3  Feature extraction & split

```bash
python preprocessingBabyCrying.py \
  --csv_file   metadata.csv \
  --data_dir   ./wav \
  --save_dir   ./features \
  --samplerate 16000 \
  --window_size 25 \
  --hop_size   10 \
  --n_fft      1024 \
  --n_mels     128
```

#### Parameter descriptions

| Flag            | Description                                                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `--samplerate`  | Target sampling rate in **Hz**. Audio will be resampled if the original differs.                                                   |
| `--window_size` | STFT analysis window length in **milliseconds**. Typical speech settings use 25 ms.                                                |
| `--hop_size`    | Frame shift between consecutive windows in **milliseconds**; controls time resolution (e.g., 10 ms).                               |
| `--n_fft`       | Number of FFT points. Must be ≥ window length (in samples); higher values improve frequency resolution at the cost of computation. |
| `--n_mels`      | Number of mel filterbank channels in the log‑mel spectrogram; sets the feature’s frequency dimension (e.g., 128).                  |

The script will:

* Generate one `*.npy` file (128 × 250 log‑mel spectrogram) per audio clip.
* Stratify the data per `classID` into 10 folds, shuffle, and write three list files:

  * `features/filenames/train.txt` — 80 %
  * `features/filenames/validation.txt` — 10 %
  * `features/filenames/evaluation.txt` — 10 %

Each line in the list has the format:

```text
<feature‑file.npy> <classID> <class‑name>
```

### Example metadata file: `UrbanSound8K_babycrying.csv`

The repository includes a ready‑to‑use CSV derived from the UrbanSound8K corpus and filtered to **baby‑cry** events. It contains every mandatory column listed above plus some UrbanSound‑specific fields explained below:

| column  | description                                                                                |
| ------- | ------------------------------------------------------------------------------------------ |
| `fold`  | Original UrbanSound8K fold index (1‑10). Useful if you want to reproduce the corpus split. |
| `start` | Event start time within the clip (s). Not used by the pipeline.                            |
| `end`   | Event end time within the clip (s). Not used by the pipeline.                              |

> **Note**  Any extra columns are safely ignored by all scripts. Only `slice_file_name`, `classID`, and `class` are required.

A minimal row example (comma‑separated):

```csv
100032-3-0-0.wav,3,1.20,3.67,0,babycry
```

---

## 3  Configuration file

All hyper‑parameters and paths live in a single JSON file. A minimal example:

```jsonc
{
  // ---------- paths ----------
  "datapath":      "features",                      // root dir that holds *.npy
  "trainlist":     "features/filenames/train.txt",
  "vallist":       "features/filenames/validation.txt",
  "evallist":      "features/filenames/evaluation.txt",
  "checkpoint_dir": "checkpoints/babycry",
  "summaries_dir":  "runs/babycry",

  // ---------- model ----------
  "model":         "efficientnetv2_s",             // choices: densenet | resnet | inception | resnet18 | resnet34 | resnet50 | efficientnetv2_s
  "pretrained":    true,                            // load ImageNet weights
  "dataset_name":  "BabyCrying",

  // ---------- training ----------
  "batch_size":    32,
  "epochs":        120,
  "lr":            3e-4,
  "weight_decay":  1e-4,
  "scheduler":     true,                            // StepLR(gamma=0.1, step_size=30)
  "num_workers":   4
}
```

Save it as, for example, `configs/babycry.json`.

---

## 4  Training

```bash
python train.py --config_path configs/babycry.json
```

The script will:

* Instantiate the selected backbone.
* Train on the **training** split and compute accuracy on the **validation** split each epoch.
* Keep the five most recent checkpoints and copy the best one to `model_best.pth.tar`.
* Write TensorBoard logs to `runs/babycry` (`tensorboard --logdir runs`).

---

## 5  Evaluation

```bash
python evaluation.py --config_path configs/babycry.json
```

Outputs:

* Clip‑level predictions and softmax confidence to `checkpoints/babycry/scores/score.txt`.
* Overall accuracy printed to console and appended to the same file.

---

## 6  Adapting to your own dataset

Follow these steps to plug‑in *any* collection of WAV files:

1. **Create a metadata CSV**

   At minimum include three columns (order doesn’t matter):

   ```csv
   slice_file_name,classID,class
   dog01.wav,0,dog
   cat01.wav,1,cat
   ...
   ```

   * `slice_file_name` – relative path from your audio root.
   * `classID` – integer starting at 0 (contiguous).
   * `class` – human‑readable label.

   You may keep extra columns (e.g., duration, speaker); the pipeline ignores them.

2. **Place raw audio clips**

   Put all `.wav` files under a folder of your choice (e.g. `./wav_raw`). Sub‑directories are allowed as long as the relative paths in the CSV match.

3. **Extract features & generate split lists**

   ```bash
   python preprocessingBabyCrying.py \
     --csv_file   your_dataset.csv \
     --data_dir   ./wav_raw \
     --save_dir   ./features_your \
     --samplerate 16000 \
     --window_size 25 --hop_size 10 \
     --n_fft 2048 --n_mels 128
   ```

   This command will create the directory structure:

   ```text
   features_your/
       filenames/
           train.txt         # 80 %
           validation.txt    # 10 %
           evaluation.txt    # 10 %
       *.npy                 # one per clip
   ```

4. **Clone and edit a config JSON**

   ```jsonc
   {
     "datapath": "features_your",
     "trainlist": "features_your/filenames/train.txt",
     "vallist": "features_your/filenames/validation.txt",
     "evallist": "features_your/filenames/evaluation.txt",

     "checkpoint_dir": "checkpoints/your",
     "summaries_dir":  "runs/your",

     "dataset_name": "YourDataset",
     "model": "efficientnetv2_s"
   }
   ```

   Copy the rest of the training hyper‑parameters or tweak as desired.

5. **Verify class count**

   The dataloader reads the list files and automatically sets the number of output neurons, so *no code change* is required if your CSV is correct.

6. **Train and evaluate**

   ```bash
   python train.py      --config_path configs/your.json
   python evaluation.py --config_path configs/your.json
   ```



