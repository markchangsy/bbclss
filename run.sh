#!/bin/bash

set -e

# Pull latest version
git pull origin main

root_path=/data
[ -d "$root_path" ] || mkdir -p "$root_path"

clean_dir=$root_path/clean_data
augment_dir=$root_path/train_data
feat_dir=$root_path/feat_data

config_path=config/babycrying_resnet34.json

clean_csv=filenames/clean_metadata.csv
augment_csv=filenames/augmentation_metadata.csv

musan_dir=$root_path/musan
rir_dir=$root_path/RIRS_NOISES

# Download clean data
python minio_download.py --objects babycrying_16k_clean.zip --output_path $clean_dir
mv $clean_dir/mixed/*.wav $clean_dir # 現在 metadata 只有檔名，沒有相對路徑，所以先手動移到同個資料夾裡
python minio_download.py --objects UrbanSound8K_16Ksr.zip --output_path $clean_dir
mv $clean_dir/audio/*.wav $clean_dir # 同上
# python minio_download.py --objects voxceleb1_dev.zip --output_path $clean_dir

# Merge metadata -> clean_metadata.csv
python preprocessing/merge_metadata.py --input_dir $clean_dir

# Download noise data
python minio_download.py --objects musan.zip --output_path $root_path
python minio_download.py --objects RIRS_NOISES.zip --output_path $root_path

# Data augmentation
python preprocessing/data_augmentation.py --input_csv $clean_csv \
                                        --audio_dir $clean_dir \
                                        --output_dir $augment_dir \
                                        --output_csv $augment_csv \
                                        --musan_dir $musan_dir \
                                        --rir_dir $rir_dir

# Put all data together
cp $clean_dir/*.wav $augment_dir

# Data preprocessing
python preprocessing/preprocessingBabyCrying.py --csv_file $augment_csv \
                                  --data_dir $augment_dir \
                                  --save_dir $feat_dir

# Training
python train.py --config $config_path

# Evaluation
python evaluation.py --config $config_path