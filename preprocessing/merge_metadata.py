import os
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_file', default="clean_metadata.csv", type=str)
args = parser.parse_args()
    
output_path = os.path.join(args.input_dir, args.output_file)

csv_dir = Path(args.input_dir)
csv_files = sorted(csv_dir.glob("*.csv"))

df_list = [pd.read_csv(f) for f in csv_files]

combined = pd.concat(df_list, ignore_index=True)

combined.to_csv(f"{output_path}", index=False)
print(f"完成：寫出 {output_path}")