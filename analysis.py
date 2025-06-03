import os
os.environ["QT_QPA_PLATFORM"] = "offscreen" 
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--score_file", type=str, required=True)
args = parser.parse_args()

# Load the uploaded score file (prediction, target, predict score)
df = pd.read_csv(args.score_file, sep=r'\s+', header=None, names=['pred', 'true', 'score'])

# Determine unique class labels (sorted for consistency)
labels = sorted(df['true'].unique())

# Compute confusion matrix
cm = confusion_matrix(df['true'], df['pred'], labels=labels)

# Wrap into a DataFrame for nicer heatmap annotation
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.1)
heatmap = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5)

heatmap.set_xlabel('Predicted Label')
heatmap.set_ylabel('True Label')
heatmap.set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig("confusion_matrix.png")
