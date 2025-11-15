import gdown
import zipfile
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def download_and_organize():
    print(">> DOWNLOAD AND ORGANIZE DATASET")
    print("="*60)

    # Remove existing
    if os.path.exists("dataset_organized"):
        print(">> Removing existing dataset_organized...")
        shutil.rmtree("dataset_organized")

    # Download
    if not os.path.exists("dataset.zip"):
        print(">> Downloading dataset...")
        FILE_ID = "1o3rl6Ap4QjxM5-C9WtiJMIgIwWVFK_ZO"  # GANTI FILE ID KAMU
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, "dataset.zip", quiet=False)

    # Extract
    print(">> Extracting...")
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")

    # Clean macosx
    mac_path = Path("dataset") / "__MACOSX"
    if mac_path.exists():
        shutil.rmtree(mac_path)

    # Read CSV
    data_path = Path("dataset")
    csv_path = data_path / "train.csv"
    df = pd.read_csv(csv_path).dropna()
    classes = sorted(df['label'].unique())

    print(f">> Found {len(df)} images, {len(classes)} classes: {classes}")

    # Create folders
    organized_path = Path("dataset_organized")
    for split in ['train', 'val']:
        for cls in classes:
            (organized_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Stratified split
    train_files, val_files = [], []
    for cls in classes:
        cls_files = df[df['label'] == cls]['filename'].tolist()
        train_names, val_names = train_test_split(cls_files, test_size=0.2, random_state=42)
        train_files.extend([(n, cls) for n in train_names])
        val_files.extend([(n, cls) for n in val_names])

    # Copy files
    print(">> Copying train files...")
    for filename, cls in train_files:
        src = data_path / "train" / filename
        dst = organized_path / "train" / cls / filename
        if src.exists():
            shutil.copy2(src, dst)

    print(">> Copying val files...")
    for filename, cls in val_files:
        src = data_path / "train" / filename
        dst = organized_path / "val" / cls / filename
        if src.exists():
            shutil.copy2(src, dst)

    # Cleanup
    os.remove("dataset.zip")
    print(f">> Dataset organized: {len(train_files)} train, {len(val_files)} val")
    print(f">> Path: dataset_organized")

if __name__ == "__main__":
    download_and_organize()
