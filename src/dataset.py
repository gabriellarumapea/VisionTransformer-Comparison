from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class ImageClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_names=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        if class_names is not None:
            self.classes = class_names
        else:
            self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        self.samples.append((img_path, self.class_to_idx[class_dir.name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
