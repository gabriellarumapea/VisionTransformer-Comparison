import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

def resume_training(checkpoint_path, model, optimizer, scheduler):
    """Load checkpoint dan resume"""
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        print(f">> Resumed from epoch {start_epoch} with best acc {best_acc:.2f}%")
        return start_epoch, best_acc, history
    return 0, 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}