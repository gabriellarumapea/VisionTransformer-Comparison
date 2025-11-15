import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import ImageClassificationDataset
from models import create_model, count_parameters
from training import train_model, evaluate_model
from evaluation import measure_inference_time, compute_metrics
from visualization import plot_learning_curves, plot_confusion_matrix

def main():
    print(">> VISION TRANSFORMER COMPARISON")
    print("="*80)

    # Config (REKOMENDED: 10 epochs untuk test cepat, ganti jadi 15 untuk full)
    CONFIG = {
        'dataset_path': 'dataset_organized',
        'img_size': 224,
        'batch_size': 32,
        'epochs': 10,  # GANTI JADI 15 UNTUK FULL TRAINING
        'learning_rate': 1e-4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_classes': None,
        'class_names': None
    }

    print(f">> DEVICE: {CONFIG['device']}")

    # Check dataset
    if not Path(CONFIG['dataset_path']).exists():
        print(">> ERROR: Dataset not found. Run 'python download_dataset.py' first")
        return

    # Load classes
    train_path = Path(CONFIG['dataset_path']) / "train"
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    CONFIG['num_classes'] = len(classes)
    CONFIG['class_names'] = classes

    print(f">> CLASSES ({len(classes)}): {classes}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DataLoaders
    train_dataset = ImageClassificationDataset(CONFIG['dataset_path'] + "/train", train_transform, classes)
    val_dataset = ImageClassificationDataset(CONFIG['dataset_path'] + "/val", val_transform, classes)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True if CONFIG['device'].type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True if CONFIG['device'].type == 'cuda' else False)

    print(f">> DATALOADERS: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Models
    MODEL_CONFIGS = {
        'ViT-Base': 'vit_base_patch16_224',
        'Swin-Base': 'swin_base_patch4_window7_224',
        'DeiT-Base': 'deit_base_patch16_224'
    }

    results = {}
    os.makedirs('results', exist_ok=True)

    for model_name, model_arch in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f">> TRAINING {model_name}")
        print(f"{'='*80}")

        model = create_model(model_arch, CONFIG['num_classes'])
        params = count_parameters(model)

        print(f">> PARAMETERS: {params['Total Parameters']:,} (total) | {params['Model Size (MB)']:.2f} MB")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

        # Train
        history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, CONFIG['device'], CONFIG['epochs'])

        # Evaluate
        eval_results = evaluate_model(model, val_loader, criterion, CONFIG['device'])
        inference_stats = measure_inference_time(model, val_loader, CONFIG['device'])
        report, cm = compute_metrics(eval_results['predictions'], eval_results['labels'], classes)

        # Save results
        results[model_name] = {
            'parameters': {k: f"{v:,}" if isinstance(v, int) else f"{v:.2f}" for k, v in params.items()},
            'performance': {
                'accuracy': f"{eval_results['val_acc']:.2f}%",
                'precision': f"{report['weighted avg']['precision']:.4f}",
                'recall': f"{report['weighted avg']['recall']:.4f}",
                'f1_score': f"{report['weighted avg']['f1-score']:.4f}"
            },
            'inference': {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in inference_stats.items()},
            'history': history
        }

        # Save metrics
        with open(f'results/metrics_{model_name}.json', 'w') as f:
            json.dump(results[model_name], f, indent=2)

        # Plots
        plot_confusion_matrix(cm, classes, model_name, f'results/cm_{model_name}.png')

        print(f">> {model_name} COMPLETED: {eval_results['val_acc']:.2f}% accuracy")

    # Final plots
    plot_learning_curves({name: r['history'] for name, r in results.items()}, 'results/learning_curves.png')

    # Summary table
    summary = []
    for name, r in results.items():
        summary.append({
            'Model': name,
            'Accuracy': r['performance']['accuracy'],
            'Params': r['parameters']['Total Parameters'],
            'Inference_ms': r['inference']['Avg Time per Image (ms)'],
            'F1_Score': r['performance']['f1_score']
        })

    # Save summary
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    import pandas as pd
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n>> ALL EXPERIMENTS COMPLETED!")
    print(">> Results saved in results/ folder")

if __name__ == "__main__":
    main()
