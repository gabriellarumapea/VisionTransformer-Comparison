import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import json
import os
import sys
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import ImageClassificationDataset
from models import create_model, count_parameters
from training import train_model, evaluate_model
from evaluation import measure_inference_time, compute_metrics
from visualization import plot_learning_curves, plot_confusion_matrix, plot_per_class_metrics

def get_hardware_info(device):
    """Get detailed hardware information"""
    info = {}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        info['GPU_Name'] = torch.cuda.get_device_name(0)
        info['GPU_Memory_GB'] = props.total_memory / 1e9
        info['GPU_Multiprocessors'] = props.multi_processor_count
        info['CUDA_Version'] = torch.version.cuda
    else:
        import platform
        import psutil
        info['CPU_Name'] = platform.processor()
        info['CPU_Cores'] = psutil.cpu_count(logical=False)
        info['RAM_GB'] = psutil.virtual_memory().total / 1e9
        info['Platform'] = platform.platform()
    return info

def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Comparison')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    args = parser.parse_args()
    
    print(">> VISION TRANSFORMER COMPARISON")
    print("="*80)
    
    # Config
    CONFIG = {
        'dataset_path': 'dataset_organized',
        'img_size': 224,
        'batch_size': 32,
        'epochs': args.epochs,
        'learning_rate': 1e-4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_classes': None,
        'class_names': None
    }
    
    print(f">> DEVICE: {CONFIG['device']}")
    
    # Hardware info
    hardware_info = get_hardware_info(CONFIG['device'])
    for k, v in hardware_info.items():
        print(f">> {k}: {v}")
    
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
    print(f">> TRAIN SIZE: {len(train_dataset)} images | VAL SIZE: {len(val_dataset)} images")
    
    # Models
    MODEL_CONFIGS = {
        'ViT-Base': 'vit_base_patch16_224',
        'Swin-Base': 'swin_base_patch4_window7_224',
        'DeiT-Base': 'deit_base_patch16_224'
    }
    
    results = {}
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    for model_name, model_arch in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f">> TRAINING {model_name}")
        print(f"{'='*80}")
        
        # Create model
        model = create_model(model_arch, CONFIG['num_classes'])
        params = count_parameters(model)
        
        print(f">> PARAMETERS: {params['Total Parameters']:,} (total)")
        print(f">> TRAINABLE: {params['Trainable Parameters']:,} (trainable)")
        print(f">> NON-TRAINABLE: {params['Non-trainable Parameters']:,} (non-trainable)")
        print(f">> MODEL SIZE: {params['Model Size (MB)']:.2f} MB")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        
        # Resume jika ada checkpoint
        start_epoch = 0
        if args.resume:
            from resume import resume_training
            start_epoch, best_acc, history = resume_training(args.resume, model, optimizer, scheduler)
        
        # Train
        history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, CONFIG['device'], CONFIG['epochs'])
        
        # Evaluate
        eval_results = evaluate_model(model, val_loader, criterion, CONFIG['device'])
        inference_stats = measure_inference_time(model, val_loader, CONFIG['device'])
        report, cm = compute_metrics(eval_results['predictions'], eval_results['labels'], classes)
        
        # Extract per-class metrics
        precision_per_class = [report[cls]['precision'] for cls in classes]
        recall_per_class = [report[cls]['recall'] for cls in classes]
        f1_per_class = [report[cls]['f1-score'] for cls in classes]
        
        # Save detailed results
        results[model_name] = {
            'parameters': {k: f"{v:,}" if isinstance(v, int) else f"{v:.2f}" for k, v in params.items()},
            'performance': {
                'accuracy': f"{eval_results['val_acc']:.2f}%",
                'precision_macro': f"{report['macro avg']['precision']:.4f}",
                'recall_macro': f"{report['macro avg']['recall']:.4f}",
                'f1_score_macro': f"{report['macro avg']['f1-score']:.4f}",
                'precision_weighted': f"{report['weighted avg']['precision']:.4f}",
                'recall_weighted': f"{report['weighted avg']['recall']:.4f}",
                'f1_score_weighted': f"{report['weighted avg']['f1-score']:.4f}",
                'precision_per_class': {cls: f"{precision_per_class[i]:.4f}" for i, cls in enumerate(classes)},
                'recall_per_class': {cls: f"{recall_per_class[i]:.4f}" for i, cls in enumerate(classes)},
                'f1_score_per_class': {cls: f"{f1_per_class[i]:.4f}" for i, cls in enumerate(classes)},
            },
            'inference': {
                **{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in inference_stats.items()},
                'Hardware': f"{torch.cuda.get_device_name(0) if CONFIG['device'].type == 'cuda' else 'CPU'}"
            },
            'history': history
        }
        
        # Save metrics
        with open(f'results/metrics_{model_name}.json', 'w') as f:
            json.dump(results[model_name], f, indent=2)
        
        # Plots
        plot_confusion_matrix(cm, classes, model_name, f'results/cm_{model_name}.png')
        
        # Per-class metrics plot (bonus)
        per_class_df = pd.DataFrame({
            'Class': classes,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class
        })
        per_class_df.to_csv(f'results/per_class_metrics_{model_name}.csv', index=False)
        plot_per_class_metrics(f'results/per_class_metrics_{model_name}.csv', model_name, f'results/per_class_{model_name}.png')
        
        print(f">> {model_name} COMPLETED: {eval_results['val_acc']:.2f}% accuracy")
        print(f">> INFERENCE: {inference_stats['Avg Time per Image (ms)']:.2f} ms/image")
    
    # Final plots
    plot_learning_curves({name: r['history'] for name, r in results.items()}, 'results/learning_curves.png')
    
    # Summary table
    summary = []
    for name, r in results.items():
        summary.append({
            'Model': name,
            'Accuracy': r['performance']['accuracy'],
            'Total_Params': r['parameters']['Total Parameters'],
            'Trainable_Params': r['parameters']['Trainable Parameters'],
            'Model_Size_MB': r['parameters']['Model Size (MB)'],
            'Inference_ms': r['inference']['Avg Time per Image (ms)'],
            'Throughput': r['inference']['Throughput (images/sec)'],
            'F1_Score': r['performance']['f1_score_weighted'],
            'Precision': r['performance']['precision_weighted'],
            'Recall': r['performance']['recall_weighted']
        })
    
    # Save summary
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # Save summary as CSV & LaTeX
    df_summary.to_csv('results/summary.csv', index=False)
    df_summary.to_latex('results/summary.tex', index=False, float_format="%.2f")
    
    print("\n>> ALL EXPERIMENTS COMPLETED!")
    print(">> Results saved in results/ folder")
    print(">> Files created:")
    print("   - metrics_[model].json (detailed metrics)")
    print("   - per_class_metrics_[model].csv (per-class metrics)")
    print("   - per_class_[model].png (per-class plots)")
    print("   - cm_[model].png (confusion matrices)")
    print("   - learning_curves.png (training curves)")
    print("   - summary.json/csv/tex (summary tables)")

if __name__ == "__main__":
    main()