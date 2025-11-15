import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(histories, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for model_name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)

        ax1.plot(epochs, history['train_loss'], label=f'{model_name} Train', linestyle='--')
        ax1.plot(epochs, history['val_loss'], label=f'{model_name} Val')

        ax2.plot(epochs, history['train_acc'], label=f'{model_name} Train', linestyle='--')
        ax2.plot(epochs, history['val_acc'], label=f'{model_name} Val')

    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, classes, model_name, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
