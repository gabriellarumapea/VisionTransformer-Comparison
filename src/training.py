import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    model.to(device)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'Loss': f'{train_loss/total:.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        train_acc = 100. * correct / total
        train_loss_avg = train_loss / len(train_loader)

        # Validation
        val_results = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_results['val_loss'])

        # Save history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['val_loss'])
        history['val_acc'].append(val_results['val_acc'])

        # Checkpoint
        if val_results['val_acc'] > best_acc:
            best_acc = val_results['val_acc']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, 'checkpoints/best_model.pth')

        print(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Val: {val_results['val_acc']:.2f}%")

    return history

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'val_loss': total_loss / len(data_loader),
        'val_acc': 100. * correct / total,
        'predictions': all_preds,
        'labels': all_labels
    }
