from sklearn.metrics import f1_score
import torch
from .plots import plot_metrics
from tqdm import tqdm
import torch.nn as nn
import copy
import time


THRESHOLD = 0.8


def train_model(model, train_loader, val_loader, real_loader, criterion, optimiser, device, epochs=10, verbose=False, return_history=False):
    history = {'train_loss': [],
               'val_loss': [],
               'real_loss': [],
               'train_acc': [],
               'val_acc': [],
               'real_acc': [],
               'train_f1': [],
               'val_f1': [],
               'real_f1': [],
               'real_acc_0s': [],
               'real_acc_1s': []}

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimiser.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > THRESHOLD).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.int().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_f1 = f1_score(all_labels, all_preds)

        # Validation (on synthetic data)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > THRESHOLD).int()
                correct += (preds == labels.int()).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.int().cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds)
        
        # Validation (on real data)
        model.eval()
        real_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []
        correct_0s, correct_1s = 0, 0
        total_0s, total_1s = 0, 0
        
        with torch.no_grad():
            for inputs, labels in real_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                real_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > THRESHOLD).int()
                correct += (preds == labels.int()).sum().item()
                total += labels.size(0)
                
                correct_0s += ((preds == 0) & (labels.int() == 0)).sum().item()
                correct_1s += ((preds == 1) & (labels.int() == 1)).sum().item()
                total_0s += (labels.int() == 0).sum().item()
                total_1s += (labels.int() == 1).sum().item()

                all_labels.extend(labels.int().cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        real_loss = real_loss / len(real_loader.dataset)
        real_acc = correct / total
        real_f1 = f1_score(all_labels, all_preds)
        
        real_acc_0s = correct_0s / total_0s if total_0s > 0 else 0
        real_acc_1s = correct_1s / total_1s if total_1s > 0 else 0

        print(f"real acc: 0s={real_acc_0s}, 1s={real_acc_1s}") if verbose else None
        
        # Store the metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['real_loss'].append(real_loss)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['real_acc'].append(real_acc)
        
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['real_f1'].append(real_f1)
        
        history['real_acc_0s'].append(real_acc_0s)
        history['real_acc_1s'].append(real_acc_1s)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] \n  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}"\
                f" | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"\
                    f" | Real Loss: {real_loss:.4f} Acc: {real_acc:.4f} F1: {real_f1:.4f}")
            
    
    if return_history:
        return history
    else:
        plot_metrics(history)

    
    
    
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > THRESHOLD).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    
    


def finetune_model(model, real_loader, val_loader, device, pos_weight, 
                   num_epochs=10, lr=1e-4, freeze_features=False, save_path='finetuned_model.pth'):
    """
    Fine-tune a pretrained model on new (real) data.

    Parameters
    ----------
        model: Pretrained model
        real_loader: DataLoader for real dataset (for training)
        val_loader: Optional DataLoader for validation set (can be None)
        device: 'cuda', 'mps', or 'cpu'
        pos_weight: Tensor for BCEWithLogitsLoss (handles class imbalance)
        num_epochs: Fine-tuning epochs
        lr: Learning rate (default lower for fine-tuning)
        freeze_features: If True, freeze feature extractor (conv layers)
        save_path: Path to save best fine-tuned model

    Returns
    -------
        model: Fine-tuned model
    """

    model = model.to(device)

    if freeze_features:
        print("Freezing feature extractor layers...")
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning ALL layers...")

    # Only train parameters that require grad
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in real_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(real_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f}', end='')

        # Optional validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            print(f' | Val Loss: {val_loss:.4f}', end='')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print('Saved best model!', end='')

        print(f' | Time: {time.time() - start_time:.1f}s')


    model.load_state_dict(best_model_wts)
    print(f'\nBest model saved to {save_path}')
    return model