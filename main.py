from src.models import *
from src.pre_process_data import prepare_data, process_data, load_torch_dataset
from src.stats import datasetStats
from src.logger import setup_logging
from src.utils import buildDirectoryStructure, train
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import numpy as np

setup_logging()

buildDirectoryStructure()

# Load dataset
try:
    torch_dataset = load_torch_dataset()
except ValueError as e:
    print(e)
    print("No extracted features found. Running process_data() to generate features...")
    process_data()
    torch_dataset = load_torch_dataset()

print(f"Dataset size: {len(torch_dataset)}")
device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = torch_dataset[0][0].numel()
num_classes = torch_dataset.get_num_classes()

# Cross-validation setup
n_splits = 10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store metrics for each fold
fold_results = {
    'fold': [],
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': [],
    'best_epoch': [],
}

print(f"\n{'='*80}")
print(f"Starting {n_splits}-Fold Cross-Validation")
print(f"{'='*80}\n")

total_cv_start = time.time()

for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(torch_dataset))), 1):
    print(f"\n{'─'*80}")
    print(f"FOLD {fold}/{n_splits}")
    print(f"{'─'*80}")
    print(f"Train samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
    
    # Create train and validation dataloaders for this fold
    train_subset = Subset(torch_dataset, train_idx)
    val_subset = Subset(torch_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    
    # Initialize fresh model for this fold
    model = AudioResNet(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    best_epoch = 0
    
    fold_start = time.time()
    
    for epoch in range(500):  # Max 500 epochs per fold
        # Training phase
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_loss_history.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
                # Collect predictions and targets
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_subset)
        val_loss_history.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            val_accuracy = accuracy_score(val_targets, val_preds)
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f} | "
                  f"patience={patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for this fold
    model.load_state_dict(best_model_state)
    
    # Final validation evaluation
    model.eval()
    val_preds = []
    val_targets = []
    final_val_loss = 0.0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            final_val_loss += loss.item() * features.size(0)
            
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    
    final_val_loss /= len(val_subset)
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    # Calculate metrics
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_precision = precision_score(val_targets, val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_targets, val_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
    
    fold_time = time.time() - fold_start
    
    # Store results
    fold_results['fold'].append(fold)
    fold_results['train_loss'].append(min(train_loss_history))
    fold_results['val_loss'].append(final_val_loss)
    fold_results['val_accuracy'].append(val_accuracy)
    fold_results['val_precision'].append(val_precision)
    fold_results['val_recall'].append(val_recall)
    fold_results['val_f1'].append(val_f1)
    fold_results['best_epoch'].append(best_epoch)
    
    print(f"\nFold {fold} Results (Time: {fold_time:.2f}s):")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall: {val_recall:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")

total_cv_time = time.time() - total_cv_start

# Print cross-validation summary
print(f"\n{'='*80}")
print(f"CROSS-VALIDATION SUMMARY ({n_splits} Folds)")
print(f"{'='*80}\n")

print(f"Average Training Loss: {np.mean(fold_results['train_loss']):.4f} ± {np.std(fold_results['train_loss']):.4f}")
print(f"Average Validation Loss: {np.mean(fold_results['val_loss']):.4f} ± {np.std(fold_results['val_loss']):.4f}")
print(f"Average Validation Accuracy: {np.mean(fold_results['val_accuracy']):.4f} ± {np.std(fold_results['val_accuracy']):.4f}")
print(f"Average Validation Precision: {np.mean(fold_results['val_precision']):.4f} ± {np.std(fold_results['val_precision']):.4f}")
print(f"Average Validation Recall: {np.mean(fold_results['val_recall']):.4f} ± {np.std(fold_results['val_recall']):.4f}")
print(f"Average Validation F1-Score: {np.mean(fold_results['val_f1']):.4f} ± {np.std(fold_results['val_f1']):.4f}")
print(f"\nTotal CV Time: {total_cv_time:.2f}s\n")

# Plot cross-validation results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'{n_splits}-Fold Cross-Validation Results', fontsize=16)

# Plot 1: Loss across folds
axes[0, 0].plot(fold_results['fold'], fold_results['train_loss'], marker='o', label='Train Loss')
axes[0, 0].plot(fold_results['fold'], fold_results['val_loss'], marker='s', label='Val Loss')
axes[0, 0].set_xlabel('Fold')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training vs Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Accuracy across folds
axes[0, 1].bar(fold_results['fold'], fold_results['val_accuracy'], color='skyblue', edgecolor='navy')
axes[0, 1].axhline(np.mean(fold_results['val_accuracy']), color='red', linestyle='--', label='Mean')
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Validation Accuracy by Fold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Precision across folds
axes[0, 2].bar(fold_results['fold'], fold_results['val_precision'], color='lightgreen', edgecolor='darkgreen')
axes[0, 2].axhline(np.mean(fold_results['val_precision']), color='red', linestyle='--', label='Mean')
axes[0, 2].set_xlabel('Fold')
axes[0, 2].set_ylabel('Precision')
axes[0, 2].set_title('Validation Precision by Fold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Recall across folds
axes[1, 0].bar(fold_results['fold'], fold_results['val_recall'], color='lightcoral', edgecolor='darkred')
axes[1, 0].axhline(np.mean(fold_results['val_recall']), color='red', linestyle='--', label='Mean')
axes[1, 0].set_xlabel('Fold')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_title('Validation Recall by Fold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: F1-Score across folds
axes[1, 1].bar(fold_results['fold'], fold_results['val_f1'], color='lightyellow', edgecolor='orange')
axes[1, 1].axhline(np.mean(fold_results['val_f1']), color='red', linestyle='--', label='Mean')
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_title('Validation F1-Score by Fold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Best epoch per fold
axes[1, 2].bar(fold_results['fold'], fold_results['best_epoch'], color='plum', edgecolor='purple')
axes[1, 2].set_xlabel('Fold')
axes[1, 2].set_ylabel('Epoch')
axes[1, 2].set_title('Best Epoch per Fold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
print("Cross-validation results plotted and saved as 'cross_validation_results.png'")
plt.show()
