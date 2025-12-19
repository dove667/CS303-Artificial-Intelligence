import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from preprocess import DataPreprocessor


class TabularDataset(Dataset):
    def __init__(self, df, y, num_cols, cat_cols):
        self.num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.num[idx], self.cat[idx]
        return self.num[idx], self.cat[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, cat_cardinalities, num_features, hidden_sizes=(8, 8), dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, min(50, int(round(cardinality ** 0.5)))) for cardinality in cat_cardinalities]
        )
        embed_dim_total = sum([emb.embedding_dim for emb in self.embeddings])
        layers = []
        input_dim = embed_dim_total + num_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, cat_x):
        embs = [emb(cat_x[:, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [num_x], dim=1)
        return self.mlp(x).squeeze(1)


def scale_numeric(train_df, val_df, test_df, num_cols):
    mean = train_df[num_cols].mean()
    std = train_df[num_cols].std().replace(0, 1)
    train_df[num_cols] = (train_df[num_cols] - mean) / std
    val_df[num_cols] = (val_df[num_cols] - mean) / std
    test_df[num_cols] = (test_df[num_cols] - mean) / std
    return train_df, val_df, test_df


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for num_x, cat_x, y in dataloader:
        num_x, cat_x, y = num_x.to(device), cat_x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(num_x, cat_x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, device, criterion=None):
    model.eval()
    preds, targets = [], []
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for num_x, cat_x, y in dataloader:
            num_x, cat_x, y = num_x.to(device), cat_x.to(device), y.to(device)
            logits = model(num_x, cat_x)
            probs = torch.sigmoid(logits)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += loss.item() * len(y)
                total_count += len(y)
            preds.extend((probs.cpu().numpy() >= 0.5).astype(int))
            targets.extend(y.cpu().numpy())
    val_loss = total_loss / total_count if total_count > 0 else None
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    return val_loss, accuracy, precision, recall, f1, preds, targets


def predict_test(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for num_x, cat_x in dataloader:
            num_x, cat_x = num_x.to(device), cat_x.to(device)
            logits = model(num_x, cat_x)
            probs = torch.sigmoid(logits)
            preds.extend((probs.cpu().numpy() >= 0.5).astype(int))
    return preds


def main():
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    preprocessor = DataPreprocessor(data_dir='.')
    X, y, X_test = preprocessor.preprocess(
        feature_engineering=False,
        remove_low_predictive=False,
        one_hot_encoding=False, # 使用label encoder
        scale=False,
        fit=True
    )

    cat_cols = preprocessor.categorical_features
    num_cols = preprocessor.numerical_features

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, X_test_scaled = scale_numeric(X_train.copy(), X_val.copy(), X_test.copy(), num_cols)

    cat_cardinalities = [len(preprocessor.label_encoders[col].classes_) for col in cat_cols]

    train_ds = TabularDataset(X_train, y_train, num_cols, cat_cols)
    val_ds = TabularDataset(X_val, y_val, num_cols, cat_cols)
    test_ds = TabularDataset(X_test_scaled, None, num_cols, cat_cols)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = MLP(cat_cardinalities, num_features=len(num_cols), hidden_sizes=(64, 32), dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    epochs = 20
    best_f1 = 0.0
    best_state = None
    val_loss_history, val_acc_history = [], []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, acc, prec, rec, f1, _, _ = eval_epoch(model, val_loader, device, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(acc)
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | acc {acc:.4f} | prec {prec:.4f} | rec {rec:.4f} | f1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, acc, prec, rec, f1, preds, targets = eval_epoch(model, val_loader, device, criterion)
    print("Validation metrics with best checkpoint:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))

    test_preds = predict_test(model, test_loader, device)
    with open('testlabel_mlp.txt', 'w') as f:
        for label in test_preds:
            f.write(f"{label}\n")
    print("Predictions saved as 'testlabel_mlp.txt'")

    # Plot validation loss and F1 curves.
    epochs_axis = list(range(1, epochs + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_axis, val_loss_history, label='Val Loss')
    plt.plot(epochs_axis, val_acc_history, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Loss and Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('val_curves.png')
    print("Saved validation curves to 'val_curves.png'")


if __name__ == '__main__':
    main()
