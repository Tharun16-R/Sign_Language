import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

from model.dataset import SequenceDataset
from model.model import LSTMClassifier


def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    losses = []
    all_y = []
    all_p = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=-1)
        all_y.extend(y.cpu().numpy().tolist())
        all_p.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(all_y, all_p) if all_y else 0.0
    return np.mean(losses) if losses else 0.0, acc


def eval_model(model, loader, criterion, device):
    model.eval()
    losses = []
    all_y = []
    all_p = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)
            all_y.extend(y.cpu().numpy().tolist())
            all_p.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(all_y, all_p) if all_y else 0.0
    return np.mean(losses) if losses else 0.0, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--labels', default='HELLO,THANKS,ILOVEYOU')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--num_features', type=int, default=63)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weights_out', default='model/weights/best_model.pth')
    args = parser.parse_args()

    labels = args.labels.split(',')
    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)

    dataset = SequenceDataset(args.data_dir, labels, args.seq_len, args.num_features)
    n = len(dataset)
    if n == 0:
        raise RuntimeError('No training samples found. Run capture_data.py first.')
    val_size = max(1, int(0.2 * n))
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(num_features=args.num_features, num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device)
        va_loss, va_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.weights_out)
            print(f"Saved best model to {args.weights_out}")


if __name__ == '__main__':
    main()
