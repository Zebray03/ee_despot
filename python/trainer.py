import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import load_config, get_model_path
from .data import prepare_dataloader
from .model import create_models, save_models


def train_step(features, labels, encoder, scorer, optimizer, criterion, batch_size):
    dataloader = prepare_dataloader(features, labels, batch_size=batch_size)
    encoder.train()
    scorer.train()
    total_loss = 0.0

    for batch_idx, (batch_features, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()

        features = encoder(batch_features)
        scores = scorer(features)

        with torch.no_grad():
            probs = torch.softmax(scores, dim=1)
            pred_dist = probs.mean(dim=0).numpy()

        unique, counts = torch.unique(batch_labels, return_counts=True)
        true_dist = np.zeros(load_config()["action_num"])
        for u, c in zip(unique, counts):
            true_dist[u] = c.item() / len(batch_labels)

        if batch_idx < 3:
            print(f"\nBatch {batch_idx + 1} 预测分布:")
            print(np.round(pred_dist, 3))
            print("真实分布:")
            print(np.round(true_dist, 3))

        loss = criterion(scores, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch平均损失: {avg_loss:.4f}")
    return avg_loss


def train_model(epochs, batch_size, problem, patience=10):
    config = load_config()

    # 创建模型
    encoder, scorer = create_models(
        config["belief_dim"],
        config["hidden_dim"],
        config["feature_dim"],
        config["action_num"]
    )

    from julia import Main
    features = np.array(Main.features_train)
    labels = np.array(Main.labels_train)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(scorer.parameters()), lr=1e-4)

    loss_history = []
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        loss = train_step(features, labels, encoder, scorer, optimizer, criterion, batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        loss_history.append(loss)
        if len(loss_history) > patience:
            loss_history.pop(0)

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch

        # 早停
        if len(loss_history) >= patience:
            min_loss = min(loss_history)
            max_loss = max(loss_history)
            if (max_loss - min_loss) / min_loss <= 1e-2:
                print(f"Early stopping at epoch {epoch + 1}. Loss variation within 10% in {patience} epochs.")
                print(f"Best loss: {best_loss:.4f}")
                print(f"Best epoch: {best_epoch + 1}")
                break

    model_path = get_model_path(subdir=problem)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # 确保目录存在
    save_models(encoder, scorer, model_path)

    return encoder, scorer
