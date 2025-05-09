import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .model import create_models, save_models
from .data import prepare_dataloader
from .config import load_config, get_model_path


def train_step(features, labels, encoder, scorer, optimizer, criterion, batch_size):
    """执行一个训练步骤"""
    dataloader = prepare_dataloader(features, labels, batch_size=batch_size)

    encoder.train()
    scorer.train()

    total_loss = 0.0
    for batch_features, batch_labels in dataloader:
        optimizer.zero_grad()

        # 前向传播
        features = encoder(batch_features)
        scores = scorer(features)

        # 计算损失
        loss = criterion(scores, batch_labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(epochs=100, batch_size=32):
    """训练模型主函数"""
    config = load_config()

    # 创建模型
    encoder, scorer = create_models(
        config["belief_dim"],
        config["hidden_dim"],
        config["feature_dim"],
        config["action_num"]
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(scorer.parameters()), lr=0.001)

    # 训练循环
    for epoch in range(epochs):
        # 从Julia获取数据（这里假设数据已经通过Main.features_train和Main.labels_train传递）
        from julia import Main
        features = Main.features_train
        labels = Main.labels_train

        # 转换为NumPy数组
        if isinstance(features, np.ndarray):
            features_np = features
        else:
            features_np = np.array(features)

        if isinstance(labels, np.ndarray):
            labels_np = labels
        else:
            labels_np = np.array(labels)

        # 训练一步
        loss = train_step(features_np, labels_np, encoder, scorer, optimizer, criterion, batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # 保存模型
    model_path = get_model_path()
    save_models(encoder, scorer, model_path)

    # 导出到Julia
    from julia import Main
    Main.extract_features = lambda x: encoder(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    Main.score_actions = lambda x: scorer(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    return encoder, scorer