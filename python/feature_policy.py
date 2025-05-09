import json

import torch
import torch.nn as nn
import torch.optim as optim
from julia import Main


class BeliefEncoder(nn.Module):
    """信念状态特征提取器"""

    def __init__(self, belief_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(belief_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class ActionScorer(nn.Module):
    """动作价值预测器"""

    def __init__(self, feature_dim, actions_num):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, actions_num)
        )

    def forward(self, x):
        return self.scorer(x)


# 读取配置文件
with open('../config/encoder_config.json', 'r') as f:
    config = json.load(f)

belief_dim = config['belief_dim']
feature_dim = config['feature_dim']
action_num = config['action_num']

# 初始化模型
encoder = BeliefEncoder(belief_dim, feature_dim)
scorer = ActionScorer(feature_dim, action_num)
optimizer = optim.Adam(list(encoder.parameters()) + list(scorer.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def extract_features(belief_np):
    """特征提取接口"""
    belief_tensor = torch.from_numpy(belief_np).float()
    with torch.no_grad():
        features = encoder(belief_tensor).numpy()
    return features


def score_actions(features_np):
    """动作评分接口"""
    features_tensor = torch.from_numpy(features_np).float()
    with torch.no_grad():
        scores = scorer(features_tensor).numpy()
    return scores


def train_step(features, labels):
    """执行单次训练步骤"""
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)

    optimizer.zero_grad()
    encoded = encoder(features_tensor)
    scores = scorer(encoded)
    loss = criterion(scores, labels_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

def save_models(prefix):
    """保存模型权重"""
    torch.save(encoder.state_dict(), f"{prefix}_encoder.pth")
    torch.save(scorer.state_dict(), f"{prefix}_scorer.pth")

def load_models(prefix):
    """加载模型权重"""
    encoder.load_state_dict(torch.load(f"{prefix}_encoder.pth"))
    scorer.load_state_dict(torch.load(f"{prefix}_scorer.pth"))


# 暴露接口给Julia
Main.extract_features = extract_features
Main.score_actions = score_actions
Main.train_step = train_step
Main.save_models = save_models