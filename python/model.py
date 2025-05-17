import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from python.config import load_config


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        out = F.relu(out)
        return out


class BeliefEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_res_blocks=3):
        super(BeliefEncoder, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = F.relu(x)
        for block in self.res_blocks:
            x = block(x)
        attn_weights = self.attention(x)
        x = x * attn_weights
        x = self.fc_out(x)
        return x


class ActionScorer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_num):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_num),
        )

    def forward(self, x):
        return self.fc(x)


def create_models(input_dim, hidden_dim, feature_dim, action_num):
    encoder = BeliefEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=feature_dim)
    scorer = ActionScorer(feature_dim=feature_dim, hidden_dim=hidden_dim, action_num=action_num)
    return encoder, scorer


def load_models(model_dir):
    config = load_config()
    encoder, scorer = create_models(
        config["belief_dim"],
        config["hidden_dim"],
        config["feature_dim"],
        config["action_num"]
    )

    encoder_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_encoder.pth")
    scorer_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_scorer.pth")

    encoder.load_state_dict(
        {k: v.float() for k, v in torch.load(encoder_path, map_location='cpu').items()},
        strict=True
    )
    scorer.load_state_dict(
        {k: v.float() for k, v in torch.load(scorer_path, map_location='cpu').items()},
        strict=True
    )
    encoder.eval()
    scorer.eval()
    return encoder, scorer


def save_models(encoder, scorer, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    encoder_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_encoder.pth")
    scorer_path = os.path.join(model_dir, f"{os.path.basename(model_dir)}_scorer.pth")

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(scorer.state_dict(), scorer_path)


def extract_features(encoder, belief_np):
    belief_tensor = torch.from_numpy(belief_np).float()
    with torch.no_grad():
        features = encoder(belief_tensor).numpy()
    return features


def score_actions(scorer, features_np):
    features_tensor = torch.from_numpy(features_np).float()
    with torch.no_grad():
        scores = scorer(features_tensor).numpy()
    return scores
