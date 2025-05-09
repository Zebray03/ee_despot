import torch
import torch.nn as nn

from python.config import load_config


class BeliefEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BeliefEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActionScorer(nn.Module):
    def __init__(self, feature_dim, action_num):
        super(ActionScorer, self).__init__()
        self.fc = nn.Linear(feature_dim, action_num)

    def forward(self, x):
        return self.fc(x)


def create_models(input_dim, hidden_dim, feature_dim, action_num):
    """创建编码器和动作评分器模型"""
    encoder = BeliefEncoder(input_dim, hidden_dim, feature_dim)
    scorer = ActionScorer(feature_dim, action_num)
    return encoder, scorer


def load_models(model_path):
    """加载预训练模型"""
    config = load_config()
    encoder, scorer = create_models(
        config["belief_dim"],
        config["hidden_dim"],
        config["feature_dim"],
        config["action_num"]
    )

    encoder.load_state_dict(torch.load(f"{model_path}_encoder.pth"))
    scorer.load_state_dict(torch.load(f"{model_path}_scorer.pth"))

    encoder.eval()
    scorer.eval()

    return encoder, scorer


def save_models(encoder, scorer, model_path):
    """保存模型"""
    torch.save(encoder.state_dict(), f"{model_path}_encoder.pth")
    torch.save(scorer.state_dict(), f"{model_path}_scorer.pth")