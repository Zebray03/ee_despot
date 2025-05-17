import torch
from julia import Main

from python.config import load_config
from python.model import create_models, load_models


class NeuralARDESPOTSolver:
    """神经增强的ARDESPOT求解器类"""

    def __init__(self, pomdp, model_path=None):
        self.pomdp = pomdp
        config = load_config()

        # 初始化模型
        if model_path:
            self.encoder, self.scorer = load_models(model_path)
        else:
            self.encoder, self.scorer = create_models(
                config["belief_dim"],
                config["hidden_dim"],
                config["feature_dim"],
                config["action_num"]
            )

        # 暴露接口给Julia
        Main.extract_features = lambda x: self.encoder(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        Main.score_actions = lambda x: self.scorer(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    def solve(self, pomdp):
        """返回使用神经增强策略的规划器"""
        from julia import Main
        return Main.eval("""
        NeuralPlanner = NeuralARDESPOTPlanner($pomdp, $(self.encoder), $(self.scorer))
        """)