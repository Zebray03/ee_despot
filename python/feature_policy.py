from .config import get_model_path
from .model import load_models

# 加载预训练模型
encoder, scorer = load_models(get_model_path("RockSample"))
