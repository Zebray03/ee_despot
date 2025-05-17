import os
import json

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config_path(filename="encoder_config.json"):
    """获取配置文件路径"""
    return os.path.join(PROJECT_ROOT,
                        "config",
                        filename)


def load_config():
    """加载模型配置"""
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        return json.load(f)


def get_julia_code_path(subdir, script_name):
    """获取Julia代码路径"""
    return os.path.join(PROJECT_ROOT,
                        "julia",
                        subdir,
                        script_name)


def get_model_path(subdir):
    """获取模型保存路径"""
    return os.path.join(PROJECT_ROOT, "model", subdir)
