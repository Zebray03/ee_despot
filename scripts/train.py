import sys

sys.path.append("..")  # 添加项目根目录到路径

from python.julia_integration import init_julia
from python.trainer import train_model
from python.feature_policy import extract_features, score_actions, train_step, save_models


def main():
    # 初始化Julia环境
    Main = init_julia()

    # 调用Julia训练函数
    Main.eval("ARDESPOT_Optimized.run_training(epochs=100)")

    print("Done")

    # 训练模型
    train_model(epochs=200)


if __name__ == "__main__":
    main()