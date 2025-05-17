import sys

sys.path.append("..")
from python.julia_integration import init_julia
from python.trainer import train_model

problem = 'RockSample'


def train():
    Main = init_julia(mode='train', problem=problem)
    Main.eval("ARDESPOT_Train.run_training()")
    train_model(epochs=10000, batch_size=32, problem=problem)
    print("Training completed. Models saved.")


if __name__ == "__main__":
    train()
