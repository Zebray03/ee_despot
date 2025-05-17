import logging

import torch
from julia import Main

from python.config import get_model_path
from python.julia_integration import init_julia
from python.model import load_models

problem = 'RockSample'


def test():
    Main = init_julia(mode='test', problem=problem)

    model_path = get_model_path(subdir=problem)
    encoder, scorer = load_models(model_path)

    device = next(encoder.parameters()).device
    Main.extract_features = lambda x: encoder(
        torch.tensor(x, dtype=torch.float32)
        .unsqueeze(0).to(device)).squeeze(0).cpu().detach().numpy()

    Main.score_actions = lambda x: scorer(
        torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy()

    Main.eval("""
            ARDESPOT_Test.run_tests()
            """)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    test()
