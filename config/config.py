import argparse
import torch

def arg_parse():
    """
    adds and parses arguments / hyperparameters
    """
    # Dataset
    data = "cora-coauthorship"
    split = 1

    # Random seed
    seed = 42

    # Model related parameters
    layers = 2
    epochs = 200
    latent_dim = 32
    alpha = 1.0
    beta = 0.8

    # parameters of optimization
    lr = 1e-2
    weight_decay = 5e-4
    dropout = 0.5
    batch_size = 96

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=seed, help="Random seed.")
    parser.add_argument('--no-cuda', action="store_true", default=False, help="Disables CUDA training.")

    parser.add_argument("--save-model", action="store_true", default=False)

    # Data settings
    parser.add_argument("--data", type=str, default=data)
    parser.add_argument("--split", type=str, default=split)
    parser.add_argument("--labels", type=str, default="")

    # Training settings
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--dropout", type=float, default=dropout)
    parser.add_argument("--weight-decay", type=float, default=weight_decay)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--num-negatives", type=int, default=5)

    # Model related
    parser.add_argument("--latent-dim", type=int, default=latent_dim)
    parser.add_argument("--num-layers", type=int, default=layers)
    parser.add_argument("--alpha", type=float, default=alpha)
    parser.add_argument("--beta", type=float, default=beta)

    # GCN option
    parser.add_argument("--weight", action="store_true", default=False)
    parser.add_argument("--non-linear", action="store_true", default=False)

    args = parser.parse_args()

    cfg = dict()
    for arg in vars(args):
       cfg[arg] = getattr(args, arg)

    cfg["cuda"] = not cfg["no_cuda"] and torch.cuda.is_available()
    cfg["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return cfg