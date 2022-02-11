from src.utils import accuracy, ensure_dir
from src.gcn import StarGCN

from config.config import arg_parse
from data.data_helper import load_data

import os, inspect, pickle
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def generate_negative(x):
    """
    Generate negative hyperedges based on the postive hyperedges.
    @param x: positive hyperedges
    @type x: dict values
    @return: hyperedge ids, hyperedge candidates (node sets), labels (0,1)
    @rtype: list of tensors
    """
    neg_list = []
    pos_h, neg_h = [], []
    for j, sample in enumerate(x):
        pos_h.append(j)
        change_list = np.random.randint(0, len(sample), neg_num) # 바꿀 node index 미리 sampling
        for i in range(neg_num): # neg_num 만큼의 negative sample 생성
            change = change_list[i] # node index to change
            temp = np.copy(sample)
            a = set()
            a.add(tuple(temp))
            trial = 0
            while not a.isdisjoint(train_dict):
                temp = np.copy(sample)
                trial += 1
                if trial >= 1000:
                    temp = ""
                    break
                temp[change] = np.random.randint(0, n_node, 1)
                temp.sort()
                a = set([tuple(temp)])
            if len(temp) > 0:
                neg_list.append(temp)
                neg_h.append(j)

    X = [torch.as_tensor(v, dtype=torch.long) for v in x]
    neg = [torch.as_tensor(v, dtype=torch.long) for v in neg_list]
    X.extend(neg)
    y = torch.cat([torch.ones((len(x), 1),  device=device), torch.zeros((len(neg), 1), device=device)], dim=0)

    pos_h = torch.LongTensor(pos_h).to(device)
    neg_h = torch.LongTensor(neg_h).to(device)
    h = torch.cat([pos_h, neg_h])

    return h, X, y

def train(model, optimizer):
    x = dataset["hypergraph"].values()
    train_idx = dataset["train_idx"]
    labels = torch.LongTensor(dataset["labels"]).to(device)

    print("Training datasets: ", len(train_idx))
    print("Params:", "alpha", cfg["alpha"], "beta", cfg["beta"])

    model.train()

    for epoch in tqdm(range(cfg["epochs"]), desc="[Training]"):
        optimizer.zero_grad()

        node, hyperedge = model.computer()

        Z = model.forward()

        ### FINAL LOSS = (1-alpha-beta) * TASK + alpha * STAR + beta * CLIQUE
        classification_loss = F.nll_loss(Z[train_idx], labels[train_idx])
        loss = (1 - cfg["beta"]) * classification_loss

        star_loss = 0
        clique_loss = 0
        if cfg["beta"] > 0:  # TOPOLOGICAL LOSS
            h, X, y = generate_negative(x)
            if cfg["alpha"] > 0:
                star_scores = model.star_(node, hyperedge, h, X)
                star_loss = F.binary_cross_entropy_with_logits(star_scores, y)
            clique_scores = model.clique_(node, X)
            clique_loss = F.binary_cross_entropy_with_logits(clique_scores, y)

        loss += cfg["beta"] * (cfg["alpha"] * star_loss + (1 - cfg["alpha"]) * clique_loss)

        loss.backward()
        optimizer.step()

def test(model):
    labels = torch.LongTensor(dataset["labels"]).to(device)

    model.eval()
    Z = model.forward()

    test_idx = dataset["test_idx"]
    test_accuracy = accuracy(Z[test_idx], labels[test_idx])

    print("[Test set results]",
          "accuracy= {:.8f}".format(test_accuracy.item()),
          "\n")

    print("Saving results...")
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # Save classification result to 'result/[data]/' directory
    result_file = os.path.join(current, "result", cfg["data"],
                               f"a{cfg['alpha']}_b{cfg['beta']}_s{cfg['split']}_d{cfg['latent_dim']}.pickle")
    ensure_dir(result_file)

    Z = Z.cpu().detach().numpy()
    with open(result_file, "wb") as f:
        pickle.dump(Z, f)

    # Save node embeddings to 'emb/[data]/' directory
    emb_file = os.path.join(current, "emb", cfg["data"],
                            f"a{cfg['alpha']}_b{cfg['beta']}_s{cfg['split']}_d{cfg['latent_dim']}_l{cfg['labels']}.emb")
    ensure_dir(emb_file)

    embeddings, _ = model.computer()
    embeddings = embeddings.cpu().detach().numpy()

    with open(emb_file, "w") as f:
        for i, emb in enumerate(embeddings):
            l = [str(e) for e in emb]
            l.insert(0, str(i))
            f.write("\t".join(l) + "\n")

    print("Done!")


#########################MAIN############################
cfg = arg_parse()

# Set seed
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
if cfg["cuda"]: torch.cuda.manual_seed(cfg["seed"])

device = cfg["device"]
neg_num = cfg["num_negatives"]
batch_size = cfg["batch_size"]

# Prepare dataset
dataset = load_data(cfg)

hypergraph = dataset["hypergraph"]
n_node = dataset["n_node"]
m_hyperedge = dataset["m_hyperedge"]

train_dict = set()
for datum in hypergraph.values():
    datum = list(datum)
    datum.sort()
    train_dict.add(tuple(datum))

# Model
model = StarGCN(cfg, dataset).to(device)

# Optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, list(model.parameters())),
    lr=cfg["lr"],
    weight_decay=cfg["weight_decay"]
)

# Train and Test
train(model, optimizer)
test(model)
