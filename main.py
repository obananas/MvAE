import random
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import dataset_with_info
from models import MvAEModel
from utils import Logger
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from utils.clusteringPerformance2 import clusteringMetrics

seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def orthogonal_loss(shared, specific):
    _shared = shared.detach()
    _shared = _shared - _shared.mean(dim=0)
    correlation_matrix = _shared.t().matmul(specific)
    norm = torch.norm(correlation_matrix, p=1)
    return norm


if __name__ == '__main__':
    datasetname = "MSRCv1"     # handwritten
    train_batch_size = 10000
    train_epoch = 300
    lr = 0.001
    logger = Logger.get_logger(__file__, datasetname)
    dataset, ins_num, view_num, nc, input_dims, _ = dataset_with_info(datasetname)
    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=100000, shuffle=False)
    ACC_array = np.zeros((6, 6))
    NMI_array = np.zeros((6, 6))
    Purity_array = np.zeros((6, 6))
    ARI_array = np.zeros((6, 6))
    params = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # alpha
    lambda_ma = params[2]
    # beta
    lambda_con = params[2]
    print("====================== start training ======================")
    print("param ", "alpha:", lambda_ma, ", param ", "beta:", lambda_con)
    do_contrast = True
    feature_dim = 64
    device = "cuda:0"
    model = MvAEModel(input_dims, view_num, feature_dim, h_dims=[500, 200]).to(device)
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    clustering_epoch = 20
    neighbors_num = int(ins_num / 4)
    pos_num = 21
    neg_num = int((neighbors_num - pos_num - 1) / 2)
    nbr_idx = []
    neg_idx = []
    losses = []
    for v in range(view_num):
        X_np = np.array(dataset.features[0][v])
        nbrs_v = np.zeros((ins_num, pos_num - 1))
        neg_v = np.zeros((ins_num, neg_num))
        nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)
        dis, idx = nbrs.kneighbors(X_np)
        for i in range(ins_num):
            for j in range(pos_num - 1):
                nbrs_v[i][j] += idx[i][j + 1]
            for j in range(neg_num):
                neg_v[i][j] += idx[i][neighbors_num - j - 1]
        nbr_idx.append(torch.LongTensor(nbrs_v))
        neg_idx.append(torch.LongTensor(neg_v))
    nbr_idx = torch.cat(nbr_idx, dim=-1)
    neg_idx = torch.cat(neg_idx, dim=-1)
    losses = []
    acc_list = []
    nmi_list = []
    pur_list = []
    for epoch in range(train_epoch):
        save_loss = True
        # Train
        for x, y, idx, pu in train_loader:
            optimizer.zero_grad()
            model.train()
            for v in range(view_num):
                x[v] = x[v].to(device)
            clustering = epoch > clustering_epoch
            hidden_share, hidden_specific, hidden, recs = model(x)
            loss_rec = 0
            loss_mi = 0
            loss_ad = 0
            labels_true = torch.ones(x[0].shape[0]).long().to(device)
            labels_false = torch.zeros(x[0].shape[0]).long().to(device)
            for v in range(view_num):
                loss_rec += mse_loss_fn(recs[v], x[v])
                loss_mi += orthogonal_loss(hidden_share, hidden_specific[v])
                loss_ad += model.discriminators_loss(hidden_specific, v)
            loss_con = 0
            if do_contrast:
                for i in range(len(idx)):
                    index = idx[i]
                    hidden_positive = hidden[nbr_idx[index]]
                    positive = torch.exp(torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_positive.detach()))
                    negative_idx = neg_idx[index]
                    hidden_negative = hidden[negative_idx]
                    negative = torch.exp(
                        torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_negative.detach())).sum()
                    loss_con -= torch.log((positive / negative)).sum()
                    torch.cuda.empty_cache()
                loss_con = loss_con / len(idx)
            loss = loss_rec + lambda_ma * (loss_mi + loss_ad) + lambda_con * loss_con
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        # Test
        with torch.no_grad():
            for x, y, idx, pu in test_loader:
                for v in range(view_num):
                    x[v] = x[v].to(device)
                model.eval()
                hidden_share, hidden_specific, hidden, recs = model(x)
                kmeans = KMeans(n_clusters=nc, n_init=50)
                datas = hidden.clone().cpu().numpy()
                y_pred = kmeans.fit_predict(datas)
                label = np.array(y)
                ACC, NMI, Purity, ARI, F_score, Precision, Recall = clusteringMetrics(label, y_pred)
                info = {"epoch": epoch, "acc": ACC, "nmi": NMI, "ari": ARI, "Purity": Purity, "fscore": F_score,
                        "percision": Precision, "recall": Recall}
                logger.info(str(info))
