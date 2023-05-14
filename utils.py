import torch
import pandas as pd
import os

MODEL_DIR = './model'


def scaling(x):
    # x has shape [batch, seq_length, num_nodes, feature dim]
    seq_length = x.shape[1]
    sum_features = torch.sum(x, dim=1, keepdim=True)
    return (1.0 + sum_features / seq_length).detach()


def pre_scaling(x):
    # x has shape [batch, seq_length, num_nodes, feature dim]
    seq_length = x.shape[1]
    sum_features = torch.sum(x, dim=0, keepdim=True)
    return (1.0 + sum_features / seq_length).flatten().detach()


def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)


def print_matrix(matrix):
    return pd.DataFrame(matrix)


def nd(target, pred):
    a = np.sum(np.abs(target - pred))
    b = np.sum(np.abs(target))
    return a / b


def list_available_models():
    for f in os.listdir(MODEL_DIR):
        if f.endswith('py') and f != 'net.py':
            print(f.replace('.py', ''))


from einops import rearrange
import numpy as np


def decompose_results(res: list):
    flatten_mu = []
    flatten_sigma = []
    flatten_y = []
    for i in range(0, len(res)):
        mu_i, sigma_i, y_true = res[i]
        mu_i = torch.concatenate(mu_i, dim=-1).cpu().numpy()
        sigma_i = torch.concatenate(sigma_i, dim=-1).cpu().numpy()
        y_true = rearrange(y_true, 'b s n f -> b n (s f)', b=1).cpu().numpy()
        flatten_mu.append(mu_i)
        flatten_sigma.append(sigma_i)
        flatten_y.append(y_true)

    return np.concatenate(flatten_mu, axis=-1), np.concatenate(flatten_sigma, axis=-1), np.concatenate(flatten_y,
                                                                                                       axis=-1)

def get_metrics(res, horizon, n_nodes):
    mus, sigmas, ys = decompose_results(res)
    print(ys.shape)
    print(mus.shape)
    ys = ys.squeeze(axis=0)
    mus = mus.squeeze(axis=0)
    print(ys.shape)
    print(mus.shape)
    assert ys.ndim == 2
    assert mus.ndim == 2
    rmse_loss = rmse_paper(ys, mus, horizon=horizon, n_nodes=n_nodes)
    nd_loss = nd(ys, mus)
    return rmse_loss, nd_loss


import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def rmse_paper(y_true, y_pred, horizon, n_nodes):
    a =  np.sqrt(np.sum((y_true - y_pred) **2) /(n_nodes * horizon))
    b =  np.sum(np.abs(y_true)) / (n_nodes * horizon)
    return a /b


def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))


DATAFRAME_DIR = './results/'


def add_metrics(name: str, rmse, nd):
    file_path = ''.join([DATAFRAME_DIR, name, '.csv'])
    data = [[rmse, nd]]
    if not os.path.exists(file_path):
        df_to_add = pd.DataFrame(data=data, columns=['RMSE', 'ND'])
        df_to_add.to_csv(file_path)
        return 1
    df = pd.read_csv(file_path, index_col=0)
    df_to_add = pd.DataFrame(data=data, columns=['RMSE', 'ND'])
    df = pd.concat((df, df_to_add), ignore_index=True)
    df.to_csv(file_path)
    return df.shape[0]


def update_dataframe(name: str, res):
    mus, sigmas, ys = decompose_results(res)
    ys = ys.squeeze(dim=0)
    mus = mus.squeeze(dim=0)
    assert ys.ndim == 2
    assert mus.ndim == 2
    rmse_loss = rmse_paper(ys, mus)
    nd_loss = nd(ys, mus)
    i = add_metrics(name, rmse_loss, nd_loss)
    print(f'df of {name} now has {i} elements')
    return i


def draw(res, horizon, n_nodes):
    for i in range(n_nodes):
        draw_single(res, i, horizon, n_nodes)



def draw_single(res, i, horizon, n_nodes):
    mus, sigmas, ys = decompose_results(res)
    fig = plt.figure(i, figsize=(16, 8))
    rmse_p = rmse_paper(ys[:, i, :], mus[:, i, :], horizon=horizon, n_nodes=1)
    nrmse_loss = nrmse(ys[:, i, :], mus[:, i, :])
    plt.plot(mus[:, i, :].flatten(), label='Predicted values')
    plt.plot(ys[:, i, :].flatten(), label='real values')
    plt.legend()
    plt.title(f'Prediction vs real value for time series {i}   RMSE (paper): {rmse_p} and NRMSE : {nrmse_loss}')
    plt.show()


from tsl.datasets.prototypes import DatetimeDataset
def compute_electric(dataset: DatetimeDataset):
    dataframe = dataset.dataframe()
