import torch


def scaling(x):
    # x has shape [batch, seq_length, num_nodes, feature dim]
    seq_length = x.shape[1]
    sum_features = torch.sum(x, dim=1, keepdim=True)
    return 1.0 + sum_features / seq_length

