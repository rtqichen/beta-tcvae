import torch

metric_name = 'MIG'


def MIG(mi_normed):
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_shapes(marginal_entropies, cond_entropies):
    factor_entropies = [6, 40, 32, 32]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric


def compute_metric_faces(marginal_entropies, cond_entropies):
    factor_entropies = [21, 11, 11]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric

