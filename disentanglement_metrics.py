import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

import lib.utils as utils
from metric_helpers.loader import load_model_and_dataset
from metric_helpers.mi_metric import compute_metric_shapes, compute_metric_faces


def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        pbar.update(batch_size)
    pbar.close()

    entropies /= S

    return entropies


def mutual_info_metric_shapes(vae, shapes_dataset):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size

    qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K)

    print('Estimating conditional entropies for scale.')
    for i in range(6):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 6, K).transpose(0, 1),
            qz_params_scale.view(N // 6, K, nparams),
            vae.q_dist)

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for orientation.')
    for i in range(40):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 40, K).transpose(0, 1),
            qz_params_scale.view(N // 40, K, nparams),
            vae.q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 40

    print('Estimating conditional entropies for pos x.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
        qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist)

        cond_entropies[2] += cond_entropies_i.cpu() / 32

    print('Estimating conditional entropies for pox y.')
    for i in range(32):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 32, K).transpose(0, 1),
            qz_params_scale.view(N // 32, K, nparams),
            vae.q_dist)

        cond_entropies[3] += cond_entropies_i.cpu() / 32

    metric = compute_metric_shapes(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


def mutual_info_metric_faces(vae, shapes_dataset):
    dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = len(dataset_loader.dataset)  # number of data samples
    K = vae.z_dim                    # number of latent variables
    nparams = vae.q_dist.nparams
    vae.eval()

    print('Computing q(z|x) distributions.')
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    for xs in dataset_loader:
        batch_size = xs.size(0)
        xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
        qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        n += batch_size

    qz_params = Variable(qz_params.view(50, 21, 11, 11, K, nparams).cuda())
    qz_samples = vae.q_dist.sample(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        vae.q_dist)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(3, K)

    print('Estimating conditional entropies for azimuth.')
    for i in range(21):
        qz_samples_pose_az = qz_samples[:, i, :, :, :].contiguous()
        qz_params_pose_az = qz_params[:, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_az.view(N // 21, K).transpose(0, 1),
            qz_params_pose_az.view(N // 21, K, nparams),
            vae.q_dist)

        cond_entropies[0] += cond_entropies_i.cpu() / 21

    print('Estimating conditional entropies for elevation.')
    for i in range(11):
        qz_samples_pose_el = qz_samples[:, :, i, :, :].contiguous()
        qz_params_pose_el = qz_params[:, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_pose_el.view(N // 11, K).transpose(0, 1),
            qz_params_pose_el.view(N // 11, K, nparams),
            vae.q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 11

    print('Estimating conditional entropies for lighting.')
    for i in range(11):
        qz_samples_lighting = qz_samples[:, :, :, i, :].contiguous()
        qz_params_lighting = qz_params[:, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_lighting.view(N // 11, K).transpose(0, 1),
            qz_params_lighting.view(N // 11, K, nparams),
            vae.q_dist)

        cond_entropies[2] += cond_entropies_i.cpu() / 11

    metric = compute_metric_faces(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpt', required=True)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-save', type=str, default='.')
    args = parser.parse_args()

    if args.gpu != 0:
        torch.cuda.set_device(args.gpu)
    vae, dataset, cpargs = load_model_and_dataset(args.checkpt)
    metric, marginal_entropies, cond_entropies = eval('mutual_info_metric_' + cpargs.dataset)(vae, dataset)
    torch.save({
        'metric': metric,
        'marginal_entropies': marginal_entropies,
        'cond_entropies': cond_entropies,
    }, os.path.join(args.save, 'disentanglement_metric.pth'))
    print('MIG: {:.2f}'.format(metric))
