# beta-TCVAE

This repository contains cleaned-up code for reproducing the quantitative experiments in Isolating Sources of Disentanglement in Variational Autoencoders \[[arxiv](https://arxiv.org/abs/1802.04942)\].

## Usage

To train a model:

```
python vae_quant.py --dataset [shapes/faces] --beta 6 --tcvae
```
Specify `--conv` to use the convolutional VAE. We used a mlp for dSprites and conv for 3d faces. To see all options, use the `-h` flag.

The main computational difference between beta-VAE and beta-TCVAE is summarized in [these lines](vae_quant.py#L220-L228).

To evaluate the MIG of a model:
```
python disentanglement_metrics.py --checkpt [checkpt]
```
To see all options, use the `-h` flag.

## Datasets

### dSprites
Download the npz file from [here](https://github.com/deepmind/dsprites-dataset) and place it into `data/`.

### 3D faces
We cannot publicly distribute this due to the [license](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). 

## Contact
Email rtqichen@cs.toronto.edu if you have questions about the code/data.
