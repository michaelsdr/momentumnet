#  Momentum Residual Neural Networks 

Paper: https://arxiv.org/abs/2102.07870

## Compat

This package has been developed and tested with `python3.8`. It is therefore not guaranteed to work with earlier versions of python.

## Install the repository on your machine


This package can easily be installed using `pip`, with the following command:

```bash
pip install numpy
pip install -e .
```

This will install the package and all its dependencies, listed in `requirements.txt`. To test that the installation has been successful, you can install `pytest` and run the test suite using

```
pip install pytest
pytest
```


## Reproducing the figures of the paper

Figure 1 - Comparison of the dynamics of a ResNet and a Momentum ResNet

```bash
python examples/plot_dynamics_1D.py
```

Figure 2 - Memory comparison on a toy example 

```bash
python examples/plot_memory.py
```

Figure 5 - Separation of nested rings using a Momentum ResNet

```bash
python examples/run_separation_nested_rings.py
python examples/plot_separation_nested_rings.py
```

## Momentu ResNets are a drop-in replacement for ResNets

To see how a Momentum ResNet can be created using a ResNet, you can run


```bash
python examples/from_resnet_to_momentumnet.py
```

This creates a Momentum ResNet-18, Momentum ResNet-34, Momentum ResNet-101 and Momentum ResNet-152.
The first two models have the same weights as pretrained ResNets on ImageNet.


## Running Image Experiments

CIFAR-10

You can train a Momentum ResNet or a ResNet on the CIFAR-10 dataset by using

```bash
python examples/run_CIFAR_10.py -m [MODEL] -g [GAMMA]
```

Available values for `[MODEL]` are `resnet18/34/101/152` for ResNets or `mresnet18/34/101/152` for Momentum ResNets
(default `mresnet18`). Available values for `[GAMMA]` are floats between 0 and 1.

## Cite

If you use this code in your project, please cite:

```bash
Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyré
Momentum Residual Neural Networks In: Proc. of ICML 2021. 
```
