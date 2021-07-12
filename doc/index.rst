.. momentumnet documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Momentum ResNets
================

Official library for using Momentum Residual Neural Networks [1].


Installation
------------

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.
Otherwise, to install ``momentumnet``, you first need to install its dependencies::

	$ pip install numpy matplotlib torch

Then install momentumnet::

	$ pip install momentumnet

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import momentumnet'

and it should not give any error message.

Quickstart
----------

To get started, you can create a toy momentumnet:

.. code:: python

   >>> from torch import nn
   >>> from momentumnet import MomentumNet
   >>> hidden = 8
   >>> d = 500
   >>> function = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))
   >>> mom_net = MomentumNet([function,] * 10, gamma=0.99)

Momentum ResNets are a drop-in replacement for ResNets
------------------------------------------------------

To see how a Momentum ResNet can be created using a ResNet, you can run:

.. code:: python

   >>> import torch
   >>> from momentumnet import transform_to_momentumnet
   >>> from torchvision.models import resnet101
   >>> resnet = resnet101(pretrained=True)
   >>> mresnet101 = transform_to_momentumnet(resnet, gamma=0.99, use_backprop=False)

This initiates a Momentum ResNet with weights of a pretrained Resnet-101 on ImageNet.

Importantly, this method also works with Pytorch Transformers module, specifying the residual layers to be turned into their Momentum counterpart.

.. code:: python

   >>> import torch
   >>> from momentumnet import transform_to_momentumnet
   >>> transformer = torch.nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
   >>> mtransformer = transform_to_momentumnet(transformer, residual_layers=["encoder.layers", "decoder.layers"], gamma=0.99,
   >>>                                          use_backprop=False, keep_first_layer=False)

This initiates a Momentum Transformer with the same weights as the original Transformer.

Dependencies
------------

These are the dependencies to use momentumnet:

* numpy (>=1.8)
* matplotlib (>=1.3)
* torch (>= 1.7)
* memory_profiler
* torchvision

Bug reports
-----------

Use the `github issue tracker <https://github.com/michaelsdr/momentumnet/issues>`_ to report bugs.

Cite
----

   [1] Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyré
      Momentum Residual Neural Networks In: Proc. of ICML 2021. https://arxiv.org/abs/2102.07870


API
---

.. toctree::
    :maxdepth: 1

    api.rst
