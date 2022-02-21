.. momentumnet documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Momentum ResNets
================

Official library for using Momentum Residual Neural Networks [1]. These models extend any Residual architecture (for instance it also work with Transformers) to a larger class of deep learning models that consume less memory. They can be initialized with the same weights as a pretrained ResNet and are promising in fine-tuning applications.


Installation
------------

To install ``momentumnet``, you first need to install its dependencies::

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

The main class is MomentumNet. It creates a Momentum ResNet that iterates

.. math::

    v_{t + 1} = \gamma * v_t + (1 - \gamma) * f_t(x_t) \\
    x_{t + 1} = x_t + v_{t + 1}


These forward equations can be reversed in closed-form,
enabling learning without standard memory consuming backpropagation.
This process trades memory for computations.

To get started, you can create a toy Momentum ResNet by specifying the functions f for the forward pass
and the value of the momentum term, gamma.

.. code:: python

   >>> from torch import nn
   >>> from momentumnet import MomentumNet
   >>> hidden = 8
   >>> d = 500
   >>> function = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))
   >>> mresnet = MomentumNet([function,] * 10, gamma=0.9)

Momentum ResNets are a drop-in replacement for ResNets
------------------------------------------------------

We can transform a ResNet into a MomentumNet with the same parameters in two lines of codes.
For instance, the following code
instantiates a Momentum ResNet with weights of a pretrained Resnet-101 on ImageNet. We set "use_backprop" to False
so that activations are not saved during the forward pass, allowing smaller memory consumptions.

.. code:: python

   >>> import torch
   >>> from momentumnet import transform_to_momentumnet
   >>> from torchvision.models import resnet101
   >>> resnet = resnet101(pretrained=True)
   >>> mresnet101 = transform_to_momentumnet(resnet, gamma=0.9, use_backprop=False)


Importantly, this method also works with Pytorch Transformers module, specifying the residual layers to be turned into their Momentum version.

.. code:: python

   >>> import torch
   >>> from momentumnet import transform_to_momentumnet
   >>> transformer = torch.nn.Transformer(num_encoder_layers=6, num_decoder_layers=6)
   >>> mtransformer = transform_to_momentumnet(transformer, sub_layers=["encoder.layers", "decoder.layers"], gamma=0.9,
   >>>                                          use_backprop=False, keep_first_layer=False)

This initiates a Momentum Transformer with the same weights as the original Transformer.

Memory savings when applying Momentum ResNets to Transformers
-------------------------------------------------------------

Here is a short `tutorial <https://colab.research.google.com/drive/1zAyNz2mSxCNcy-rIXLDYS8B2CJXqDYA3?usp=sharing>`_ showing the memory gains using Momentum Transformers.

Dependencies
------------

These are the dependencies to use momentumnet:

* numpy (>=1.8)
* matplotlib (>=1.3)
* torch (>= 1.9)
* memory_profiler
* torchvision
* vit_pytorch

Bug reports
-----------

Use the `github issue tracker <https://github.com/michaelsdr/momentumnet/issues>`_ to report bugs.

Cite
----

   [1] Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyre. Momentum Residual Neural Networks.
      Proceedings of the 38th International Conference
      on Machine Learning, PMLR 139:9276-9287

      https://arxiv.org/abs/2102.07870


API
---

.. toctree::
    :maxdepth: 1

    api.rst
