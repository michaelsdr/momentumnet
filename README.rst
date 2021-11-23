Momentum ResNets: Drop-in replacement for any ResNet with reduced memory footprint 
==================================================================================================

|GHActions|_ |PyPI|_ |Downloads|_

.. |GHActions| image:: https://github.com/michaelsdr/momentumnet/workflows/unittests/badge.svg?branch=main&event=push
.. _GHActions: https://github.com/michaelsdr/momentumnet/actions

.. |PyPI| image:: https://badge.fury.io/py/momentumnet.svg
.. _PyPI: https://badge.fury.io/py/momentumnet

.. |Downloads| image:: http://pepy.tech/badge/momentumnet
.. _Downloads: http://pepy.tech/project/momentumnet

This repository hosts Python code for Momentum ResNets.

See the `documentation <https://michaelsdr.github.io/momentumnet/index.html>`_, our `ICML 2021 paper <https://arxiv.org/abs/2102.07870>`_ and a `5 min presentation <https://www.youtube.com/watch?v=4PQR7ErASNo>`_.

Model
---------

Installation
------------

pip
~~~

To install ``momentumet``, you first need to install its dependencies::

	$ pip install numpy matplotlib torch

Then install momentumnet with pip::

	$ pip install momentumnet

or to get the latest version of the code::

  $ pip install git+https://github.com/michaelsdr/momentumnet.git#egg=momentumnet

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.


check
~~~~~

To check if everything worked fine, you can do::

	$ python -c 'import momentumnet'

and it should not give any error message.


Quickstart
----------

The main class is MomentumNet. It creates a Momentum ResNet for which
forward equations can be reversed in closed-form,
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

Here is a short `tutorial <https://colab.research.google.com/drive/1zAyNz2mSxCNcy-rIXLDYS8B2CJXqDYA3?usp=sharing>`_ showing the memory gains when using Momentum Transformers.



Dependencies
------------

These are the dependencies to use momentumnet:

* numpy (>=1.8)
* matplotlib (>=1.3)
* torch (>= 1.9)
* memory_profiler
* vit_pytorch



Cite
----

If you use this code in your project, please cite::

    Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyré
    Momentum Residual Neural Networks
    Proceedings of the 38th International Conference on Machine Learning, PMLR 139:9276-9287
    https://arxiv.org/abs/2102.07870

