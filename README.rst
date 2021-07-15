Momentum ResNets: Drop-in replacement for any ResNet with a significantly reduced memory footprint 
==================================================================================================

|GHActions|_ |PyPI|_ |Downloads|_

.. |GHActions| image:: https://github.com/michaelsdr/momentumnet/workflows/unittests/badge.svg?branch=main&event=push
.. _GHActions: https://github.com/michaelsdr/momentumnet/actions

.. |PyPI| image:: https://badge.fury.io/py/momentumnet.svg
.. _PyPI: https://badge.fury.io/py/momentumnet

.. |Downloads| image:: http://pepy.tech/badge/momentumnet
.. _Downloads: http://pepy.tech/project/momentumnet

This repository hosts Python code for Momentum ResNets.

See the `documentation <https://michaelsdr.github.io/momentumnet/index.html>`_ and our `ICML 2021 paper <https://arxiv.org/abs/2102.07870>`_.

Model
---------

Installation
------------

pip
~~~

To install ``momentumet``, you first need to install its dependencies::

	$ pip install numpy matplotlib numexpr scipy

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

Reproducing the figures of the paper
------------------------------------

You can download the directory examples_paper and reproduce some figures of the paper. 

Figure 1 - Comparison of the dynamics of a ResNet and a Momentum ResNet::

 python examples_paper/plot_dynamics_1D.py

Figure 2 - Memory comparison on a toy example:: 

$ python examples_paper/plot_memory.py

Figure 5 - Separation of nested rings using a Momentum ResNet::

$ python examples_paper/run_separation_nested_rings.py
$ python examples_paper/plot_separation_nested_rings.py

You can also train a Momentum ResNet or a ResNet on the CIFAR-10 dataset by using::

$ python examples_paper/run_CIFAR_10.py -m [MODEL] -g [GAMMA]

Available values for `[MODEL]` are `resnet18/34/101/152` for ResNets or `mresnet18/34/101/152` for Momentum ResNets
(default `mresnet18`). Available values for `[GAMMA]` are floats between 0 and 1.

Dependencies
------------

These are the dependencies to use momentumnet:

* numpy (>=1.8)
* matplotlib (>=1.3)
* torch (>= 1.7)
* memory_profiler
* vit_pytorch



Cite
----

If you use this code in your project, please cite::

    Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyré
    Momentum Residual Neural Networks
    In: Proc. of ICML 2021. 
    https://arxiv.org/abs/2102.07870

