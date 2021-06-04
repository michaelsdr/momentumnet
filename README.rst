Momentum ResNets: Drop-in replacement for any ResNet with a significantly reduced memory footprint and better representation capabilities.
=========================================

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

Picard is an algorithm for maximum likelihood independent component analysis.
It shows state of the art speed of convergence, and solves the same problems as the widely used FastICA, Infomax and extended-Infomax, faster.

.. image:: comparison.png
  :scale: 50 %
  :alt: Comparison
  :align: center

The parameter `ortho` choses whether to work under orthogonal constraint (i.e. enforce the decorrelation of the output) or not.
It also comes with an extended version just like extended-infomax, which makes separation of both sub and super-Gaussian signals possible.
It is chosen with the parameter `extended`.

* `ortho=False, extended=False`: same solution as Infomax
* `ortho=False, extended=True`: same solution as extended-Infomax
* `ortho=True, extended=True`: same solution as FastICA
* `ortho=True, extended=False`: finds the same solutions as Infomax under orthogonal constraint.




Installation
------------

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_.


conda
~~~~~

Picard can be installed with `conda-forge <https://conda-forge.org/docs/user/introduction.html>`_.
You need to add `conda-forge` to your conda channels, and then do::

  $ conda install python-picard


pip
~~~

Otherwise, to install ``picard``, you first need to install its dependencies::

	$ pip install numpy matplotlib numexpr scipy

Then install Picard with pip::

	$ pip install python-picard

or to get the latest version of the code::

  $ pip install git+https://github.com/pierreablin/picard.git#egg=picard

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.


check
~~~~~

To check if everything worked fine, you can do::

	$ python -c 'import picard'

and it should not give any error message.


matlab/octave
~~~~~~~~~~~~~

The Matlab/Octave version of Picard and Picard-O is `available here <https://github.com/pierreablin/picard/tree/master/matlab_octave>`_.

Quickstart
----------

To get started, you can build a synthetic mixed signals matrix:

.. code:: python

   >>> import numpy as np
   >>> N, T = 3, 1000
   >>> S = np.random.laplace(size=(N, T))
   >>> A = np.random.randn(N, N)
   >>> X = np.dot(A, S)

And then use Picard to separate the signals:

.. code:: python

   >>> from picard import picard
   >>> K, W, Y = picard(X)

Picard outputs the whitening matrix, K, the estimated unmixing matrix, W, and
the estimated sources Y. It means that Y = WKX

NEW: scikit-learn compatible API
--------------------------------

Introducing `picard.Picard`, which mimics `sklearn.decomposition.FastICA` behavior:

.. code:: python

    >>> from sklearn.datasets import load_digits
    >>> from picard import Picard
    >>> X, _ = load_digits(return_X_y=True)
    >>> transformer = Picard(n_components=7)
    >>> X_transformed = transformer.fit_transform(X)
    >>> X_transformed.shape


Dependencies
------------

These are the dependencies to use Picard:

* numpy (>=1.8)
* matplotlib (>=1.3)
* numexpr (>= 2.0)
* scipy (>=0.19)


These are the dependencies to run the EEG example:

* mne (>=0.14)

Cite
----

If you use this code in your project, please cite::

    Pierre Ablin, Jean-Francois Cardoso, Alexandre Gramfort
    Faster independent component analysis by preconditioning with Hessian approximations
    IEEE Transactions on Signal Processing, 2018
    https://arxiv.org/abs/1706.08171

    Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort
    Faster ICA under orthogonal constraint
    ICASSP, 2018
    https://arxiv.org/abs/1711.10873
