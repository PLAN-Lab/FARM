Bootstrap Documentation
=======================

:mod:`Bootstrap` is a high-level framework for starting deep learning projects.
It aims at accelerating research projects and prototyping by providing a powerful workflow focused on your dataset and model only.

And it is:

- Scalable
- Modular
- Shareable
- Extendable
- Uncomplicated
- Built for reproducibility
- Easy to log and plot anything

It's not a wrapper over pytorch, it's a powerful extension.

Installation
============

First, install python3 and pytorch with Anaconda:

- `python with anaconda <https://www.continuum.io/downloads>`_
- `pytorch with CUDA <http://pytorch.org>`_

Attention: You need `pytorch <http://pytorch.org>`_ version 0.4.1 or superior in order to use Bootstrap.

There are two ways of using bootstrap.pytorch: (1) as a standalone project, or (2) as a python library.

1. As a standalone project
+++++++++++++++++++++++++++++++

We advise you to clone bootstrap to start a new project.
This way, it will be easier to prototype and debug your code, as you will have direct access to bootstrap core functions:

.. code:: bash

    git clone https://github.com/Cadene/bootstrap.pytorch.git
    cd bootstrap.pytorch
    pip install -r requirements.txt


2. As a python library
+++++++++++++++++++++++

Using bootstrap like a python library is also possible. You can use pip install:

.. code:: bash

    pip install bootstrap.pytorch

Or install from source:

.. code:: bash

    git clone https://github.com/Cadene/bootstrap.pytorch.git
    cd bootstrap.pytorch
    python setup.py install


.. toctree::
   :maxdepth: 2
   :caption: Notes

   concepts
   quickstart
   directories
   examples 

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   engines
   models
   networks
   criterions
   metrics
   datasets
   optimizers
   views
   options
   logger
   lib

.. toctree::
   :maxdepth: 2
   :caption: Package Reference for Submodules
    
   submodules/mnist/models/networks

.. automodule:: bootstrap
   :members:

Few words from the authors
--------------------------

Bootstrap is the result of the time we spent engineering stuff since the beginning of our PhDs.
We have worked with different libraries and languages (Torch7, Keras, Tensorflow, Pytorch, Torchnet, and others), and they all inspired the development of bootstrap.pytorch.

Part of this inspiration also comes from the modularity of modern web frameworks.
We came up with a nice workflow and good practicies that we wanted like to share.

Last but not least, criticism is always welcome, feel free to send us a message or a pull request. :-)

`Remi Cadene <http://remicadene.com/>`_, `Micael Carvalho <http://micaelcarvalho.com/>`_, Hedi Ben-Younes, and `Thomas Robert <http://www.thomas-robert.fr/en/>`_
