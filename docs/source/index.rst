.. AIJack documentation master file, created by
   sphinx-quickstart on Sat Jan 14 12:10:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AIJack's documentation!
==================================

.. raw:: html

   <div align="left">
   <img src="https://badgen.net/github/stars/Koukyosyumei/AIjack?color=green">
   <img src="https://badgen.net/github/forks/Koukyosyumei/AIjack">
   <img src="https://badgen.net/github/watchers/Koukyosyumei/AIjack">
   <img src="https://img.shields.io/github/commit-activity/m/Koukyosyumei/AIJack">
   <img src="https://img.shields.io/github/languages/code-size/Koukyosyumei/AIJack">
   <img src="https://app.codacy.com/project/badge/Grade/529a33e6523e4bf2a79c26e815a070f4"/>
   <img src="https://img.shields.io/github/license/Koukyosyumei/AIJack">
   </div>

.. raw:: html

   <h1 align="center">

   <br>
   <img src="_static/logo_wide_gif.gif" width=1000>
   <br>
   <br>

   </h1>

AIJack is an easy-to-use open-source simulation tool for testing the security of your AI system against hijackers.
It provides advanced security techniques like Differential Privacy, Homomorphic Encryption, and Federated Learning to guarantee protection for your AI.
With AIJack, you can test and simulate defenses against various attacks such as Poisoning, Model Inversion, Backdoor, and Free-Rider. We support more than 30 state-of-the-art methods.
Start securing your AI today with AIJack.

Key Features
============

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: All-around abilities
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      AIJack offers flexible API for more than 30 attack and defense algorithms. You can easily experiment various combinations of these methods.

   .. grid-item-card:: PyTorch-friendly design
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      AIJack supports many models of PyTorch. You can integrate most attacks and defenses with minimal modifications of the original codes.

   .. grid-item-card:: Compatible with sklearn
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      AIJack also supports scikit-kearn so that you can simulate not only deep learning but also other machine learning models.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Fast Implementation
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      AIJack uses C++ backend for many components like Differential Privacy and Homomorphic Encryption to enhance scalability.

   .. grid-item-card:: MPI-backend for FL
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      AIJack supports MPI-backed for Federated Learning so that you can deploy AIJack in your High Performance Computing system.

   .. grid-item-card:: Extensible
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      AIJack consists of simple modular APIs. All source codes are available on GitHub. Everyone is welcome to contribute.

Resources
=========

.. grid:: 3

   .. grid-item-card:: :material-regular:`rocket_launch;2em` Tutorial
      :columns: 12 6 6 4
      :link: tutorial
      :link-type: ref
      :class-card: getting-started

   .. grid-item-card:: :material-regular:`library_books;2em` API Docs
      :columns: 12 6 6 4
      :link: api
      :link-type: ref
      :class-card: user-guides

   .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Developer Docs
      :columns: 12 6 6 4
      :link: contribution
      :link-type: ref
      :class-card: developer-docs

Installation
============

.. note::
   AIJack requires Boost and pybind11.

   .. code-block:: bash

      apt install -y libboost-all-dev
      pip install -U pip
      pip install "pybind11[global]"

You can install aijack via pip.

.. code-block:: bash

   pip install aijack


Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :caption: Tutorial
   :hidden:
   :numbered:
   :maxdepth: 2

   notebooks/federated_learning
   notebooks/modelinversion
   notebooks/labelleakage
   notebooks/membershipinference
   notebooks/poisoning
   notebooks/evasion
   notebooks/diferentialprivacy
   notebooks/kanonymity
   notebooks/debugging

.. toctree::
   :caption: API Docs
   :hidden:
   :numbered:
   :maxdepth: 2

   aijack.attack
   aijack.defense
   aijack.collaborative
   aijack.utils

.. toctree::
   :caption: Developer Docs
   :hidden:
   :numbered:
   :maxdepth: 2

   contribution
