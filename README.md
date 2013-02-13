Playdoh: pure Python library for distributed computing and optimization 
=======================================================================

Playdoh is a pure Python library for distributing computations across the free computing units (CPUs and GPUs) available in a small network of multicore computers. Playdoh supports independent (embarassingly) parallel problems as well as loosely coupled tasks such as global optimizations, Monte Carlo simulations and numerical integration of partial differential equations. It is designed to be *lightweight* and *easy-to-use* and should be of interest to scientists wanting to turn their lab computers into a small cluster at no cost.

## Features

  * **Standalone, lightweight, easy-to-use pure Python library** for small-scale distributed scientific computing
  * **Parallel/distributed version of `map`**: call any Python function with different parameters in parallel over different CPUs/computers interconnected within a standard Ethernet network

    import playdoh
    result = playdoh.map(lambda x: x * x, [1, 2], cpu=2)  # result == [1, 4]


  * **Built-in support for GPU through PyCUDA**: if you provide the PyCUDA/CUDA code, Playdoh can run it on several GPUs (on one or more computers) in parallel
  * **Built-in distributed optimization toolbox**: you provide an objective Python function, and Playdoh will minimize or maximize it in parallel over several CPUs/computers (or GPUs if you give the PyCUDA/CUDA code) using a particle-based gradient-free optimization algorithm (CMA-ES, Genetic Algorithms or Particle Swarm Optimization algorithm)
  * **Simple coarse-grained distributed computing interface**: you can distribute your computations into loosely-coupled tasks and run them in parallel over CPUs and computers, inter-node communication happening through blocking FIFO tubes exposing a straightforward push/pull interface (examples include distributed PDE solvers, cellular automata, Monte Carlo simulators, optimizations...)
  * **Code handling**: Playdoh handles the transport of your Python code between machines
  * **Resources sharing**: specify how many CPUs you dedicate to others' computations and how many you keep for yourself

## Documentation

The documentation can be found here: [http://playdoh.googlecode.com/svn/docs/index.html Documentation of Playdoh].

## Paper

Rossant C, Fontaine B, Goodman DFM (2011). [**Playdoh: a lightweight Python package for distributed computing and optimisation**](http://www.sciencedirect.com/science/article/pii/S1877750311000561). *Journal of Computational Science*

### Abstract

Parallel computing is now an essential paradigm for high performance scientific computing. Most existing hardware and software solutions are expensive or difficult to use. We developed Playdoh, a Python library for distributing computations across the free computing units available in a small network of multicore computers. Playdoh supports independent and loosely coupled parallel problems such as global optimisations, Monte Carlo simulations and numerical integration of partial differential equations. It is designed to be lightweight and easy to use and should be of interest to scientists wanting to turn their lab computers into a small cluster at no cost.

## Contribute

Playdoh is an open-source project and anyone is welcome to contribute to the project. Here are some info about the source code.

  * There's a unit testing suite in the `test` directory, it can be launched with the `test.py` script.

  * A document explaining how to add new features to Playdoh (like a new optimization algorithm) will be available soon.

  * A document explaining the internal implementation details of Playdoh will be available soon, it will be the starting point if you want to contribute to the code.

  * You can submit issues, ideas, comments on the [Playdoh Google group](http://groups.google.com/group/playdoh-library).


