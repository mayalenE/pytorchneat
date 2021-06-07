## Differentiable pytorch-based implementation of CPPNs

Summary and references:  
**1) NEAT (NeuroEvolution of Augmenting Topologies):** while previous work proposed genetic algorithms to evolve the weights of fixed-topologt NN, NEAT is a neuroevolution algorithm that evolves the architectures of networks in addition to the weights [[Stanley et al., 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)].  
**2) CPPN (Compositional Pattern Producing Network):** CPPN is a recurrent NN (generally small with few neurons/connections) of the form *f(input,structural bias)=output* which is designed to represent patterns with regularities such as symmetry, repetition, and repetition with variation. CPPN must fully activate once for each coordinate in the phenotype, making its complexity O(N) (O(N2) for images) [[Stanley 2007 paper](https://eplex.cs.ucf.edu/papers/stanley_gpem07.pdf)].  
**3) CPPN-NEAT:**  Slight modifications of NEAT (several activation functions instead of one) to evolve CPPN networks. CPPN are small resolution-agnostic encoding networks making the NEAT optimization less complex than its previous application on big network phenotypes.  

While it was initially proposed to evolve 2D image phenotypes, the output of the CPPN network can take many forms and other applications have been proposed:
* **HyperNEAT** propose to use CPPN to indirectly encode the weights of the network (called the *substrate*) and to evolve it with CPPN-NEAT. See [Stanley et al. 2009 paper](https://www.researchgate.net/publication/23986881_A_Hypercube-Based_Encoding_for_Evolving_Large-Scale_Neural_Networks) and [website](http://eplex.cs.ucf.edu/hyperNEATpage/).
* **Evolving soft-robot morphologies**: while the output was previously considered as continuous and 1-dimensional(image intensity for CPPN or connection weight in HyperNEAT), the output can be modularized and divided into binary/continuous outputs as proposed in [Cheney et al. 2014](https://www.researchgate.net/publication/270696982_Unshackling_evolution).
* Other applications have considered interactive evolution-schemes implicating humans to select  phenotypes and use this as fitness function to evolve "interesting" 2D or 3D patterns (see [picbreeder](http://picbreeder.org/) and [endlessforms](http://endlessforms.com/).

## Examples
With this repository you can:
* Reproduce the results of the official NEAT-Python package in `tests/test_rnn.py`
* Differentiate the CPPNs toward target images`tests/test_differentiable_cppn.py`

## Acknowledgements
Official repositoriy of CPPNs is  [NEAT-Python](https://neat-python.readthedocs.io/en/latest/).   
This repository builds upon [PyTorch-Neat](https://github.com/uber-research/PyTorch-NEAT) by Uber-Research and [autodisc](https://github.com/flowersteam/automated_discovery_of_lenia_patterns/tree/master/autodisc/autodisc/cppn) CPPN implementation by Chris Reinke (FLOWERS Team, Inria). Those repositories are based on pytorch implementation and allow to use GPU-acceleration, but do not consider differentiating  though the network evolved weights.  
