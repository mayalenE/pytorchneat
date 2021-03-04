# Adaptation of pytorch-neat from autodisc and uber-research packages 

## Activations
### Autodisc
* def delphineat_gauss_torch_activation(z):
    '''PyTorch implementation of gauss activation as defined by SharpNEAT, which is also as in DelphiNEAT.'''
    return 2.0 * torch.exp(-1 * (z * 2.5) ** 2) - 1
* def delphineat_sigmoid_torch_activation(z):
    '''PyTorch implementation of sigmoidal activation function as defined in DelphiNEAT'''
    return 2.0 * (1.0 / (1.0 + torch.exp(-z*5)))-1

### Uber-research
* def sigmoid_activation(x):    return torch.sigmoid(5 * x)
* def tanh_activation(x):    return torch.tanh(2.5 * x)
* def abs_activation(x):    return torch.abs(x)
* def gauss_activation(x):   return torch.exp(-5.0 * x**2)
* def identity_activation(x):  return x
* def sin_activation(x):    return torch.sin(x)
* def relu_activation(x):    return F.relu(x)

### Morphosearch
* all 

## Aggregations
### Chris
* sum

### Uber-Research
* sum
* product

### Morphosearch
* sum
* product
* TODO: others from NEAT package (min, max, median)

## RecurrentNet
### Chris 
* inherit from nn.Module (forward function, etc) 
* allows self-connections genomes
* only one arg for *connections = [[from_id, to_id, weight], ...]*
* only sum aggregation implemented
### Uber-research
* Sparse tensors but converted to dense  (faster? less memory?)
* Different aggregations functions
* one arg per type of connection *input_to_hidden, hidden_to_hidden, output_to_hidden, input_to_output, hidden_to_output, output_to_output*
* specific implementation for CPPNs (why?)
### Morphosearch

## NEAT instantiation / evolution
### Autodisc
* allows for population of size 1 and no fitness (which we want to do too)
### Uber-research
* Use neat and multiprocessing, no specific updates. Example here: https://github.com/uber-research/PyTorch-NEAT/blob/master/examples/adaptive/main.py
### Morphosearch
* One for simcells input parametrisation 
* One for toysystem update rule