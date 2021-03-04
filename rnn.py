import torch
torch.set_default_dtype(torch.float64)
from pytorchneat.activations import str_to_activation
from pytorchneat.aggregations import str_to_aggregation
from numbers import Number

def str_to_tuple_key(str_key):
    return (int(str_key.split(",")[0]), int(str_key.split(",")[1]))

def tuple_to_str_key(tuple_key):
    return f"{tuple_key[0]},{tuple_key[1]}"


class RecurrentNetwork(torch.nn.Module):

    def __init__(self, input_neuron_ids, hidden_neuron_ids, output_neuron_ids, biases, responses, activations, aggregations, connections):
        """
        :param input_neuron_ids: list of ids of input neurons
        :param hidden_neuron_ids: list of ids of hidden neurons
        :param output_neuron_ids: list of ids of output neurons
        """
        super().__init__()

        self.input_neuron_ids = input_neuron_ids # input neurons are indexed from -n_inputs
        self.n_inputs = len(input_neuron_ids)
        self.output_neuron_ids = output_neuron_ids # output neurons are indexed from 0
        self.n_outputs = len(output_neuron_ids)
        self.hidden_neuron_ids = hidden_neuron_ids # hidden neurons are indexed from n_outputs
        self.n_hidden = len(hidden_neuron_ids)

        # connection parameters
        connections_with_str_keys = dict() # torch accept only str as parameter name and not Tuple of int
        for k, v in connections.items():
            try:
                connections_with_str_keys[tuple_to_str_key(k)] = torch.nn.Parameter(torch.tensor(v))
            except:
                print('break')
        self.connections = torch.nn.ParameterDict(connections_with_str_keys)

        # node parameters : just list where ids are node keys
        biases_with_str_keys = dict()
        for k, v in biases.items():
            biases_with_str_keys[str(k)] = torch.nn.Parameter(torch.tensor(v))
        self.biases = torch.nn.ParameterDict(biases_with_str_keys)
        responses_with_str_keys = dict()
        for k, v in responses.items():
            responses_with_str_keys[str(k)] = torch.nn.Parameter(torch.tensor(v))
        self.responses = torch.nn.ParameterDict(responses_with_str_keys)
        self.activation_functions = dict()
        for k, v in activations.items():
            self.activation_functions[str(k)] = v
        self.aggregation_functions = dict()
        for k, v in aggregations.items():
            self.aggregation_functions[str(k)] = v

        self.input_key2idx = dict.fromkeys(self.input_neuron_ids)
        v = 0
        for k in self.input_key2idx.keys():
            self.input_key2idx[k] = v
            v += 1

        self.neuron_key2idx = dict.fromkeys(self.output_neuron_ids+self.hidden_neuron_ids)
        v = 0
        for k in self.neuron_key2idx.keys():
            self.neuron_key2idx[k] = v
            v +=1


    def forward(self, inputs):
        '''
        :param inputs: tensor of size (batch_size, n_inputs)

        Note: output of a node as follows: activation(bias+(response∗aggregation(inputs)))

        returns: (batch_size, n_outputs)
        '''
        batch_size = len(inputs)
        after_pass_node_activs = torch.zeros_like(self.node_activs)

        for neuron_key in self.output_neuron_ids+self.hidden_neuron_ids:
            incoming_connections = [c for c in self.connections.keys() if str_to_tuple_key(c)[1] == neuron_key]
            incoming_inputs = torch.empty((0, batch_size))

            # multiply by connection weights
            for conn in incoming_connections:
                input_key = str_to_tuple_key(conn)[0]
                # input_to_neuron case
                if input_key < 0:
                    incoming_inputs = torch.vstack([incoming_inputs, self.connections[conn] * inputs[:, self.input_key2idx[input_key]]])
                # neuron_to_neuron case
                else:
                    incoming_inputs = torch.vstack([incoming_inputs, self.connections[conn] * self.node_activs[:, self.neuron_key2idx[input_key]]])

            # aggregate incoming inputs
            if len(incoming_connections)>0:
                node_outputs = self.activation_functions[str(neuron_key)](self.biases[str(neuron_key)] + self.responses[str(neuron_key)] * self.aggregation_functions[str(neuron_key)](incoming_inputs))
            else:
                node_outputs = torch.zeros((batch_size))
            after_pass_node_activs[:, self.neuron_key2idx[neuron_key]] = node_outputs

        self.node_activs = after_pass_node_activs
        return after_pass_node_activs[:, self.output_neuron_ids]


    def activate(self, inputs, n_passes=1):
        '''
        :param inputs: tensor of size (batch_size, n_inputs)
        :param n_passes: number of passes in the RNN TODO: currently global passes should it be internal passes as in pytorch-neat by uber research ?
        returns: (batch_size, n_outputs)
        '''
        batch_size = inputs.shape[0]
        assert inputs.shape[1] == self.n_inputs

        self.node_activs = torch.zeros(batch_size, self.n_outputs+self.n_hidden)

        for _ in range(n_passes):
            outputs = self.forward(inputs)

        return outputs



    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns the RecurrentCPPN()
        """
        genome_config = config.genome_config

        input_neuron_ids = list(reversed(range(-genome_config.num_inputs, 0))) # list of ids of input neurons
        output_neuron_ids = list(range(genome_config.num_outputs)) # list of ids of output neurons
        hidden_neuron_ids = list(genome.nodes.keys())[len(output_neuron_ids):] # list of ids of hidden neurons (neither input nor ouput)

        biases = dict.fromkeys(genome.nodes.keys()) # biases = {node_id: nodes[node_id].bias}
        responses = dict.fromkeys(genome.nodes.keys()) # responses = {node_id: nodes[node_id].response}
        activations = dict.fromkeys(genome.nodes.keys()) # activations = {node_id: nodes[node_id].activation}
        aggregations = dict.fromkeys(genome.nodes.keys()) # aggregations = {node_id: nodes[node_id].aggregation}

        for node_id, node in genome.nodes.items():
            biases[node_id] = node.bias
            responses[node_id] = node.response
            activations[node_id] = str_to_activation[node.activation]
            aggregations[node_id] = str_to_aggregation[node.aggregation]

        connections = dict.fromkeys(genome.connections.keys())  # connections = {(from_node_id, to_node_id): connections[(from_node_id, to_node_id)].weight}
        for (from_id, to_id), connection in genome.connections.items():
            if connection.enabled:
                connections[(from_id, to_id)] = connection.weight
            else:
                del connections[(from_id, to_id)]

        return RecurrentNetwork(input_neuron_ids, hidden_neuron_ids, output_neuron_ids, biases, responses, activations, aggregations, connections)