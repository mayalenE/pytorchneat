import torch
torch.set_default_dtype(torch.float64)
from pytorchneat.activations import str_to_activation
from pytorchneat.aggregations import str_to_aggregation
import copy
import graphviz

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
            connections_with_str_keys[tuple_to_str_key(k)] = torch.nn.Parameter(torch.tensor(v))
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
                node_outputs = self.activation_functions[str(neuron_key)](self.biases[str(neuron_key)] + self.responses[str(neuron_key)] * self.aggregation_functions[str(neuron_key)](incoming_inputs.t()))
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


    def draw_net(self, view=False, filename=None, node_names=None, prune_unused=False, node_colors=None, fmt='svg'):
        if node_names is None:
            node_names = {}

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2'}

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        inputs = set()
        for k in self.input_neuron_ids:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in self.output_neuron_ids :
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightblue'),
                          'fontsize': '9', 'fontcolor': node_colors.get(k, 'blue'),
                          'xlabel': f'{self.activation_functions[str(k)].__name__[:-11]}({self.biases[str(k)]:.1f}+\\n{self.responses[str(k)]:.1f}*{self.aggregation_functions[str(k)].__name__[:-12]}(inputs))'}

            dot.node(name, _attributes=node_attrs)

        if prune_unused:
            connections = set()
            for cg in self.connections.keys():
                connections.add((str_to_tuple_key(cg)[0], str_to_tuple_key(cg)[1]))

            used_nodes = copy.copy(outputs)
            pending = copy.copy(outputs)
            while pending:
                new_pending = set()
                for a, b in connections:
                    if b in pending and a not in used_nodes:
                        new_pending.add(a)
                        used_nodes.add(a)
                pending = new_pending
        else:
            used_nodes = set(self.input_neuron_ids+self.output_neuron_ids+self.hidden_neuron_ids)

        for n in used_nodes:
            if n in inputs or n in outputs:
                continue
            attrs = {'style': 'filled',
                     'fillcolor': node_colors.get(n, 'white'),
                     'fontsize': '9',
                     'xlabel': f'{self.activation_functions[str(n)].__name__[:-11]}({self.biases[str(n)]:.1f}+\\n{self.responses[str(n)]:.1f}*{self.aggregation_functions[str(n)].__name__[:-12]}(inputs))'}
            dot.node(str(n), _attributes=attrs)

        for cg, cg_weight in self.connections.items():
                # if cg.input not in used_nodes or cg.output not in used_nodes:
                #    continue
                input = str_to_tuple_key(cg)[0]
                output = str_to_tuple_key(cg)[1]
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid'
                color = 'green' if cg_weight > 0 else 'red'
                width = str(abs(cg_weight))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'fontcolor': color, 'penwidth': width, 'fontsize': '9', 'label': f'w={cg_weight:.1f}'})

        dot.render(filename, view=view)

        return dot



    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns the RecurrentCPPN()
        """
        genome_config = config.genome_config

        input_neuron_ids = genome_config.input_keys  # list of ids of input neurons
        output_neuron_ids = genome_config.output_keys  # list of ids of output neurons
        hidden_neuron_ids = list(genome.nodes.keys())[
                            len(output_neuron_ids):]  # list of ids of hidden neurons (neither input nor ouput)

        biases = dict.fromkeys(genome.nodes.keys())  # biases = {node_id: nodes[node_id].bias}
        responses = dict.fromkeys(genome.nodes.keys())  # responses = {node_id: nodes[node_id].response}
        activations = dict.fromkeys(genome.nodes.keys())  # activations = {node_id: nodes[node_id].activation}
        aggregations = dict.fromkeys(genome.nodes.keys())  # aggregations = {node_id: nodes[node_id].aggregation}

        for node_id, node in genome.nodes.items():
            biases[node_id] = node.bias
            responses[node_id] = node.response
            activations[node_id] = str_to_activation[node.activation]
            aggregations[node_id] = str_to_aggregation[node.aggregation]

        connections = dict.fromkeys(
            genome.connections.keys())  # connections = {(from_node_id, to_node_id): connections[(from_node_id, to_node_id)].weight}
        for (from_id, to_id), connection in genome.connections.items():
            if connection.enabled:
                connections[(from_id, to_id)] = connection.weight
            else:
                del connections[(from_id, to_id)]

        return RecurrentNetwork(input_neuron_ids, hidden_neuron_ids, output_neuron_ids, biases, responses, activations,
                                aggregations, connections)


'''
class RecurrentNetwork(torch.nn.Module):
    ### TODO: MOVE TO SPARSE BECAUSE here with SGD you can add unexisting connections!!

    def __init__(self, input_neuron_ids, hidden_neuron_ids, output_neuron_ids, biases, responses, activations, aggregations, connections):
        """
        :param input_neuron_ids: list of ids of input neurons
        :param hidden_neuron_ids: list of ids of hidden neurons
        :param output_neuron_ids: list of ids of output neurons
        """
        super().__init__()

        self.input_key_to_idx = {k: i for i, k in enumerate(input_neuron_ids)}
        self.n_inputs = len(input_neuron_ids)
        self.hidden_key_to_idx = {k: i for i, k in enumerate(hidden_neuron_ids)}
        self.n_outputs = len(output_neuron_ids)
        self.output_key_to_idx = {k: i for i, k in enumerate(output_neuron_ids)}
        self.n_hidden = len(hidden_neuron_ids)

        # connection parameters
        self.input_to_hidden = ([], [])
        self.hidden_to_hidden = ([], [])
        self.output_to_hidden = ([], [])
        self.input_to_output = ([], [])
        self.hidden_to_output = ([], [])
        self.output_to_output = ([], [])
        for k, v in connections.items():
            if k[1] in hidden_neuron_ids:
                if k[0] in input_neuron_ids:
                    self.input_to_hidden[0].append((self.input_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]))
                    self.input_to_hidden[1].append(v)
                elif k[0] in hidden_neuron_ids:
                    self.hidden_to_hidden[0].append((self.hidden_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]))
                    self.hidden_to_hidden[1].append(v)
                elif k[0] in output_neuron_ids:
                    self.output_to_hidden[0].append((self.output_key_to_idx[k[0]], self.hidden_key_to_idx[k[1]]))
                    self.output_to_hidden[1].append(v)
            elif k[1] in output_neuron_ids:
                if k[0] in input_neuron_ids:
                    self.input_to_output[0].append((self.input_key_to_idx[k[0]], self.output_key_to_idx[k[1]]))
                    self.input_to_output[1].append(v)
                elif k[0] in hidden_neuron_ids:
                    self.hidden_to_output[0].append((self.hidden_key_to_idx[k[0]], self.output_key_to_idx[k[1]]))
                    self.hidden_to_output[1].append(v)
                elif k[0] in output_neuron_ids:
                    self.output_to_output[0].append((self.output_key_to_idx[k[0]], self.output_key_to_idx[k[1]]))
                    self.output_to_output[1].append(v)
        self.input_to_hidden = torch.nn.Parameter(torch.sparse_coo_tensor(torch.as_tensor(self.input_to_hidden[0]).t(), self.input_to_hidden[1], (self.n_inputs, self.n_hidden)).to_dense())
        self.hidden_to_hidden = torch.nn.Parameter(torch.sparse_coo_tensor(torch.as_tensor(self.hidden_to_hidden[0]).t(), self.hidden_to_hidden[1], (self.n_hidden, self.n_hidden)).to_dense())
        self.input_to_output = torch.nn.Parameter(torch.sparse_coo_tensor(torch.as_tensor(self.input_to_output[0]).t(), self.input_to_output[1], (self.n_inputs, self.n_outputs)).to_dense())
        self.hidden_to_output = torch.nn.Parameter(torch.sparse_coo_tensor(torch.as_tensor(self.hidden_to_output[0]).t(), self.hidden_to_output[1], (self.n_hidden, self.n_outputs)).to_dense())
        self.output_to_output = torch.nn.Parameter(torch.sparse_coo_tensor(torch.as_tensor(self.output_to_output[0]).t(), self.output_to_output[1], (self.n_outputs, self.n_outputs)).to_dense())


        # node parameters
        ## biases
        self.hidden_biases = []
        self.output_biases = []
        for k, v in biases.items():
            if k in hidden_neuron_ids:
                self.hidden_biases.append(v)
            elif k in output_neuron_ids:
                self.output_biases.append(v)
        self.hidden_biases = torch.nn.Parameter(torch.as_tensor(self.hidden_biases))
        self.output_biases = torch.nn.Parameter(torch.as_tensor(self.output_biases))
        ## responses
        self.hidden_responses = []
        self.output_responses = []
        for k, v in responses.items():
            if k in hidden_neuron_ids:
                self.hidden_responses.append(v)
            elif k in output_neuron_ids:
                self.output_responses.append(v)
        self.hidden_responses = torch.nn.Parameter(torch.as_tensor(self.hidden_responses))
        self.output_responses = torch.nn.Parameter(torch.as_tensor(self.output_responses))
        ## aggregations
        self.hidden_aggregations = []
        self.output_aggregations = []
        for k, v in aggregations.items():
            if k in hidden_neuron_ids:
                self.hidden_aggregations.append(v)
            elif k in output_neuron_ids:
                self.output_aggregations.append(v)
        ## activations
        self.hidden_activations = []
        self.output_activations = []
        for k, v in activations.items():
            if k in hidden_neuron_ids:
                self.hidden_activations.append(v)
            elif k in output_neuron_ids:
                self.output_activations.append(v)


    def forward(self, inputs):
        """
        :param inputs: tensor of size (batch_size, n_inputs)
        Note: output of a node as follows: activation(bias+(response∗aggregation(inputs)))
        returns: (batch_size, n_outputs)
        """
        batch_size = len(inputs)
        after_pass_hidden_activs = torch.zeros_like(self.hidden_activs)
        after_pass_output_activs = torch.zeros_like(self.output_activs)

        # Step 1: multiply incoming inputs with connection weights
        hidden_incoming_activs = torch.cat([
            self.input_to_hidden * inputs.unsqueeze(-1).repeat(1, 1, self.n_hidden),
            self.hidden_to_hidden * self.hidden_activs.unsqueeze(-1).repeat(1, 1, self.n_hidden),
        ], 1) #batch_size, n_inputs+n_hidden, n_hidden
        output_incoming_activs = torch.cat([
            self.input_to_output * inputs.unsqueeze(-1).repeat(1, 1, self.n_outputs),
            self.hidden_to_output * self.hidden_activs.unsqueeze(-1).repeat(1, 1, self.n_outputs),
            self.output_to_output * self.output_activs.unsqueeze(-1).repeat(1, 1, self.n_outputs),
        ], 1)  # batch_size, n_inputs+n_hidden+n_output, n_hidden

        # Step 2: x = agg(x)
        for hidden_neuron_idx in range(self.n_hidden):
            after_pass_hidden_activs[:, hidden_neuron_idx] = self.hidden_aggregations[hidden_neuron_idx](hidden_incoming_activs[:, :, hidden_neuron_idx])
        for output_neuron_idx in range(self.n_outputs):
            after_pass_output_activs[:, output_neuron_idx] = self.output_aggregations[output_neuron_idx](output_incoming_activs[:, :, output_neuron_idx])

        # Step 3: x = bias + reponse * x
        after_pass_hidden_activs = self.hidden_biases + self.hidden_responses * after_pass_hidden_activs
        after_pass_output_activs = self.output_biases + self.output_responses * after_pass_output_activs

        # Step 3: x = act(x)
        for hidden_neuron_idx in range(self.n_hidden):
            after_pass_hidden_activs[:, hidden_neuron_idx] = self.hidden_activations[hidden_neuron_idx](after_pass_hidden_activs[:, hidden_neuron_idx])
        for output_neuron_idx in range(self.n_outputs):
            after_pass_output_activs[:, output_neuron_idx] = self.output_activations[output_neuron_idx](after_pass_output_activs[:, output_neuron_idx])


        self.hidden_activs = after_pass_hidden_activs
        self.output_activs = after_pass_output_activs

        return after_pass_output_activs


    def activate(self, inputs, n_passes=1):
        """
        :param inputs: tensor of size (batch_size, n_inputs)
        :param n_passes: number of passes in the RNN TODO: currently global passes should it be internal passes as in pytorch-neat by uber research ?
        returns: (batch_size, n_outputs)
        """
        batch_size = inputs.shape[0]
        assert inputs.shape[1] == self.n_inputs

        self.hidden_activs = torch.zeros((batch_size, self.n_hidden))
        self.output_activs = torch.zeros((batch_size, self.n_outputs))
        for _ in range(n_passes):
            outputs = self.forward(inputs)

        return outputs


    def draw_net(self, view=False, filename=None, node_names=None, prune_unused=False, node_colors=None, fmt='svg'):
        if node_names is None:
            node_names = {}

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2'}

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        inputs = set()
        for k in self.input_neuron_ids:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in self.output_neuron_ids :
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightblue'),
                          'fontsize': '9', 'fontcolor': node_colors.get(k, 'blue'),
                          'xlabel': f'{self.activation_functions[str(k)].__name__[:-11]}({self.biases[str(k)]:.1f}+\\n{self.responses[str(k)]:.1f}*{self.aggregation_functions[str(k)].__name__[:-12]}(inputs))'}

            dot.node(name, _attributes=node_attrs)

        if prune_unused:
            connections = set()
            for cg in self.connections.keys():
                connections.add((str_to_tuple_key(cg)[0], str_to_tuple_key(cg)[1]))

            used_nodes = copy.copy(outputs)
            pending = copy.copy(outputs)
            while pending:
                new_pending = set()
                for a, b in connections:
                    if b in pending and a not in used_nodes:
                        new_pending.add(a)
                        used_nodes.add(a)
                pending = new_pending
        else:
            used_nodes = set(self.input_neuron_ids+self.output_neuron_ids+self.hidden_neuron_ids)

        for n in used_nodes:
            if n in inputs or n in outputs:
                continue
            attrs = {'style': 'filled',
                     'fillcolor': node_colors.get(n, 'white'),
                     'fontsize': '9',
                     'xlabel': f'{self.activation_functions[str(n)].__name__[:-11]}({self.biases[str(n)]:.1f}+\\n{self.responses[str(n)]:.1f}*{self.aggregation_functions[str(n)].__name__[:-12]}(inputs))'}
            dot.node(str(n), _attributes=attrs)

        for cg, cg_weight in self.connections.items():
                # if cg.input not in used_nodes or cg.output not in used_nodes:
                #    continue
                input = str_to_tuple_key(cg)[0]
                output = str_to_tuple_key(cg)[1]
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid'
                color = 'green' if cg_weight > 0 else 'red'
                width = str(abs(cg_weight))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'fontcolor': color, 'penwidth': width, 'fontsize': '9', 'label': f'w={cg_weight:.1f}'})

        dot.render(filename, view=view)

        return dot



    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns the RecurrentCPPN()
        """
        genome_config = config.genome_config

        input_neuron_ids = genome_config.input_keys  # list of ids of input neurons
        output_neuron_ids = genome_config.output_keys  # list of ids of output neurons
        hidden_neuron_ids = list(genome.nodes.keys())[
                            len(output_neuron_ids):]  # list of ids of hidden neurons (neither input nor ouput)

        biases = dict.fromkeys(genome.nodes.keys())  # biases = {node_id: nodes[node_id].bias}
        responses = dict.fromkeys(genome.nodes.keys())  # responses = {node_id: nodes[node_id].response}
        activations = dict.fromkeys(genome.nodes.keys())  # activations = {node_id: nodes[node_id].activation}
        aggregations = dict.fromkeys(genome.nodes.keys())  # aggregations = {node_id: nodes[node_id].aggregation}

        for node_id, node in genome.nodes.items():
            biases[node_id] = node.bias
            responses[node_id] = node.response
            activations[node_id] = str_to_activation[node.activation]
            aggregations[node_id] = str_to_aggregation[node.aggregation]

        connections = dict.fromkeys(
            genome.connections.keys())  # connections = {(from_node_id, to_node_id): connections[(from_node_id, to_node_id)].weight}
        for (from_id, to_id), connection in genome.connections.items():
            if connection.enabled:
                connections[(from_id, to_id)] = connection.weight
            else:
                del connections[(from_id, to_id)]

        return RecurrentNetwork(input_neuron_ids, hidden_neuron_ids, output_neuron_ids, biases, responses, activations,
                                aggregations, connections)
'''