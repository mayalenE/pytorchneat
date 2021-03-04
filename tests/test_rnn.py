from unittest import TestCase
import neat
import os
import torch
import torchvision
from pytorchneat import selfconnectiongenome, rnn, activations, aggregations
from morphosearch.utils.sampling import set_seed
from morphosearch.utils import cppn_utils
import matplotlib.pyplot as plt
import math


def delphineat_gauss_activation(z):
    '''Gauss activation as defined by SharpNEAT, which is also as in DelphiNEAT.'''
    return 2 * math.exp(-1 * (z * 2.5)**2) - 1

def delphineat_sigmoid_activation(z):
    '''Sigmoidal activation function as defined in DelphiNEAT'''
    return 2.0 * (1.0 / (1.0 + math.exp(-z*5)))-1



class TestRecurrentNetwork(TestCase):

    def test_pytorchneat_vs_pythonneat(self):
        """
        Given the test_neat.cfg, create a number of random networks 
        and test if the neat rnn and the pytorch rnn generate the same output for each
        """

        set_seed(0)

        num_networks = 20
        recurrent_net_passes = 4
        show_plots = False

        # load test configuration
        config_path = os.path.join(os.path.dirname(__file__), 'test_neat.cfg')
        neat_config = neat.Config(
            selfconnectiongenome.SelfConnectionGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )

        # add custom activation functions for python-neat version:
        neat_config.genome_config.add_activation('delphineat_gauss', delphineat_gauss_activation)
        neat_config.genome_config.add_activation('delphineat_sigmoid', delphineat_sigmoid_activation)

        # create the cppn input (image_height, image_width,num_inputs)
        img_height = 50
        img_width = 50
        cppn_input = cppn_utils.create_image_cppn_input((img_height, img_width)) # here input for each pixel location is b=1,x,y,d

        for net_idx in range(num_networks):

            # create a genome
            ## instantiate genome=SelfConnectionGenome(key=net_idx) where key is the identifier for this individual/genome
            genome = neat_config.genome_type(net_idx)
            ## required interface to configure a new genome (itself) based on the given configuration object
            genome.configure_new(neat_config.genome_config)

            # Pytorch-Neat Output
            pytorch_cppn = rnn.RecurrentNetwork.create(genome, neat_config)
            # in pytorch cppn, the activate functions takes an input:
            ## the cppn input of size (N, num_inputs) and activate for each of the N locations
            ## the number of passes in the cppn recurrent network
            pytorch_cppn_output = pytorch_cppn.activate(cppn_input, recurrent_net_passes)
            pytorch_cppn_output = (1.0 - pytorch_cppn_output.abs()).cpu().detach().view(img_height, img_width)

            # Python-Neat Output
            neat_cppn = neat.nn.RecurrentNetwork.create(genome, neat_config)
            # the activate function must be applied at each pixel location
            neat_cppn_output = torch.zeros((img_height*img_width, neat_config.genome_config.num_outputs))
            for idx, input in enumerate(cppn_input):
                neat_cppn.reset()
                for _ in range(recurrent_net_passes):
                    neat_cppn_output[idx, :] = neat_cppn.activate(input)[0]
            neat_cppn_output = (1.0 - neat_cppn_output.abs()).cpu().detach().view(img_height, img_width)

            assert torch.allclose(neat_cppn_output, pytorch_cppn_output) # returns True if two tensors are element-wise equal within a tolerance.
            if show_plots:
                fig = plt.subplots(1,2)
                plt.subplot(121)
                plt.imshow(pytorch_cppn_output, cmap='gray')
                plt.axis("off")
                plt.title("Pytorch-Neat")
                plt.subplot(122)
                plt.imshow(neat_cppn_output, cmap='gray')
                plt.axis("off")
                plt.title("Python-Neat")
                plt.tight_layout()
                plt.show()

        for net_idx in range(num_networks):

            ###########################################################
            # Create a network which has a node that has no intput nodes
            genome = neat_config.genome_type(net_idx)
            genome.configure_new(neat_config.genome_config)

            # change network structure
            trg_node = list(genome.nodes.keys())[-1]
            delete_keys = []
            for key, _ in genome.connections.items():
                if key[1] == trg_node:
                    delete_keys.append(key)
            for key in delete_keys:
                del genome.connections[key]

            # test its outputs
            # Pytorch-Neat Output
            pytorch_cppn = rnn.RecurrentNetwork.create(genome, neat_config)
            pytorch_cppn_output = pytorch_cppn.activate(cppn_input, recurrent_net_passes)
            pytorch_cppn_output = (1.0 - pytorch_cppn_output.abs()).cpu().detach().view(img_height, img_width)

            # Python-Neat Output
            neat_cppn = neat.nn.RecurrentNetwork.create(genome, neat_config)
            neat_cppn_output = torch.zeros((img_height * img_width, neat_config.genome_config.num_outputs))
            for idx, input in enumerate(cppn_input):
                neat_cppn.reset()
                for _ in range(recurrent_net_passes):
                    neat_cppn_output[idx, :] = neat_cppn.activate(input)[0]
            neat_cppn_output = (1.0 - neat_cppn_output.abs()).cpu().detach().view(img_height, img_width)

            assert torch.allclose(neat_cppn_output, pytorch_cppn_output)
            if show_plots:
                fig = plt.subplots(1, 2)
                plt.subplot(121)
                plt.imshow(pytorch_cppn_output, cmap='gray')
                plt.axis("off")
                plt.title("1 node without input connections: Pytorch-Neat")
                plt.subplot(122)
                plt.imshow(neat_cppn_output, cmap='gray')
                plt.axis("off")
                plt.title("Python-Neat")
                plt.tight_layout()
                plt.show()

        for net_idx in range(num_networks):
            ###############################################################
            # Create and test a network without connections
            genome = neat_config.genome_type(net_idx)
            genome.configure_new(neat_config.genome_config)

            # change network structure
            genome.connections.clear()

            # test its outputs
            pytorch_cppn = rnn.RecurrentNetwork.create(genome, neat_config)
            pytorch_cppn_output = pytorch_cppn.activate(cppn_input, recurrent_net_passes)
            pytorch_cppn_output = (1.0 - pytorch_cppn_output.abs()).cpu().detach().view(img_height, img_width)

            neat_cppn = neat.nn.RecurrentNetwork.create(genome, neat_config)
            neat_cppn_output = torch.zeros((img_height * img_width, neat_config.genome_config.num_outputs))
            for idx, input in enumerate(cppn_input):
                neat_cppn.reset()
                for _ in range(recurrent_net_passes):
                    neat_cppn_output[idx, :] = neat_cppn.activate(input)[0]
            neat_cppn_output = (1.0 - neat_cppn_output.abs()).cpu().detach().view(img_height, img_width)

            assert torch.allclose(neat_cppn_output, pytorch_cppn_output)
            if show_plots:
                fig = plt.subplots(1, 2)
                plt.subplot(121)
                plt.imshow(pytorch_cppn_output, cmap='gray')
                plt.axis("off")
                plt.title("No connections at all: Pytorch-Neat")
                plt.subplot(122)
                plt.imshow(neat_cppn_output, cmap='gray')
                plt.axis("off")
                plt.title("Python-Neat")
                plt.tight_layout()
                plt.show()

        return