from unittest import TestCase
import neat
import os
import torch
import torchvision
from torchvision import transforms
from pytorchneat import selfconnectiongenome, rnn, activations, aggregations
from utils import create_image_cppn_input
import matplotlib.pyplot as plt
import math
import time


def delphineat_gauss_activation(z):
    '''Gauss activation as defined by SharpNEAT, which is also as in DelphiNEAT.'''
    return 2 * math.exp(-1 * (z * 2.5)**2) - 1

def delphineat_sigmoid_activation(z):
    '''Sigmoidal activation function as defined in DelphiNEAT'''
    return 2.0 * (1.0 / (1.0 + math.exp(-z*5)))-1


class TestRecurrentNetwork(TestCase):
    def test_pytorchneat_differentiability(self):

        config_path = os.path.join(os.path.dirname(__file__), 'test_neat.cfg')
        neat_config = neat.Config(
            selfconnectiongenome.SelfConnectionGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        neat_config.genome_config.add_activation('delphineat_gauss', delphineat_gauss_activation)
        neat_config.genome_config.add_activation('delphineat_sigmoid', delphineat_sigmoid_activation)

        # create the cppn input (image_height, image_width,num_inputs)
        img_height = 56
        img_width = 56
        cppn_input = create_image_cppn_input((img_height, img_width))
        mnist_dataset = torchvision.datasets.MNIST(root="/home/mayalen/data/pytorch_datasets/mnist/", download=False, train=True)
        upscale_target_tansform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_height, img_width)), transforms.ToTensor()])

        for mnist_target_idx in range(1,21):
            if not os.path.exists('test_differentiability_outputs'):
                os.makedirs('test_differentiability_outputs')
            if not os.path.exists('test_differentiability_outputs/mnist_targets'):
                os.makedirs('test_differentiability_outputs/mnist_targets')
            cur_output_dir = f'test_differentiability_outputs/mnist_targets/target_{mnist_target_idx}'
            if not os.path.exists(cur_output_dir):
                os.makedirs(cur_output_dir)
            target_img = upscale_target_tansform(mnist_dataset.data[mnist_target_idx]).squeeze().float()
            target_img = target_img / target_img.max()
            plt.figure()
            plt.imshow(target_img,cmap='gray')
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(cur_output_dir, 'target.png'))
            plt.close()

            def eval_genomes(genomes, neat_config):
                genomes_train_losses = []
                genomes_train_images = []

                for genome_idx, genome in genomes:
                    t0 = time.time()
                    cppn_net = rnn.RecurrentNetwork.create(genome, neat_config)
                    opt = torch.optim.Adam(cppn_net.parameters(), 1e-2)
                    train_losses = []
                    train_images = []
                    for train_step in range(45):
                        cppn_net_output = cppn_net.activate(cppn_input, 2)
                        cppn_net_output = (1.0 - cppn_net_output.abs()).view(img_height, img_width)
                        loss = (target_img - cppn_net_output).pow(2).sum()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        train_losses.append(loss.item())
                        train_images.append(cppn_net_output.cpu().detach().unsqueeze(0))
                    t1 = time.time()
                    print(f"Training genome {genome_idx} took {t1-t0} secs")

                    genome.fitness = - train_losses[-1]
                    plt.figure()
                    plt.subplot(211)
                    plt.plot(train_losses)
                    plt.subplot(212)
                    plt.imshow(torchvision.utils.make_grid(train_images, 15).permute(1,2,0))
                    plt.tight_layout()
                    plt.savefig(os.path.join(cur_output_dir, f'individual_{genome.key}.png'))
                    plt.close()

                    # rewrite trained values in genome:
                    cppn_net.update_genome(genome)

                genomes_train_losses.append(train_losses)
                genomes_train_images.append(train_images)

                return genomes_train_losses, genomes_train_images


            pop = neat.Population(neat_config)
            stats = neat.StatisticsReporter()
            pop.add_reporter(stats)
            reporter = neat.StdOutReporter(True)
            pop.add_reporter(reporter)

            n_generations = 10
            pop.run(eval_genomes, n_generations)

        return