# Adapted from https://github.com/uber-research/PyTorch-NEAT

from functools import reduce
from operator import mul


def sum_aggregation(inputs):
    return sum(inputs)


def product_aggregation(inputs):
    return reduce(mul, inputs, 1)


str_to_aggregation = {
    'sum': sum_aggregation,
    'product': product_aggregation,
}
