import math

import numpy as np
from math_function.function import tanh, softmax

input_dimension = 28 * 28
output_dimension = 10

dimension = [input_dimension, output_dimension]
function = [tanh, softmax]

distribution = [
    {"b": [0, 0]},
    {"b": [0, 0], "w": [-math.sqrt(6 / (input_dimension + output_dimension)),
                        math.sqrt(6 / (input_dimension + output_dimension))]}
]





