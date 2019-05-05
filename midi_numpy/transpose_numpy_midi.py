import sys
from os import path, mkdir
import numpy as np
from common import debug, read_numpy_midi, save_numpy_midi


def transpose(numpy_midi, steps):
    """
    shifts notes up/down by given step (1 = half-step in musical terms)
    may lose top #steps lowest/highiest notes 
    """
    res = np.zeros_like(numpy_midi)
    fill_value = 0
    if steps > 0:
        res[:, :steps] = fill_value
        res[:, steps:] = numpy_midi[:, :-steps]
    elif steps < 0:
        res[:, steps:] = fill_value
        res[:, :steps] = numpy_midi[:, -steps:]
    else:
        res = numpy_midi
    return res


if __name__ == '__main__':
    # parsing arguments
    arguments = sys.argv[1:]
    if len(arguments) < 1 or (
        '.csv' not in arguments[0] and
        '.npy' not in arguments[0] and
        '.npz' not in arguments[0]
    ):
        raise Exception('Please specify valid file path')
    input_path = arguments[0]

    if len(arguments) < 2:
        raise Exception('Invalid output path')
    try:
        step_range_min = int(arguments[1])
    except ValueError:
        raise Exception('Please specify valid range')

    if len(arguments) < 3:
        raise Exception('Invalid output path')
    try:
        step_range_max = int(arguments[2])
    except ValueError:
        raise Exception('Please specify valid range')

    input_numpy = read_numpy_midi(input_path)
    file_basename = path.basename(input_path)
    output_dir = path.dirname(path.abspath(input_path)) + '\\transposed\\'
    if not path.exists(output_dir):
        mkdir(output_dir)

    for i in range(step_range_min, step_range_max + 1):
        new_filename = file_basename.replace('.', f'_tr_{str(i)}.')
        output_path = f'{output_dir}{new_filename}'
        transposed = transpose(input_numpy, i)
        save_numpy_midi(output_path, transposed)
