import os
import sys

if __name__ == '__main__':
    # parsing arguments
    arguments = sys.argv[1:]
    if len(arguments) < 1:
        raise Exception('Please specify dir')
    input_dir = arguments[0]

    file_names = os.listdir(input_dir)
    file_names = filter(lambda fn: '.npz' in fn, file_names)

    for fn in file_names:
        input_file_path = f'{input_dir}\\{fn}'
        os.system(
            f'python transpose_numpy_midi.py {input_file_path} -12 12')
