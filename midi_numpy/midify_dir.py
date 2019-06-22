import os
import sys

if __name__ == '__main__':
    # parsing arguments
    arguments = sys.argv[1:]
    if len(arguments) < 1:
        raise Exception('Please specify dir')
    input_dir = arguments[0]

    if len(arguments) < 2:
        raise Exception('Invalid output dir')
    output_dir = arguments[1]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_names = os.listdir(input_dir)
    file_names = filter(
        lambda fn: '.csv' in fn or '.npy' in fn or '.npz' in fn, file_names)

    for fn in file_names:
        input_file_path = f'{input_dir}\\{fn}'
        output_file_name = fn.replace('.csv', '.mid').replace(
            '.npy', '.mid').replace('.npz', '.mid').replace(' ', '_')
        output_file_path = f'{output_dir}\\{output_file_name}'
        os.system(
            f'python numpy_to_midi.py {input_file_path} {output_file_path}')
