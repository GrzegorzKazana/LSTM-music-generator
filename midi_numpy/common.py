MSECS_PER_FRAME = 10
NUM_NOTES = 128
NUM_VELOCITY = 128
DEFAULT_BPM = 120


def compose(*functions):
    """
    composes functions left to right
    """
    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg
    return inner


def debug(messages):
    if 'numpy' in str(type(messages)):
        print(messages.shape)
        print(messages)
    else:
        for msg in messages:
            print(msg)
        print(len(messages))
    return messages


def read_numpy_midi(input_path):
    import numpy as np
    from scipy import sparse
    if 'csv' in input_path:
        return np.loadtxt(input_path, delimiter=",", dtype=np.int32)
    elif 'npy' in input_path:
        return np.load(input_path).astype(np.int32)
    elif 'npz' in input_path:
        sparse_numpy = sparse.load_npz(input_path)
        return sparse_numpy.toarray().astype(np.int32)


def save_numpy_midi(output_path, res):
    import numpy as np
    from scipy import sparse
    if '.csv' in output_path:
        np.savetxt(output_path, res, delimiter=",", fmt='%i')
        print(f'Saved to: {output_path}')
    elif '.npy' in output_path:
        np.save(output_path, res.astype(np.int8))
        print(f'Saved to: {output_path}')
    elif '.npz' in output_path:
        res_sparse = sparse.coo_matrix(res)
        sparsity_factor = 1 - (res_sparse.getnnz() / res.size)
        sparse.save_npz(output_path, res_sparse)
        print(
            f'Saved to: {output_path}, {int(100 * sparsity_factor)}% sparsity')
