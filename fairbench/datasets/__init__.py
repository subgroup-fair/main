from .adult import load_adult, load_sparse_adult

def load_dataset(args):
    name = args.dataset.lower()
    if name == "adult":
        data = load_adult(args)
    elif name == "sparse_adult":
        data = load_sparse_adult(args)
    else:
        raise ValueError(name)
    return data
