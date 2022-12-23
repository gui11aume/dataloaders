import itertools
import json
import gzip
import os
import torch


class IterableData(torch.utils.data.IterableDataset):
    """
    Defines the logic for iterable datasets (working over streams of
    data) in parallel multi-processing environments, e.g., multi-GPU.
    """

    @property
    def iterator(self):
        # Extend this class to define the stream.
        raise NotImplementedError

    def __iter__(self):
        # Get worker info if in multi-processing context.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.iterator

        # In multi-processing context, use 'os.environ' to
        # find global worker rank. Then use 'islice' to allocate
        # the items of the stream to the workers.
        process_rk = int(os.environ.get("LOCAL_RANK", 0))
        process_nb = int(os.environ.get("WORLD_SIZE", 1))
        local_worker_rk = worker_info.id
        local_worker_nb = worker_info.num_workers
        # Assume that each process has the same number of local workers.
        worker_rk = process_rk * local_worker_nb + local_worker_rk
        worker_nb = process_nb * local_worker_nb

        return itertools.islice(self.iterator, worker_rk, None, worker_nb)


class DataFromIterator(IterableData):

    def __init__(self, iterator, train=True)
        torch.utils.data.IterableDataset.__init__(self)
        self._iterator = iterator
        self.train = train

    @property
    def iterator(self):
        return self._iterator


class IterableDataFromFile(IterableData):

    def __init__(self, path, train=True, encoding="ascii"):
        torch.utils.data.IterableDataset.__init__(self)
        self.path = path
        self.train = train
        self.encoding = encoding

    def load(self, x):
        raise NotImplementedError

    @property
    def iterator(self):
        # Read the magic number.
        with open(self.data_path, "rb") as f:
            magic_number = f.read(2)
        # If file is gzipped, uncompress it on the fly.
        if magic_number == b'\x1f\x8b':
            iterator = map(
                    lambda line: self.load(line.decode(self.encoding)),
                    gzip.open(self.path)
            )
        else:
            iterator = map(
                    lambda line: self.load(line),
                    open(self.path)
            )
        return iterator


class IterableTextData(IterableDataFromFile):
    def load(self, x):
        return x

class IterableJSONData(IterableDataFromFile):
    def load(self, x):
        return json.loads(x)
