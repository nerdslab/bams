import abc
import os
import pickle
import logging

from torch.utils.data import Dataset


class CachedDataset(Dataset, abc.ABC):
    r"""A simple dataset class that caches processed data. This is useful for
    large datasets that require a lot of processing to be done before they can be
    used. The processed data is cached in a pickle file.

    This class is meant to be subclassed. The subclass should implement the
    `process` method, which should return a dictionary of data to be cached.
    
    Args:
        cache_path (str): Path to the cache file.
        cache (bool): Whether to cache the processed data and/or use the cached data.
    """
    def __init__(self, cache_path, cache=True):
        self.cache_path = cache_path

        if not os.path.exists(self.cache_path) or not cache:
            data = self.process()
            if cache:
                self.save_processed_data(data)
        else:
            data = self.load_from_processed()

        self.__dict__.update(data)

    @staticmethod
    def cache_is_available(cache_path):
        return os.path.exists(cache_path)

    def load_from_processed(self):
        logging.warn(
            "Loading processed data from {}. If raw data or processing scripts changed,"
            " please delete this file to force the data to be re-processed, or use"
            " `cache=False`".format(self.cache_path)
        )
        with open(self.cache_path, "rb") as fp:
            data = pickle.load(fp)["data"]
        return data

    def save_processed_data(self, data):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as output:
            pickle.dump({"data": data}, output)
        logging.info("Processed data was saved to {}.".format(self.cache_path))

    @abc.abstractmethod
    def process(self):
        ...
