# My imports
from datasets.common import TranscriptionDataset

# Regular imports
import numpy as np
import os


class MAESTRO(TranscriptionDataset):
    def __init__(self, base_dir):
        super().__init__()

    @staticmethod
    def available_splits():
        # TODO - alternative year splits?
        return ['train', 'validation', 'test']

    @staticmethod
    def download(save_dir):
        # TODO - "flac" option which download flac instead
        pass
