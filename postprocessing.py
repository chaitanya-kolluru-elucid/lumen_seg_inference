import patchify
import numpy
import itk

class Postprocessing:
    def __init__(self, config):
        self.config = config['postprocessing_config']
        self.common_config = config['common_config']


    