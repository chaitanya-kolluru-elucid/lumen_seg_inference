import numpy as np
import itk
import torch
import os
import pickle
import datetime
from scipy.ndimage import gaussian_filter
from utils.gpu_memtrack import MemTracker
from utils.logger_config import logger

class Inferencing:
    def __init__(self, config):

        self.gpu_mem_tracker = MemTracker()

        self.config = config['inferencing_config']
        self.common_config = config['common_config']

        if self.config['device'] == 'cpu':

            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            self.device = torch.device('cpu')

        elif self.config['device'] == 'gpu':
            torch.set_num_threads(self.common_config['torch_num_threads'])
            torch.set_num_interop_threads(1)
            self.device = torch.device('cuda:' + self.config['gpu_to_use'])

        self.checkpoint =torch.load(os.path.join('model_files', 
                                    self.config['model_checkpoint_filename']),
                                    map_location=torch.device('cpu'))
        
        with open(os.path.join('model_files', self.config['model_architecture_filename']), 'rb') as f:
            self.model = pickle.load(f)    

        if self.config['results_device'] == 'cpu':
            self.results_device = torch.device('cpu')

        elif self.config['results_device'] == 'gpu':
            self.results_device = torch.device('cuda:' + self.config['gpu_to_use'])

        if self.config['use_gaussian_smoothing']:
            self.weights = self.compute_gaussian()
        else:
            self.weights = 1

        if self.config['disable_cuda_allocator_caching']:
            os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

        self.use_mirroring = self.config['use_mirroring']
        
        # Create a logger object
        self.logger = logger.getLogger('Inferencing')        
        self.logger.info('Inference will be done on device: ' + self.config['device'])
        self.logger.info('Results will be processed on device: ' + self.config['results_device'])
        self.logger.info('Pytorch cuda caching allocator disable status: ' + str(self.config['disable_cuda_allocator_caching']))

    def empty_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            from torch import mps
            mps.empty_cache()
        else:
            pass

    def compute_gaussian(self):
        
        sigma_scale = 1. / 8
        value_scaling_factor = 1.

        tmp = np.zeros(self.common_config['patch_size'])
        center_coords = [i // 2 for i in self.common_config['patch_size']]
        sigmas = [i * sigma_scale for i in self.common_config['patch_size']]
        tmp[tuple(center_coords)] = 1

        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

        gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
        gaussian_importance_map = gaussian_importance_map.to(device=self.results_device, dtype=torch.float16)
        
        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        mask = gaussian_importance_map == 0
        gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
        return gaussian_importance_map

    def infer(self, arr, slicers):     

        self.empty_cache()

        start = datetime.datetime.now()

        self.gpu_mem_tracker.track()
        self.model.load_state_dict(self.checkpoint['network_weights'])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.gpu_mem_tracker.track()

        arr_tensor = torch.from_numpy(arr)
        predictions = torch.zeros((self.config['num_output_labels'], *arr.shape[1:]), dtype=torch.half, device=self.results_device)
        n_predictions = torch.zeros(arr.shape[1:], dtype=torch.half, device=self.results_device)
        self.gpu_mem_tracker.track()

        with torch.inference_mode():

            for sl in slicers:
                patch = arr_tensor[sl][None]
                patch = patch.to(self.device)

                prediction = self.model(patch)

                if self.use_mirroring:
                    axes_combinations = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]

                    for axes in axes_combinations:
                        prediction += torch.flip(self.network(torch.flip(patch, axes)), axes)
                    prediction /= (len(axes_combinations) + 1)

                predictions[sl] +=  torch.softmax(prediction.squeeze(0), axis = 0).to(self.results_device)
                n_predictions[sl[1:]] += self.weights

            self.logger.info('Inference took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

            start = datetime.datetime.now()
            predictions /= n_predictions
            del n_predictions
            self.empty_cache()
            self.gpu_mem_tracker.track()
            self.logger.info('Division took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')            
            
            start = datetime.datetime.now()
            predictions = torch.argmax(predictions, axis = 0).cpu()
            predictions = np.array(predictions, dtype=np.uint8)
            self.logger.info('Predictions argmax took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

        self.empty_cache()
        self.gpu_mem_tracker.track()

        return predictions