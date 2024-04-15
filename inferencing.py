import numpy as np
import itk
import torch
import os
import pickle
import datetime
from scipy.ndimage import gaussian_filter

class Inferencing:
    def __init__(self, config):
        self.config = config['inferencing_config']
        self.common_config = config['common_config']

        if self.config['device'] == 'cpu':

            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            self.device = torch.device('cpu')

        #TODO: For hendrix, use gpu 1 for now
        elif self.config['device'] == 'cuda':
            torch.set_num_threads(self.common_config['torch_num_threads'])
            torch.set_num_interop_threads(1)
            self.device = torch.device('cuda:1')

        self.checkpoint =torch.load(os.path.join('model_files', 
                                    self.config['model_checkpoint_filename']),
                                    map_location=torch.device('cpu'))
        
        with open(os.path.join('model_files', self.config['model_architecture_filename']), 'rb') as f:
            self.model = pickle.load(f)
    
        if self.config['use_gaussian_smoothing']:
            self.weights = self.compute_gaussian()
        else:
            self.weights = 1

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
        gaussian_importance_map = gaussian_importance_map.to(device='cpu', dtype=torch.float16)
        
        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        mask = gaussian_importance_map == 0
        gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
        return gaussian_importance_map

    def infer(self, arr, slicers):     

        self.empty_cache()

        start = datetime.datetime.now()

        self.model.load_state_dict(self.checkpoint['network_weights'])
        self.model = self.model.to(self.device)
        self.model.eval()

        arr_tensor = torch.from_numpy(arr)
        predictions = torch.zeros((self.config['num_output_labels'], *arr.shape[1:]), dtype=torch.half)
        n_predictions = torch.zeros(arr.shape[1:], dtype=torch.half)

        with torch.inference_mode():

            for sl in slicers:
                patch = arr_tensor[sl][None]
                patch = patch.to(self.device)

                predictions[sl] +=  torch.softmax(self.model(patch).squeeze(0), axis = 0).cpu()
                n_predictions[sl[1:]] += self.weights

            print('Inference took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

            start = datetime.datetime.now()
            predictions /= n_predictions
            print('Division took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')            
            
            start = datetime.datetime.now()
            predictions = predictions.to(self.device)
            print('Moving predictions from cpu to gpu took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

            start = datetime.datetime.now()
            predictions = torch.argmax(predictions, axis = 0).cpu()
            predictions = np.array(predictions, dtype=np.uint8)
            print('Predictions argmax took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

        self.empty_cache()
        return predictions