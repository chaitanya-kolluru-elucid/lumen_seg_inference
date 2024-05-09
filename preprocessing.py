import numpy as np
import itk
from torch.nn import functional as F
import torch
from utils.logger_config import logger
import datetime

class Preprocessing:
    def __init__(self, config):
        self.config = config['preprocessing_config']
        self.common_config = config['common_config']

        # Create a logger object
        self.logger = logger.getLogger('Preprocessing')

    def preprocess(self, im):

        start = datetime.datetime.now()

        im = self.resample_image(im)
        arr = itk.GetArrayFromImage(im)
        arr = arr.astype(np.float16)
        arr = self.normalize_intensity(arr)

        self.logger.info('Reading image and normalization took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')  

        # Pad the array so that the array size is at least the patch size
        start = datetime.datetime.now()
        arr, slicer_to_revert_padding = self.pad_array(arr)
        self.logger.info('Padding array took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')  

        # Get slicers for patches
        start = datetime.datetime.now()
        slicers = self.get_sliding_window_slicers(arr.shape)        
        self.logger.info('Calculating slicers for patches took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')  

        # Add an extra axis in the front for channels
        arr = arr[np.newaxis, :]

        return arr, slicer_to_revert_padding, slicers
    
    def normalize_intensity(self, arr):

        arr = np.clip(arr, self.config['percentile_00_5'], self.config['percentile_99_5'])
        arr = (arr - self.config['mean']) / self.config['std']

        return arr

    def resample_image(self, im):

        # Define the linear interpolator
        interpolator = itk.LinearInterpolateImageFunction.New(im)

        # Create a resampling filter
        resampler = itk.ResampleImageFilter.New(Input=im)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(tuple(self.common_config['target_spacing']))
        resampler.SetSize([int(sz * (spc_old / spc_new)) for sz, spc_old, spc_new in 
                           zip(im.GetLargestPossibleRegion().GetSize(), im.GetSpacing(), 
                               tuple(self.common_config['target_spacing']))])
        resampler.SetOutputDirection(im.GetDirection())
        resampler.SetOutputOrigin(im.GetOrigin())
        resampler.Update()

        return resampler.GetOutput()
    
    def pad_array(self, arr):

        old_shape = np.array(arr.shape)
        patch_size =  self.common_config['patch_size']

        patch_size = [max(patch_size[i], old_shape[i]) for i in range(len(patch_size))]

        difference = patch_size - old_shape
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = [list(i) for i in zip(pad_below, pad_above)]

        if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
            res = np.pad(arr, pad_list, 'constant', kwargs= {'value':0})
        else:
            res = arr

        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer
    
    def get_sliding_window_slicers(self, image_size):

        slicers = []
        
        steps = self.compute_steps_for_sliding_window(image_size, self.common_config['patch_size'],
                                                    self.common_config['patch_step_size_ratio'])
        
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                zip((sx, sy, sz), self.common_config['patch_size'])]]))
        return slicers

    def compute_steps_for_sliding_window(self, image_size, tile_size, tile_step_size):

        target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

        steps = []
        for dim in range(len(tile_size)):
            max_step_value = image_size[dim] - tile_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 0  

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps