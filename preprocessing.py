from patchify import patchify
import numpy as np
import itk

class Preprocessing:
    def __init__(self, config):
        self.config = config['preprocessing_config']
        self.common_config = config['common_config']

    def preprocess(self, im):

        im = self.resample_image(im)
        arr = itk.GetArrayFromImage(im)
        arr = arr.astype(np.float32)
        arr = self.normalize_intensity(arr)

        patch_step_size = tuple([int(ps * por) for ps, por in 
                                 zip(self.common_config['patch_size'], 
                                     self.common_config['patch_step_size_ratio'])])
        
        # TODO: Assuming same step size in all three dimensions
        patch_step_size = patch_step_size[0]

        patches = patchify(arr, tuple(self.common_config['patch_size']), step = patch_step_size)

        num_patches = [patches.shape[0], patches.shape[1], patches.shape[2]]

        patches = patches.reshape(-1, self.common_config['patch_size'][0],
                                  self.common_config['patch_size'][1],
                                  self.common_config['patch_size'][2])

        return patches, num_patches, arr.shape
    
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
        resampler.Update()

        return resampler.GetOutput()
