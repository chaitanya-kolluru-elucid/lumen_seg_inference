from patchify import unpatchify
import numpy as np
import itk

class Postprocessing:
    def __init__(self, config):
        self.config = config['postprocessing_config']
        self.common_config = config['common_config']

    def postprocess(self, prediction_patches, num_patches, arr_shape, original_im):

        prediction_patches = prediction_patches.reshape(num_patches[0], num_patches[1], num_patches[2],
                                                        prediction_patches.shape[1], prediction_patches.shape[2],
                                                        prediction_patches.shape[3])
        
        s_h = int(((arr_shape[0] - self.common_config['patch_size'][0] )/ (num_patches[0] - 1)) + 1)
        s_w = int(((arr_shape[1] - self.common_config['patch_size'][1] )/ (num_patches[1] - 1)) + 1)
        s_c = int(((arr_shape[2] - self.common_config['patch_size'][2] )/ (num_patches[2] - 1)) + 1)

        pad_size_h = int(((num_patches[0] - 1) * s_h) + self.common_config['patch_size'][0])
        pad_size_w = int(((num_patches[1] - 1) * s_w) + self.common_config['patch_size'][1])
        pad_size_c = int(((num_patches[2] - 1) * s_c) + self.common_config['patch_size'][2])

        prediction_patches = unpatchify(prediction_patches, (pad_size_h, pad_size_w, pad_size_c)).astype(np.uint8)

        im = itk.GetImageFromArray(prediction_patches)
        im.SetDirection(original_im.GetDirection())
        im.SetOrigin(original_im.GetOrigin())
        im.SetSpacing(self.common_config['target_spacing'])

        im = self.resample_image(im, original_im)
        return im

    def resample_image(self, im, original_im):

        # Create a ResampleImageFilter instance
        resampler = itk.ResampleImageFilter.New(im)

        # Set the input image to be resampled
        resampler.SetInput(im)

        # Set the reference image as the reference space
        resampler.SetReferenceImage(original_im)

        # Set the interpolation method (e.g., linear, nearest neighbor, etc.)
        resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction.New(im))

        resampler.UseReferenceImageOn()
        
        # Perform the resampling
        resampler.Update()

        return resampler.GetOutput()