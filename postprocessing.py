import numpy as np
import itk

class Postprocessing:
    def __init__(self, config):
        self.config = config['postprocessing_config']
        self.common_config = config['common_config']

    def postprocess(self, prediction, slicer_to_revert_padding, original_im):

        prediction = prediction[*slicer_to_revert_padding]

        im = itk.GetImageFromArray(prediction)
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