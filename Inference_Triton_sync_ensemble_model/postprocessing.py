import itk.itkLabelShapeKeepNObjectsImageFilterPython
import numpy as np
import itk
from utils.logger_config import logger

class Postprocessing:
    def __init__(self, config):
        self.config = config['postprocessing_config']
        self.common_config = config['common_config']

        # Create a logger object
        self.logger = logger.getLogger('Postprocessing')

    def postprocess(self, prediction, slicer_to_revert_padding, original_im):

        prediction = prediction[tuple(slicer_to_revert_padding)]

        im = itk.GetImageFromArray(prediction)
        im.SetDirection(original_im.GetDirection())
        im.SetOrigin(original_im.GetOrigin())
        im.SetSpacing(self.common_config['target_spacing'])

        im = self.resample_image(im, original_im)

        if self.config['aorta_keep_largest']:
            im = self.keep_largest_cc_for_aorta(im)

        return im

    def keep_largest_cc_for_aorta(self, im):

        # Threshold the image to isolate the class with value equal to 3
        thresholdFilter = itk.BinaryThresholdImageFilter.New(im)
        thresholdFilter.SetLowerThreshold(self.config['aorta_class_value'])
        thresholdFilter.SetUpperThreshold(self.config['aorta_class_value'])
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        aorta = thresholdFilter.GetOutput()

        # Create connected component filter
        InputImageType = itk.Image[itk.UC, 3]
        OutputImageType = itk.Image[itk.SS, 3]
        connected = itk.ConnectedComponentImageFilter[InputImageType, OutputImageType].New(aorta)
        connected.Update()

        # Print number of objects
        self.logger.info("Number of connected objects in aorta mask: " + str(connected.GetObjectCount()))

        # Create label shape keep N objects filter
        labelShapeKeepNObjectsImageFilter = itk.LabelShapeKeepNObjectsImageFilter[OutputImageType].New()
        labelShapeKeepNObjectsImageFilter.SetInput(connected.GetOutput())
        labelShapeKeepNObjectsImageFilter.SetBackgroundValue(0)
        labelShapeKeepNObjectsImageFilter.SetNumberOfObjects(1)
        labelShapeKeepNObjectsImageFilter.SetAttribute(100)

        # Create rescale filter
        RescaleFilterType = itk.RescaleIntensityImageFilter[OutputImageType, InputImageType]
        rescaleFilter = RescaleFilterType.New()
        rescaleFilter.SetOutputMinimum(0)
        rescaleFilter.SetOutputMaximum(self.config['aorta_class_value'])
        rescaleFilter.SetInput(labelShapeKeepNObjectsImageFilter.GetOutput())

        # Use the mask to set corresponding pixels to zero
        maskFilter = itk.MaskImageFilter[InputImageType, InputImageType, InputImageType].New()
        maskFilter.SetInput(im)
        maskFilter.SetMaskImage(aorta)
        maskFilter.SetMaskingValue(1)
        maskFilter.SetOutsideValue(0)
        maskFilter.Update()

        addFilter = itk.AddImageFilter[InputImageType, InputImageType, InputImageType].New()
        addFilter.SetInput1(maskFilter.GetOutput())
        addFilter.SetInput2(rescaleFilter.GetOutput())
        addFilter.Update()

        return addFilter.GetOutput()

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