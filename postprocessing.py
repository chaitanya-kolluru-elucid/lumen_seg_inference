import numpy as np
import itk
from utils.logger_config import logger
import datetime
import os
import json

class Postprocessing:
    def __init__(self, config):
        self.config = config['postprocessing_config']
        self.common_config = config['common_config']

        if not os.path.exists(self.config['probs_save_dir']):
            os.makedirs(self.config['probs_save_dir'])

        # Create a logger object
        self.logger = logger.getLogger('Postprocessing')

    def postprocess(self, prediction, slicer_to_revert_padding, original_im, case_name):

        prediction = prediction[tuple(slicer_to_revert_padding)]

        im = itk.GetImageFromArray(prediction)
        im.SetDirection(original_im.GetDirection())
        im.SetOrigin(original_im.GetOrigin())
        im.SetSpacing(self.common_config['target_spacing'])

        im = self.resample_image(im, original_im)

        if self.config['aorta_keep_largest']:

            start = datetime.datetime.now()

            im = self.keep_largest_cc_for_aorta(im)

            self.logger.info('Keeping largest connected aorta component took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

        if self.config['ensure_connectivity_lumen_calc_aorta']:

            start = datetime.datetime.now()

            im = self.ensure_connectivity_lumen_calc_aorta(im)

            self.logger.info('Ensuring connectivity for lumen, calc, and aorta took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

        if self.config['find_root_node_coordinates']:

            start = datetime.datetime.now()

            root_mask_arr, reoriented_im, aorta = self.get_root_mask_array(im)
            root_coordinates = self.get_root_coordinates(root_mask_arr, reoriented_im, aorta)

            if self.config['root_coordinates_dir'] is not None:

                if not os.path.exists(self.config['root_coordinates_dir']):
                    os.makedirs(self.config['root_coordinates_dir'])

                self.write_root_coordinates_to_file(root_coordinates, case_name)
            else:
                self.logger.warning('Find root coordinates flag set to true, but root coordinates directory left empty.')


            self.logger.info('Finding root node coordinates took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

        if self.config['split_lca_rca_and_match_rsip']:

            start = datetime.datetime.now()

            im = self.split_lca_rca(im, root_coordinates)
            im = self.change_labels_to_match_rsip(im)

            self.logger.info('Split LCA/RCA and changing labels to match RSIP took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

        return im
    
    def change_labels_to_match_rsip(self, im):

        relabel_filter = itk.ChangeLabelImageFilter[type(im), type(im)].New()
        relabel_filter.SetInput(im)

        # RSIP convention: 1 - aorta, 2, 3 - left/right lumen, 5,6 - left/right calc
        change_map = {4:2, 5:3, 6:5, 7:6, 3:1}
        relabel_filter.SetChangeMap(change_map)   

        relabel_filter.Update()
    
    def split_lca_rca(self, im, root_coordinates):
        thresholdFilter = itk.BinaryThresholdImageFilter.New(im)
        thresholdFilter.SetLowerThreshold(self.config['lumen_class_value'])
        thresholdFilter.SetUpperThreshold(self.config['lumen_class_value'])
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()
        lumen = thresholdFilter.GetOutput()

        thresholdFilter.SetLowerThreshold(self.config['calc_class_value'])
        thresholdFilter.SetUpperThreshold(self.config['calc_class_value'])
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()
        calc = thresholdFilter.GetOutput()

        addFilter = itk.AddImageFilter[type(lumen), type(calc), type(calc)].New()
        addFilter.SetInput1(lumen)
        addFilter.SetInput2(calc)
        addFilter.Update()
        lumen_and_calc = addFilter.GetOutput()

        # Find the connected lumen and calc component in which the ostia points are inside
        # Label them appropriately as left and right
        ccFilter = itk.ConnectedComponentImageFilter.New(lumen_and_calc)
        ccFilter.Update()
        lumen_and_calc_labels = ccFilter.GetOutput()
        lumen_and_calc_labels_arr = itk.GetArrayFromImage(lumen_and_calc_labels)
        left_and_right_labels_arr = np.zeros_like(lumen_and_calc_labels_arr)

        if root_coordinates["LeftOstium"] is not None:
            current_point = root_coordinates["LeftOstium"]
            left_label_value = lumen_and_calc_labels.GetPixel(lumen_and_calc_labels.TransformPhysicalPointToIndex(current_point))
            left_and_right_labels_arr[lumen_and_calc_labels_arr == left_label_value] = 1

        if root_coordinates["RightOstium"] is not None:
            current_point = root_coordinates["RightOstium"]
            right_label_value = lumen_and_calc_labels.GetPixel(lumen_and_calc_labels.TransformPhysicalPointToIndex(current_point))
            left_and_right_labels_arr[lumen_and_calc_labels_arr == right_label_value] = 2
        
        lumen_arr = itk.GetArrayFromImage(lumen)
        lumen_arr[left_and_right_labels_arr == 1] = 4
        lumen_arr[left_and_right_labels_arr == 2] = 5

        calc_arr = itk.GetArrayFromImage(calc)
        calc_arr[left_and_right_labels_arr == 1] = 6
        calc_arr[left_and_right_labels_arr == 2] = 7

        # Sanity check to remove any lumen/calc that is not left/right
        lumen_arr[lumen_arr == 1] = 0
        calc_arr[calc_arr == 1] = 0

        thresholdFilter.SetLowerThreshold(self.config['aorta_class_value'])
        thresholdFilter.SetUpperThreshold(self.config['aorta_class_value'])
        thresholdFilter.SetInsideValue(3)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()
        aorta = thresholdFilter.GetOutput()
        aorta_arr = itk.GetArrayFromImage(aorta)

        # Left lumen - 4, Right lumen - 5, Left calc - 6, Right calc - 7, aorta - 3
        combined_mask_arr = lumen_arr + calc_arr + aorta_arr
        combined_mask_arr = combined_mask_arr.astype(np.uint8)
        combined_mask = itk.GetImageFromArray(combined_mask_arr)
        combined_mask.SetSpacing(im.GetSpacing())
        combined_mask.SetDirection(im.GetDirection())
        combined_mask.SetOrigin(im.GetOrigin())

        return combined_mask            
    
    def write_root_coordinates_to_file(self, root_coordinates, case_name):

        with open(os.path.join(self.config['root_coordinates_dir'], case_name + '_root_coords.json'), 'w') as f:
            json.dump(root_coordinates, f, indent = 2, sort_keys = True)
    
    def get_root_coordinates(self, root_mask_arr, reoriented_im, aorta):

        # Keep the two largest connected components in the array
        # Find the centroid, find which is right/left based on anterior/posterior

        root_im = itk.GetImageFromArray(root_mask_arr.astype(np.uint8))
        root_im.SetSpacing(reoriented_im.GetSpacing())
        root_im.SetDirection(reoriented_im.GetDirection())
        root_im.SetOrigin(reoriented_im.GetOrigin())

        ccFilter = itk.ConnectedComponentImageFilter.New(root_im)
        ccFilter.Update()
        labels = ccFilter.GetOutput()

        labelShapeStatisticsFilter = itk.LabelImageToShapeLabelMapFilter.IUC3LM3.New(labels.astype(itk.UC))
        labelShapeStatisticsFilter.SetBackgroundValue(0)
        labelShapeStatisticsFilter.SetComputeOrientedBoundingBox(False)
        labelShapeStatisticsFilter.SetComputePerimeter(False)
        labelShapeStatisticsFilter.SetComputeFeretDiameter(False)
        labelShapeStatisticsFilter.Update()
        labelShapeStatistics = labelShapeStatisticsFilter.GetOutput()

        num_labels = labelShapeStatistics.GetNumberOfLabelObjects()
        self.logger.info('Found ' + str(num_labels) + ' connected components in the root mask, will use largest two.')

        match num_labels:

            case 0:
                return dict()
            
            case 1:
                centroids = []
                centroids.append(np.array(labelShapeStatistics.GetCentroid(1)))

                ccFilter = itk.ConnectedComponentImageFilter.New(aorta)
                ccFilter.Update()

                # Ensure only one connected aorta component
                labelShapeKeepNObjectsImageFilter = itk.LabelShapeKeepNObjectsImageFilter[itk.Image[itk.UC, 3]].New()
                labelShapeKeepNObjectsImageFilter.SetInput(ccFilter.GetOutput().astype(itk.UC))
                labelShapeKeepNObjectsImageFilter.SetBackgroundValue(0)
                labelShapeKeepNObjectsImageFilter.SetNumberOfObjects(1)
                labelShapeKeepNObjectsImageFilter.SetAttribute(100)
                labelShapeKeepNObjectsImageFilter.Update()
                aorta_labels = labelShapeKeepNObjectsImageFilter.GetOutput()

                aortaLabelShapeStatisticsFilter = itk.LabelImageToShapeLabelMapFilter.IUC3LM3.New(aorta_labels.astype(itk.UC))
                aortaLabelShapeStatisticsFilter.SetBackgroundValue(0)
                aortaLabelShapeStatisticsFilter.SetComputeOrientedBoundingBox(False)
                aortaLabelShapeStatisticsFilter.SetComputePerimeter(False)
                aortaLabelShapeStatisticsFilter.SetComputeFeretDiameter(False)
                aortaLabelShapeStatisticsFilter.Update()
                aortaLabelShapeStatistics = aortaLabelShapeStatisticsFilter.GetOutput()

                aorta_centroid = np.array(aortaLabelShapeStatistics.GetNthLabelObject(0).GetCentroid())

                if aorta_centroid[0][1] >= centroids[0][1]:
                    root_coordinates['RightOstium'] = list(centroids[0])
                    root_coordinates['LeftOstium'] = []
                else:
                    root_coordinates['LeftOstium'] = list(centroids[0])
                    root_coordinates['RightOstium'] = []

                return root_coordinates

            case _:

                # Get the indices and sizes of connected components
                labels_sizes = [(labelShapeStatistics.GetNthLabelObject(label_id).GetNumberOfPixels(), label_id) for label_id in range(0, num_labels)]
                labels_sizes.sort(reverse=True)  # Sort by size in descending order

                # Get the two largest connected components
                largest_labels = [labels_sizes[i][1] for i in range(2)]

                # Calculate centroids of the two largest connected components
                root_coordinates = {}
                centroids = []

                for label_id in largest_labels:
                    centroid = np.array(labelShapeStatistics.GetNthLabelObject(label_id).GetCentroid())
                    centroids.append(centroid)

                if centroids[0][1] >= centroids[1][1]:
                    root_coordinates['LeftOstium'] = list(centroids[0])
                    root_coordinates['RightOstium'] = list(centroids[1])

                else:
                    root_coordinates['LeftOstium'] = list(centroids[1])
                    root_coordinates['RightOstium'] = list(centroids[0])

                return root_coordinates
            
    def get_root_mask_array(self, im):

        # Orient the image to ITK_COORDINATE_ORIENTATION_RIP
        itk_so_enums = itk.SpatialOrientationEnums 
        itk_rai = itk_so_enums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAI

        # This allows us to find which positions are anatomically anterior/posterior based on indices
        orientFilter = itk.OrientImageFilter.New(im)
        orientFilter.UseImageDirectionOn()
        orientFilter.SetDesiredCoordinateOrientation(itk_rai)
        orientFilter.Update()

        reoriented_im = orientFilter.GetOutput()

        # Find a maximum of two root nodes in the lumen + calc mask, touching the aorta
        # If only one, use the aorta mask to find if it is left or right
        InputImageType = itk.Image[itk.UC, 3]

        # Get the aorta and lumen masks individually
        aortaThresholdFilter = itk.BinaryThresholdImageFilter.New(reoriented_im)
        aortaThresholdFilter.SetLowerThreshold(self.config['aorta_class_value'])
        aortaThresholdFilter.SetUpperThreshold(self.config['aorta_class_value'])
        aortaThresholdFilter.SetInsideValue(1)
        aortaThresholdFilter.SetOutsideValue(0)
        aortaThresholdFilter.Update()
        aorta = aortaThresholdFilter.GetOutput()

        lumenThresholdFilter = itk.BinaryThresholdImageFilter.New(reoriented_im)
        lumenThresholdFilter.SetLowerThreshold(self.config['lumen_class_value'])
        lumenThresholdFilter.SetUpperThreshold(self.config['lumen_class_value'])
        lumenThresholdFilter.SetInsideValue(1)
        lumenThresholdFilter.SetOutsideValue(0)
        lumenThresholdFilter.Update()
        lumen = lumenThresholdFilter.GetOutput()

        calcThresholdFilter = itk.BinaryThresholdImageFilter.New(reoriented_im)
        calcThresholdFilter.SetLowerThreshold(self.config['calc_class_value'])
        calcThresholdFilter.SetUpperThreshold(self.config['calc_class_value'])
        calcThresholdFilter.SetInsideValue(1)
        calcThresholdFilter.SetOutsideValue(0)
        calcThresholdFilter.Update()
        calc = calcThresholdFilter.GetOutput()

        addFilter = itk.AddImageFilter[type(lumen), type(calc), type(calc)].New()
        addFilter.SetInput1(lumen)
        addFilter.SetInput2(calc)
        addFilter.Update()
        lumen_and_calc = addFilter.GetOutput()

        # Dilate the aorta by one voxel
        StructuringElementType = itk.FlatStructuringElement[3]
        structuringElement = StructuringElementType.Ball(1)

        DilateFilterType = itk.BinaryDilateImageFilter[InputImageType, InputImageType, StructuringElementType].New()
        dilateFilter = DilateFilterType.New()
        dilateFilter.SetInput(aorta)
        dilateFilter.SetKernel(structuringElement)
        dilateFilter.SetForegroundValue(1)
        dilateFilter.Update()
        aorta_dilated = dilateFilter.GetOutput()

        aorta_dilated_arr = itk.GetArrayFromImage(aorta_dilated)
        lumen_and_calc_arr = itk.GetArrayFromImage(lumen_and_calc)

        # Get the mask from which root coordinates should be picked
        root_mask_arr = np.zeros_like(aorta_dilated_arr)
        root_mask_arr[(aorta_dilated_arr == 1) & (lumen_and_calc_arr == 1)] = 1

        return root_mask_arr, reoriented_im, aorta

    def ensure_connectivity_lumen_calc_aorta(self, im):

        # Given that we have a singly connected aorta at this point
        # we now want to ensure lumen + calc + aorta mask is all connected and just one object

        # TODO: Assumes that the labels are consecutive numbers, else threshold logic below will fail
        minimum_label = np.min([self.config['aorta_class_value'], self.config['lumen_class_value'], self.config['calc_class_value']])
        maximum_label = np.max([self.config['aorta_class_value'], self.config['lumen_class_value'], self.config['calc_class_value']])
        arr = itk.GetArrayFromImage(im)

        thresholdFilter = itk.BinaryThresholdImageFilter.New(im)
        thresholdFilter.SetLowerThreshold(minimum_label)
        thresholdFilter.SetUpperThreshold(maximum_label)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()

        ccFilter = itk.ConnectedComponentImageFilter.New(thresholdFilter.GetOutput())
        ccFilter.Update()
        combined_mask_im = ccFilter.GetOutput()

        OutputImageType = itk.Image[itk.UC, 3]
        labelShapeKeepNObjectsImageFilter = itk.LabelShapeKeepNObjectsImageFilter[OutputImageType].New()
        labelShapeKeepNObjectsImageFilter.SetInput(combined_mask_im.astype(itk.UC))
        labelShapeKeepNObjectsImageFilter.SetBackgroundValue(0)
        labelShapeKeepNObjectsImageFilter.SetNumberOfObjects(1)
        labelShapeKeepNObjectsImageFilter.SetAttribute(100)
        labelShapeKeepNObjectsImageFilter.Update()
        largest_component_combined_mask = labelShapeKeepNObjectsImageFilter.GetOutput()

        largest_component_combined_mask_arr = itk.GetArrayFromImage(largest_component_combined_mask)
        updated_arr = np.zeros_like(arr)
        updated_arr[largest_component_combined_mask_arr != 0] = arr[largest_component_combined_mask_arr != 0]

        output_im = itk.GetImageFromArray(updated_arr)
        output_im.SetSpacing(im.GetSpacing())
        output_im.SetDirection(im.GetDirection())
        output_im.SetOrigin(im.GetOrigin())

        return output_im

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
    
    def postprocess_and_save_probabilities(self, prediction_probabilities, slicer_to_revert_padding, im, case_name):

        if self.common_config['save_probabilities']:
            start = datetime.datetime.now()

            for k in range(self.common_config['num_output_labels']):
                class_name = self.config['class_names'][k]

                current_channel_prediction_probabilities = prediction_probabilities[k, :, :, :]
                current_channel_prediction_probabilities = current_channel_prediction_probabilities[tuple(slicer_to_revert_padding)]

                image_to_save = itk.GetImageFromArray(current_channel_prediction_probabilities.astype(np.float32))
                image_to_save.SetDirection(im.GetDirection())
                image_to_save.SetOrigin(im.GetOrigin())
                image_to_save.SetSpacing(self.common_config['target_spacing'])

                image_to_save = self.resample_image(image_to_save, im)

                itk.imwrite(image_to_save, os.path.join(self.config['probs_save_dir'], case_name + '_' + class_name + '.nii.gz'))

            self.logger.info('Saving inference probabilities took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')

    