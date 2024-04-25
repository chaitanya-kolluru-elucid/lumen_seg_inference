import glob
import os
import itk
import json
import numpy as np
import datetime

import preprocessing 
import inferencing 
import postprocessing 

from utils.logger_config import logger

if __name__ == '__main__':

    logging_handle = logger.getLogger('__main__')

    # Read and set config parameters
    with open('parameters.json', 'r') as f:
        config = json.load(f)

    preprocess_filter = preprocessing.Preprocessing(config)
    inference_filter = inferencing.Inferencing(config)
    postprocess_filter = postprocessing.Postprocessing(config)

    image_filelist = sorted(glob.glob(os.path.join(config['common_config']['images_dir'], '*.nii.gz')))
    for k in range(len(image_filelist)):

        # Get the case name
        case_name = os.path.basename(image_filelist[k]).split('.nii.gz')[0]

        logging_handle.info('Processing case: ' + case_name)

        start = datetime.datetime.now()

        # Get the image
        im = itk.imread(image_filelist[k])

        # Run preprocessing steps
        arr, slicer_to_revert_padding, slicers = preprocess_filter.preprocess(im)

        # Run inference steps
        prediction = inference_filter.infer(arr, slicers)

        # Run postprocessing steps
        postprocessed = postprocess_filter.postprocess(prediction, slicer_to_revert_padding, im)

        # Save the result
        itk.imwrite(postprocessed, os.path.join(config['common_config']['preds_dir'], case_name + '_pred.nii.gz'))

        logging_handle.info('Processing this case took ' + str((datetime.datetime.now() - start).seconds)+ ' seconds')