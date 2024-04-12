import glob
import os
import itk
import json
import numpy as np

import preprocessing 
import inferencing 
import postprocessing 

if __name__ == '__main__':

    # Read and set config parameters
    with open('parameters.json', 'r') as f:
        config = json.load(f)

    preprocess_filter = preprocessing.Preprocessing(config)
    inference_filter = inferencing.Inferencing(config)
    postprocess_filter = postprocessing.Postprocessing(config)

    image_filelist = sorted(glob.glob(os.path.join('images', '*.nii.gz')))
    
    for k in range(len(image_filelist)):

        # Get the case name
        case_name = os.path.basename(image_filelist[k]).split('_0000')[0]

        print('Processing case: ' + case_name)

        # Get the image
        im = itk.imread(image_filelist[k])

        # Run preprocessing steps
        patches, num_patches, arr_shape = preprocess_filter.preprocess(im)

        # Run inference steps
        prediction_patches = inference_filter.infer(patches)

        # Run postprocessing steps
        postprocessed = postprocess_filter.postprocess(prediction_patches, num_patches, 
                                                       arr_shape, im)

        # Save the result
        itk.imwrite(postprocessed, os.path.join('predictions', case_name + '.nii.gz'))

