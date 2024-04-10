import glob
import os
import itk
import json

import preprocessing 
import inferencing 
import postprocessing 

if __name__ == '__main__':

    # Read and set config parameters
    with open('parameters.json', 'r') as f:
        config = json.load(f)

    preprocessing.set_parameters(config['preprocessing'])
    inferencing.set_parameters(config['inferencing'])
    postprocessing.set_parameters(config['postprocessing'])

    image_filelist = sorted(glob.glob(os.path.join('images', '*.nii.gz')))
    
    for k in range(len(image_filelist)):

        # Get the case name
        case_name = os.path.basename(image_filelist[k]).split('_0000')[0]

        # Get the image
        im = itk.imread(image_filelist[k])

        # Set parameters


        # Run preprocessing steps
        preprocessed = preprocessing.preprocess(im)

        # Run inference steps
        prediction = inferencing.infer(preprocessed)

        # Run postprocessing steps
        postprocessed = postprocessing.postprocess(prediction)

        # Save the result
        itk.imwrite(postprocessed, os.path.join('predictions', case_name + '.nii.gz'))

