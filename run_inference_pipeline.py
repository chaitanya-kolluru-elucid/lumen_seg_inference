import glob
import os
import itk

from preprocessing import preprocess
from inferencing import infer
from postprocessing import postprocess

if __name__ == '__main__':

    image_filelist = sorted(glob.glob(os.path.join('images', '*.nii.gz')))
    
    for k in range(len(image_filelist)):

        case_name = os.path.basename(image_filelist[k]).split('_0000')[0]

        # Get the image
        im = itk.imread(image_filelist[k])

        # Run preprocessing steps
        preprocessed = preprocess(im)

        # Run inference steps
        prediction = infer(preprocessed)

        # Run postprocessing steps
        postprocessed = postprocess(prediction)

        # Save the result
        itk.imwrite(postprocessed, os.path.join('predictions', case_name + '.nii.gz'))

