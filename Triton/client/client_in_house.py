from tritonclient.utils import *
import tritonclient.http as httpclient
import nibabel as nb
import argparse
import numpy as np
import os
import sys
import time
from skimage.transform import resize

model_name = "in_house_model"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="in_house model prediction")
    parser.add_argument(
        "input",
        type=str,
        help="Path to NIFTI file to send for in_house_model prediction",
    )
    args = parser.parse_args()
    in_image = args.input

    if not os.path.exists(in_image):
        print (f'Input file {in_image} does not exist')
        sys.exit(1)

    with httpclient.InferenceServerClient("localhost:8000") as client:
        # load input image to get header and affine
        input_folder = os.path.dirname(in_image)
        input_filename = os.path.basename(in_image)
        in_image_nii = nb.load(in_image)
        image_data = in_image_nii.get_fdata()
        # to be extra sure of not overwriting data:
        temp_data = np.copy(image_data)
        # update data type:
        new_dtype = np.float32
        temp_data = temp_data.astype(new_dtype)

        # resize the image to 160x160x160
        new_data = resize(temp_data, (160, 160, 160))

        # add the batch dimension to the image
        new_data_res = new_data[np.newaxis, ...]

        inputs = httpclient.InferInput("input", new_data_res.shape, datatype="FP32")
        inputs.set_data_from_numpy(new_data_res)

        outputs = httpclient.InferRequestedOutput("output")

        inference_start_time = time.time() * 1000
        response = client.infer(model_name, inputs=[inputs], outputs=[outputs])
        inference_time = time.time() * 1000 - inference_start_time

        print (f'inference time = {inference_time}')

        result = response.as_numpy("output")
        print (f'shape of inference output is: {result.shape}')

        # output from in_house model has dimension (batch#,classification,img_x_dim,img_y_dim,img_z_dim)
        # remove the batch dimension from the output
        result_sq = np.squeeze(result, axis=0)
        print (f'shape of inference output after squeeze is: {result_sq.shape}')

        # extract the indices of the maximum values along axis 0.
        inference_output = np.argmax(result_sq, axis=0)
        print (f'shape of inference output after argmax is: {inference_output.shape}')

        # save the final inference output as NIFTI image
        ni_img = nb.Nifti1Image(inference_output, header=in_image_nii.header, affine=in_image_nii.affine)
        nb.save(ni_img, os.path.join(input_folder,input_filename.split(".nii.gz")[0]+"_pred_Triton.nii.gz"))
