# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import json
import math

import numpy as np

import torch
from torch.utils.dlpack import from_dlpack
from torch.utils.dlpack import to_dlpack

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "in_house_model_preprocessing_output"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        self.preprocessing_config = {
            "mean": 427.54730224609375,
            "percentile_00_5": 81.0,
            "percentile_99_5": 823.0,
            "std": 139.19976806640625
        }

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        logger = pb_utils.Logger
        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_1 = pb_utils.get_input_tensor_by_name(
                request, "in_house_model_preprocessing_input"
            )
            request_id = request.request_id()
            image_out = None
            if pb_utils.Tensor.is_cpu(in_1):
                in_image = in_1.as_numpy()
                logger.log_info(f"In {__file__}:execute(), request_id = {request_id}, input is on CPU, image mean intensity = {np.mean(in_image)}")
                image_proc = np.clip(in_image, self.preprocessing_config['percentile_00_5'], self.preprocessing_config['percentile_99_5'])
                image_proc = (image_proc - self.preprocessing_config['mean']) / self.preprocessing_config['std']
                # Add an extra axis in front for batch dimension
                image_proc = image_proc[np.newaxis, :]
                # Convert numpy to PyTorch tensor and put on GPU
                image_out = torch.from_numpy(image_proc).to(device='cuda')
            else:
                logger.log_info(f"In {__file__}:execute(), request_id = {request_id}, input is on GPU")
                # convert a Python backend tensor to a PyTorch tensor without making any copies
                in_1_pytorch_tensor = from_dlpack(in_1.to_dlpack())
                image_proc = torch.clip(in_1_pytorch_tensor, self.preprocessing_config['percentile_00_5'], self.preprocessing_config['percentile_99_5'])
                image_proc = torch.div(torch.sub(image_proc, self.preprocessing_config['mean']), self.preprocessing_config['std'])
                # Add an extra axis in front for batch dimension
                image_out = torch.unsqueeze(image_proc, 0)

            # convert a PyTorch tensor to a Python backend tensor
            out_tensor_0 = pb_utils.Tensor.from_dlpack("in_house_model_preprocessing_output", to_dlpack(image_out))
            # out_tensor_0 = pb_utils.Tensor("in_house_model_preprocessing_output", image_proc.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
