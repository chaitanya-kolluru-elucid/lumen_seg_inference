import numpy as np
import itk
import torch
import os
import sys
import pickle
import datetime
from scipy.ndimage import gaussian_filter
from utils.gpu_memtrack import MemTracker
from utils.logger_config import logger
from tritonclient.utils import *
import tritonclient.http as httpclient

class Inferencing:
    def __init__(self, config):

        self.model_name = "ensemble_model"
        # self.gpu_mem_tracker = MemTracker()
        self.config = config['inferencing_config']
        self.common_config = config['common_config']

        if self.config['device'] == 'cpu':
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            self.device = torch.device('cpu')

        elif self.config['device'] == 'gpu':
            torch.set_num_threads(self.common_config['torch_num_threads'])
            torch.set_num_interop_threads(1)
            self.device = torch.device('cuda:' + self.config['gpu_to_use'])

        self.checkpoint =torch.load(os.path.join('model_files', 
                                    self.config['model_checkpoint_filename']),
                                    map_location=torch.device('cpu'))
        
        with open(os.path.join('model_files', self.config['model_architecture_filename']), 'rb') as f:
            self.model = pickle.load(f)    

        if self.config['results_device'] == 'cpu':
            self.results_device = torch.device('cpu')

        elif self.config['results_device'] == 'gpu':
            self.results_device = torch.device('cuda:' + self.config['gpu_to_use'])

        if self.config['use_gaussian_smoothing']:
            self.weights = self.compute_gaussian()
        else:
            self.weights = 1

        if self.config['disable_cuda_allocator_caching']:
            os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

        self.use_mirroring = self.config['use_mirroring']
        
        # Create a logger object
        self.logger = logger.getLogger('Inferencing')        
        self.logger.info('Inference will be done on device: ' + self.config['device'])
        self.logger.info('Results will be processed on device: ' + self.config['results_device'])
        self.logger.info('Pytorch cuda caching allocator disable status: ' + str(self.config['disable_cuda_allocator_caching']))
        self.logger.info('Use mirroring as a test time augmentation: ' + str(self.config['use_mirroring']))

    def empty_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            from torch import mps
            mps.empty_cache()
        else:
            pass

    def compute_gaussian(self):
        
        sigma_scale = 1. / 8
        value_scaling_factor = 1.

        tmp = np.zeros(self.common_config['patch_size'])
        center_coords = [i // 2 for i in self.common_config['patch_size']]
        sigmas = [i * sigma_scale for i in self.common_config['patch_size']]
        tmp[tuple(center_coords)] = 1

        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

        gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
        gaussian_importance_map = gaussian_importance_map.to(device=self.results_device, dtype=torch.float16)
        
        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        mask = gaussian_importance_map == 0
        gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
        return gaussian_importance_map

    def infer(self, arr, slicers):     

        self.empty_cache()

        start = datetime.datetime.now()
        predictions = np.zeros(arr.shape, dtype=np.uint8)
        # self.gpu_mem_tracker.track()

        # with torch.inference_mode():
        request_count = len(slicers)
        self.logger.info(f"Number of requests = {request_count}")
        try:
            # Need to specify large enough concurrency to issue all the
            # inference requests to the server in parallel, let's try 20
            triton_client = httpclient.InferenceServerClient(url="localhost:8000", concurrency=20)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        async_requests = []
        sent_count = 0
        for sl in slicers:
            startpatch = datetime.datetime.now()
            patch = arr[sl]
            # Add an extra axis in front for batch dimension in Triton
            patch = patch[np.newaxis, :]
            endtime1 = datetime.datetime.now()
            inputs = httpclient.InferInput("input_image", patch.shape, datatype="FP16")
            inputs.set_data_from_numpy(patch)
            outputs = httpclient.InferRequestedOutput("postprocessed_output")
            # Asynchronous inference call.
            sent_count += 1
            async_requests.append({"sl_patch": sl,
                                   "infer": triton_client.async_infer(model_name=self.model_name, request_id=str(sent_count), inputs=[inputs], outputs=[outputs])
                                  }
                                 )
            endtime2 = datetime.datetime.now()
            self.logger.info('Request ID: '  + str(sent_count))
            self.logger.info('Step1-InferInput ' + str((endtime1 - startpatch).microseconds/1000.0) + ' ms.')
            self.logger.info('Step2-Inference '  + str((endtime2 - endtime1).microseconds/1000.0) + ' ms.')
        
        for async_request in async_requests:
            # Get the result from the initiated asynchronous inference request.
            # Note the call will block till the server responds.
            resp_starttime = datetime.datetime.now()
            response = async_request["infer"].get_result()
            self.logger.info(f"Response ID: {response.get_response()['id']}")
            predictions[async_request["sl_patch"]] = response.as_numpy("postprocessed_output")
            resp_endtime = datetime.datetime.now()
            self.logger.info('Step3-Response ' + str((resp_endtime - resp_starttime).microseconds/1000.0) + ' ms.')

        self.logger.info('Total inference took ' + str((datetime.datetime.now() - start).seconds) + ' seconds.')
        predictions = np.array(predictions)

        self.empty_cache()
        # self.gpu_mem_tracker.track()

        return predictions