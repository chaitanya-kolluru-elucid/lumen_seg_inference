# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import numpy as np
import tensorrt as trt
import common_runtime
import cuda
from cuda import cudart

class TensorRTInfer:

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common_runtime.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """
        # Prepare the output data
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        common_runtime.memcpy_host_to_device(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        common_runtime.memcpy_device_to_host(output, self.outputs[0]["allocation"])

        # Process the results
        predictions = np.argmax(output, axis = 1)
        predictions = np.array(predictions, dtype=np.uint8)

        return predictions
    
def main():
    trt.init_libnvinfer_plugins(None,'')
    trt_infer = TensorRTInfer(os.path.join('model_files', 'trt.engine'))
    input = np.random.uniform(0, 1.0, size=(1, 1, 160, 160, 160))
    results= trt_infer.infer(input)
    
if __name__ == '__main__':   
    main()