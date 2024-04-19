import onnx
import onnxruntime as nxrun
import numpy as np
import torch

if __name__ == '__main__':


    sess = nxrun.InferenceSession("Model_4x19.onnx")
    input_name = sess.get_inputs()[0].name

    input_image = np.random.randn(1, 1, 160, 160, 160).astype(np.float32)

    with torch.no_grad():
        out = sess.run(None, {input_name: input_image})

    result_run_one = out[0]

    with torch.no_grad():
        out = sess.run(None, {input_name: input_image})

    result_run_two = out[0]

    print('Are the two results identical?: ' + str(np.array_equal(result_run_one, result_run_two)))