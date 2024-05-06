#import torch_tensorrt
import tensorrt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.onnx
import pickle
import os
import onnx
from onnxsim import simplify
import tensorrt as trt

def main():
    
    checkpoint =torch.load(os.path.join('model_files', 'checkpoint_best.pth'),map_location=torch.device('cpu'))
    with open(os.path.join('model_files', 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
    model.load_state_dict(checkpoint['network_weights'])
    model.to(torch.device('cuda:0'))
    model.eval()    
   
    #input = torch.randn(1, 1, 160, 160, 160).cuda()

    with torch.no_grad():
        input_names =['input']
        output_names=['output']
        #dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        # A dummy input is needed to generate the ONNX model
        dummy_input = torch.randn(1, 1, 160, 160, 160).cuda()

        # A example output needed to tell ONNX export how to properly define output
        #example_output = torch.randn(1*160*160*160, 4)

        torch.onnx.export(model, dummy_input, "inhouse.onnx", verbose=True,  export_params=True, training=torch.onnx.TrainingMode.EVAL,
                    do_constant_folding=False, input_names=input_names, output_names=output_names)
        
    """  torch_out = torch.onnx._export(model,              # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               "inhouse.onnx", export_params=True) # store the trained parameter weights inside the model file  """


    onnx_model = onnx.load("inhouse.onnx")
    simplified_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated."
    onnx.save(simplified_model, 'inhouse_simplified.onnx')
    
    #tensorrt conversion start here
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.max_workspace_size = 1<<30
    profile = builder.create_optimization_profile()

    profile.set_shape('input', [1, 1, 160, 160, 160], [1, 1, 160, 160, 160], [1, 1, 160, 160, 160])
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)

    with open('trt.engine', mode='wb') as f:
        f.write(bytearray(engine))#f.write(bytearray(engine.serialize()))
        print("generating file done!")
    
if __name__ == '__main__':   
    main()
