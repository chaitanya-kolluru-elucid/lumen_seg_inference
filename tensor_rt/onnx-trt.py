import numpy as np
import os
import onnx
import tensorrt as trt

def main():

    onnx_model = onnx.load("inhouse_simplified.onnx")
    
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