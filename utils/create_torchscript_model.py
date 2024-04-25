import pickle
import torch
import os

if __name__ == '__main__':

    # load the network weights from the checkpoint
    checkpoint =torch.load(os.path.join('model_files', 'checkpoint_best.pth'), map_location=torch.device('cpu'))

    # load the model architecture
    with open(os.path.join('model_files', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)      

    # load the weights into the model
    model.load_state_dict(checkpoint['network_weights'])
    model.to(torch.device('cuda:1'))
    model.eval()
    model.train(False)

    # sample input size
    input = torch.randn(1, 1, 160, 160, 160).type(torch.float16).to(torch.device('cuda:1'))

    # export the model
    torch_out = torch.onnx.export(model.half(),              # model being run
                                input,                       # model input (or a tuple for multiple inputs)
                                "Model_4x19.onnx", export_params=True) # where to save the model (can be a file or file-like object)