import numpy as np
import itk
import torch
import os
import pickle

class Inferencing:
    def __init__(self, config):
        self.config = config['inferencing_config']
        self.common_config = config['common_config']

        if self.config['device'] == 'cpu':

            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            self.device = torch.device('cpu')

        elif self.config['device'] == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.device = torch.device('cuda:1')

        self.checkpoint =torch.load(os.path.join('model_files', 
                                    self.config['model_checkpoint_filename']),
                                    map_location=torch.device('cpu'))
        
        with open(os.path.join('model_files', self.config['model_architecture_filename']), 'rb') as f:
            self.model = pickle.load(f)
    
    def infer(self, patches):       

        self.model.load_state_dict(self.checkpoint['network_weights'])
        self.model = self.model.to(self.device)
        self.model.eval()

        patches_predictions = np.zeros_like(patches)
        patches_tensor = torch.from_numpy(patches)
        patches_tensor = patches_tensor.to(self.device)

        for k in range(patches_tensor.shape[0]):

            patches_predictions[k,...] = torch.argmax(
                                            torch.softmax(
                                            self.model(patches_tensor[k,...].unsqueeze(0).unsqueeze(0)).squeeze(0),
                                            axis = 0), axis = 0).cpu()

        return patches_predictions