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

        #TODO: For hendrix, use gpu 1 for now
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

        for k in range(0, patches_tensor.shape[0], self.config['batch_size']):            

            if k + self.config['batch_size'] <= patches_tensor.shape[0]:    
                patches_batch = patches_tensor[k:k+self.config['batch_size'],...]        
                patches_batch = patches_batch.to(self.device)

                patches_predictions[k:k+self.config['batch_size'],...] = torch.argmax(
                                                    torch.softmax(
                                                    self.model(patches_batch.unsqueeze(1)).squeeze(0),
                                                    axis = 1), axis = 1).cpu()
            else:                            
                patches_batch = patches_tensor[k:,...]
                patches_batch = patches_batch.to(self.device)

                patches_predictions[k:,...] = torch.argmax(
                                                    torch.softmax(
                                                    self.model(patches_batch.unsqueeze(1)).squeeze(0),
                                                    axis = 1), axis = 1).cpu()


        return patches_predictions