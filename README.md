## Requirements

Create a virtual environment and install the packages described in requirements.txt

```
pip install virtualenv (if you don't already have virtualenv installed)
virtualenv venv to create your new environment (called 'venv' here)
source venv/bin/activate to enter the virtual environment
pip install -r requirements.txt to install the requirements in the current environment
```

## Saving the model from a trained nnUNet

Save the network as a python pickle file  
Get the latest/best checkpoint in .pth format

Use load_state_dict to load in the weights

```
Convert the model to torchscript and then save to a file for Triton
model_scripted = torch.jit.trace(self.model, torch.rand(1, 1, 160, 160, 160).to(self.device))
torch.jit.save(model_scripted, 'model_files/in_house_model.pt')
```
## Memory usage and logging info
Log files are written to a directory called 'logs' in the repo
