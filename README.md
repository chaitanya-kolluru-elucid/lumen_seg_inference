# Saving the model from a trained nnUNet

Save the network as a python pickle file
Get the latest/best checkpoint in .pth format

Use load_state_dict to load in the weights

Convert the model to torchscript and then save to a file
model_scripted = torch.jit.trace(self.model, torch.rand(1, 1, 160, 160, 160).to(self.device))
torch.jit.save(model_scripted, 'model_files/in_house_model.pt')