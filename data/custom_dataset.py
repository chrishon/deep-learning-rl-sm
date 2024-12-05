import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self,path):
        # Storing each data item separately for easy access
        self.path = path
        data = torch.load(path)
        self.states = data['states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.time_steps = data['time_steps']
        self.action_masks = data['action_masks']
        self.returns_to_go = data['returns_to_go']
        
    def __len__(self):
        # Return the number of samples (assuming all lists have the same length)
        return len(self.states)

    def __getitem__(self, idx):
        # Return a tuple of each item type for a given index
        return (self.states[idx,:,:], 
                self.actions[idx,:,:], 
                self.rewards[idx,:,:], 
                self.dones[idx,:,:], 
                self.time_steps[idx,:,:], 
                self.action_masks[idx,:,:], 
                self.returns_to_go[idx,:,:])