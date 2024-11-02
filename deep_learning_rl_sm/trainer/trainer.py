import torch
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, dataset,criterion ,optimizer: Optimizer, batch_size: int = 32, learning_rate: float = 1e-3, num_epochs: int = 10, device=None):
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define DataLoader
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Define optimizer and loss function
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_states, batch_actions, batch_rewards in self.data_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)

                # Forward pass
                action_logits = self.model(batch_states)
                
                # Flatten tensors for computing loss
                action_logits = action_logits.view(-1, action_logits.size(-1))  # (batch_size * sequence_length, num_actions)
                batch_actions = batch_actions.view(-1)  # (batch_size * sequence_length)
                
                # Compute loss
                loss = self.criterion(action_logits, batch_actions)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.data_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
    
    def evaluate(self, dataset):
        """Evaluate the model on a given dataset."""
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_states, batch_actions, batch_rewards in data_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)

                action_logits = self.model(batch_states)
                
                # Flatten tensors for computing loss
                action_logits = action_logits.view(-1, action_logits.size(-1))
                batch_actions = batch_actions.view(-1)
                
                # Compute loss
                loss = self.criterion(action_logits, batch_actions)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def predict(self, states):
        self.model.eval()
        with torch.no_grad():
            states = states.to(self.device)
            action_logits = self.model(states)
            predicted_actions = torch.argmax(action_logits, dim=-1)
        return predicted_actions.cpu().numpy()

