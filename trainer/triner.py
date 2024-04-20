
import torch
import torch.nn as nn


class Trainer(object):
    def __init__(self, cfg, model, train_loaer, test_loader, optimizer, device):
        
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device


        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def train(self, epoch):

        self.model.train()
        for epoch in range(self.cfg['epochs']):
            for i, (input, target) in enumerate(self.train_loader):
                
                if cfg['cuda']:
                    input, target = input.to(self.device), target.to(self.device)
    
                output = self.model(input)
                loss = criterion(output, target)
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
            
            
            

    def test(self):
        pass

        
        
