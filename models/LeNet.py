## CODE FOR LENET ADAPTED FROM: https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define relevant variables for the ML task
class LeNet():
    def __init__(self, weigths_pth = "models/lenet5_model.pth"):
        self.batch_size = 64
        self.num_classes = 10
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.trained_mdl_pth = weigths_pth
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.requires_denorm = True
        self.mean = [0.1307]
        self.std = [0.3081]
        
        self.transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    ])

        #Loading the dataset and preprocessing
        # self.train_dataset = torchvision.datasets.MNIST(root = './data',
        #                                         train = True,
        #                                         transform = transforms.Compose([
        #                                                 transforms.Resize((32,32)),
        #                                                 transforms.ToTensor(),
        #                                                 transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
        #                                         download = True)


        # self.test_dataset = torchvision.datasets.MNIST(root = './data',
        #                                         train = False,
        #                                         transform = transforms.Compose([
        #                                                 transforms.Resize((32,32)),
        #                                                 transforms.ToTensor(),
        #                                                 transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
        #                                         download=True)


        # self.train_loader = torch.utils.data.DataLoader(dataset = self.train_dataset,
        #                                         batch_size = self.batch_size,
        #                                         shuffle = True)


        # MNIST Test dataset and dataloader declaration
        self.testloader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
                batch_size=1, shuffle=True)
        
        self.model = torch_model().to(self.device)
        self.model.load_state_dict(torch.load(self.trained_mdl_pth, map_location=self.device))
        self.model.eval()
        self.cost = nn.CrossEntropyLoss()

        # #Setting the optimizer with the model parameters and learning rate
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


    # def train(self, mdl_pth = None):
    #     total_step = len(self.train_loader)
    #     for epoch in range(self.num_epochs):
    #         for i, (images, labels) in enumerate(self.train_loader):  
    #             images = images.to(self.device)
    #             labels = labels.to(self.device)
                
    #             #Forward pass
    #             outputs = self.model(images)
    #             loss = self.cost(outputs, labels)
                    
    #             # Backward and optimize
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
                        
    #             if (i+1) % 400 == 0:
    #                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
    #                             .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))
                
    #             # Save the trained model
    #     if mdl_pth is not None:
    #         torch.save(self.model.state_dict(), mdl_pth)
    #         print(f"Model weights saved to {mdl_pth}")
    #         self.trained_mdl_pth = mdl_pth
    
    def test(self):
        with torch.no_grad(): # note this should not be left on for WB attacks
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
    # restores the tensors to their original scale
    def denorm(self, batch):
        """
        Convert a batch of tensors to their original scale.

        Args:
            batch (torch.Tensor): Batch of normalized tensors.
            mean (torch.Tensor or list): Mean used for normalization.
            std (torch.Tensor or list): Standard deviation used for normalization.

        Returns:
            torch.Tensor: batch of tensors without normalization applied to them.
        """
        mean, std = self.mean, self.std
        if isinstance(mean, list):
            mean = torch.tensor(mean).to(self.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(self.device)
        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    
    def renorm(self, batch):
        return transforms.Normalize(tuple(self.mean), tuple(self.std))(batch).detach()
        

# LeNet Model definition
class torch_model(nn.Module):
    def __init__(self):
        super(torch_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



