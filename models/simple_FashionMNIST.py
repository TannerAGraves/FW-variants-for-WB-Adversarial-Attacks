# ADAPTED FROM TUTORIAL: https://github.com/junaidaliop/pytorch-fashionMNIST-tutorial/tree/main
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import os

class simple_FashionMNIST():
    def __init__(self, trained_mdl_pth):
        self.trained_mdl_pth = trained_mdl_pth
        # For reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Setting up device for GPU usage if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}.")

        # Define a transformation pipeline. 
        # Here, we're only converting the images to PyTorch tensor format.
        transform = transforms.Compose([transforms.ToTensor()])

        # Using torchvision, load the Fashion MNIST training dataset.
        # root specifies the directory where the dataset will be stored.
        # train=True indicates that we want the training dataset.
        # download=True will download the dataset if it's not present in the specified root directory.
        # transform applies the defined transformations to the images.
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

        # Create a data loader for the training set.
        # It will provide batches of data, in this case, batches of size 4.
        # shuffle=True ensures that the data is shuffled at the start of each epoch.
        # num_workers=2 indicates that two subprocesses will be used for data loading.
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        # Similarly, load the Fashion MNIST test dataset.
        self.testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=2)
        # dataset is nicely 0-1 scaled so we don't need to scale it when performing the attack
        self.requires_denorm = False

        # Define the class labels for the Fashion MNIST dataset.
        self.classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
        self.num_classes = len(self.classes)

        self.model = BasicCNN().to(self.device)
        self.cost = nn.CrossEntropyLoss()

        if os.path.exists(self.trained_mdl_pth):
            self.load_model()
        else:
            print(f"No model file found at {self.trained_mdl_pth}.\nTraining...")
            self.train(self.trained_mdl_pth)

    def load_model(self, mdl_pth=None):
        mdl_pth = self.trained_mdl_pth if mdl_pth is None else mdl_pth
        #self.model = BasicCNN(self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.trained_mdl_pth))
        self.model.eval()
        print("Model weights loaded successfully")
        return
    
    def train(self):
        # Number of complete passes through the dataset
        num_epochs = 5

        # Start the training loop
        for epoch in range(num_epochs):
            # Set the model to training mode
            self.model.train()
            
            # Iterate over each batch of the training data
            for images, labels in self.trainloader:
                # Move the images and labels to the computational device (CPU or GPU)
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Clear the gradients from the previous iteration
                self.optimizer.zero_grad()
                
                # Forward pass: Pass the images through the model to get the predicted outputs
                outputs = model(images)
                
                # Compute the loss between the predicted outputs and the true labels
                loss = self.criterion(outputs, labels)
                
                # Backward pass: Compute the gradient of the loss w.r.t. model parameters
                loss.backward()
                
                # Update the model parameters
                self.optimizer.step()

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Input: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, 3)  # Output: [batch_size, 32, 26, 26]
        
        # Input: [batch_size, 32, 26, 26]
        self.conv2 = nn.Conv2d(32, 64, 3) # Output: [batch_size, 64, 11, 11]
        
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Flattening: [batch_size, 64*5*5]
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: [batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        # Shape: [batch_size, 32, 26, 26]
        x = F.max_pool2d(x, 2)
        # Shape: [batch_size, 32, 13, 13]
        
        x = F.relu(self.conv2(x))
        # Shape: [batch_size, 64, 11, 11]
        x = F.max_pool2d(x, 2)
        # Shape: [batch_size, 64, 5, 5]
        
        x = x.view(-1, 64 * 5 * 5) # Flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)