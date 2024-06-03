## CODE FOR LENET ADAPTED FROM: https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/

# Load in relevant libraries, and alias where appropriate
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define relevant variables for the ML task
class LeNet():
    def __init__(self):
        self.batch_size = 64
        self.num_classes = 10
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        self.trained_mdl_pth = "lenet5_model.pth"
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        #Loading the dataset and preprocessing
        self.train_dataset = torchvision.datasets.MNIST(root = './data',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.Resize((32,32)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                                download = True)


        self.test_dataset = torchvision.datasets.MNIST(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        transforms.Resize((32,32)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                                download=True)


        self.train_loader = torch.utils.data.DataLoader(dataset = self.train_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = True)


        self.test_loader = torch.utils.data.DataLoader(dataset = self.test_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = True)
        
        self.model = torch_model(self.num_classes).to(self.device)
        self.cost = nn.CrossEntropyLoss()

        #Setting the optimizer with the model parameters and learning rate
        self.optimizer = torch.optim.Adam(torch_model.parameters(), lr=self.learning_rate)

        #this is defined to print how many steps are remaining when training
        self.total_step = len(self.train_loader)

    def train(self, mdl_pth = None):
        total_step = len(self.train_loader)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):  
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                #Forward pass
                outputs = torch_model(images)
                loss = self.cost(outputs, labels)
                    
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                        
                if (i+1) % 400 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                                .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))
                
                # Save the trained model
                if mdl_pth is not None:
                    torch.save(torch_model.state_dict(), mdl_pth)
                    print(f"Model weights saved to {mdl_pth}")
                    self.trained_mdl_pth = mdl_pth

    def load_model(self, mdl_pth=None):
        mdl_pth = self.trained_mdl_pth if mdl_pth is None else mdl_pth
        torch_model = torch_model(self.num_classes).to(self.device)
        torch_model.load_state_dict(torch.load(self.trained_mdl_pth))
        torch_model.eval()
        print("Model weights loaded successfully")
        

class torch_model(nn.Module):
    def __init__(self, num_classes):
        super(torch_model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out