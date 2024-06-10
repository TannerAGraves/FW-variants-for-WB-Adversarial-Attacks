import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
import random

# Pretrained model from: https://github.com/chenyaofo/pytorch-cifar-models/

class ResNet20():
    
    def __init__(self):
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        self.model.eval()
        self.num_classes = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.classes = [
                        'airplane', 										
                        'automobile', 									
                        'bird', 									
                        'cat', 										
                        'deer',										
                        'dog', 										
                        'frog', 										
                        'horse', 										
                        'ship', 										
                        'truck'
                        ]

        self.cost = nn.CrossEntropyLoss()
        
        self.testset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=False, 
                                                download=True,
                                                #transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))]))
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])
        )
        self.testloader = torch.utils.data.DataLoader(self.testset,
                                                batch_size=1, 
                                                shuffle=True)