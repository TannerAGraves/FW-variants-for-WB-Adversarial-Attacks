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
        
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.testset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=False, 
                                                download=True,
                                                #transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))]))
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(self.mean, self.std)
                                                ])
        )
        self.testloader = torch.utils.data.DataLoader(self.testset,
                                                batch_size=1, 
                                                shuffle=True)
        
    # restores the tensors to their original scale
    def denorm(self, batch):
        """
        Convert a batch of tensors to their original scale.

        Args:
            batch (torch.Tensor): Batch of normalized tensors.

        Returns:
            torch.Tensor: batch of tensors without normalization applied to them.
        """
        mean, std = self.mean, self.std
        mean = torch.tensor(mean).to(self.device)
        std = torch.tensor(std).to(self.device)
        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    
    def renorm(self, batch):
        return transforms.Normalize(self.mean, self.std)(batch).detach()