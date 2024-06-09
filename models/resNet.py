import torch
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt

class ResNet20():
    
    def __init__(self):
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        self.num_classes = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.cost = nn.CrossEntropyLoss()
        
        self.testset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=False, 
                                                download=True,
                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        self.testloader = torch.utils.data.DataLoader(self.testset,
                                                batch_size=4, 
                                                shuffle=True)