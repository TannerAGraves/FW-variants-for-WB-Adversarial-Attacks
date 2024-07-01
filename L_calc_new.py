import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def lipschitz(model, data_loader, device, epsilon=1e-5):
    model.eval()
    max_ratio = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            perturbed_images = images + torch.randn_like(images) * epsilon
            outputs = model(images)
            perturbed_outputs = model(perturbed_images)

            for i in range(len(images)):
                original_output = outputs[i].cpu().numpy()
                perturbed_output = perturbed_outputs[i].cpu().numpy()
                ratio = np.linalg.norm(perturbed_output - original_output) / np.linalg.norm(
                    perturbed_images[i].cpu().numpy() - images[i].cpu().numpy())
                max_ratio = max(max_ratio, ratio)

    return max_ratio


from models.LeNet import *
target_mdl = LeNet("models/lenet_mnist_model.pth")
model = target_mdl.model
device = target_mdl.device
test_loader = target_mdl.testloader
mdl_name='MNIST'

#L_MNIST = lipschitz(model, test_loader, device, epsilon=1e-5) # 2.63
L_MNIST = 2.6258278

from models.simple_FashionMNIST import *
target_mdl = simple_FashionMNIST("models/simple_FashionMNIST.pth")
model = target_mdl.model
device = target_mdl.device
test_loader = target_mdl.testloader
mdl_name = 'FMNIST'


#L_FMNIST = lipschitz(model, test_loader, device, epsilon=1e-5) # 46.06
L_FMNIST = 46.06396


from models.resNet import ResNet20
target_mdl = ResNet20()
model = target_mdl.model
device = target_mdl.device
test_loader = target_mdl.testloader
mdl_name = 'ResNet'

#L_CIFAR = lipschitz(model, test_loader, device, epsilon=1e-5) # 2.46
L_CIFAR = 2.4623497