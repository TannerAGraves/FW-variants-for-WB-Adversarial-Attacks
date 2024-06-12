from models.LeNet import *
target_lenet = LeNet("models/lenet_mnist_model.pth")
mnist_model = target_lenet.model
mnist_device = target_lenet.device
mnist_test_loader = target_lenet.testloader
from models.simple_FashionMNIST import *

#%% raw
target_simplefmnist = simple_FashionMNIST("models/simple_FashionMNIST.pth")
fmnist_model = target_simplefmnist.model
fmnist_device = target_simplefmnist.device
fmnist_test_loader = target_simplefmnist.testloader
from models.resNet import ResNet20

#%% raw
target_resnet = ResNet20()
cifar10_model = target_resnet.model
cifar10device = target_resnet.device
cifar10test_loader = target_resnet.testloader

def lipschitz_constant(data):
    L = 0  # initialize

    for layer in data.children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weight = layer.weight.data  # extract weight data

            singular_values = torch.svd(weight).S  # as we did initially in the homework, we approximate the Lipschitz
            L = max(L, singular_values.max().item())  # constant as the maximum singular value of the weight matrix

    return L


L_mnist_value = lipschitz_constant(mnist_model)
L_fashionmnist_value = lipschitz_constant(fmnist_model)
L_cifar10_value = lipschitz_constant(cifar10_model)

L_constants = [L_mnist_value, L_fashionmnist_value, L_cifar10_value] # [6.701272487640381, 19.250612258911133, 3.085362672805786]