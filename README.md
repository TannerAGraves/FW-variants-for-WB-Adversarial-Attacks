# FW-variants-for-WB-Adversarial-Attacks

## GROUP 3
### Tanner Aaron Graves - 2073559
### Alessandro Pala - 2107800

0. Analyze in depth the theory of the papers (FW.pdf, FW_variants.pdf, FW_survey.pdf)

1. Develop the codes for the following algorithms:
1.1 Frank-Wolfe;
1.2 Pairwise Frank-Wolfe;
1.3 Away-Step Frank-Wolfe.
1.4 Implement caveats to make the algos fast and accurate - e.g., line search, LMO, ... 

2. Test the algorithms on the Adversarial-Attack problem  (Section 3.5 in FW_survey.pdf) using 3 Real-World datasets (suitably choose the data you like the most): follow the lines of attacks.pdf (section 5  - WB attacks).

3. Analyze the results using plots and tables.

### 
Model and data loading for new methods (lipschitz=fixed, model dependent)

    from models.LeNet import *
    mnist = LeNet("models/lenet_mnist_model.pth")
    mnist_model = target_lenet.model
    mnist_device = target_lenet.device
    mnist_test_loader = target_lenet.testloader
    
    from models.simple_FashionMNIST import *
    fmnist = simple_FashionMNIST("models/simple_FashionMNIST.pth")
    fmnist_model = fmnist.model
    fmnist_device = fmnist.device
    fmnist_test_loader = fmnist.testloader
    
    from models.resNet import ResNet20
    cifar = ResNet20()
    cifar10 = cifar.model
    cifar10device = cifar.device
    cifar10test_loader = cifar.testloader


###
example

    accuracies = []
    examples = []
    hist_dfs = []
    final_hist_dfs = []
    fw_iters = [15]
    eps = 0.05
    adv_examples = []
    hist = None
    debug = True
    
    algs = ['fw_pair']
    for alg in algs:
        acc, ex, hist = test_fw(mnist, mnist_device, eps, 15, method=alg, early_stopping='gap_PW', fw_stepsize_rule='fixed', gap_FW_tol=0.1)
        accuracies.append(acc)
        examples.append(ex)
        hist_dfs.append(hist)
        final_hist = hist.groupby('example_idx').tail(1)
        final_hist_dfs.append(final_hist)
      
    final_hist_dfs[-1]
