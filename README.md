Sure! Here's the README content you can directly copy and paste:
Comparison of Frank-Wolfe Variants for White-Box Adversarial Attacks
Project Overview

This project implements and compares several variants of the Frank-Wolfe algorithm for generating white-box adversarial attacks on deep neural networks. The goal is to explore the efficiency of Frank-Wolfe and its variants in creating adversarial examples while staying within predefined constraints. We compare their performance on popular datasets including MNIST, FashionMNIST, and CIFAR-10, focusing on attack success rates and convergence times.
Authors

    Tanner Aaron Graves (2073559)
    Alessandro Pala (2107800)

Date: June 2024
Abstract

Deep neural networks (DNNs) are vulnerable to adversarial examples, which are subtle modifications to input data that cause incorrect model predictions. This project explores white-box adversarial attacks, where access to model gradients enables efficient adversarial example generation. We apply Frank-Wolfe (FW) and its variants—Away-step FW and Pairwise FW—to this constrained optimization problem and compare their performance across three datasets. The project evaluates the effectiveness and convergence behavior of each algorithm under both targeted and untargeted attack scenarios.
Table of Contents

    Introduction
    Algorithms
        Frank-Wolfe (FW)
        Away-Step FW
        Pairwise FW
    Results
    Step-size Exploration
    Convergence Analysis
    Conclusion
    References

Introduction

Adversarial attacks on neural networks are often expressed as constrained optimization problems. We use the Frank-Wolfe algorithm to generate adversarial examples within a predefined constraint space, such as an LpLp​-norm ball. The advantage of Frank-Wolfe is that it respects the constraint at each iteration while effectively exploiting the structure of the constraint set.

We implemented both targeted and untargeted attacks, comparing results for various model architectures trained on MNIST, FashionMNIST, and CIFAR-10 datasets.
Algorithms

This project implements several variants of the Frank-Wolfe algorithm:

    Frank-Wolfe (FW): Standard FW that minimizes the objective function using a linear minimization oracle (LMO).
    Away-Step Frank-Wolfe (AFW): An extension that adds "away steps" to improve convergence near the boundary of the constraint set.
    Pairwise Frank-Wolfe (PFW): A variant that simultaneously reduces the contribution of a "bad" atom while increasing the contribution of a new atom.

Key Features:

    Support for both L∞L∞​ and L2L2​-norm constraints.
    Step-size techniques including decaying step-size and line-search methods.
    Momentum implementation for accelerated convergence.

Results

The algorithms were tested on pre-trained models for the MNIST, FashionMNIST, and CIFAR-10 datasets. The results show:

    Success Rate: The percentage of successful adversarial attacks for each method.
    Convergence Time: The number of iterations required for convergence.

Visual Results:

The following image shows examples of successful adversarial attacks generated by the algorithms: Adversarial Examples
Step-size Exploration

We explored several step-size strategies:

    Decaying Step-size: A simple step-size that decreases over time.
    Exact Line Search: Finds the optimal step-size at each iteration.
    Armijo-rule Line Search: A heuristic method to balance progress with computational efficiency.

The table below shows the success rates and average iterations for each method:
Step Rule	ASR (%)	Avg. Iters	Attacks/Second
Decay	62.4	13.52	4.5
Armijo	63.6	13.13	2.5
Exact-LS	63.8	12.98	0.34
Convergence Analysis

The Frank-Wolfe algorithm guarantees sublinear convergence for constrained optimization problems. However, the non-convexity of neural network loss functions complicates this behavior. Variants like AFW and PFW attempt to address this by reducing oscillations, but momentum-based FW emerged as the most efficient variant.
Conclusion

Frank-Wolfe and its variants are highly effective for generating adversarial attacks on deep learning models. The momentum variant consistently outperformed others in terms of convergence speed, making it a highly recommended approach for adversarial attack generation. Future work could focus on extending these methods to other architectures and attack scenarios.
