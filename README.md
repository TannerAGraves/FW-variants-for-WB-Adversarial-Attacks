# FW-variants-for-WB-Adversarial-Attacks

## GROUP 3
### Tanner Aaron Graves - 2073559
### Alessandro Pala - 2107800


# USAGE
Examples of attacks can be seem in test attacks.ipynb  
There you may choose to attack one of three models: LeNet5, FMNIST classifier, or ResNet-20.  
Additionally, many parameters may be experimented with, including: epsilon, Constraint norm, and Stepsize rule.

Implementations of attacks can be seen in the test method implemented in utils.py and several helper methods in AttackStep.py.

# Deliverables
0. Analyze in depth the theory of the papers (FW.pdf, FW_variants.pdf, FW_survey.pdf)

1. Develop the codes for the following algorithms:
1.1 Frank-Wolfe;
1.2 Pairwise Frank-Wolfe;
1.3 Away-Step Frank-Wolfe.
1.4 Implement caveats to make the algos fast and accurate - e.g., line search, LMO, ... 

2. Test the algorithms on the Adversarial-Attack problem  (Section 3.5 in FW_survey.pdf) using 3 Real-World datasets (suitably choose the data you like the most): follow the lines of attacks.pdf (section 5  - WB attacks).

3. Analyze the results using plots and tables.
