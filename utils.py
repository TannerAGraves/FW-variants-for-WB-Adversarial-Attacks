import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods import *
from normedMethods import *

### UTILS
class AdversarialLoss(nn.Module):
    def __init__(self, num_classes, specific_label=None):
        """
        Initialize the AdversarialLoss.

        Args:
        - num_classes (int): Total number of classes in the classification problem.
        - specific_label (int, optional): A specific incorrect label to target. If None, the loss will consider all incorrect labels.
        """
        super(AdversarialLoss, self).__init__()
        self.num_classes = num_classes
        self.specific_label = specific_label

    def forward(self, outputs, targets):
        """
        Compute the adversarial loss.

        Args:
        - outputs (torch.Tensor): The model outputs (logits) of shape (batch_size, num_classes).
        - targets (torch.Tensor): The true labels of shape (batch_size,).

        Returns:
        - loss (torch.Tensor): The computed adversarial loss.
        """
        batch_size = outputs.size(0)
        if self.specific_label is not None:
            # Targeting a specific incorrect label
            incorrect_labels = torch.full_like(targets, self.specific_label)
            mask = (incorrect_labels != targets).float()
            specific_log_probs = F.log_softmax(outputs, dim=1).gather(1, incorrect_labels.unsqueeze(1)).squeeze(1)
            loss = -specific_log_probs * mask
            return loss.mean()
        else:
            # Averaging over all incorrect labels
            log_probs = F.log_softmax(outputs, dim=1)
            incorrect_log_probs = log_probs.clone()
            correct_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            for i in range(batch_size):
                incorrect_log_probs[i, targets[i]] = float('-inf')
            average_incorrect_log_probs = incorrect_log_probs.logsumexp(dim=1) - torch.log(
                torch.tensor(self.num_classes - 1, dtype=torch.float))
            loss = -average_incorrect_log_probs
            return loss.mean()

class LMO():
    def __init__(self, norm, epsilon, x_0):
        self.x_0 = x_0
        self.epsilon = epsilon

        self.meth = None
        if norm == 'inf':
            self.p = None #don't refer to this
            self.meth = self.l_inf
        elif norm == 1:
            self.p = 1
        elif isinstance(norm, (int, float)) and norm > 1:
            self.p = norm
        else:
            raise Exception("Invalid norm choice")
        
    def get(self, g_t, epsilon):
        return self.norm_method(g_t, epsilon)
    
    def l_inf(x_0, g_t, epsilon):
        g_t_sign = g_t.sign()
        return -epsilon * g_t_sign + x_0


class stepsize():
    def __init__(self, model, strat, fixed_size=1, ls_criterion=None, ls_target=None, ls_num_samples=10):

        lipschitz_stepsizes = {            # from L_calc
            'mnist': 0.14922538993069884,
            'fmnist': 0.051946399758641376,
            'cifar': 0.32411100607845683
        }

        if isinstance(strat, (float, int)):
            fixed_size = strat
            strat = 'fixed'

        self.model = model
        self.strat = strat
        self.fixed_size = fixed_size
        self.ls_criterion = ls_criterion
        self.ls_target = ls_target
        self.ls_num_samples = ls_num_samples

        if self.strat == 'fixed' and model in lipschitz_stepsizes:   # set fixed stepsize as lipschitz if we inster a specific model as input
            self.stepsize = lipschitz_stepsizes[model]
        else:
            self.stepsize = fixed_size

        if self.strat not in ['fixed', 'ls', 'decay', 'armijo']:
            raise Exception("Accepted stepsize rules are ['fixed', 'ls', 'decay', 'armijo']")

    def set_stepsize_decay(self, t):
        self.stepsize = 2 / (t + 2)
        return

    def armijo_rule(self, x_t, d_t, max_step=1, delta=0.5, gamma=0.1):
        alpha = max_step
        while alpha > 1e-8:  #small threshold to avoid infinite loops

            x_tc = x_t.clone().detach()
            d_tc = d_t.clone().detach()

            # calcs for the current point
            output = self.model(x_tc)
            loss = self.ls_criterion(output, self.ls_target)
            loss.backward()
            f_x = loss.item()
            grad_f_x = x_tc.grad.clone().detach()

            # new point with step size alpha
            x_tc_alpha = x_tc + alpha * d_tc
            output_alpha = self.model(x_tc_alpha)
            loss_alpha = self.ls_criterion(output_alpha, self.ls_target)
            f_x_alpha = loss_alpha.item()

            # armijo rule
            if f_x_alpha <= f_x + gamma * alpha * torch.dot(grad_f_x.view(-1), d_tc.view(-1)).item():
                self.stepsize = alpha
                return self.stepsize

            alpha *= delta # reduce step size

        self.stepsize = alpha
        return self.stepsize

    def stepsize_linesearch(self, x_t, d_t, max_step=0.1):  # updated max_step to 0.1
        x_tc = x_t.clone().detach()
        d_tc = d_t.clone().detach()
        losses = []
        with torch.no_grad():
            steps = [max_step * (i + 1) / self.ls_num_samples for i in range(self.ls_num_samples)]
            for step in steps:
                output = self.model(x_tc + step * d_tc)
                loss = self.ls_criterion(output, self.ls_target)
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = float('inf')  # penalize invalid losses
                losses.append(loss)
        best_idx = np.argmin(losses)
        self.stepsize = steps[best_idx]
        return self.stepsize

    def get_stepsize(self, x_t, d_t, max_step=1):
        if self.strat == 'ls':
            fw_stepsize = self.stepsize_linesearch(x_t, d_t, max_step)
        elif self.strat == 'armijo':
            fw_stepsize = self.armijo_rule(x_t, d_t, max_step)
        else:
            fw_stepsize = self.stepsize
        fw_stepsize = min(fw_stepsize, max_step)
        return fw_stepsize

def test_fw(target_model, device, epsilon, num_fw_iter, num_test=1000, method='fw', early_stopping=None,
            fw_stepsize_rule=1, gap_FW_tol=0.05, targeted=False):
    testloader = target_model.testloader
    model = target_model.model
    num_classes = target_model.num_classes

    # Accuracy counter
    correct = 0
    adv_examples = []
    hist = []
    ex_num = 0
    # Loop over all examples in test set
    for x0, target in testloader:

        x_t = x0.detach().clone().to(device)  # Clone and move to device
        # Send the data and label to the device
        x0, target = x0.to(device), target.to(device)
        x0_denorm = target_model.denorm(x0)
        # x_t.requires_grad = True  #Set requires_grad attribute of tensor. Important for Attack
        m_t_last = None  # gradient from previous iter
        had_first_success = False
        gap_FW = None
        S_t = [x0_denorm]
        A_t = [1]
        info = None
        adv_target = None
        if targeted:
            # select a random target for attack that is not the true target.
            adv_target = random.randint(0, num_classes - 2)
            adv_target = adv_target if adv_target < target else adv_target + 1
            criterion = AdversarialLoss(num_classes, specific_label=adv_target)
        else:
            criterion = AdversarialLoss(num_classes)
        stepsize_method = stepsize(model, fw_stepsize_rule, ls_criterion=criterion, ls_target=target)

        for t in range(num_fw_iter):
            # Step size calculation
            if stepsize_method.strat == 'decay':
                stepsize_method.set_stepsize_decay(t)
            x_t.requires_grad = True
            # Forward pass the data through the model
            output = model(x_t)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, don't bother attacking, just move on
            if (init_pred.item() != target.item()) and (t == 0):
                break

            # Calculate the loss
            loss = criterion(output, target)
            # 1 - F.nll_loss(output, target) # DNN maximizing POSITIVE log liklihood
            # criterion = nn.CrossEntropyLoss()
            # loss = criterion(output, target)
            # loss = untargeted_attack_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()
            x_t_grad = x_t.grad  # .data

            # Restore the data to its original scale
            x_t_denorm = target_model.denorm(x_t)

            # Call Attack
            with torch.no_grad():
                if method == 'fgsm':
                    perturbed_image = fgsm_attack(x_t_denorm, epsilon, x_t_grad)

                elif method == 'fw':
                    perturbed_image, gap_FW, info = fw_step(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                            stepsize_method=stepsize_method)
                elif method == 'fw_momentum':
                    perturbed_image, m_t_last, gap_FW = fw_step_momentum(x_t_denorm, epsilon, x_t_grad, m_t_last,
                                                                         x0_denorm, stepsize_method=stepsize_method)
                elif method == 'fw_away':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_away(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                           S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_pair':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_pairwise(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                               S_t, A_t,
                                                                               stepsize_method=stepsize_method)
                elif method == 'fw_l1':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_l1(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                        S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_l2':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_l2(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                        S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_p2':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_p2(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                        S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_away_l1':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_away_l1(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                             S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_away_l2':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_away_l2(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                             S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_away_p2':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_away_p2(x_t_denorm, epsilon, x_t_grad, x0_denorm,
                                                                             S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_pair_l1':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_pairwise_l1(x_t_denorm, epsilon, x_t_grad,
                                                                                 x0_denorm, S_t, A_t,
                                                                                 stepsize_method=stepsize_method)
                elif method == 'fw_pair_l2':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_pairwise_l2(x_t_denorm, epsilon, x_t_grad,
                                                                                 x0_denorm, S_t, A_t,
                                                                                 stepsize_method=stepsize_method)
                elif method == 'fw_pair_p2':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_pairwise_p2(x_t_denorm, epsilon, x_t_grad,
                                                                                 x0_denorm, S_t, A_t,
                                                                                 stepsize_method=stepsize_method)

            # Reapply normalization
            x_t = target_model.renorm(
                perturbed_image)  # transforms.Normalize((0.1307,), (0.3081,))(perturbed_image).detach()

            # Re-classify the perturbed image
            x_t.requires_grad = False
            output = model(x_t)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                success = False
                first_success = False
                # if t == num_fw_iter - 1:
                #     correct += 1
                # Special case for saving 0 epsilon examples
                if epsilon == 0 and len(adv_examples) < 5:
                    adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                first_success = not had_first_success
                had_first_success = True
                success = True
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            if (early_stopping == 'pred') and first_success:
                # The attack was successful so the classification was not correct
                stop = True
                stop_reason = 'pred'
            elif (early_stopping == 'gap_FW') and (gap_FW < gap_FW_tol):
                if not success:  # attack failed
                    correct += 1
                stop = True
                stop_reason = 'gap'
            elif (t == num_fw_iter - 1):  # Stop condition: Hit max FW iters
                stop = True
                stop_reason = 'max_iter'
                if not success:  # attack failed
                    correct += 1
            elif (early_stopping == 'gap_pairwise') and (info is not None) and (info['gap_pairwise'] < gap_FW_tol):
                if not success:  # attack failed
                    correct += 1
                stop = True
                stop_reason = 'gap_pairwise'
            else:
                # no stop criteria met, continue
                stop = False
                stop_reason = None
            hist_iter = {
                'example_idx': ex_num,
                'FW_iter': t + 1,  # original example is 0
                'gap_FW': gap_FW if gap_FW is not None else None,
                'success': success,
                'first_success': first_success,
                'target': target.item(),
                'pred': final_pred.item(),
                'stop_cond': stop_reason
            }
            if targeted:
                info['adv_target'] = adv_target
            if info is not None:
                hist_iter.update(info)  # some methods output dict containing info at each step
            hist.append(hist_iter)
            if stop:
                break
        ex_num += 1
        if ex_num >= num_test:  # limit test set for speed
            break

    # Calculate final accuracy for this epsilon
    final_acc = correct / num_test
    print(f"Epsilon: {epsilon}\tCorrect Classifications (Failed Attacks) = {correct} / {num_test} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, pd.DataFrame(hist)