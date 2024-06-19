import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from AttackStep import AttackStep

class LMO:
    def __init__(self, epsilon, x0):
        self.x0 = x0.clone().detach()  # Ensure x0 is not modified elsewhere
        self.epsilon = epsilon

    def get(self, g_t):
        g_t_sign = g_t.sign()
        s_t = -self.epsilon * g_t_sign + self.x0
        return s_t

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
            average_incorrect_log_probs = incorrect_log_probs.logsumexp(dim=1) - torch.log(torch.tensor(self.num_classes - 1, dtype=torch.float))
            loss = -average_incorrect_log_probs
            return loss.mean()
        
class stepsize():
    def __init__(self, model, strat, x0, fixed_size = 1, ls_criterion=None, ls_target = None, ls_num_samples=10):
        if isinstance(strat, (float, int)):
            fixed_size = strat
            strat = 'fixed'
        self.model = model
        self.strat = strat
        self.fixed_size = fixed_size
        self.x0 = x0
        self.ls_criterion = ls_criterion
        self.ls_target = ls_target
        self.ls_num_samples = ls_num_samples
        self.stepsize = fixed_size # will be updated if using other method
        if self.strat not in ['fixed', 'ls', 'decay']:
            raise Exception("Accepted stepsize rules are ['fixed', 'ls', 'decay']")
    
    def set_stepsize_decay(self, t):
        self.stepsize = 2 / (t + 2)
        return

    def stepsize_linesearch(self, x_t, d_t, max_step = 1):
        x_tc = x_t.clone().detach()
        d_tc = d_t.clone().detach()
        losses = []
        with torch.no_grad():
            steps = [max_step * (i + 1) / self.ls_num_samples for i in range(self.ls_num_samples)]
            for step in steps:
                output = self.model(x_tc + step * d_tc)
                losses.append(self.ls_criterion(output, self.ls_target))
        best_idx = np.argmin(losses) # check if this is min or max
        self.stepsize = steps[best_idx]
        return self.stepsize
    
    def get_stepsize(self, x_t, d_t, max_step = 1):
        if self.strat == 'ls':
            fw_stepsize = self.stepsize_linesearch(x_t, d_t, max_step)
        else:
            fw_stepsize = self.stepsize
        fw_stepsize = min(fw_stepsize, max_step)
        return fw_stepsize

def test(target_model, device, epsilon,num_fw_iter, num_test = 1000, method='fw', early_stopping = None, fw_stepsize_rule = 1, gap_FW_tol = 0.05, targeted = False):
    testloader = target_model.testloader
    model = target_model.model

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
        if targeted:
            # select a random target for attack that is not the true target.
            adv_target = random.randint(0, target_model.num_classes - 2)
            adv_target = adv_target if adv_target < target else adv_target + 1
            criterion = AdversarialLoss(target_model.num_classes, specific_label=adv_target)
        else:
            criterion = AdversarialLoss(target_model.num_classes)
        lmo = LMO(epsilon, x0_denorm)
        stepsize_method = stepsize(model, fw_stepsize_rule, x0_denorm, ls_criterion=criterion, ls_target=target)
        attackStep = AttackStep(method, epsilon, x0_denorm, lmo, stepsize_method)
        #x_t.requires_grad = True  #Set requires_grad attribute of tensor. Important for Attack
        had_first_success = False
        gap_FW = None
        info = None

        for t in range(num_fw_iter):
            # Step size calculation
            if stepsize_method.strat == 'decay':
                stepsize_method.set_stepsize_decay(t)
            x_t.requires_grad = True
            # Forward pass the data through the model
            output = model(x_t)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, don't bother attacking, just move on
            if (init_pred.item() != target.item()) and (t == 0):
                break

            # Calculate the loss
            loss = criterion(output, target)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            x_t_grad = x_t.grad#.data
            # Restore the data to its original scale
            x_t_denorm = target_model.denorm(x_t)
            # Call Attack
            with torch.no_grad():
                perturbed_image, gap_FW, info = attackStep.step(x_t_denorm, x_t_grad)
            
            # Reapply normalization
            x_t = target_model.renorm(perturbed_image)#transforms.Normalize((0.1307,), (0.3081,))(perturbed_image).detach()
            # Re-classify the perturbed image
            x_t.requires_grad = False
            output = model(x_t)
            info['l_inf'] = torch.max(torch.abs(x0_denorm - perturbed_image)).item()
            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if final_pred.item() == target.item():
                success = False
                first_success = False
                # if t == num_fw_iter - 1:
                #     correct += 1
                # Special case for saving 0 epsilon examples
                if epsilon == 0 and len(adv_examples) < 5:
                    adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            else:
                first_success =  not had_first_success
                had_first_success = True
                success = True
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            if (early_stopping == 'pred') and first_success:
                # The attack was successful so the classification was not correct
                stop = True
                stop_reason = 'pred'
            elif (early_stopping == 'gap_FW') and (gap_FW < gap_FW_tol):
                if not success: # attack failed
                    correct +=1
                stop = True
                stop_reason = 'gap'
            elif (t == num_fw_iter - 1): # Stop condition: Hit max FW iters
                stop = True
                stop_reason = 'max_iter'
                if not success: # attack failed
                    correct +=1
            elif (early_stopping == 'gap_pairwise') and (info is not None) and (info['gap_pairwise'] < gap_FW_tol):
                if not success: # attack failed
                    correct +=1
                stop = True
                stop_reason = 'gap_pairwise'
            else:
                # no stop criteria met, continue
                stop = False
                stop_reason = None
            hist_iter = {
                'example_idx':ex_num,
                'FW_iter': t + 1, # original example is 0
                'gap_FW': gap_FW if gap_FW is not None else None,
                'success': success,
                'first_success': first_success,
                'target': target.item(),
                'pred': final_pred.item(),
                'stop_cond': stop_reason
            }
            if info is not None:
                hist_iter.update(info) # some methods output dict containing info at each step
            hist.append(hist_iter)
            if stop:
                break
        ex_num += 1
        if ex_num >= num_test: # limit test set for speed
            break

    # Calculate final accuracy for this epsilon
    final_acc = correct/num_test
    print(f"Epsilon: {epsilon}\tCorrect Classifications (Failed Attacks) = {correct} / {num_test} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, pd.DataFrame(hist)