import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


### OPTIMIZATION METHODS
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def fw_step(x_t, epsilon, g_t, x0, stepsize_method):
    # alg from attacks.pdf. Modified to remove momentum.
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t

    # determine stepsize
    if stepsize_method.strat == 'ls':
        fw_stepsize = stepsize_method.stepsize_linesearch(x_t, d_t)
    else:
        fw_stepsize = stepsize_method.stepsize

    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #gap_FW = torch.sum(d_t * g_t).item()#torch.dot(d_t, g_t)
    gap_FW = torch.sum(-d_t * g_t).item()
    return perturbed_image, gap_FW

def fw_step_altgap(x_t, epsilon, g_t, x0, stepsize_method):
    # alg from attacks.pdf. Modified to remove momentum.
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t

    # determine stepsize
    if stepsize_method.strat == 'ls':
        fw_stepsize = stepsize_method.stepsize_linesearch(x_t, d_t)
    else:
        fw_stepsize = stepsize_method.stepsize

    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #gap_FW = torch.sum(d_t * g_t).item()#torch.dot(d_t, g_t)
    #gap_FW = torch.sum(-d_t * g_t).item()
    gap_FW = torch.sum(-(perturbed_image - x_t)*g_t).item()
    return perturbed_image, gap_FW

def fw_step_momentum(x_t, epsilon, g_t, m_t_last, x0, stepsize_method, momentum = 0.2):
    # alg from attacks.pdf
    m_t = (1 - momentum)*g_t
    if m_t_last is not None:
        m_t += momentum*m_t_last
    m_t_sign = m_t.sign()
    v_t = -epsilon * m_t_sign + x0
    d_t = v_t - x_t

    # determine stepsize
    if stepsize_method.strat == 'ls':
        fw_stepsize = stepsize_method.stepsize_linesearch(x_t, d_t)
    else:
        fw_stepsize = stepsize_method.stepsize

    gap_FW = torch.sum(-d_t * g_t).item()
    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, m_t, gap_FW

def fw_step_momentum1(x_t, epsilon, g_t, m_t_last, x0, momentum = 0.2,fw_stepsize = 1):
    # alg from attacks.pdf
    m_t = (1 - momentum)*g_t
    if m_t_last is not None:
        m_t += momentum*m_t_last
    m_t_sign = m_t.sign()
    v_t = -epsilon * m_t_sign + x0
    d_t = v_t - x_t
    perturbed_image = x_t - momentum * epsilon * m_t_sign - momentum * (x_t - x0)#x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, m_t

def fw_step_away(x_t, epsilon, g_t, x0, S_t):
    info = {}
    # alg from FW_varients.pdf
    # FW direction
    g_t_sign = g_t.sign()
    s_t = -epsilon * g_t_sign + x0
    d_t_FW = s_t - x_t
    # AWAY direction. From set of vertices already visited
    away_costs = []
    for v in S_t:
        away_costs.append(torch.sum(g_t*v).item())
    v_t_idx = np.argmax(away_costs)
    v_t = S_t[v_t_idx]
    # at each iter x_t expressed by convex combination of active verticies
    #alpha_v_t = alphas_t[v_t_idx]
    d_t_AWAY = x_t - v_t
    #check optimality (FW gap)
    gap_FW = torch.sum(-g_t * d_t_FW).item()
    gap_AWAY = torch.sum(-g_t*d_t_AWAY).item()
    info['gap_FW'] = gap_FW
    info['gap_AS'] = gap_AWAY

    
    # check which direction is closer to the gradient
    if gap_FW >= gap_AWAY:
        info['step'] = 'FW'
        info['S_idx'] = -1 #idicate last vertex is S_t is used which is the current FW direction
        d_t = d_t_FW
        max_step = 1
        S_t.append(d_t.clone().detach())
    else:
        info['step'] = 'AS'
        info['S_idx'] = v_t_idx
        d_t = d_t_AWAY
        alpha_v_t = 0.1 # REMOVE ME or implement line searching or solve system to get alpha coeffs
        max_step = 0.1 #alpha_v_t / (1 - alpha_v_t) # this is a safe step size to remain in C. Need to verify this

    # line-search for the best gamma (FW stepsize)
    perturbed_image = x_t + max_step * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, S_t, info


def fw_step_pairwise():
    # alg from FW_varients.pdf
    return



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
            average_incorrect_log_probs = incorrect_log_probs.logsumexp(dim=1) - torch.log(torch.tensor(self.num_classes - 1, dtype=torch.float))
            loss = -average_incorrect_log_probs
            return loss.mean()
        
class stepsize():
    def __init__(self, model, strat, fixed_size = 1, ls_criterion=None, ls_target = None, ls_num_samples=10):
        if isinstance(strat, (float, int)):
            fixed_size = strat
            strat = 'fixed'
        self.model = model
        self.strat = strat
        self.fixed_size = fixed_size
        self.ls_criterion = ls_criterion
        self.ls_target = ls_target
        self.ls_num_samples = ls_num_samples
        self.stepsize = fixed_size # will be updated if using other method
        if self.strat not in ['fixed', 'ls', 'decay']:
            raise Exception("Accepted stepsize rules are ['fixed', 'ls', 'decay']")
    
    def set_stepsize_decay(self, t):
        self.stepsize = 2 / (t + 2)
        return

    def stepsize_linesearch(self, x_t, d_t):
        x_tc = x_t.clone().detach()
        d_tc = d_t.clone().detach()
        losses = []
        with torch.no_grad():
            steps = [(i + 1) / self.ls_num_samples for i in range(self.ls_num_samples)]
            for step in steps:
                output = self.model(x_tc + step * d_tc)
                losses.append(self.ls_criterion(output, self.ls_target))
        best_idx = np.argmin(losses) # check if this is min or max
        self.stepsize = steps[best_idx]
        return self.stepsize

def test_fw(target_model, device, epsilon,num_fw_iter, num_test = 1000, method='fw', early_stopping = None, fw_stepsize_rule = 1, gap_FW_tol = 0.05):
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
        #x_t.requires_grad = True  #Set requires_grad attribute of tensor. Important for Attack
        m_t_last = None # gradient from previous iter
        had_first_success = False
        gap_FW = None
        S_t = [x0_denorm]
        info = None
        criterion = AdversarialLoss(10)
        stepsize_method = stepsize(model, fw_stepsize_rule, ls_criterion=criterion, ls_target=target)

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
            #1 - F.nll_loss(output, target) # DNN maximizing POSITIVE log liklihood
            #criterion = nn.CrossEntropyLoss()
            #loss = criterion(output, target)
            #loss = untargeted_attack_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()
            x_t_grad = x_t.grad#.data

            # Restore the data to its original scale
            x_t_denorm = target_model.denorm(x_t)

            # Call Attack
            if method == 'fgsm':
                perturbed_image = fgsm_attack(x_t_denorm, epsilon, x_t_grad)
            elif method == 'fw':
                perturbed_image, gap_FW = fw_step(x_t_denorm, epsilon, x_t_grad, x0_denorm, stepsize_method=stepsize_method)
            elif method == 'fw_altgap':
                perturbed_image, gap_FW = fw_step_altgap(x_t_denorm, epsilon, x_t_grad, x0_denorm, stepsize_method=stepsize_method)
            elif method == 'fw_AWAY':
                perturbed_image, S_t, info = fw_step_away(x_t_denorm, epsilon, x_t_grad, x0_denorm, S_t)
            elif method == 'fw_momentum':
                perturbed_image, m_t_last, gap_FW = fw_step_momentum(x_t_denorm, epsilon, x_t_grad, m_t_last, x0_denorm, stepsize_method=stepsize_method)
            # Reapply normalization
            x_t = target_model.renorm(perturbed_image)#transforms.Normalize((0.1307,), (0.3081,))(perturbed_image).detach()

            # Re-classify the perturbed image
            x_t.requires_grad = False
            output = model(x_t)

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