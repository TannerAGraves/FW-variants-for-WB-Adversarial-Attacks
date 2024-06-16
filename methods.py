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
    info = {}
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t

    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t)

    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #gap_FW = torch.sum(d_t * g_t).item()#torch.dot(d_t, g_t)
    gap_FW = torch.sum(-d_t * g_t).item()
    return perturbed_image, gap_FW, info

def fw_step_momentum(x_t, epsilon, g_t, m_t_last, x0, stepsize_method, momentum = 0.2):
    # alg from attacks.pdf
    m_t = (1 - momentum)*g_t
    if m_t_last is not None:
        m_t += momentum*m_t_last
    m_t_sign = m_t.sign()
    v_t = -epsilon * m_t_sign + x0
    d_t = v_t - x_t

    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t)

    gap_FW = torch.sum(-d_t * g_t).item()
    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, m_t, gap_FW

def update_active_away(gamma, gamma_max, S_t, A_t, s_t, v_t_idx, step_type, epsilon, debug = True):
    """
    Args:
        gamma (float): stepsize
        gamma_max (int): Max stepsize informs when FW step will make S_t singular or AS dropstep
        S_t (list(torch.Tensor)): Active set of directions s.t. x_t in conv{S_t}
        A_t (list(float)): coefficients corresponding to atoms in S_t. x_t = A_t .* S_t
        v_t_idx (int): index of away atom in S_t
    """
    info = {}
    debug_info = {}
    if step_type == 'FW':
        if abs(gamma - 1) < 0.001:
            # drop step
            S_t = [s_t]
            A_t = [1]
            debug_info['drop_step'] = 'FW'
        else:
            ## UPDATE S
            # need to check if vertex is already in S
            diffs = [torch.sum(torch.abs(s_t - s)).item() for s in S_t]#[torch.max(torch.abs(s_t - s)).item() for s in S_t]
            min_diff = min(diffs)
            arg = np.argmin(diffs)
            if min_diff < 0.9*epsilon:
                # s_t already in S_t
                s_t_idx = arg
                debug_info['FW_revisit'] = True
            else:
                S_t.append(s_t)
                A_t.append(0.0)
                s_t_idx = -1
            debug_info["min_revisit_diff"] = min_diff
            ## UPDATE ALPHAS
            A_t = [(1 - gamma) * alpha for alpha in A_t]
            A_t[s_t_idx] += gamma
    elif step_type == 'AS':
        if gamma >= gamma_max:
            # drop step: remove atom and alpha
            A_t.pop(v_t_idx)
            S_t.pop(v_t_idx)
            debug_info['drop_step'] = 'AS'
        else:
            ## UPDATE ALPHAS
            A_t = [(1 + gamma) * alpha for alpha in A_t]
            A_t[s_t_idx] -= gamma
    else:
        raise Exception("Step must be FW or AS")
    if debug:
        info.update(debug_info)
    return S_t, A_t, info

def fw_step_away(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method, debug = True):
    info = {}
    # alg from FW_varients.pdf
    # FW direction
    g_t_sign = g_t.sign()
    s_t = -epsilon * g_t_sign + x0
    d_t_FW = s_t - x_t
    # AWAY direction. From set of vertices already visited
    away_costs = []
    for v in S_t:
        away_costs.append(torch.sum(-g_t*v).item()) # negative here because were attacking
    v_t_idx = np.argmin(away_costs) # docs have arg max
    v_t = S_t[v_t_idx]
    # at each iter x_t expressed by convex combination of active verticies
    #alpha_v_t = alphas_t[v_t_idx]
    d_t_AWAY = x_t - v_t
    #check optimality (FW gap)
    gap_FW = torch.sum(-g_t * d_t_FW).item()
    gap_AWAY = torch.sum(-g_t*d_t_AWAY).item()
    info['gap_FW'] = gap_FW
    info['gap_AS'] = gap_AWAY
    #info['gap_ASmax'] = torch.sum(g_t*(x_t - S_t[np.argmax(away_costs)])).item()

    
    # check which direction is closer to the gradient
    if (gap_FW >= gap_AWAY) or (len(S_t) == 1):
        step_type = 'FW'
        d_t = d_t_FW
        max_step = 1
    else:
        step_type = 'AS'
        d_t = d_t_AWAY
        alpha_v_t = A_t[v_t_idx]
        max_step = 1 if alpha_v_t == 1 else alpha_v_t / (1 - alpha_v_t) # avoid divide by zero when alpha = 1
    info['step_type'] = step_type
    info['max_step'] = max_step
    # determine stepsize according to rule
    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t, max_step)
    info['stepsize'] = fw_stepsize
    
    S_t, A_t, update_info = update_active_away(fw_stepsize, max_step, S_t, A_t, s_t, v_t_idx, step_type, epsilon, debug=debug)

    info['alphas'] = A_t
    perturbed_image = x_t + fw_stepsize * d_t
    #perturbed_image = sum([alpha * v for alpha, v in zip(A_t, S_t)])
    info['diff'] = torch.max(torch.abs(perturbed_image - x_t + fw_stepsize * d_t)).item()
    #info['minmax'] = (torch.min(perturbed_image).item(),torch.max(perturbed_image).item()) # debug
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    info.update(update_info)
    return perturbed_image, gap_FW, S_t, A_t, info

def fw_step_away_old(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method, alpha_remove_tol = 0.001):
    info = {}
    # alg from FW_varients.pdf
    # FW direction
    g_t_sign = g_t.sign()
    s_t = -epsilon * g_t_sign + x0
    d_t_FW = s_t - x_t
    # AWAY direction. From set of vertices already visited
    away_costs = []
    for v in S_t:
        away_costs.append(torch.sum(-g_t*v).item())
    v_t_idx = np.argmax(away_costs)
    v_t = S_t[v_t_idx]
    # at each iter x_t expressed by convex combination of active verticies
    #alpha_v_t = alphas_t[v_t_idx]
    d_t_AWAY = x_t - v_t
    #check optimality (FW gap)
    gap_FW = torch.sum(-g_t * d_t_FW).item()
    gap_AWAY = torch.sum(g_t*d_t_AWAY).item()
    info['gap_FW'] = gap_FW
    info['gap_AS'] = gap_AWAY


    # check which direction is closer to the gradient
    if (gap_FW >= gap_AWAY) or (len(S_t) == 1):
        info['step'] = 'FW'
        d_t = d_t_FW
        max_step = 1
        diffs = np.max([torch.max(torch.abs(s - s_t)).item() for s in S_t]) # check the inf norm of difference to see if vertex already active
        diff = np.max(diffs)
        idx = np.argmax(diffs)
        if diff < 0.001:
            print(f'whoo{diff}')
            s_t_idx = idx
        else:
            s_t_idx = -1
            S_t.append(d_t.clone().detach())
            A_t.append(max_step)
        info['S_idx'] = s_t_idx
    else:
        info['step'] = 'AS'
        info['S_idx'] = v_t_idx
        d_t = d_t_AWAY
        alpha_v_t = A_t[v_t_idx]
        max_step = 1 if alpha_v_t == 1 else alpha_v_t / (1 - alpha_v_t) # avoid divide by zero when alpha = 1

    # determine stepsize according to rule
    if stepsize_method.strat == 'ls':
        fw_stepsize = stepsize_method.stepsize_linesearch(x_t, d_t)
    else:
        fw_stepsize = stepsize_method.stepsize
    # clip stepsize according to rule to max_step as defined above
    fw_stepsize = min(fw_stepsize, max_step)

    if info['step'] == 'AS':
        # adjust the alpha value if doing away step
        A_t[v_t_idx] -= fw_stepsize

    # check if directions need to be removed from the active set
    remove_idx = list(np.where(np.array(A_t) < alpha_remove_tol))
    info['removed'] = remove_idx
    S_t = [s_k for s_k, a_k in zip(S_t, A_t) if a_k >= alpha_remove_tol]
    A_t = [a_k for a_k in A_t if a_k >= alpha_remove_tol]
    info['alphas'] = A_t
    # line-search for the best gamma (FW stepsize)
    perturbed_image = x_t + fw_stepsize * d_t
    #perturbed_image = sum([alpha * v for alpha, v in zip(A_t, S_t)])
    info['diff'] = torch.max(torch.abs(perturbed_image - x_t + fw_stepsize * d_t))
    #info['minmax'] = (torch.min(perturbed_image).item(),torch.max(perturbed_image).item()) # debug
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, gap_FW, S_t, A_t, info

def update_active_pair(gamma, S_t, A_t, s_t, epsilon):
    diffs = [torch.sum(torch.abs(s_t - s)).item() for s in S_t]#[torch.max(torch.abs(s_t - s)).item() for s in S_t]
    min_diff = min(diffs)
    arg = np.argmin(diffs)
    if min_diff < 0.9*epsilon:
        # s_t already in S_t
        s_t_idx = arg
    else:
        S_t.append(s_t)
        A_t.append(0.0)
        s_t_idx = len(S_t) - 1
    A_t = [a + gamma if i == s_t_idx else a - gamma for i, a in enumerate(A_t)]
    return S_t, A_t

def fw_step_pairwise(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method):
    info = {}
    # FW direction
    g_t_sign = g_t.sign()
    s_t = -epsilon * g_t_sign + x0
    d_t_FW = s_t - x_t
    # AWAY direction. From set of vertices already visited
    away_costs = []
    for v in S_t:
        away_costs.append(torch.sum(-g_t*v).item())
    v_t_idx = np.argmin(away_costs)
    v_t = S_t[v_t_idx]
    alpha_v_t = A_t[v_t_idx]
    max_step = alpha_v_t
    d_t_AWAY = x_t - v_t
    gap_FW = torch.sum(-g_t * d_t_FW).item()
    gap_AWAY = torch.sum(-g_t*d_t_AWAY).item()
    info['gap_FW'] = gap_FW
    info['gap_AS'] = gap_AWAY
    d_t = s_t - v_t
    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t, max_step)
    update_active_pair(fw_stepsize, S_t, A_t, s_t, epsilon)
    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, gap_FW, S_t, A_t, info

def fw_step_pairwise_alt(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method, alpha_remove_tol=0.01):
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t
    s_t = -epsilon * g_t_sign + x0
    d_t_FW = s_t - x_t

    d_t_PW = d_t + d_t_FW

    if stepsize_method.strat == 'ls':
        fw_stepsize = stepsize_method.stepsize_linesearch(x_t, d_t_PW)
    else:
        fw_stepsize = stepsize_method.stepsize

    perturbed_image = x_t + fw_stepsize * d_t_PW
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    v_t_tuple = tuple(v_t.tolist())

    if v_t_tuple not in S_t:
        S_t.append(v_t_tuple)
        A_t.append(0.0)  # initialize

    # update coefficients
    for i in range(len(A_t)):
        A_t[i] = (1 - fw_stepsize) * A_t[i]

    index_v_t = S_t.index(v_t_tuple)
    A_t[index_v_t] += fw_stepsize

    indices_to_remove = [i for i, alpha in enumerate(A_t) if alpha < alpha_remove_tol]
    for i in sorted(indices_to_remove, reverse=True):
        del A_t[i]
        del S_t[i]

    gap_FW = torch.sum(-d_t_PW * g_t).item()
    return perturbed_image, gap_FW, S_t, A_t, None



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
        A_t = [1]
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
            with torch.no_grad():
                if method == 'fgsm':
                    perturbed_image = fgsm_attack(x_t_denorm, epsilon, x_t_grad)
                elif method == 'fw':
                    perturbed_image, gap_FW, info = fw_step(x_t_denorm, epsilon, x_t_grad, x0_denorm, stepsize_method=stepsize_method)
                elif method == 'fw_momentum':
                    perturbed_image, m_t_last, gap_FW = fw_step_momentum(x_t_denorm, epsilon, x_t_grad, m_t_last, x0_denorm, stepsize_method=stepsize_method)
                elif method == 'fw_away':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_away(x_t_denorm, epsilon, x_t_grad, x0_denorm, S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_pair':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_pairwise(x_t_denorm, epsilon, x_t_grad, x0_denorm, S_t, A_t, stepsize_method=stepsize_method)
                elif method == 'fw_away_old':
                    perturbed_image, gap_FW, S_t, A_t, info = fw_step_away_old(x_t_denorm, epsilon, x_t_grad, x0_denorm, S_t, A_t, stepsize_method=stepsize_method)
            # Reapply normalization
            x_t = target_model.renorm(perturbed_image)#transforms.Normalize((0.1307,), (0.3081,))(perturbed_image).detach()

            # Re-classify the perturbed image
            x_t.requires_grad = False
            output = model(x_t)
            info['l_inf'] = torch.max(torch.abs(x0_denorm - perturbed_image))
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