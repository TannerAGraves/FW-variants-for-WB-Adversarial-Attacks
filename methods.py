import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

### ATTACKS
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def pgd_attack(x_t, epsilon, g_t, stepsize_method):
    perturbed_image = x_t + stepsize_method.get_stepsize(x_t, g_t) * g_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fw_step(x_t, epsilon, g_t, x0, stepsize_method):
    info = {}
    # alg from attacks.pdf. Modified to remove momentum.
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t

    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t)
    info['stepsize'] = fw_stepsize
    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # gap_FW = torch.sum(d_t * g_t).item()#torch.dot(d_t, g_t)
    gap_FW = torch.sum(-d_t * g_t).item()
    return perturbed_image, gap_FW, info

def fw_step_momentum(x_t, epsilon, g_t, m_t_last, x0, stepsize_method, momentum=0.2):
    # alg from attacks.pdf
    m_t = (1 - momentum) * g_t
    if m_t_last is not None:
        m_t += momentum * m_t_last
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

def fw_step_away(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method, debug=True):
    info = {}
    # alg from FW_varients.pdf
    # FW direction
    g_t_sign = g_t.sign()
    s_t = -epsilon * g_t_sign + x0
    d_t_FW = s_t - x_t
    # AWAY direction. From set of vertices already visited
    away_costs = []
    for v in S_t:
        away_costs.append(torch.sum(-g_t * v).item())  # negative here because were attacking
    v_t_idx = np.argmin(away_costs)  # docs have arg max
    v_t = S_t[v_t_idx]
    # at each iter x_t expressed by convex combination of active verticies
    # alpha_v_t = alphas_t[v_t_idx]
    d_t_AWAY = x_t - v_t
    # check optimality (FW gap)
    gap_FW = torch.sum(-g_t * d_t_FW).item()
    gap_AWAY = torch.sum(-g_t * d_t_AWAY).item()
    info['gap_FW'] = gap_FW
    info['gap_AS'] = gap_AWAY
    # info['gap_ASmax'] = torch.sum(g_t*(x_t - S_t[np.argmax(away_costs)])).item()

    # check which direction is closer to the gradient
    if (gap_FW >= gap_AWAY) or (len(S_t) == 1):
        step_type = 'FW'
        d_t = d_t_FW
        max_step = 1
    else:
        step_type = 'AS'
        d_t = d_t_AWAY
        alpha_v_t = A_t[v_t_idx]
        max_step = 1 if alpha_v_t == 1 else alpha_v_t / (1 - alpha_v_t)  # avoid divide by zero when alpha = 1
    info['step_type'] = step_type
    info['max_step'] = max_step
    # determine stepsize according to rule
    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t, max_step)
    info['stepsize'] = fw_stepsize

    S_t, A_t, update_info = update_active_away(fw_stepsize, max_step, S_t, A_t, s_t, v_t_idx, step_type, epsilon,
                                               debug=debug)

    info['alphas'] = A_t
    perturbed_image = x_t + fw_stepsize * d_t
    # perturbed_image = sum([alpha * v for alpha, v in zip(A_t, S_t)])
    info['diff'] = torch.max(torch.abs(perturbed_image - x_t + fw_stepsize * d_t)).item()
    # info['minmax'] = (torch.min(perturbed_image).item(),torch.max(perturbed_image).item()) # debug
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    info.update(update_info)
    return perturbed_image, gap_FW, S_t, A_t, info

def update_active_pair(gamma, S_t, A_t, s_t, epsilon):
    diffs = [torch.sum(torch.abs(s_t - s)).item() for s in S_t]  # [torch.max(torch.abs(s_t - s)).item() for s in S_t]
    min_diff = min(diffs)
    arg = np.argmin(diffs)
    if min_diff < 0.9 * epsilon:
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
        away_costs.append(torch.sum(-g_t * v).item())
    v_t_idx = np.argmin(away_costs)
    v_t = S_t[v_t_idx]
    alpha_v_t = A_t[v_t_idx]
    max_step = alpha_v_t
    d_t_AWAY = x_t - v_t
    gap_FW = torch.sum(-g_t * d_t_FW).item()
    gap_AWAY = torch.sum(-g_t * d_t_AWAY).item()
    info['gap_FW'] = gap_FW
    info['gap_AS'] = gap_AWAY
    d_t = s_t - v_t
    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t, max_step)
    update_active_pair(fw_stepsize, S_t, A_t, s_t, epsilon)
    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, gap_FW, S_t, A_t, info