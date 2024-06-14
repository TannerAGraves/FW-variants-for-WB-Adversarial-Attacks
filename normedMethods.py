import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods import *
### NORMS
def p2_norm(grad):
    norm = torch.sum(torch.abs(grad)**2)**(1/2)
    return grad / norm
def l1_norm(grad):
    norm = torch.sum(torch.abs(grad))
    return grad / norm

def l2_norm(grad):
    norm = torch.sqrt(torch.sum(grad ** 2))
    return grad / norm

### NORMED FUNCTIONS
def fw_step_p2(x_t, epsilon, g_t, x0, stepsize_method):
    # alg from attacks.pdf. Modified to remove momentum.
    g_t = p2_norm(g_t)
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t

    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t)

    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # gap_FW = torch.sum(d_t * g_t).item()#torch.dot(d_t, g_t)
    gap_FW = torch.sum(-d_t * g_t).item()
    return perturbed_image, gap_FW
def fw_step_l1(x_t, epsilon, g_t, x0, stepsize_method):
    # alg from attacks.pdf. Modified to remove momentum.
    g_t = l1_norm(g_t)
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t

    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t)

    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # gap_FW = torch.sum(d_t * g_t).item()#torch.dot(d_t, g_t)
    gap_FW = torch.sum(-d_t * g_t).item()
    return perturbed_image, gap_FW
def fw_step_l2(x_t, epsilon, g_t, x0, stepsize_method):
    # alg from attacks.pdf. Modified to remove momentum.
    g_t = l2_norm(g_t)
    g_t_sign = g_t.sign()
    v_t = -epsilon * g_t_sign + x0
    d_t = v_t - x_t

    fw_stepsize = stepsize_method.get_stepsize(x_t, d_t)

    perturbed_image = x_t + fw_stepsize * d_t
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # gap_FW = torch.sum(d_t * g_t).item()#torch.dot(d_t, g_t)
    gap_FW = torch.sum(-d_t * g_t).item()
    return perturbed_image, gap_FW

def fw_step_away_p2(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method, debug=True):
    info = {}
    # alg from FW_varients.pdf
    # FW direction
    g_t = p2_norm(g_t)
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
def fw_step_away_l1(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method, debug=True):
    info = {}
    # alg from FW_varients.pdf
    # FW direction
    g_t = l1_norm(g_t)
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
def fw_step_away_l2(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method, debug=True):
    info = {}
    # alg from FW_varients.pdf
    # FW direction
    g_t = l2_norm(g_t)
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

def fw_step_pairwise_p2(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method):
    info = {}
    # FW direction
    g_t = p2_norm(g_t)
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
def fw_step_pairwise_l1(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method):
    info = {}
    # FW direction
    g_t = l1_norm(g_t)
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
def fw_step_pairwise_l2(x_t, epsilon, g_t, x0, S_t, A_t, stepsize_method):
    info = {}
    # FW direction
    g_t = l2_norm(g_t)
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