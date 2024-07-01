import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

### ATTACKS
class AttackStep:
    def __init__(self, method, epsilon, x0_denorm, lmo, stepsize_method=None, momentum=0.2):
        self.method = method
        self.epsilon = epsilon
        self.x0_denorm = x0_denorm
        self.stepsize_method = stepsize_method
        self.lmo = lmo
        self.momentum = momentum
        self.m_t_last = None
        self.S_t = [x0_denorm]
        self.A_t = [1]

    def step(self, x_t_denorm, x_t_grad):
        if self.method == 'fgsm':
            return self.fgsm_attack(x_t_denorm, x_t_grad)
        elif self.method == 'fw':
            return self.fw_step(x_t_denorm, x_t_grad)
        elif self.method == 'fw_momentum':
            return self.fw_step_momentum(x_t_denorm, x_t_grad)
        elif self.method == 'fw_away':
            return self.fw_step_away(x_t_denorm, x_t_grad)
        elif self.method == 'fw_pair':
            return self.fw_step_pairwise(x_t_denorm, x_t_grad)
        elif self.method == 'fw_pair_test':
            return self.fw_step_pairwise_test(x_t_denorm, x_t_grad)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
    def fgsm_attack(self, image, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + self.epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    def pgd_attack(self, x_t, g_t):
        perturbed_image = x_t + self.stepsize_method.get_stepsize(x_t, g_t) * g_t
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def fw_step(self, x_t, g_t):
        info = {}
        # Use LMO to compute the attack direction
        v_t = self.lmo.get(g_t)
        d_t = v_t - x_t
        self.d_t = d_t
        fw_stepsize = self.stepsize_method.get_stepsize(x_t, d_t)
        info['stepsize'] = fw_stepsize
        perturbed_image = x_t + fw_stepsize * d_t
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        gap_FW = torch.sum(-d_t * g_t).item()
        return perturbed_image, gap_FW, info

    def fw_step_momentum(self, x_t, g_t,  momentum=0.2):
        # alg from attacks.pdf
        info = {}
        m_t = (1 - momentum) * g_t
        if self.m_t_last is not None:
            m_t += momentum * self.m_t_last
        v_t = self.lmo.get(m_t)
        d_t = v_t - x_t

        fw_stepsize = self.stepsize_method.get_stepsize(x_t, d_t)

        gap_FW = torch.sum(-d_t * g_t).item()
        perturbed_image = x_t + fw_stepsize * d_t
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        self.m_t_last = m_t.clone().detach()
        return perturbed_image, gap_FW, info

    def update_active_away(self, gamma, s_t, v_t_idx, step_type, debug = True):
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
            if abs(gamma - 1) < 0.0001:
                # drop step
                self.S_t = [s_t]
                self.A_t = [1]
                debug_info['drop_step'] = 'FW'
            else:
                ## UPDATE S
                # need to check if vertex is already in S
                diffs = [torch.sum(torch.abs(s_t - s)).item() for s in self.S_t]
                min_diff = min(diffs)
                arg = np.argmin(diffs)
                if min_diff < 0.9*self.epsilon:
                    # s_t already in S_t
                    s_t_idx = arg
                    debug_info['FW_revisit'] = True
                else:
                    self.S_t.append(s_t)
                    self.A_t.append(0.0)
                    s_t_idx = -1
                #debug_info["min_revisit_diff"] = min_diff
                
                ## UPDATE ALPHAS
                self.A_t = [(1 - gamma) * alpha for alpha in self.A_t]
                self.A_t[s_t_idx] += gamma
        elif step_type == 'AS':
            if False: #gamma >= gamma_max:
                # drop step: remove atom and alpha
                # logic changed to check if alpha is zeroed (see below)
                self.A_t.pop(v_t_idx)
                self.S_t.pop(v_t_idx)
                debug_info['drop_step'] = 'AS'
            else:
                ## UPDATE ALPHAS
                self.A_t = [(1 + gamma) * alpha for alpha in self.A_t]
                self.A_t[v_t_idx] -= gamma
        else:
            raise Exception("Step must be FW or AS")
        
        # Detect if the away step was a dropstep
        # done by seeing if any of the alphas where zeroed out.
        i = 0
        while i < len(self.A_t):
            alpha_i = self.A_t[i]
            if alpha_i <= 0:
                debug_info['drop_stepAS'] = 'AS'
                self.A_t.pop(i)
                self.S_t.pop(i)
            else:
                i += 1

        debug_info['v_t_idx'] = v_t_idx
        if debug:
            info.update(debug_info)
        return self.S_t, self.A_t, info

    def fw_step_away(self, x_t, g_t, debug=True):
        # alg from FW_varients.pdf
        use_conv_comb_x_t = False
        info = {}
        debug_info = {}
        
        # FW direction
        s_t = self.lmo.get(g_t)
        d_t_FW = s_t - x_t
        # AWAY direction. From set of vertices already visited
        away_sign = 1
        away_costs = []
        for v in self.S_t:
            away_costs.append(torch.sum(away_sign * g_t * v).item()) 
        v_t_idx = np.argmax(away_costs) 
        v_t = self.S_t[v_t_idx]
        d_t_AWAY = x_t - v_t
        # check optimality (FW gap)
        gap_FW = torch.sum(-g_t * d_t_FW).item()
        gap_AWAY = torch.sum(-g_t * d_t_AWAY).item()
        info['gap_FW'] = gap_FW
        info['gap_AS'] = gap_AWAY
        debug_info['awayCosts'] = away_costs

        # check which direction is closer to the -gradient
        if (gap_FW >= gap_AWAY) or (len(self.S_t) == 1): # don't step away if only one vertex in S_t
            step_type = 'FW'
            d_t = d_t_FW
            max_step = 1
        else:
            step_type = 'AS'
            d_t = d_t_AWAY
            alpha_v_t = self.A_t[v_t_idx]
            max_step = 1 if alpha_v_t == 1 else alpha_v_t / (1 - alpha_v_t)  # avoid divide by zero when alpha = 1
        self.d_t = d_t
        info['step_type'] = step_type
        debug_info['max_step'] = max_step
        # determine stepsize according to rule
        fw_stepsize = self.stepsize_method.get_stepsize(x_t, d_t, max_step)
        info['stepsize'] = fw_stepsize

        self.S_t, self.A_t, update_info = self.update_active_away(fw_stepsize, s_t, v_t_idx, step_type,
                                                debug=debug)

        ## UPDATE x_t
        perturbed_image_step = x_t + fw_stepsize * d_t
        perturbed_image_alpha = sum([alpha * v for alpha, v in zip(self.A_t, self.S_t)])
        if use_conv_comb_x_t: # use x_t = A_t.T * S_t or x_t + gamma * d_t ?
            perturbed_image = perturbed_image_alpha
        else:
            perturbed_image = perturbed_image_step
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # LOGGING and DEBUG info
        info['alphas'] = self.A_t
        debug_info['L_inf_step'] = torch.max(torch.abs(perturbed_image_step - self.x0_denorm)).item()
        debug_info['L_inf_alpha'] = torch.max(torch.abs(perturbed_image_alpha - self.x0_denorm)).item()
        alpha_np = ((perturbed_image_alpha).squeeze(0).permute(1, 2, 0).numpy()).clip(0,1)
        step_np = ((perturbed_image_step).squeeze(0).permute(1, 2, 0).numpy()).clip(0,1)
        debug_info['step_alpha_diffFactor'] = (alpha_np - step_np).sum() / self.epsilon
        info.update(update_info)
        if debug:
            info.update(debug_info)
        self.last_d = d_t
        return perturbed_image, gap_FW, info

    def update_active_pair(self, gamma, s_t, v_t_idx, info):
        drop = True
        diffs = [torch.sum(torch.abs(s_t - s)).item() for s in self.S_t]  # [torch.max(torch.abs(s_t - s)).item() for s in S_t]
        min_diff = min(diffs)
        arg = np.argmin(diffs)
        if min_diff < 0.9 * self.epsilon:
            # s_t already in S_t
            s_t_idx = arg
        else:
            self.S_t.append(s_t)
            self.A_t.append(0.0)
            s_t_idx = -1
        #self.A_t = [a + gamma if i == s_t_idx else a - gamma for i, a in enumerate(self.A_t)]
        self.A_t[s_t_idx] += gamma
        self.A_t[v_t_idx] -= gamma
        
        tol = 0.001
        i=0
        while i < len(self.A_t):
            alpha_v_i = self.A_t[i]
            s_t_i = self.S_t[i]
            if abs(1 - alpha_v_i) < tol:
                self.A_t = [1.0]
                self.S_t = [s_t_i]
                break
            elif self.A_t[i] < tol:
                if drop:
                    self.A_t.pop(i)
                    self.S_t.pop(i)
                else:
                    pass
            else:
                i += 1
        info['A_t'] = copy.deepcopy(self.A_t)
        return self.S_t, self.A_t

    def fw_step_pairwise(self, x_t, g_t):
        use_conv_comb_x_t = True
        info = {}
        debug_info = {}

        # Using LMO to compute s_t
        s_t = self.lmo.get(g_t)
        d_t_FW = s_t - x_t

        # AWAY direction. From set of vertices already visited
        away_costs = []
        for v in self.S_t:
            away_costs.append(torch.sum(g_t * v).item())
        v_t_idx = np.argmax(away_costs)
        v_t = self.S_t[v_t_idx]
        alpha_v_t = self.A_t[v_t_idx]
        max_step = alpha_v_t
        d_t_AWAY = x_t - v_t
        self.away_costs = away_costs

        gap_FW = torch.sum(-g_t * d_t_FW).item()
        gap_AWAY = torch.sum(-g_t * d_t_AWAY).item()
        info['gap_FW'] = gap_FW
        info['gap_AS'] = gap_AWAY

        d_t = d_t_FW + d_t_AWAY #s_t - v_t
        self.d_t = d_t
        fw_stepsize = self.stepsize_method.get_stepsize(x_t, d_t, max_step)

        self.S_t, self.A_t = self.update_active_pair(fw_stepsize, s_t, v_t_idx, info)
        perturbed_image_step = x_t + fw_stepsize * d_t
        perturbed_image_alpha = sum([alpha * v for alpha, v in zip(self.A_t, self.S_t)])
        if use_conv_comb_x_t: 
            perturbed_image = perturbed_image_alpha
        else:
            perturbed_image = perturbed_image_step
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        info.update(debug_info)

        return perturbed_image, gap_FW, info
    
    def update_active_pair_test(self, gamma, s_t, v_t_idx, debug = True):
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
        if abs(gamma - 1) < 0.0001:
            # drop step
            self.S_t = [s_t]
            self.A_t = [1.0]
            debug_info['drop_step'] = 'FW'
        else:
            ## UPDATE S
            # need to check if vertex is already in S
            diffs = [torch.sum(torch.abs(s_t - s)).item() for s in self.S_t]
            min_diff = min(diffs)
            arg = np.argmin(diffs)
            debug_info['min_revisit_diff'] = min_diff
            if min_diff < 0.9*self.epsilon:
                # s_t already in S_t
                s_t_idx = arg
                debug_info['FW_revisit'] = True
            else:
                self.S_t.append(s_t)
                self.A_t.append(0.0)
                s_t_idx = -1
            #debug_info["min_revisit_diff"] = min_diff
            
            ## UPDATE ALPHAS
            #self.A_t = [(1 - gamma) * alpha for alpha in self.A_t]
            self.A_t[s_t_idx] += gamma
            if False: #gamma >= gamma_max:
                # drop step: remove atom and alpha
                # logic changed to check if alpha is zeroed (see below)
                self.A_t.pop(v_t_idx)
                self.S_t.pop(v_t_idx)
                debug_info['drop_step'] = 'AS'
            else:
                ## UPDATE ALPHAS
                #self.A_t = [(1 + gamma) * alpha for alpha in self.A_t]
                self.A_t[v_t_idx] -= gamma
        
        # Detect if the away step was a dropstep
        # done by seeing if any of the alphas where zeroed out.
        i = 0
        while i < len(self.A_t):
            alpha_i = self.A_t[i]
            if (alpha_i <= 0) and (len(self.A_t) > 1):
                debug_info['drop_step'] = 'AS'
                self.A_t.pop(i)
                self.S_t.pop(i)
            else:
                i += 1

        debug_info['v_t_idx'] = v_t_idx
        if debug:
            info.update(debug_info)
        return self.S_t, self.A_t, info
    
    def fw_step_pairwise_test(self, x_t, g_t, debug=True):
        # alg from FW_varients.pdf
        use_conv_comb_x_t = True
        info = {}
        debug_info = {}
        
        # FW direction
        s_t = self.lmo.get(g_t)
        d_t_FW = s_t - x_t
        # AWAY direction. From set of vertices already visited
        away_sign = 1
        away_costs = []
        for v in self.S_t:
            away_costs.append(torch.sum(away_sign * g_t * v).item()) 
        v_t_idx = np.argmax(away_costs) 
        v_t = self.S_t[v_t_idx]
        d_t_AWAY = x_t - v_t
        # check optimality (FW gap)
        gap_FW = torch.sum(-g_t * d_t_FW).item()
        gap_AWAY = torch.sum(-g_t * d_t_AWAY).item()
        info['gap_FW'] = gap_FW
        info['gap_AS'] = gap_AWAY
        debug_info['awayCosts'] = away_costs

        # # check which direction is closer to the -gradient
        # if (gap_FW >= gap_AWAY) or (len(self.S_t) == 1): # don't step away if only one vertex in S_t
        #     step_type = 'FW'
        #     d_t = d_t_FW
        #     max_step = 1
        # else:
        #     step_type = 'AS'
        #     d_t = d_t_AWAY
        #     alpha_v_t = self.A_t[v_t_idx]
        #     max_step = 1 if alpha_v_t == 1 else alpha_v_t / (1 - alpha_v_t)  # avoid divide by zero when alpha = 1
        # info['step_type'] = step_type
        # debug_info['max_step'] = max_step
        # DEFINE PAIRWISE DIRECTION
        d_t = s_t - v_t
        alpha_v_t = self.A_t[v_t_idx]
        max_step = alpha_v_t
        info['max_stepsize'] = max_step
        # determine stepsize according to rule
        fw_stepsize = self.stepsize_method.get_stepsize(x_t, d_t, max_step)
        info['stepsize'] = fw_stepsize

        self.S_t, self.A_t, update_info = self.update_active_pair_test(fw_stepsize, s_t, v_t_idx,
                                                debug=debug)

        ## UPDATE x_t
        perturbed_image_step = x_t + fw_stepsize * d_t
        perturbed_image_alpha = sum([alpha * v for alpha, v in zip(self.A_t, self.S_t)])
        if use_conv_comb_x_t: # use x_t = A_t.T * S_t or x_t + gamma * d_t ?
            perturbed_image = perturbed_image_alpha
        else:
            perturbed_image = perturbed_image_step
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # LOGGING and DEBUG info
        info['alphas'] = self.A_t
        debug_info['L_inf_step'] = torch.max(torch.abs(perturbed_image_step - self.x0_denorm)).item()
        debug_info['L_inf_alpha'] = torch.max(torch.abs(perturbed_image_alpha - self.x0_denorm)).item()
        alpha_np = ((perturbed_image_alpha).squeeze(0).permute(1, 2, 0).numpy()).clip(0,1)
        step_np = ((perturbed_image_step).squeeze(0).permute(1, 2, 0).numpy()).clip(0,1)
        debug_info['step_alpha_diffFactor'] = (alpha_np - step_np).sum() / self.epsilon
        info.update(update_info)
        if debug:
            info.update(debug_info)
        self.last_d = d_t
        return perturbed_image, gap_FW, info