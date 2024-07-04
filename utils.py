import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from AttackStep import AttackStep
from tqdm import tqdm
import matplotlib.pyplot as plt

class LMO:
    def __init__(self, epsilon, x0, p):
        self.x0 = x0.clone().detach()  # Ensure x0 is not modified elsewhere
        self.epsilon = epsilon
        self.p = p
        # Select the appropriate LMO method based on the norm p
        if p == -1:
            self.method = self._LMO_inf
        elif p == 1:
            self.method = self._LMO_l1
        elif p == 2:
            self.method = self._LMO_l2
        else:
            raise Exception(f"invalid choice of norm {p}")

    def get(self, g_t):
        return self.method(g_t)

    def _LMO_inf(self, g_t):
        g_t_sign = g_t.sign()  # Get the sign of the gradient
        s_t = -self.epsilon * g_t_sign + self.x0  # Update step for l_inf norm
        return s_t

    def _LMO_l1(self, gradient):
        abs_gradient = gradient.abs()  # Get the absolute value of the gradient
        sign_gradient = gradient.sign()  # Get the sign of the gradient
        perturbation = torch.zeros_like(gradient)
        # For each example in the batch, select the component with the maximum absolute gradient
        for i in range(gradient.size(0)):
            _, idx = torch.topk(abs_gradient[i].view(-1), 1)
            perturbation[i].view(-1)[idx] = sign_gradient[i].view(-1)[idx]
        return self.epsilon * perturbation

    def _LMO_l2(self, g_t): # from arxiv version of attacks.pdf
        g_t_norm = torch.norm(g_t, p=2, dim=-1, keepdim=True)
        s_t = -self.epsilon * g_t / g_t_norm + self.x0
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

        # if self.specific_label is not None:
        #     # Targeting a specific incorrect label
        #     specific_labels = torch.full_like(targets, self.specific_label)
            
        #     # Compute log probabilities
        #     log_probs = F.log_softmax(outputs, dim=1)
            
        #     # Get the log probabilities of the specific label
        #     specific_log_probs = log_probs.gather(1, specific_labels.unsqueeze(1)).squeeze(1)
            
        #     # Compute the negative log likelihood
        #     loss = -specific_log_probs
        #     return loss.mean()
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
        
    def forward1(self, outputs, targets):
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
            adv_target = torch.full((outputs.size(0),), self.specific_label, dtype=torch.long)
            if isinstance(targets, int):
                targets = torch.full((outputs.size(0),), targets, dtype=torch.long)
            
            return F.cross_entropy(outputs, adv_target)
        else:
            if isinstance(targets, int):
                targets = torch.full((outputs.size(0),), targets, dtype=torch.long)
            return -F.nll_loss(outputs, targets)
        
class stepsize():
    def __init__(self, model, strat, x0, fixed_size = 1, ls_criterion=None, ls_target = None, ls_num_samples=50, renorm = None):
        if isinstance(strat, (float, int)):
            fixed_size = strat
            strat = 'fixed'
        self.model = model
        self.strat = strat
        self.fixed_size = fixed_size
        # used for amjo
        self.x0 = x0
        self.x_t_grad = None
        self.loss0 = None

        # used for ls
        self.ls_criterion = ls_criterion
        self.ls_target = ls_target
        self.ls_num_samples = ls_num_samples
        self.renorm = renorm

        self.stepsize = fixed_size # will be updated if using other method
        if self.strat not in ['fixed', 'ls', 'decay', 'amjo']:
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
            self._sizes_ls = steps
            for step in steps:
                new_x = self.renorm(x_tc + step * d_tc)
                output = self.model(new_x)
                loss = self.ls_criterion(output, self.ls_target).item()
                losses.append(loss)
        best_idx = np.argmin(losses)
        self.stepsize = steps[best_idx]
        
        self._loss_ls = losses
        return self.stepsize
    

    # def stepsize_linesearch(self, x_t, d_t, max_step=1):
    #     x_tc = x_t.clone().detach()
    #     d_tc = d_t.clone().detach()
    #     losses = []
    #     with torch.no_grad():
    #         # Create a tensor of steps
    #         steps = torch.linspace(max_step / self.ls_num_samples, max_step, self.ls_num_samples).to(x_t.device).view(-1, 1, 1, 1)

    #         # Generate all possible x_t + step * d_t combinations in parallel
    #         x_t_steps = x_tc + steps * d_tc

    #         # Flatten the batch and step dimensions for efficient processing
    #         batch_size = x_t.size(0)
    #         num_steps = steps.size(0)
    #         #x_t_steps = x_t_steps.view(batch_size * num_steps, *x_t.size()[1:])

    #         # Pass through the model in parallel
    #         output = self.model(x_t_steps)

    #         # # Compute losses
    #         output = output.view(batch_size, num_steps, -1)
    #         losses = []
    #         for out in output:
    #             losses.append(self.ls_criterion(out, self.ls_target).item())
            
    #         # Find the index of the minimum loss for each example in the batch
    #         best_idx = np.argmin(losses)
            
    #         # Get the corresponding steps
    #         self.stepsize = steps[best_idx].item()
            
    #     return self.stepsize




    def stepsize_armijo(self, x_t, d_t, max_step = 1):
        info_step = {}
        x_tc = x_t.clone().detach()
        d_tc = d_t.clone().detach()
        step_size = max_step
        best_stepsize = max_step
        gamma = 0.5
        delta = 0.5
        initial_loss = self.ls_criterion(self.model(self.renorm(x_tc)), self.ls_target)#F.cross_entropy(self.model(x_k), target).item()
        min_loss = float('inf')
        
        while step_size > 1e-4:
            new_point = self.renorm(x_tc + step_size * d_tc)
            new_loss = self.ls_criterion(self.model(new_point), self.ls_target)#F.cross_entropy(self.model(new_point), target).item()
            RHS = initial_loss + gamma * step_size * torch.sum(self.x_t_grad * d_tc).item()
            if new_loss < min_loss:
                min_loss = new_loss
                best_stepsize = step_size
            if new_loss <= RHS:
                return step_size
            
            step_size *= delta
        
        return best_stepsize


    def get_stepsize(self, x_t, d_t, max_step = 1):
        if self.strat == 'ls':
            #fw_stepsize = self.exact_ls(x_t, d_t, max_step)
            fw_stepsize = self.stepsize_linesearch(x_t, d_t, max_step)
        elif self.strat == 'amjo':
            fw_stepsize = self.stepsize_armijo(x_t, d_t, max_step)
        else:
            fw_stepsize = self.stepsize
        fw_stepsize = min(fw_stepsize, max_step)
        return fw_stepsize
    
def early_stopper(criterion, t, success, first_success, info, gap_FW_tol, max_fw_iter, gap_FW):
    new_correct = 0
    if (criterion == 'pred') and first_success:
        # The attack was successful so the classification was not correct
        stop = True
        stop_reason = 'pred'
    elif (criterion == 'gap_FW') and (gap_FW < gap_FW_tol):
        if not success: # attack failed
            new_correct +=1
        stop = True
        stop_reason = 'gap'
    elif (t == max_fw_iter - 1): # Stop condition: Hit max FW iters
        stop = True
        stop_reason = 'max_iter'
        if not success: # attack failed
            new_correct +=1
    elif (criterion == 'gap_pairwise') and (info is not None) and (info['gap_pairwise'] < gap_FW_tol):
        if not success: # attack failed
            new_correct +=1
        stop = True
        stop_reason = 'gap_pairwise'
    else:
        # no stop criteria met, continue
        stop = False
        stop_reason = None
    return stop, stop_reason, new_correct

class example_saver():
    def __init__(self, num_adv_ex = 10, num_failed_ex = 10) -> None:
        self.num_adv_ex = num_adv_ex
        self.num_failed_ex = num_failed_ex
        self.adv_true = []
        self.adv_pred = []
        self.adv_x0 = []
        self.adv_atk = []
        self.adv_xt = []
        self.adv_true_init_prob = []
        self.adv_final_prob = []
        self.adv_ex = []
        self.failed_ex = []
        self.info = []
        pass

    def save_ex(self, perturbed_image, x0, true, final_pred, success, true_class_prob0, pred_class_prob):
        self.info.append(true)
        
        if (len(self.adv_ex) >= self.num_adv_ex) and (len(self.failed_ex) >= self.num_failed_ex):
            return

        atk = perturbed_image - x0
        atk = atk.detach().squeeze().cpu().numpy()
        x0 = x0.detach().squeeze().cpu().numpy()
        ex = perturbed_image.squeeze().detach().cpu().numpy()
        if len(atk.shape) > 2:
            atk = np.transpose(atk, (1, 2, 0))
            x0 = np.transpose(x0, (1, 2, 0))
            ex = np.transpose(ex, (1, 2, 0))


        # Save some adv examples for visualization later
        if success and (len(self.adv_ex) < self.num_adv_ex):
            self.adv_ex.append( (true.item(), final_pred.item(), ex) )
            self.adv_true.append(true.item())
            self.adv_pred.append(final_pred.item())
            self.adv_x0.append(x0)
            self.adv_atk.append(atk)
            self.adv_true_init_prob.append(true_class_prob0)
            self.adv_final_prob.append(pred_class_prob)
            self.adv_xt.append(ex)
        if (not success) and (len(self.failed_ex) < self.num_failed_ex):
            self.failed_ex.append( (true.item(), final_pred.item(), ex) )


def test(target_model, device, epsilon,num_fw_iter, num_test = 1000, method='fw', early_stopping = None, fw_stepsize_rule = 1, gap_FW_tol = 0.05, targeted = False, ex_saver=None, norm_p=-1, seed=42):
    testloader = target_model.remake_testloader(seed)
    model = target_model.model

    # Accuracy counter
    correct = 0
    adv_examples = []
    hist = []
    ex_num = 0
    # Loop over all examples in test set
    for x0, target in tqdm(testloader):
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
        lmo = LMO(epsilon, x0_denorm, norm_p)
        stepsize_method = stepsize(model, fw_stepsize_rule, x0_denorm, ls_criterion=criterion, ls_target=target, renorm = target_model.renorm)
        attackStep = AttackStep(method, epsilon, x0_denorm, lmo, stepsize_method)
        #x_t.requires_grad = True  #Set requires_grad attribute of tensor. Important for Attack
        had_first_success = False
        gap_FW = None
        info = None
        true_class_prob0 = 0

        for t in range(num_fw_iter):
            # Step size calculation
            if stepsize_method.strat == 'decay':
                stepsize_method.set_stepsize_decay(t)
            x_t.requires_grad = True
            # Forward pass the data through the model
            output = model(x_t)
            class_probs = torch.softmax(output,dim=1)
            # Calculate the loss
            loss = criterion(output, target)

            if t==0:
                # save init confidence in true class
                true_class_prob0 = class_probs[0, target.item()].item()
                stepsize_method.loss0 = loss.item()
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, don't bother attacking, just move on
            if (init_pred.item() != target.item()) and (t == 0):
                ex_num -= 1 # don't count this example
                break
            
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            x_t_grad = x_t.grad#.data
            # Restore the data to its original scale
            x_t_denorm = target_model.denorm(x_t)

            # save information needed for linesearching stepsize rules
            stepsize_method.x_t_grad = x_t_grad.clone().detach()

            # Call Attack
            with torch.no_grad():
                perturbed_image, gap_FW, info = attackStep.step(x_t_denorm, x_t_grad)
            
            # Reapply normalization
            x_t = target_model.renorm(perturbed_image)#transforms.Normalize((0.1307,), (0.3081,))(perturbed_image).detach()
            # Re-classify the perturbed image
            x_t.requires_grad = False
            output = model(x_t)
            info['l_inf'] = torch.max(torch.abs(x0_denorm - perturbed_image)).item()
            info['mdlLoss'] = loss.item()

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if final_pred.item() == target.item():
                success = False
                first_success = False
            else:
                first_success =  not had_first_success
                had_first_success = True
                success = True
            stop, stop_reason, new_correct = early_stopper(early_stopping, t, success, first_success, info, gap_FW_tol, num_fw_iter, gap_FW)
            correct += new_correct

            # metric logging
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
            if targeted:
                hist_iter['adv_target'] = adv_target
                targeted_success = (final_pred.item() == adv_target)
                info['targeted_success'] = targeted_success
            else:
                targeted_success = False
        
            if info is not None:
                hist_iter.update(info) # some methods output dict containing info at each step
            hist.append(hist_iter)
            if stop:
                class_probs = torch.softmax(output,dim=1)
                pred_class_prob = class_probs[0, final_pred.item()].item()
                if ex_saver is not None:
                    save_as_adv = targeted_success if targeted else success
                    ex_saver.save_ex(perturbed_image, x0_denorm, target, final_pred, save_as_adv, true_class_prob0, pred_class_prob)
                break
        ex_num += 1
        if ex_num >= num_test: # limit test set for speed
            break

    # Calculate final accuracy for this epsilon
    final_acc = correct/num_test
    print(f"Epsilon: {epsilon}\tCorrect Classifications (Failed Attacks) = {correct} / {num_test} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, pd.DataFrame(hist)

def plot_convergence(hist_dfs, algs):
    final_hist_dfs = [hist.groupby('example_idx').tail(1) for hist in hist_dfs]
    for i, final_hist in enumerate(final_hist_dfs):
        targeted = 'targeted_success' in final_hist.columns
        ASR_text = 'Targeted Attack Success Rate' if targeted else 'Attack Success Rate'
        alg = algs[i]
        print(alg)
        print(f"\t{ASR_text}: {final_hist['targeted_success' if targeted else 'success'].mean()}")
        print(f"\tAvg iters: {final_hist['FW_iter'].mean()}")
        if alg == 'fw_away':
            st = hist_dfs[i].groupby('step_type').size().to_dict()
            if len(st.keys()) > 1:
                print(f"\tStep Types: FW {st['FW']}, AS {st['AS']}. {100 * st['AS'] / (st['FW'] + st['AS']):.1f}% Away Steps.")
            else:
                print("\t100% FW steps")
        plt.plot(hist_dfs[i].groupby('FW_iter')['gap_FW'].mean(), label=algs[i])
    plt.xlabel("Iteration t")
    plt.ylabel("Average FW gap")
    plt.legend()
    plt.show()

def display_examples(ex_saver, epsilon, classes, show_atk_mag = False, n_col = 3, offset = 2):
    # classes either int corresponding to number of classes, or list with class names
    if isinstance(classes, int):
        classes = list(range(classes))
    dim = len(ex_saver.adv_x0[0].shape)
    
    
    n_col = min(n_col, len(ex_saver.adv_pred))
    cmap = None if dim > 2 else 'gray'

    fig, axs = plt.subplots(3,n_col)
    for i in range(n_col):
        ex_idx = i + offset
        true = ex_saver.adv_true[ex_idx]
        pred = ex_saver.adv_pred[ex_idx]
        x0 = ex_saver.adv_x0[ex_idx]
        atk = (ex_saver.adv_atk[ex_idx]+epsilon)/(2*epsilon)
        if show_atk_mag:
            atk = np.abs(atk-0.5)
            atk *= 1/np.max(atk)
        atk = np.clip(atk,0,1)
        xt = ex_saver.adv_xt[ex_idx]
        prob_true = ex_saver.adv_true_init_prob[ex_idx]
        prob_adv = ex_saver.adv_final_prob[ex_idx]
        axs[0, i].imshow(x0, cmap = cmap)
        axs[0, i].axis('off')
        axs[0, i].set_title(f"Orginal: {classes[true]}\np = {prob_true:.2f}", fontsize=10)
        axs[1, i].imshow(atk, cmap = cmap)
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Scaled offset:\nϵ = {epsilon}", fontsize=10)
        axs[2, i].imshow(xt, cmap = cmap)
        axs[2, i].axis('off')
        axs[2, i].set_title(f"Adv Pred: {classes[pred]}\np = {prob_adv:.2f}", fontsize=10)
    plt.tight_layout(pad=1.0, w_pad=-15, h_pad=1.0)
    plt.show()