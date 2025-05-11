import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize
from itertools import product
from tqdm import tqdm
from matplotlib.font_manager import FontProperties
def lr(current_step, max_lr=2e-4, min_lr=2e-5, total_steps=60000, warmup_steps=500, lr_method='cosine'):
    
    if lr_method == 'constant':
        return max_lr
    
    if current_step <= warmup_steps:
        # Note here we compute S1 and S2 regarding lr as max_lr in warmup stages (but linear warmup in real training). We discuss the reason in section 3.4 of our paper.
        return max_lr
    num_steps_ = current_step - warmup_steps
    annealing_steps_ = total_steps - warmup_steps
    delta_lr = max_lr - min_lr
    
    if lr_method == 'linear':
        decay_ratio = float(num_steps_) / float(annealing_steps_)
        coeff = (1.0 - decay_ratio)
        current_lr = min_lr + coeff * delta_lr

    elif lr_method == 'cosine':
        decay_ratio = float(num_steps_) / float(annealing_steps_)
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        current_lr = min_lr + coeff * delta_lr

    else:
        raise Exception('{} decay style is not supported.'.format(lr_method))
    
    return current_lr

def Howe_Scaling_Law(step, lr_method, L0, A, C, alpha):
    predict_loss = L0 + A*(1/S1[lr_method][step])**alpha - C*S2[lr_method][step]
    return predict_loss

def huber_loss(residual, delta):
    return np.where(np.abs(residual) < delta , 0.5*((residual)**2), delta*np.abs(residual) - 0.5*(delta**2))

def objective(params):
    loss = 0
    for fitting_lr_method in fitting_lr_methods:
        indices = [i for i, lr_method in enumerate(lr_methods_data) if fitting_lr_method == lr_method]
        predict_losses = Howe_Scaling_Law(fitting_steps[indices], fitting_lr_method, *params)
        residual = np.log(fitting_losses[indices]) - np.log(predict_losses)
        loss += huber_loss(residual, huber_delta).sum() 
    return loss

# Load or replace your data here ...

fitting_steps = [i for i in range(1000, 20001, 1000)] + [i for i in range(1000, 20001, 1000)]
fitting_losses = [
    3.6696358, 3.3254106, 3.1908393, 3.1158533, 3.0604768, 3.020384, 2.9880967, 2.9612265, 2.9395165, 2.9195616, 2.9030452, 2.8881907, 2.8767889, 2.8654318, 2.8565135, 2.8489892, 2.843639, 2.8398669, 2.8380618, 2.8376071,
    3.6714418, 3.325626, 3.1884103, 3.111947, 3.058354, 3.0211415, 2.9874816, 2.96082, 2.9406552, 2.9241402, 2.9054413, 2.8942103, 2.8807929, 2.8702695, 2.862175, 2.8530996, 2.8448784, 2.8383856, 2.8326995, 2.8294308
    ]
lr_methods_data = ['cosine']*20 + ['constant']*20
fitting_steps = np.array(fitting_steps)
fitting_losses = np.array(fitting_losses)
lrs = {}
S1 = {}
momentum = {}
S2 = {}
fitting_lr_methods = ["cosine", "constant"]

# initialization grid search
L0_init_range = np.linspace(0.1, 2.1, 2)
A_init_range = np.linspace(1, 22, 3)
C_init_range = np.linspace(1, 22, 3)
alpha_init_range = np.linspace(0, 0.8, 3)
decay_factor = 0.999
huber_delta = 1e-3

# Compute S1 and S2
for lr_method in fitting_lr_methods:
    steps = np.arange(0, 20000+1, 1)
    lrs[lr_method] = np.array([lr(step, lr_method=lr_method, max_lr=2e-4, min_lr=0, total_steps=20000) for step in steps]).astype(np.float64)
    
    # compute S1
    S1[lr_method] = np.cumsum(lrs[lr_method])
    
    # compute S2
    current_lrs = lrs[lr_method]
    n = len(current_lrs)
    momentum[lr_method] = np.zeros(n)
    for i in range(1, n):
        momentum[lr_method][i] = decay_factor * momentum[lr_method][i-1] + (current_lrs[i-1] - current_lrs[i])
    S2[lr_method] = np.cumsum(momentum[lr_method])

# Fitting
best_params = None
best_loss = np.inf
initial_params = product(L0_init_range, A_init_range, C_init_range, alpha_init_range)
for initial_param in tqdm(initial_params):
    result = minimize(objective, initial_param, method='L-BFGS-B', bounds=[(0, np.inf),(0,np.inf),(0,np.inf),(0, np.inf)], options={'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6, 'eps': 1e-8})
    if result.fun < best_loss:
        best_loss = result.fun
        best_params = result.x

# Compute R2
predict_losses = []
for fitting_step, lr_method, in zip(fitting_steps, lr_methods_data):
    predict_loss = Howe_Scaling_Law(fitting_step, lr_method, *best_params)
    predict_losses.append(predict_loss)
predict_losses = np.array(predict_losses).astype(np.float32)
ss_res = np.sum((fitting_losses - predict_losses) ** 2)
ss_tot = np.sum((fitting_losses - np.mean(fitting_losses)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f'L0, A, C, alpha = {best_params}')
print(f'R^2 = {r2}')
L0, A, C, alpha = best_params

# Draw 
plt.figure()
for i, lr_method in enumerate(fitting_lr_methods):
    predict_losses = []
    steps = np.arange(0, 20000, 1)
    for step in steps:
        predict_losses.append(Howe_Scaling_Law(step, lr_method, *best_params))
    predict_losses = np.array(predict_losses).astype(np.float32)
    
    indices = [i for i, x in enumerate(lr_methods_data) if x == lr_method]
    plt.plot(fitting_steps[indices], fitting_losses[indices], 'x', markersize=4, label=f'{lr_method} LRS, Ground Truth Loss', color=f"C{i}")
    plt.plot(steps, predict_losses, '--', label=f'{lr_method} Fitting Curve', color=f"C{i}")
    
plt.yticks(np.arange(2.8, 4.0, 0.1))
plt.ylim(2.8, 4.0)
plt.xlim(0, 20000)
x1, x2, y1, y2 = 10000, 20000, 2.81, 2.9
plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r--')
plt.text(0.2, 0.5, f'Fitting Curve: L = {L0:.3f} + {A:.3f}*S1^(-{alpha:.3f}) - {C:.3f}*S2', fontsize=10, transform=plt.gca().transAxes)
plt.text(0.2, 0.4, r'R^2=' + f'{r2:.5f}', fontsize=10, transform=plt.gca().transAxes)
plt.grid()
plt.legend(prop=FontProperties(size=12))
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig(f"fit.pdf", bbox_inches='tight', pad_inches=0.1)
