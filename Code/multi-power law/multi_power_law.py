import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
from scipy.stats import linregress
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"运行设备: {device}\n")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(path):
    with open(path, 'rb') as f:
        raw_data = pickle.load(f)
    
    key_mapping = {
        'M:100M_gpt_D:20B_scheduler:811_rope': '8-1-1',
        'M:100M_gpt_D:20B_scheduler:wsd_rope': 'WSD',
        'M:100M_gpt_D:20B_scheduler:cosine_rope': 'cosine'
    }
    
    return {
        new_key: {
            'steps': raw_data[orig_key]['step'].values.astype(int),
            'losses': raw_data[orig_key]['Metrics/loss'].values.astype(np.float64),
            'lrs': np.clip(raw_data[orig_key]['lr'].values.astype(np.float64), 1e-10, None)
        } for orig_key, new_key in key_mapping.items() if orig_key in raw_data
    }

class PowerLawModel(torch.nn.Module):
    def __init__(self, L0, A, B, C, alpha, beta, gamma):
        super().__init__()
        self.log_L0 = torch.nn.Parameter(torch.log(torch.tensor(L0, device=device)))
        self.log_A = torch.nn.Parameter(torch.log(torch.tensor(A, device=device)))
        self.log_B = torch.nn.Parameter(torch.log(torch.tensor(B, device=device)))
        self.log_C = torch.nn.Parameter(torch.log(torch.tensor(C, device=device)))
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(alpha, device=device)))
        self.log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta, device=device)))
        self.log_gamma = torch.nn.Parameter(torch.log(torch.tensor(gamma, device=device)))
    
    def forward(self, lrs):
        L0 = torch.exp(self.log_L0)
        A = torch.exp(self.log_A)
        B = torch.exp(self.log_B)
        C = torch.exp(self.log_C)
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)
        gamma = torch.exp(self.log_gamma)
        
        lrs = torch.clamp(lrs, min=1e-10)
        S1 = torch.cumsum(lrs, dim=0)
        
        base_term = A * (S1 + 1e-10) ** (-alpha)
        
        delta_eta = torch.cat([torch.zeros(1, device=device), lrs[:-1] - lrs[1:]])
        G = 1 - (1 + C * lrs ** (-gamma) * S1) ** (-beta)
        dynamic_term = B * torch.cumsum(delta_eta * G, dim=0)
        
        return torch.clamp(L0 + base_term - dynamic_term, min=1e-5)

def initialize_params(train_data):
    losses = train_data['losses']
    lrs = train_data['lrs']
    
    valid_idx = (losses > 1e-3) & (lrs > 1e-6)
    log_y = np.log(losses[valid_idx])
    log_x = np.log(np.cumsum(lrs)[valid_idx])
    _, _ = linregress(log_x, log_y)[:2]
    
    return {
        'L0': np.linspace(2.0, 3.0, 2),
        'A': np.linspace(1.0, 2.0, 2),
        'B': np.linspace(1, 20, 2),  
        'C': np.linspace(0.001, 0.1, 2),
        'alpha': np.linspace(0.3, 0.8, 2),
        'beta': np.linspace(0.2, 0.8, 2),  
        'gamma': np.linspace(0.1, 0.5, 2)  
    }

def train_model(train_data, rho=0.5, lr1=4e-3, lr2=1e-4, max_steps=1000):
    param_grid = initialize_params(train_data)
    p_combs = list(product(*param_grid.values()))
    
    final_params, min_loss = None, float('inf')
    full_size = len(train_data['lrs'])
    sample_size = int(full_size * rho)
    
    for idx, params in enumerate(p_combs, 1):
        current_model = PowerLawModel(*params).to(device)
        optimzier = optim.AdamW([
            {'params': [current_model.log_L0, current_model.log_A, current_model.log_B, current_model.log_C], 'lr': lr1},
            {'params': [current_model.log_alpha, current_model.log_beta, current_model.log_gamma], 'lr': lr2}
        ])
        
        lrs_t = torch.tensor(train_data['lrs'], dtype=torch.float32, device=device)
        losses_t = torch.tensor(train_data['losses'], dtype=torch.float32, device=device)
        idxs = torch.randperm(full_size, device=device)[:sample_size]
        
        try:
            print(f"\r拟合进度: {idx}/{len(p_combs)}", end='')
            
            for _ in range(max_steps):
                def closure(m=current_model, l=lrs_t, t=losses_t, i=idxs):
                    optimzier.zero_grad()
                    p = m(l)[i]
                    loss = torch.mean((torch.log(p) - torch.log(t[i])) ** 2)
                    loss.backward()
                    return loss
                
                loss_val = optimzier.step(closure)
                
                if loss_val.item() < min_loss:
                    min_loss = loss_val.item()
                    final_params = {
                        'L0': torch.exp(current_model.log_L0).item(),
                        'A': torch.exp(current_model.log_A).item(),
                        'B': torch.exp(current_model.log_B).item(),
                        'C': torch.exp(current_model.log_C).item(),
                        'alpha': torch.exp(current_model.log_alpha).item(),
                        'beta': torch.exp(current_model.log_beta).item(),
                        'gamma': torch.exp(current_model.log_gamma).item()
                    }
        
        except Exception as e:
            print(f"\n参数组合异常: {params} - {str(e)}")
            continue
    
    print("\r" + " " * 40 + "\r", end='')
    return final_params, min_loss

def plot_results(dataset, params, fit_type='8-1-1'):
    fig = plt.figure(figsize=(14,7))
    style_config = {
        '8-1-1': {'c':'#1f77b4','m':'o','l':'8-1-1_LRS'},
        'WSD': {'c':'#ff7f0e','m':'s','l':'WSD_LRS'},
        'cosine': {'c':'#2ca02c','m':'D','l':'余弦_LRS'}
    }
    
    predictor = PowerLawModel(**params).to(device)
    with torch.no_grad():
        predictions = predictor(torch.tensor(dataset[fit_type]['lrs'], device=device)).cpu().numpy()
    actual = dataset[fit_type]['losses']
    r_squared = 1 - np.sum((actual - predictions)**2) / np.sum((actual - np.mean(actual))**2)
    
    for lt in ['8-1-1', 'WSD', 'cosine']:
        target_data = dataset[lt]
        with torch.no_grad():
            pred_values = predictor(torch.tensor(target_data['lrs'], device=device)).cpu().numpy()
        
        plt.plot(target_data['steps'], pred_values, color=style_config[lt]['c'], lw=2,
                 ls='--' if lt != fit_type else '-', label=f"{style_config[lt]['l']}预测")
        plt.scatter(target_data['steps'][::1000], target_data['losses'][::1000], s=10,
                    ec=style_config[lt]['c'], fc='white', marker=style_config[lt]['m'])
    
    os.makedirs('./figures', exist_ok=True)
    plt.title(f"基于{fit_type}的多幂律预测 (R²={r_squared:.4f})", fontsize=14)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.yscale('log')
    
    param_display = "\n".join([
        f"L0 = {params['L0']:.4f}",
        f"A  = {params['A']:.4f}",
        f"B  = {params['B']:.4f}",
        f"C  = {params['C']:.4f}",
        f"α  = {params['alpha']:.4f}",
        f"β  = {params['beta']:.4f}",
        f"γ  = {params['gamma']:.4f}"
    ])
    plt.text(0.97, 0.95, param_display, transform=fig.gca().transAxes,
             va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend(ncol=3, loc='lower left', fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('./figures/multi_power_law_fit.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    FIT_TYPE = '8-1-1'
    dataset = load_data('../loss curves/gpt_loss+lrs.pkl')
    
    best_params, best_loss = train_model(dataset[FIT_TYPE], rho=1.0)
    
    if best_params:
        print("\n最佳参数:")
        print(f"L0 = {best_params['L0']:.4f}")
        print(f"A  = {best_params['A']:.4f}")
        print(f"B  = {best_params['B']:.4f}") 
        print(f"C  = {best_params['C']:.4f}")
        print(f"α  = {best_params['alpha']:.4f}")
        print(f"β  = {best_params['beta']:.4f}")
        print(f"γ  = {best_params['gamma']:.4f}")
        plot_results(dataset, best_params)
    else:
        print("所有参数组合优化失败！")
