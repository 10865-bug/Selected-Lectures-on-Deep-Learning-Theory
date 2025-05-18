import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
import sys
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

def initialize_params(data, train_key):
    losses = data[train_key]['losses']
    lrs = data[train_key]['lrs']
    
    valid_idx = (losses > 1e-3) & (lrs > 1e-6)
    log_y = np.log(losses[valid_idx])
    log_x = np.log(np.cumsum(lrs)[valid_idx])
    slope, intercept = linregress(log_x, log_y)[:2]
    
    return {
        'L0': np.linspace(1.0, 3.0, 2),
        'A': np.linspace(1.0, 3.0, 2),
        'B': np.linspace(100, 500, 4),  
        'C': np.linspace(0.01, 0.5, 2),
        'alpha': np.linspace(0.1, 1.0, 2),
        'beta': np.linspace(0.3, 1.0, 2),  
        'gamma': np.linspace(0.1, 0.8, 2)  
    }

def train_model(data, train_key, rho=0.5, lr1=1e-3, lr2=1e-4, max_steps=1000):
    param_grid = initialize_params(data, train_key)
    p_combs = list(product(*param_grid.values()))
    
    best_params, best_loss = None, float('inf')
    full_size = len(data[train_key]['lrs'])
    sample_size = int(full_size * rho)
    
    for idx, params in enumerate(p_combs, 1):
        try:
            print(f"\r拟合进度: {idx}/{len(p_combs)}", end='')
            
            model = PowerLawModel(*params).to(device)
            optimizer = optim.AdamW([
                {'params': [model.log_L0, model.log_A, model.log_B, model.log_C], 'lr': lr1},
                {'params': [model.log_alpha, model.log_beta, model.log_gamma], 'lr': lr2}
            ])
            
            lrs_tensor = torch.tensor(data[train_key]['lrs'], dtype=torch.float32, device=device)
            losses_tensor = torch.tensor(data[train_key]['losses'], dtype=torch.float32, device=device)
            
            indices = torch.randperm(full_size, device=device)[:sample_size]
            
            for step in range(max_steps):
                def closure():
                    optimizer.zero_grad()
                    pred = model(lrs_tensor)[indices]
                    loss = torch.mean((torch.log(pred) - torch.log(losses_tensor[indices])) ** 2)
                    loss.backward()
                    return loss
                
                loss = optimizer.step(closure)
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params = {
                        'L0': torch.exp(model.log_L0).item(),
                        'A': torch.exp(model.log_A).item(),
                        'B': torch.exp(model.log_B).item(),
                        'C': torch.exp(model.log_C).item(),
                        'alpha': torch.exp(model.log_alpha).item(),
                        'beta': torch.exp(model.log_beta).item(),
                        'gamma': torch.exp(model.log_gamma).item()
                    }
        
        except Exception as e:
            print(f"\n警告: 参数组合 {params} 优化失败 - {str(e)}")
            continue
    
    print("\r" + " " * 40 + "\r", end='')
    return best_params, best_loss

def plot_results(data, best_params, fit_type='8-1-1'):
    plt.figure(figsize=(14,7))
    scfg = {
        '8-1-1': {'c':'#1f77b4','m':'o','l':'8-1-1_LRS'},
        'WSD': {'c':'#ff7f0e','m':'s','l':'WSD_LRS'},
        'cosine': {'c':'#2ca02c','m':'D','l':'余弦_LRS'}
    }
    
    model = PowerLawModel(**best_params).to(device)
    with torch.no_grad():
        pred = model(torch.tensor(data[fit_type]['lrs'], device=device)).cpu().numpy()
    ta = data[fit_type]['losses']
    r2 = 1 - np.sum((ta - pred)**2) / np.sum((ta - np.mean(ta))**2)
    
    for lt in ['8-1-1', 'WSD', 'cosine']:
        td = data[lt]
        with torch.no_grad():
            pred = model(torch.tensor(td['lrs'], device=device)).cpu().numpy()
        
        plt.plot(td['steps'], pred, color=scfg[lt]['c'], lw=2,
                 ls='--' if lt != fit_type else '-', label=f"{scfg[lt]['l']}预测")
        plt.scatter(td['steps'][::1000], td['losses'][::1000], s=10,
                    ec=scfg[lt]['c'], fc='white', marker=scfg[lt]['m'])
    
    os.makedirs('./figures', exist_ok=True)
    plt.title(f"基于{fit_type}的多幂律预测 (R²={r2:.4f})", fontsize=14)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.yscale('log')
    
    param_text = "\n".join([
        f"L0 = {best_params['L0']:.4f}",
        f"A  = {best_params['A']:.4f}",
        f"B  = {best_params['B']:.4f}",
        f"C  = {best_params['C']:.4f}",
        f"α  = {best_params['alpha']:.4f}",
        f"β  = {best_params['beta']:.4f}",
        f"γ  = {best_params['gamma']:.4f}"
    ])
    plt.text(0.97, 0.95, param_text, transform=plt.gca().transAxes,
             va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend(ncol=3, loc='lower left', fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('./figures/multi_power_law_fit.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    FIT_TYPE = '8-1-1'
    data = load_data('../loss curves/gpt_loss+lrs.pkl')
    
    best_params, best_loss = train_model(data, FIT_TYPE, rho=1.0)
    
    if best_params:
        print("\n最佳参数:")
        print(f"L0 = {best_params['L0']:.4f}")
        print(f"A  = {best_params['A']:.4f}")
        print(f"B  = {best_params['B']:.4f}") 
        print(f"C  = {best_params['C']:.4f}")
        print(f"α  = {best_params['alpha']:.4f}")
        print(f"β  = {best_params['beta']:.4f}")
        print(f"γ  = {best_params['gamma']:.4f}")
        plot_results(data, best_params)
    else:
        print("所有参数组合优化失败！")
