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
    def __init__(self, L0, A, B, C, alpha, beta, gamma, Sw):
        super().__init__()
        self.log_L0 = torch.nn.Parameter(torch.log(torch.tensor(L0, device=device)))
        self.log_A = torch.nn.Parameter(torch.log(torch.tensor(A, device=device)))
        self.log_B = torch.nn.Parameter(torch.log(torch.tensor(B, device=device)))
        self.log_C = torch.nn.Parameter(torch.log(torch.tensor(C, device=device)))
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(alpha, device=device)))
        self.log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta, device=device)))
        self.log_gamma = torch.nn.Parameter(torch.log(torch.tensor(gamma, device=device)))
        self.log_Sw = torch.nn.Parameter(torch.log(torch.tensor(Sw, device=device)))
    
    def forward(self, lrs):
        L0 = torch.exp(self.log_L0)
        A = torch.exp(self.log_A)
        B = torch.exp(self.log_B)
        C = torch.exp(self.log_C)
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)
        gamma = torch.exp(self.log_gamma)
        Sw = torch.exp(self.log_Sw)
        
        lrs = torch.clamp(lrs, min=1e-10)
        S1 = torch.cumsum(lrs, dim=0)
        
        base_term = A * (S1 + Sw) ** (-alpha)
        
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
        'B': np.linspace(1.0, 500.0, 2),  
        'C': np.linspace(1.0, 5.0, 2),
        'alpha': np.linspace(0.5, 1.5, 2),
        'beta': np.linspace(0.01, 0.1, 2),  
        'gamma': np.linspace(0.01, 0.3, 2),
        'Sw': np.linspace(0.1, 0.2, 2)
    }

def train_model(train_data, rho=0.5, lr1=1e-3, lr2=1e-4, max_steps=1000):
    param_grid = initialize_params(train_data)
    p_combs = list(product(*param_grid.values()))
    
    final_params, min_loss = None, float('inf')
    full_size = len(train_data['lrs'])
    sample_size = int(full_size * rho)
    
    for idx, params in enumerate(p_combs, 1):
        current_model = PowerLawModel(*params).to(device)
        optimzier = optim.AdamW([
            {'params': [current_model.log_L0, current_model.log_A, current_model.log_B, current_model.log_C, current_model.log_Sw], 'lr': lr1},
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
                        'gamma': torch.exp(current_model.log_gamma).item(),
                        'Sw': torch.exp(current_model.log_Sw).item()
                    }
        
        except Exception as e:
            print(f"\n参数组合异常: {params} - {str(e)}")
            continue
    
    print("\r" + " " * 40 + "\r", end='')
    return final_params, min_loss

if __name__ == "__main__":
    FIT_TYPE = '8-1-1'
    PLOT_SKIP = 100
    PREDICT_TYPES = ['8-1-1', 'WSD', 'cosine']
    
    dataset = load_data('../loss curves/gpt_loss+lrs.pkl')
    
    best_params, best_loss = train_model(dataset[FIT_TYPE], rho=1.0)
    
    if not best_params:
        print("所有参数组合优化失败！")
        exit()
    
    print("\n最佳参数:")
    print(f"L0 = {best_params['L0']:.4f}")
    print(f"A  = {best_params['A']:.4f}")
    print(f"B  = {best_params['B']:.4f}") 
    print(f"C  = {best_params['C']:.4f}")
    print(f"α  = {best_params['alpha']:.4f}")
    print(f"β  = {best_params['beta']:.4f}")
    print(f"γ  = {best_params['gamma']:.4f}")
    print(f"Sw = {best_params['Sw']:.4f}")
    
    os.makedirs('./figures', exist_ok=True)
    model = PowerLawModel(**best_params).to(device)
    
    results = {}
    for lt in PREDICT_TYPES:
        td = dataset[lt]
        with torch.no_grad():
            pvals = model(torch.tensor(td['lrs'], device=device)).cpu().numpy()
        ta = td['losses']
        
        ssr = np.sum((ta - pvals) ** 2)
        sst = np.sum((ta - np.mean(ta)) ** 2)
        r2 = 1 - (ssr / sst) if sst != 0 else 0.0
        
        abs_errors = np.abs(pvals - ta)
        relative_errors = abs_errors / ta
        mape = np.mean(relative_errors) * 100
        
        results[lt] = {'R2': r2, 'MAPE': mape}
        
        n = len(ta)
        n_groups = n // PLOT_SKIP
        
        avg_steps = np.zeros(n_groups)
        avg_losses = np.zeros(n_groups)
        avg_pvals = np.zeros(n_groups)
        
        for i in range(n_groups):
            start = i * PLOT_SKIP
            end = start + PLOT_SKIP
            avg_steps[i] = td['steps'][start]
            avg_losses[i] = np.mean(ta[start:end]) if end <= n else 0.0
            avg_pvals[i] = np.mean(pvals[start:end]) if end <= n else 0.0
        
        plt.figure(figsize=(14,7))
        plt.plot(avg_steps, avg_losses, 'b-', lw=1.5, label='ground truth')
        plt.plot(avg_steps, avg_pvals, 'r--', lw=1.5, label='prediction')
        
        plt.title(f"基于{FIT_TYPE}拟合的Multi-power law作用于{lt}", fontsize=14, pad=15)
        plt.xlabel("Step", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.yscale('log')
        plt.legend(fontsize=12, loc='best')
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'./figures/multi_power_law_fit_{lt}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        last_steps = 25000
        if len(td['steps']) > last_steps:
            start_idx = len(td['steps']) - last_steps
            n = last_steps
            n_groups = n // PLOT_SKIP
                
            last_avg_steps = np.zeros(n_groups)
            last_avg_losses = np.zeros(n_groups)
            last_avg_pvals = np.zeros(n_groups)
                
            for i in range(n_groups):
                start = start_idx + i * PLOT_SKIP
                end = start + PLOT_SKIP
                last_avg_steps[i] = td['steps'][start]
                last_avg_losses[i] = np.mean(ta[start:end]) if end <= len(ta) else 0.0
                last_avg_pvals[i] = np.mean(pvals[start:end]) if end <= len(pvals) else 0.0
                
            plt.figure(figsize=(14,7))
            plt.plot(last_avg_steps, last_avg_losses, 'b-', lw=1.5, label='ground truth')
            plt.plot(last_avg_steps, last_avg_pvals, 'r--', lw=1.5, label='prediction')
                
            plt.title(f"基于{FIT_TYPE}拟合的Multi-power law作用于{lt} (最后{last_steps}步)", fontsize=14, pad=15)
            plt.xlabel("Step", fontsize=20)
            plt.ylabel("Loss", fontsize=20)
            plt.yscale('log')
            plt.legend(fontsize=12, loc='best')
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(f'./figures/multi_power_law_fit_{lt}_last.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("\n模型评估指标:")
    for lt in PREDICT_TYPES:
        print(f"{lt}: R² = {results[lt]['R2']:.4f}, MAPE = {results[lt]['MAPE']:.4f}%")
