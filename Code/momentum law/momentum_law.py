import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    def process_df(df_key):
        df = data[df_key]
        return {
            'steps': df['step'].values.astype(int),
            'losses': df['Metrics/loss'].values.astype(np.float64),
            'lrs': df['lr'].values.astype(np.float64)
        }
    
    return {
        '8-1-1': process_df('M:100M_gpt_D:20B_scheduler:811_rope'),
        'WSD': process_df('M:100M_gpt_D:20B_scheduler:wsd_rope'),
        'cosine': process_df('M:100M_gpt_D:20B_scheduler:cosine_rope')
    }

def compute_S1_S2(lr_sequence, decay_factor=0.999):
    S1 = np.cumsum(lr_sequence).astype(np.float64)
    momentum = np.zeros_like(lr_sequence)
    for i in range(1, len(lr_sequence)):
        momentum[i] = decay_factor * momentum[i-1] + (lr_sequence[i-1] - lr_sequence[i])
    return S1, np.cumsum(momentum)

def scaling_law(steps, S1, S2, L0, A, C, alpha):
    valid_steps = np.clip(steps, 0, len(S1)-1)
    return L0 + A * (np.clip(S1[valid_steps], 1e-10, None) ** (-alpha)) - C * S2[valid_steps]

def log_mse_loss(pred, true):
    log_pred = np.log(np.clip(pred, 1e-10, None))
    log_true = np.log(np.clip(true, 1e-10, None))
    return np.mean((log_pred - log_true)**2)

def objective(params, fit_data):
    L0, A, C, alpha = params
    pred_loss = scaling_law(fit_data['steps'], fit_data['S1'], fit_data['S2'], L0, A, C, alpha)
    return log_mse_loss(pred_loss, fit_data['losses'])

if __name__ == "__main__":
    FIT_LRS_TYPE = '8-1-1'
    PREDICT_LRS_TYPES = ['8-1-1', 'WSD', 'cosine']
    SAMPLE_RATIO = 0.1
    DISPLAY_INTERVAL = 400
    
    GRID_CONFIG = {
        'L0': np.linspace(0.01, 5.0, 5),
        'A': np.linspace(0.01, 10.0, 5),
        'C': np.linspace(0.01, 5.0, 5),
        'alpha': np.linspace(0.01, 1.0, 5)
    }
    
    OPTIM_CONFIG = {
        'method': 'L-BFGS-B',
        'bounds': [(0.01, 5.0), (0.01, 10.0), (0.01, 5.0), (0.01, 1.0)],
        'options': {
            'maxiter': 1000,
            'ftol': 1e-6,
            'gtol': 1e-5,
            'disp': False
        }
    }

    data = load_data('../loss curves/gpt_loss+lrs.pkl')
    
    # 拟合数据
    raw_data = data[FIT_LRS_TYPE]
    n_total = len(raw_data['steps'])
    sample_size = int(n_total * SAMPLE_RATIO)
    sample_idx = np.sort(np.random.choice(n_total, sample_size, replace=False))
    
    S1_full, S2_full = compute_S1_S2(raw_data['lrs'])
    fit_data = {
        'steps': np.arange(n_total),
        'losses': raw_data['losses'],
        'S1': S1_full,
        'S2': S2_full
    }
    sampled_data = {k: v[sample_idx] for k, v in fit_data.items()}

    param_combinations = list(product(*GRID_CONFIG.values()))
    best_params, best_loss = None, np.inf
    
    for params in param_combinations:
        res = minimize(objective, params, args=(sampled_data,), **OPTIM_CONFIG)
        if res.success and res.fun < best_loss:
            best_params, best_loss = res.x, res.fun

    all_true, all_pred = [], []
    for lrs_type in PREDICT_LRS_TYPES:
        target_data = data[lrs_type]
        lr_seq = target_data['lrs']
        S1, S2 = compute_S1_S2(lr_seq)
        pred_loss = scaling_law(np.arange(len(S1)), S1, S2, *best_params)
        all_true.extend(target_data['losses'])
        all_pred.extend(pred_loss)
    
    # 计算总体R²
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # 绘图配置
    plt.figure(figsize=(14, 7))
    style_config = {
        '8-1-1': {'color': '#1f77b4', 'marker': 'o', 'label': '8-1-1_LRS'},
        'WSD': {'color': '#ff7f0e', 'marker': 's', 'label': 'WSD_LRS'},
        'cosine': {'color': '#2ca02c', 'marker': 'D', 'label': '余弦_LRS'}
    }
    
    for lrs_type in PREDICT_LRS_TYPES:
        target_data = data[lrs_type]
        lr_seq = target_data['lrs']
        steps_full = target_data['steps']
        S1, S2 = compute_S1_S2(lr_seq)
        pred_loss = scaling_law(np.arange(len(S1)), S1, S2, *best_params)
        
        plt.plot(steps_full, pred_loss,
                 color=style_config[lrs_type]['color'],
                 linewidth=2,
                 linestyle='--' if lrs_type != FIT_LRS_TYPE else '-',
                 label=f"{style_config[lrs_type]['label']}预测损失")
        
        display_idx = np.arange(0, len(steps_full), DISPLAY_INTERVAL)
        plt.scatter(steps_full[display_idx], target_data['losses'][display_idx],
                    s=10, edgecolor=style_config[lrs_type]['color'],
                    facecolor='white',
                    marker=style_config[lrs_type]['marker'],
                    linewidth=1.5,
                    label=f"{style_config[lrs_type]['label']}实际损失")

    os.makedirs('./figures', exist_ok=True) 
    
    plt.title(f"基于{FIT_LRS_TYPE}拟合的Scaling law (R²={r_squared:.4f})", fontsize=14, pad=15)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.yscale('log')
    
    param_text = (f"拟合参数:\n"
                 f"L0 = {best_params[0]:.3f}\n"
                 f"A = {best_params[1]:.3f}\n"
                 f"C = {best_params[2]:.3f}\n"
                 f"α = {best_params[3]:.3f}")
    plt.text(0.97, 0.95, param_text,
             transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(ncol=3, loc='lower left', fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('./figures/cross_scheduler_prediction.png', dpi=300, bbox_inches='tight') 
    plt.show()
