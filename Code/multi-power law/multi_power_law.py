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
        ld = pickle.load(f)
    
    def proc_df(k):
        df = ld[k]
        return {
            'steps': df['step'].values.astype(int),
            'losses': df['Metrics/loss'].values.astype(np.float64),
            'lrs': df['lr'].values.astype(np.float64)
        }
    
    return {
        '8-1-1': proc_df('M:100M_gpt_D:20B_scheduler:811_rope'),
        'WSD': proc_df('M:100M_gpt_D:20B_scheduler:wsd_rope'),
        'cosine': proc_df('M:100M_gpt_D:20B_scheduler:cosine_rope')
    }

def compute_terms(lr_arr, gamma, beta, B, C):
    """优化后的向量化计算函数"""
    lr_cumsum = np.concatenate([[0], np.cumsum(lr_arr)])
    delta_eta = np.concatenate([[0], lr_arr[:-1] - lr_arr[1:]])
    eta_k_pow = lr_arr ** (-gamma)
    S_kt = lr_cumsum[1:] - lr_cumsum[:-1]
    G = 1 - (1 + C * eta_k_pow * S_kt) ** (-beta)
    return np.cumsum(lr_arr), B * np.cumsum(delta_eta * G)

def multi_power_law(steps, S1, LD, L0, A, alpha):
    valid_steps = np.clip(steps, 0, len(S1)-1)
    return L0 + A * (np.clip(S1[valid_steps], 1e-10, None)**-alpha) - LD[valid_steps]

def log_mse(pred, target):
    return np.mean((np.log(np.clip(pred, 1e-10, None)) - np.log(np.clip(target, 1e-10, None)))**2)

def obj_func(params, data):
    L0, A, B, C, alpha, beta, gamma = params
    S1, LD = compute_terms(data['lrs'], gamma, beta, B, C)
    pred = multi_power_law(data['steps'], S1, LD, L0, A, alpha)
    return log_mse(pred, data['losses'])

if __name__ == "__main__":
    FIT_TYPE = '8-1-1'
    PREDICT_TYPES = ['8-1-1', 'WSD', 'cosine']
    SAMPLE_RATIO = 0.4    
    DISPLAY_INT = 400     
    
    GRID_CFG = {
        'L0':    np.linspace(0.1, 2.0, num=2),   
        'A':     np.linspace(0.5, 2.5, num=2),   
        'B':     np.linspace(0.1, 500, num=5),   
        'C':     np.linspace(0.1, 0.9, num=2),   
        'alpha': np.linspace(0.2, 0.8, num=2),   
        'beta':  np.linspace(0.3, 0.7, num=2),   
        'gamma': np.linspace(0.3, 0.7, num=2)   
    }
    
    OPTIM_CFG = {
        'method': 'L-BFGS-B',
        'bounds': [
            (0.1, 3.0),    
            (0.1, 3.0),    
            (0.1, 500),    
            (0.01, 1.0),   
            (0.01, 1.0),  
            (0.2, 0.8),    
            (0.2, 0.8)    
        ],
        'options': {
            'maxiter': 500,
            'ftol': 1e-5,
            'gtol': 1e-4,
            'disp': False
        }
    }

    data = load_data('../loss curves/gpt_loss+lrs.pkl')
    raw = data[FIT_TYPE]
    
    n_total = len(raw['steps'])
    s_size = int(n_total * SAMPLE_RATIO)
    sidx = np.sort(np.random.choice(n_total, s_size, replace=False))
    
    sampled = {
        'steps': np.arange(n_total)[sidx],
        'losses': raw['losses'][sidx],
        'lrs': raw['lrs'][sidx]
    }

    p_combs = list(product(*GRID_CFG.values()))
    total_comb = len(p_combs)
    bp, bl = None, np.inf

    # 网格搜索优化
    for idx, pc in enumerate(p_combs, 1):
        try:          
            print(f"\r拟合进度: {idx:03d}/{total_comb} ", end='', flush=True)
            
            res = minimize(obj_func, pc, args=(sampled,), **OPTIM_CFG)
            
            if res.success and res.fun < bl:
                bp, bl = res.x, res.fun
                
        except Exception as e:
            print(f"\n[警告] 参数组合 {pc} 失败: {str(e)}")
            continue
        
    print("\r" + " " * 40 + "\r", end='')

    if bp is None:
        raise RuntimeError("所有参数组合优化失败")

    print("\r最佳参数组合:")
    param_names = ['L0', 'A', 'B', 'C', 'alpha', 'beta', 'gamma']
    for name, value in zip(param_names, bp):
        print(f"{name:6} = {value:.4f}")

    # 可视化
    plt.figure(figsize=(14, 7))
    scfg = {
        '8-1-1': {'c':'#1f77b4','m':'o','l':'8-1-1_LRS'},
        'WSD': {'c':'#ff7f0e','m':'s','l':'WSD_LRS'},
        'cosine': {'c':'#2ca02c','m':'D','l':'余弦_LRS'}
    }
    
    ta, pa = [], []
    for lt in PREDICT_TYPES:
        td = data[lt]
        lr = td['lrs']
        S1, LD = compute_terms(lr, bp[6], bp[5], bp[2], bp[3])
        pvals = multi_power_law(np.arange(len(S1)), S1, LD, bp[0], bp[1], bp[4])
        
        # 绘制预测曲线
        plt.plot(td['steps'], pvals,
                 color=scfg[lt]['c'],
                 lw=2,
                 ls='--' if lt != FIT_TYPE else '-',
                 label=f"{scfg[lt]['l']}预测")
        
        # 绘制实际散点
        didx = np.arange(0, len(td['steps']), DISPLAY_INT)
        plt.scatter(td['steps'][didx], td['losses'][didx],
                    s=10, ec=scfg[lt]['c'],
                    fc='white',
                    marker=scfg[lt]['m'],
                    lw=1.5,
                    label=f"{scfg[lt]['l']}实际")
        
        ta.extend(td['losses'])
        pa.extend(pvals)

    ta, pa = np.array(ta), np.array(pa)
    r2 = 1 - np.sum((ta - pa)**2) / np.sum((ta - np.mean(ta))**2)
    
    # 图表装饰
    plt.title(f"基于{FIT_TYPE}的多策略预测 (R²={r2:.4f})", fontsize=14)
    plt.xlabel("训练步数", fontsize=12)
    plt.ylabel("损失值", fontsize=12)
    plt.yscale('log')
    plt.grid(alpha=0.2)
    
    # 参数标注框
    param_text = "\n".join([f"{n}: {v:.3f}" for n,v in zip(param_names, bp)])
    plt.annotate(param_text, xy=(0.97, 0.65), xycoords='axes fraction',
                ha='right', va='top', 
                bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    
    plt.legend(ncol=2, loc='upper right', fontsize=9)
    plt.tight_layout()
    
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/multi_power_law_fit.png', dpi=300, bbox_inches='tight')
