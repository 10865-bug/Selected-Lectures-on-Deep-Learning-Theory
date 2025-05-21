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

def compute_s1s2(lr_arr, df=0.999):
    s1_cum = np.cumsum(lr_arr).astype(np.float64)
    mom = np.zeros_like(lr_arr)
    for i in range(1, len(lr_arr)):
        mom[i] = df * mom[i-1] + (lr_arr[i-1] - lr_arr[i])
    return s1_cum, np.cumsum(mom)

def scaling_func(steps, s1_arr, s2_arr, L0, A, C, alpha):
    vs = np.clip(steps, 0, len(s1_arr)-1)
    return L0 + A * (np.clip(s1_arr[vs], 1e-10, None) ** (-alpha)) - C * s2_arr[vs]

def log_mse(p, t):
    lp = np.log(np.clip(p, 1e-10, None))
    lt = np.log(np.clip(t, 1e-10, None))
    return np.mean((lp - lt)**2)

def obj_func(prms, dt):
    L0, A, C, alpha = prms
    p = scaling_func(dt['steps'], dt['S1'], dt['S2'], L0, A, C, alpha)
    return log_mse(p, dt['losses'])

if __name__ == "__main__":
    FIT_TYPE = '8-1-1'
    PREDICT_TYPES = ['8-1-1', 'WSD', 'cosine']
    SAMPLE_RATIO = 1.0
    DISPLAY_INT = 1000
    
    GRID_CFG = {
        'L0': np.linspace(2.3, 2.7, 5),
        'A': np.linspace(1.0, 1.5, 5),
        'C': np.linspace(0.01, 0.1, 5),
        'alpha': np.linspace(0.3, 0.7, 5)
    }
    
    OPTIM_CFG = {
        'method': 'L-BFGS-B',
        'bounds': [(0.001, 10.0), (0.001, 10.0), (0.001, 10.0), (0.001, 10.0)],
        'options': {
            'maxiter': 1000,
            'ftol': 1e-6,
            'gtol': 1e-5,
            'disp': False
        }
    }

    data = load_data('../loss curves/gpt_loss+lrs.pkl')
    
    raw = data[FIT_TYPE]
    n_total = len(raw['steps'])
    s_size = int(n_total * SAMPLE_RATIO)
    sidx = np.sort(np.random.choice(n_total, s_size, replace=False))
    
    s1f, s2f = compute_s1s2(raw['lrs'])
    fdata = {
        'steps': np.arange(n_total),
        'losses': raw['losses'],
        'S1': s1f,
        'S2': s2f
    }
    sampled = {k: v[sidx] for k, v in fdata.items()}

    p_combs = list(product(*GRID_CFG.values()))
    bp, bl = None, np.inf
    
    total_comb = len(p_combs)
    
    for idx, pc in enumerate(p_combs, 1):
        try:
            print(f"\r拟合进度: {idx}/{total_comb}", end='', flush=True)
            res = minimize(obj_func, pc, args=(sampled,), **OPTIM_CFG)
            if res.success and res.fun < bl:
                bp, bl = res.x, res.fun
        except Exception as e:
            print(f"\n警告: 参数组合 {pc} 优化失败 - {str(e)}")
            continue
    
    print("\r" + " " * 40 + "\r", end='')
    
    if bp is None:
        raise RuntimeError("所有参数组合优化失败！")

    print("\r最佳参数组合:")
    param_names = ['L0', 'A', 'C', 'alpha']
    for name, value in zip(param_names, bp):
        print(f"{name:6} = {value:.4f}")

    td = data[FIT_TYPE]
    lr = td['lrs']
    s1v, s2v = compute_s1s2(lr)
    pa = scaling_func(np.arange(len(s1v)), s1v, s2v, *bp)
    ta = td['losses']

    ta = np.array(ta)
    pa = np.array(pa)
    ssr = np.sum((ta - pa) ** 2)
    sst = np.sum((ta - np.mean(ta)) ** 2)
    r2 = 1 - (ssr / sst) if sst != 0 else 0.0

    plt.figure(figsize=(14,7))
    scfg = {
        '8-1-1': {'c':'#1f77b4','m':'o','l':'8-1-1_LRS'},
        'WSD': {'c':'#ff7f0e','m':'s','l':'WSD_LRS'},
        'cosine': {'c':'#2ca02c','m':'D','l':'余弦_LRS'}
    }
    
    for lt in PREDICT_TYPES:
        td = data[lt]
        lr = td['lrs']
        s1v, s2v = compute_s1s2(lr)
        pvals = scaling_func(np.arange(len(s1v)), s1v, s2v, *bp)
        
        plt.plot(td['steps'], pvals,
                 color=scfg[lt]['c'],
                 lw=2,
                 ls='--' if lt != FIT_TYPE else '-',
                 label=f"{scfg[lt]['l']}预测")
        
        didx = np.arange(0, len(td['steps']), DISPLAY_INT)
        plt.scatter(td['steps'][didx], td['losses'][didx],
                    s=10, ec=scfg[lt]['c'],
                    fc='white',
                    marker=scfg[lt]['m'],
                    lw=1.5,
                    label=f"{scfg[lt]['l']}实际")

    os.makedirs('./figures', exist_ok=True)
    
    plt.title(f"基于{FIT_TYPE}拟合的Scaling law (R²={r2:.4f})", fontsize=14, pad=15)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.yscale('log')
    
    ptxt = (f"拟合参数:\nL0={bp[0]:.3f}\nA={bp[1]:.3f}\nC={bp[2]:.3f}\nα={bp[3]:.3f}")
    plt.text(0.97,0.95, ptxt, transform=plt.gca().transAxes,
             va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.legend(ncol=3, loc='lower left', fontsize=10)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('./figures/momentum_law_fit.png', dpi=300, bbox_inches='tight')
