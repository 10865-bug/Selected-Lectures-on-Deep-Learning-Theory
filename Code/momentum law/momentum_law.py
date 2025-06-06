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

def scaling_func(steps, s1_arr, s2_arr, L0, A, C, alpha, Sw):
    vs = np.clip(steps, 0, len(s1_arr)-1)
    return L0 + A * (np.clip(s1_arr[vs] + Sw, 1e-10, None) ** (-alpha)) - C * s2_arr[vs]

def huber_log_loss(p, t, delta=5):
    lp = np.log(np.clip(p, 1e-10, None))
    lt = np.log(np.clip(t, 1e-10, None))
    e = lt - lp
    abs_e = np.abs(e)
    loss = np.where(abs_e <= delta, 0.5 * e**2, delta * (abs_e - 0.5 * delta))
    return np.mean(loss)

def obj_func(prms, dt):
    L0, A, C, alpha, Sw = prms
    p = scaling_func(dt['steps'], dt['S1'], dt['S2'], L0, A, C, alpha, Sw)
    return huber_log_loss(p, dt['losses'])

if __name__ == "__main__":
    FIT_TYPE = 'cosine'
    PREDICT_TYPES = ['8-1-1', 'WSD', 'cosine']
    SAMPLE_RATIO = 1.0
    PLOT_SKIP = 100
    
    GRID_CFG = {
        'L0': np.linspace(2.0, 3.0, 2),
        'A': np.linspace(1.0, 2.0, 2),
        'C': np.linspace(0.01, 0.3, 3),
        'alpha': np.linspace(0.1, 2.0, 3),
        'Sw': np.linspace(0.1, 0.2, 10)
    }
    
    OPTIM_CFG = {
        'method': 'L-BFGS-B',
        'bounds': [(0.001, 10.0), (0.0001, 10.0), (0.0001, 10.0), (0.001, 10.0), (0.01, 1.0)],
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
    param_names = ['L0', 'A', 'C', 'alpha', 'Sw']
    for name, value in zip(param_names, bp):
        print(f"{name:6} = {value:.4f}")

    os.makedirs('./figures', exist_ok=True)
    scfg = {
        '8-1-1': {'l':'8-1-1_LRS'},
        'WSD': {'l':'WSD_LRS'},
        'cosine': {'l':'余弦_LRS'}
    }
    
    results = {}
    for lt in PREDICT_TYPES:
        td = data[lt]
        lr = td['lrs']
        s1v, s2v = compute_s1s2(lr)
        pvals = scaling_func(np.arange(len(s1v)), s1v, s2v, *bp)
        ta = td['losses']
        
        ssr = np.sum((ta - pvals) ** 2)
        sst = np.sum((ta - np.mean(ta)) ** 2)
        r2 = 1 - (ssr / sst) if sst != 0 else 0.0
        
        abs_errors = np.abs(pvals - ta)
        relative_errors = abs_errors / ta
        mape = np.mean(relative_errors) * 100
        
        results[lt] = {'R2': r2, 'MAPE': mape}
        
        n = len(ta)
        group_size = PLOT_SKIP
        n_groups = n // group_size
        
        avg_steps = np.zeros(n_groups)
        avg_losses = np.zeros(n_groups)
        avg_pvals = np.zeros(n_groups)
        
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size
            avg_steps[i] = td['steps'][start]
            avg_losses[i] = np.mean(ta[start:end]) if end <= n else 0.0
            avg_pvals[i] = np.mean(pvals[start:end]) if end <= n else 0.0
        
        plt.figure(figsize=(14,7))
        plt.plot(avg_steps, avg_losses, 'b-', lw=1.5, label='ground truth')
        plt.plot(avg_steps, avg_pvals, 'r--', lw=1.5, label='prediction')
        
        plt.title(f"基于{FIT_TYPE}拟合的Momentum law作用于{lt}", fontsize=14, pad=15)
        plt.xlabel("Step", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.yscale('log')
        plt.legend(fontsize=12, loc='best')
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'./figures/momentum_law_fit_{lt}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        
        last_steps = 25000
        if len(td['steps']) > last_steps:
            start_idx = len(td['steps']) - last_steps
            n = last_steps
            n_groups = n // group_size
                
            last_avg_steps = np.zeros(n_groups)
            last_avg_losses = np.zeros(n_groups)
            last_avg_pvals = np.zeros(n_groups)
                
            for i in range(n_groups):
                start = start_idx + i * group_size
                end = start + group_size
                last_avg_steps[i] = td['steps'][start]
                last_avg_losses[i] = np.mean(ta[start:end]) if end <= len(ta) else 0.0
                last_avg_pvals[i] = np.mean(pvals[start:end]) if end <= len(pvals) else 0.0
                
            plt.figure(figsize=(14,7))
            plt.plot(last_avg_steps, last_avg_losses, 'b-', lw=1.5, label='ground truth')
            plt.plot(last_avg_steps, last_avg_pvals, 'r--', lw=1.5, label='prediction')
                
            plt.title(f"基于{FIT_TYPE}拟合的Momentum law作用于{lt} (最后{last_steps}步)", fontsize=14, pad=15)
            plt.xlabel("Step", fontsize=20)
            plt.ylabel("Loss", fontsize=20)
            plt.yscale('log')
            plt.legend(fontsize=12, loc='best')
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(f'./figures/our_law_fit_{lt}_last.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("\n模型评估指标:")
    for lt in PREDICT_TYPES:
        print(f"{lt}: R² = {results[lt]['R2']:.4f}, MAPE = {results[lt]['MAPE']:.4f}%")
