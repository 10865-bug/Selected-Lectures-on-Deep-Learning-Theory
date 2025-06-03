import pickle
import numpy as np
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    data = load_data('../loss curves/gpt_loss+lrs.pkl')
    plt.figure(figsize=(14, 7)) 
    scheduler_styles = {
        '8-1-1': {'name': '8-1-1_LRS'},
        'WSD': {'name': 'WSD_LRS'},
        'cosine': {'name': 'cosine_LRS'}
    }

    selected_scheduler = '8-1-1' 
    
    scheduler_data = data[selected_scheduler]
    steps = scheduler_data['steps']
    lrs = scheduler_data['lrs']
    
    plt.plot(steps, lrs, 
         color='#1f77b4',
         lw=1.5,
         label='Learning Rate Curve')  
    
    plt.title(scheduler_styles[selected_scheduler]['name'], fontsize=14, pad=15)  
    plt.xlabel("Step", fontsize=20)  
    plt.ylabel("Learning Rate", fontsize=20)  
    plt.legend(fontsize=12, frameon=True, borderaxespad=0.5)
    plt.grid(alpha=0.2)
    os.makedirs('./figures', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'./figures/{selected_scheduler}_lrs.png', dpi=300, bbox_inches='tight')