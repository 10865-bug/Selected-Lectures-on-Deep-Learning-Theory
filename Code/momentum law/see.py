import pickle
import numpy as np
from pprint import pprint

def inspect_pkl(path):
    """查看PKL文件结构"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print("="*40)
    print("PKL文件结构摘要：")
    print("-"*40)
    
    # 递归打印数据结构
    def print_structure(d, indent=0, max_samples=3):
        prefix = "  "*indent
        if isinstance(d, dict):
            for k, v in d.items():
                print(f"{prefix}├── Key: {k} ({type(v)})")
                if isinstance(v, (dict, list, np.ndarray)):
                    print_structure(v, indent+1)
                else:
                    if hasattr(v, '__len__') and len(v) > max_samples:
                        print(f"{prefix}│   └── Samples: {v[:max_samples]}... (total {len(v)})")
                    else:
                        print(f"{prefix}│   └── Value: {v}")
        elif isinstance(d, (list, np.ndarray)):
            print(f"{prefix}├── Type: {type(d)}")
            print(f"{prefix}│   └── Length: {len(d)}")
            if len(d) > 0:
                print(f"{prefix}│   └── First element: {type(d[0])}")
                if isinstance(d[0], (np.generic, int, float)):
                    print(f"{prefix}│       └── Samples: {d[:max_samples]}...")
        else:
            print(f"{prefix}└── Value: {d}")

    print_structure(data)
    print("="*40)

if __name__ == "__main__":
    inspect_pkl('../loss curves/gpt_loss+lrs.pkl')