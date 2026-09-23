"""Global pytest configuration for AuditRAG."""

import os

# Disable tqdm monitor background thread to prevent thread conflicts in Python 3.13 on macOS
os.environ["TQDM_DISABLE_MONITOR"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    from tqdm import tqdm
    tqdm.monitor_interval = 0
except ImportError:
    pass

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass
