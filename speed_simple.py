import torch
import time
import numpy as np
from utils.common import get_model, merge_config

torch.backends.cudnn.benchmark = True
args, cfg = merge_config()
net = get_model(cfg)
net.eval()

x = torch.ones((1,3,cfg.train_height,cfg.train_width)).cuda()
for i in range(10):
    y = net(x)

t_all = []
for i in range(100):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))


