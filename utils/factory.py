from utils.loss import SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis, MeanLoss, TokenSegLoss, VarLoss, EMDLoss, RegLoss
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, Mae
from utils.dist_utils import DistSummaryWriter

import torch


def get_optimizer(net,cfg):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, cfg, iters_per_epoch):
    if cfg.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg.steps, cfg.gamma, iters_per_epoch, cfg.warmup, iters_per_epoch if cfg.warmup_iters is None else cfg.warmup_iters)
    elif cfg.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg.epoch * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)
    else:
        raise NotImplementedError
    return scheduler

def get_loss_dict(cfg):
    if cfg.dataset == 'CurveLanes':
        loss_dict = {
            'name': ['cls_loss', 'relation_loss', 'relation_dis','cls_loss_col','cls_ext','cls_ext_col', 'mean_loss_row', 'mean_loss_col','var_loss_row', 'var_loss_col', 'lane_token_seg_loss_row', 'lane_token_seg_loss_col'],
            'op': [SoftmaxFocalLoss(2, ignore_lb=-1), ParsingRelationLoss(), ParsingRelationDis(), SoftmaxFocalLoss(2, ignore_lb=-1), torch.nn.CrossEntropyLoss(),  torch.nn.CrossEntropyLoss(), MeanLoss(), MeanLoss(), VarLoss(cfg.var_loss_power), VarLoss(cfg.var_loss_power), TokenSegLoss(), TokenSegLoss()],
            'weight': [1.0, cfg.sim_loss_w, cfg.shp_loss_w, 1.0, 1.0, 1.0, cfg.mean_loss_w, cfg.mean_loss_w, 0.01, 0.01, 1.0, 1.0],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',), ('cls_out_col', 'cls_label_col'), 
            ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label') , ('cls_out', 'cls_label'),('cls_out_col', 'cls_label_col'),('cls_out', 'cls_label'),('cls_out_col', 'cls_label_col'), ('seg_out_row', 'seg_label'), ('seg_out_col', 'seg_label')
            ],
        }
    elif cfg.dataset in ['Tusimple', 'CULane']:
        loss_dict = {
            'name': ['cls_loss', 'relation_loss', 'relation_dis','cls_loss_col','cls_ext','cls_ext_col', 'mean_loss_row', 'mean_loss_col'],
            'op': [SoftmaxFocalLoss(2, ignore_lb=-1), ParsingRelationLoss(), ParsingRelationDis(), SoftmaxFocalLoss(2, ignore_lb=-1), torch.nn.CrossEntropyLoss(),  torch.nn.CrossEntropyLoss(), MeanLoss(), MeanLoss(),],
            'weight': [1.0, cfg.sim_loss_w, cfg.shp_loss_w, 1.0, 1.0, 1.0, cfg.mean_loss_w, cfg.mean_loss_w,],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',), ('cls_out_col', 'cls_label_col'), 
            ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label') , ('cls_out', 'cls_label'),('cls_out_col', 'cls_label_col'),
            ],
        }
    else:
        raise NotImplementedError

    
    if cfg.use_aux:
        loss_dict['name'].append('seg_loss')
        loss_dict['op'].append(torch.nn.CrossEntropyLoss(weight = torch.tensor([0.6, 1., 1., 1., 1.])).cuda())
        loss_dict['weight'].append(1.0)
        loss_dict['data_src'].append(('seg_out', 'seg_label'))

    assert len(loss_dict['name']) == len(loss_dict['op']) == len(loss_dict['data_src']) == len(loss_dict['weight'])
    return loss_dict

def get_metric_dict(cfg):

    metric_dict = {
        'name': ['top1', 'top2', 'top3', 'ext_row', 'ext_col'],
        'op': [AccTopk(-1, 1), AccTopk(-1, 2), AccTopk(-1, 3), MultiLabelAcc(),MultiLabelAcc()],
        'data_src': [('cls_out', 'cls_label'), ('cls_out', 'cls_label'), ('cls_out', 'cls_label'), 
        ('cls_out_ext', 'cls_out_ext_label'),('cls_out_col_ext','cls_out_col_ext_label')]
    }
    metric_dict['name'].extend(['col_top1', 'col_top2', 'col_top3'])
    metric_dict['op'].extend([AccTopk(-1, 1), AccTopk(-1, 2), AccTopk(-1, 3),])
    metric_dict['data_src'].extend([('cls_out_col', 'cls_label_col'), ('cls_out_col', 'cls_label_col'), ('cls_out_col', 'cls_label_col'), ])

    if cfg.use_aux:
        metric_dict['name'].append('iou')
        metric_dict['op'].append(Metric_mIoU(5))
        metric_dict['data_src'].append(('seg_out', 'seg_label'))

    assert len(metric_dict['name']) == len(metric_dict['op']) == len(metric_dict['data_src'])
    return metric_dict


class MultiStepLR:
    def __init__(self, optimizer, steps, gamma = 0.1, iters_per_epoch = None, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.steps = steps
        self.steps.sort()
        self.gamma = gamma
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # multi policy
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)
            power = -1
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)
            # print(self.iters, self.iters_per_epoch, self.steps, power)
            
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)
import math
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2

        