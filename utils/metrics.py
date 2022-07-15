import numpy as np
import torch
import time,pdb

def converter(data):
    if isinstance(data,torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()
def fast_hist(label_pred, label_true,num_classes):
    #pdb.set_trace()
    hist = np.bincount(num_classes * label_true.astype(int) + label_pred, minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

class Metric_mIoU():
    def __init__(self,class_num):
        self.class_num = class_num
        self.hist = np.zeros((self.class_num,self.class_num))
    def update(self,predict,target):
        predict,target = converter(predict),converter(target)

        self.hist += fast_hist(predict,target,self.class_num)

    def reset(self):
        self.hist = np.zeros((self.class_num,self.class_num))
    def get_miou(self):
        miou = np.diag(self.hist) / (
                    np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) -
                    np.diag(self.hist))
        miou = np.nanmean(miou)
        return miou

    def get_acc(self):
        acc = np.diag(self.hist) / self.hist.sum(axis=1)
        acc = np.nanmean(acc)
        return acc
    def get(self):
        return self.get_miou()
class MultiLabelAcc():
    def __init__(self):
        self.cnt = 0
        self.correct = 0
    def reset(self):
        self.cnt = 0
        self.correct = 0
    def update(self,predict,target):
        predict = predict.argmax(1)
        predict,target = converter(predict),converter(target)
        self.cnt += len(predict)
        self.correct += np.sum(predict==target)
    def get_acc(self):
        return self.correct * 1.0 / self.cnt
    def get(self):
        return self.get_acc()
class AccTopk():
    def __init__(self,background_classes,k):
        self.background_classes = background_classes
        self.k = k
        self.cnt = 0
        self.top5_correct = 0
    def reset(self):
        self.cnt = 0
        self.top5_correct = 0
    def update(self,predict,target):
        predict = predict.argmax(1)
        predict,target = converter(predict),converter(target)
        self.cnt += len(predict)
        background_idx = (target == self.background_classes)
        # self.top5_correct += np.sum(predict[background_idx] == target[background_idx])
        not_background_idx = np.logical_not(background_idx)
        self.top5_correct += np.sum(np.absolute(predict[not_background_idx]-target[not_background_idx])<self.k)
    def get(self):
        return self.top5_correct * 1.0 / self.cnt

class Mae():
    def __init__(self, dim_sel, ignore=-1):
        self.dim_sel = dim_sel
        self.ignore = ignore
        self.all_res = []
    def reset(self):
        self.all_res = []
    def update(self,predict,target):
        # import pdb; pdb.set_trace()
        predict = predict[..., self.dim_sel]
        target = target[..., self.dim_sel]

        cls_dim = predict.shape[1]
        grid = torch.arange(cls_dim, device = predict.device).view(1, cls_dim, 1)
        predict = predict.softmax(1)
        predict = (predict * grid).sum(1) / (cls_dim - 1)
        res = (predict - target).abs()[target!= self.ignore]
        res = converter(res)
        if len(res) != 0:
            self.all_res.append(res)
    def get(self):
        if len(self.all_res) == 0:
            return 1
        return np.mean(np.concatenate(self.all_res))

def update_metrics(metric_dict, pair_data):
    for i in range(len(metric_dict['name'])):
        metric_op = metric_dict['op'][i]
        data_src = metric_dict['data_src'][i]
        metric_op.update(pair_data[data_src[0]], pair_data[data_src[1]])


def reset_metrics(metric_dict):
    for op in metric_dict['op']:
        op.reset()


if __name__ == '__main__':

    # p = np.random.randint(5, size=(800, 800))
    # t = np.zeros((800, 800))
    # me = Metric_mIoU(5)
    # me.update(p,p)
    # me.update(p,t)
    # me.update(p,p)
    # me.update(p,t)
    # print(me.get_miou())
    # print(me.get_acc())
    
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    b = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 0])
    me = AccTopk(0,5)
    me.update(b,a)
    print(me.get())