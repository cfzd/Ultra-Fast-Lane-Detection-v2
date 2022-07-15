import torch
from model.backbone import resnet
import numpy as np
from utils.common import initialize_weights
from model.seg_model import SegHead
from model.layer import CoordConv

class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row = None, num_cls_row = None, num_grid_col = None, num_cls_col = None, num_lane_on_row = None, num_lane_on_col = None, use_aux=False,input_height = None, input_width = None, fc_norm = False):
        super(parsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet(backbone, pretrained=pretrained)

        # for avg pool experiment
        # self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.pool = torch.nn.AdaptiveMaxPool2d(1)

        # self.register_buffer('coord', torch.stack([torch.linspace(0.5,9.5,10).view(-1,1).repeat(1,50), torch.linspace(0.5,49.5,50).repeat(10,1)]).view(1,2,10,50))

        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18', '34fca'] else torch.nn.Conv2d(2048,8,1)
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
        initialize_weights(self.cls)
    def forward(self, x):

        x2,x3,fea = self.model(x)
        if self.use_aux:
            seg_out = self.seg_head(x2, x3,fea)
        fea = self.pool(fea)
        # print(fea.shape)
        # print(self.coord.shape)
        # fea = torch.cat([fea, self.coord.repeat(fea.shape[0],1,1,1)], dim = 1)
        
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)

        pred_dict = {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}
        if self.use_aux:
            pred_dict['seg_out'] = seg_out
        
        return pred_dict

    def forward_tta(self, x):
        x2,x3,fea = self.model(x)

        pooled_fea = self.pool(fea)
        n,c,h,w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:,:,:,:w-1] = pooled_fea[:,:,:,1:]
        left_pooled_fea[:,:,:,-1] = pooled_fea.mean(-1)
        
        right_pooled_fea[:,:,:,1:] = pooled_fea[:,:,:,:w-1]
        right_pooled_fea[:,:,:,0] = pooled_fea.mean(-1)

        up_pooled_fea[:,:,:h-1,:] = pooled_fea[:,:,1:,:]
        up_pooled_fea[:,:,-1,:] = pooled_fea.mean(-2)

        down_pooled_fea[:,:,1:,:] = pooled_fea[:,:,:h-1,:]
        down_pooled_fea[:,:,0,:] = pooled_fea.mean(-2)
        # 10 x 25
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim = 0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        return {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}

def get_model(cfg):
    return parsingNet(pretrained = True, backbone=cfg.backbone, num_grid_row = cfg.num_cell_row, num_cls_row = cfg.num_row, num_grid_col = cfg.num_cell_col, num_cls_col = cfg.num_col, num_lane_on_row = cfg.num_lanes, num_lane_on_col = cfg.num_lanes, use_aux = cfg.use_aux, input_height = cfg.train_height, input_width = cfg.train_width, fc_norm = cfg.fc_norm).cuda()