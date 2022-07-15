import torch
from model.backbone import resnet
import numpy as np
from utils.common import initialize_weights
from model.seg_model import SegHead

class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row = None, num_cls_row = None, num_grid_col = None, num_cls_col = None, 
                num_lane_on_row = None, num_lane_on_col = None, use_aux=False, input_height = None, input_width = None):
        super(parsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux

        self.input_height = input_height
        self.input_width = input_width


        self.dim1 = self.num_grid_row * self.num_cls_row
        self.dim2 = 2 * self.num_cls_row
        self.dim3 = self.num_grid_col * self.num_cls_col
        self.dim4 = 2 * self.num_cls_col
        self.total_dim_row = self.dim1 + self.dim2
        self.total_dim_col = self.dim3 + self.dim4
        mlp_mid_dim = 2048
        
        self.input_dim = (self.input_height//32) * (self.input_width//32) * 9

        self.model = resnet(backbone, pretrained=pretrained)

        self.cls_distribute = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 20, 3, padding=1),
        )
        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU()
        )
        self.cls_row = torch.nn.Linear(mlp_mid_dim, self.total_dim_row)
        self.cls_col = torch.nn.Linear(mlp_mid_dim, self.total_dim_col)
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18', '34fca'] else torch.nn.Conv2d(2048,8,1)
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
        initialize_weights(self.cls_distribute)
        initialize_weights(self.cls)
        initialize_weights([self.cls_row])
        initialize_weights([self.cls_col])

    def forward(self, x):
        x2,x3,fea = self.model(x)
        if self.use_aux:
            seg_out = self.seg_head(x2, x3,fea)
        lane_token = self.cls_distribute(fea).reshape(-1, 20, 1, self.input_height//32, self.input_width//32)
        fea = self.pool(fea).unsqueeze(1).repeat(1, 20, 1, 1, 1)
        fea = torch.cat([fea, lane_token], 2)

        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea).reshape(-1, 20, 2048)
        out_row = self.cls_row(out[:, :10, :]).permute(0, 2, 1)
        out_col = self.cls_col(out[:, 10:, :]).permute(0, 2, 1)

        pred_dict = {'loc_row': out_row[:,:self.dim1, :].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out_col[:,:self.dim3, :].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out_row[:,self.dim1:self.dim1+self.dim2, :].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out_col[:,self.dim3:self.dim3+self.dim4, :].view(-1, 2, self.num_cls_col, self.num_lane_on_col),
                'lane_token_row': lane_token[:, :10, :, :].sum(1), 'lane_token_col': lane_token[:, 10:, :, :].sum(1)}
        if self.use_aux:
            pred_dict['seg_out'] = seg_out
        
        return pred_dict

    def forward_tta(self, x):
        raise NotImplementedError

def get_model(cfg):
    return parsingNet(pretrained = True, backbone=cfg.backbone, num_grid_row = cfg.num_cell_row, num_cls_row = cfg.num_row, num_grid_col = cfg.num_cell_col, num_cls_col = cfg.num_col, num_lane_on_row = cfg.num_lanes, num_lane_on_col = cfg.num_lanes, use_aux = cfg.use_aux, input_height = cfg.train_height, input_width = cfg.train_width).cuda()