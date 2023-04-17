import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config


class UFLDv2:
    def __init__(self, engine_path, config_path, ori_size):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        cfg = Config.fromfile(config_path)
        self.ori_img_w, self.ori_img_h = ori_size
        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col
        self.row_anchor = np.linspace(0.42, 1, self.num_row)
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def pred2coords(self, pred):
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1)
        # n , num_cls, num_lanes
        valid_row = pred['exist_row'].argmax(1)
        # n, num_cls, num_lanes

        max_indices_col = pred['loc_col'].argmax(1)
        # n , num_cls, num_lanes
        valid_col = pred['exist_col'].argmax(1)
        # n, num_cls, num_lanes

        pred['loc_row'] = pred['loc_row']
        pred['loc_col'] = pred['loc_col']

        coords = []
        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]

        for i in row_lane_idx:
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - self.input_width),
                                                          min(num_grid_row - 1,
                                                              max_indices_row[0, k, i] + self.input_width) + 1)))

                        out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        tmp.append((int(out_tmp), int(self.row_anchor[k] * self.ori_img_h)))
                coords.append(tmp)

        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - self.input_width),
                                                          min(num_grid_col - 1,
                                                              max_indices_col[0, k, i] + self.input_width) + 1)))
                        out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * self.ori_img_h
                        tmp.append((int(self.col_anchor[k] * self.ori_img_w), int(out_tmp)))
                coords.append(tmp)
        return coords

    def forward(self, img):
        im0 = img.copy()
        img = img[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 2, 0, 1))
        img = np.ascontiguousarray(img)
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        preds = {}
        for out in self.outputs:
            output = np.zeros(out['shape'], out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            preds[out['name']] = torch.tensor(output)
        coords = self.pred2coords(preds)
        for lane in coords:
            for coord in lane:
                cv2.circle(im0, coord, 2, (0, 255, 0), -1)
        cv2.imshow("result", im0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py', help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weights/culane_res34.engine',
                        help='path to engine file', type=str)
    parser.add_argument('--video_path', default='example.mp4', help='path to video file', type=str)
    parser.add_argument('--ori_size', default=(1600, 320), help='size of original frame', type=tuple)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video_path)
    isnet = UFLDv2(args.engine_path, args.config_path, args.ori_size)
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1600, 903))
        img = img[380:700, :, :]
        isnet.forward(img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
