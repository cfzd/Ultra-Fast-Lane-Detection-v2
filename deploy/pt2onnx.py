import torch
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
from utils.common import get_model
from utils.config import Config
from utils.dist_utils import dist_print



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py', help='path to config file', type=str)
    parser.add_argument('--model_path', default='weights/culane_res34.pth',
                        help='path to model file', type=str)
    parser.add_argument('--accuracy', default='fp32', choices=['fp16', 'fp32'], type=str)
    parser.add_argument('--size', default=(1600, 320), help='size of original frame', type=tuple)
    return parser.parse_args()


def convert(model, args):
    dist_print('start convert...')
    images = torch.ones((1, 3, args.size[1], args.size[0])).cuda()
    onnx_path = args.model_path[:-4] + ".onnx"
    with torch.no_grad():
        torch.onnx.export(model, images,
                          onnx_path,
                          verbose=False,
                          input_names=['input'],
                          output_names=["loc_row", "loc_col", "exist_row", "exist_col"])
        dist_print("Export ONNX successful. Model is saved at", onnx_path)

        if args.accuracy == 'fp16':
            onnx_model = onnxmltools.utils.load_model(onnx_path)
            onnx_model = convert_float_to_float16(onnx_model)

            onnx_half_path = args.model_path[:-4] + "_fp16.onnx"
            onnxmltools.utils.save_model(onnx_model, onnx_half_path)
            dist_print("Half model is saved at", onnx_half_path)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = get_args()
    cfg = Config.fromfile(args.config_path)
    cfg.batch_size = 1

    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = get_model(cfg)

    state_dict = torch.load(args.model_path, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    convert(net, args)
