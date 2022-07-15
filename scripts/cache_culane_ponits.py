import os
import cv2
import numpy as np
import tqdm
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the dataset')
    return parser


if __name__ == '__main__':
    args = get_args().parse_args()
    culane_root = args.root
    train_list = os.path.join(culane_root, 'list/train_gt.txt')
    with open(train_list, 'r') as fp:
        res = fp.readlines()
    cache_dict = {}
    for line in tqdm.tqdm(res):
        info = line.split(' ')


        label_path = os.path.join(culane_root, info[1][1:])
        label_img = cv2.imread(label_path)[:,:,0]

        txt_path = info[0][1:].replace('jpg','lines.txt')
        txt_path = os.path.join(culane_root, txt_path)
        lanes = open(txt_path, 'r').readlines()

        all_points = np.zeros((4,35,2), dtype=np.float)
        the_anno_row_anchor = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590])
        all_points[:,:,1] = np.tile(the_anno_row_anchor, (4,1))
        all_points[:,:,0] = -99999
        # init using no lane


        for lane_idx , lane in enumerate(lanes):
            ll = lane.strip().split(' ')
            point_x = ll[::2]
            point_y = ll[1::2]

            mid_x = int(float(point_x[int(len(point_x)/2)]))
            mid_y = int(float(point_y[int(len(point_x)/2)]))
            lane_order = label_img[mid_y-1, mid_x - 1]
            if lane_order == 0:
                import pdb; pdb.set_trace()

            for i in range(len(point_x)):
                p1x = float(point_x[i])
                pos = (int(point_y[i]) - 250) / 10
                all_points[lane_order - 1, int(pos), 0] = p1x
        cache_dict[info[0][1:]] = all_points.tolist()
    with open(os.path.join(culane_root, 'culane_anno_cache.json'), 'w') as f:
        json.dump(cache_dict, f)

        


        

