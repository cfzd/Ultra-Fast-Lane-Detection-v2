import os
import cv2
import tqdm
import json
import numpy as np
import json, argparse
import imagesize

def calc_k(line, height, width, angle=False):
    '''
    Calculate the direction of lanes
    '''
    line_x = line[::2]
    line_y = line[1::2]

    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
    if length < 90:
        return -10
    p = np.polyfit(line_x, line_y, deg = 1)
    rad = np.arctan(p[0])

    if angle:
        return rad

    try:
        curve = np.polyfit(line_x[:2], line_y[:2], deg = 1)
    except Exception:
        curve = np.polyfit(line_x[:3], line_y[:3], deg = 1)

    try:
        curve1 = np.polyfit(line_y[:2], line_x[:2], deg = 1)
    except Exception:
        curve1 = np.polyfit(line_y[:3], line_x[:3], deg = 1)

    if rad < 0:
        y = np.poly1d(curve)(0)
        if y > height:
            result = np.poly1d(curve1)(height)
        else:
            result = -(height-y)
    else:
        y = np.poly1d(curve)(width)
        if y > height:
            result = np.poly1d(curve1)(height)
        else:
            result = width+(height-y)
    
    return result
    
def draw(im, line, idx, ratio_height = 1, ratio_width = 1, show = False):
    '''
    Generate the segmentation label according to json annotation
    '''
    line_x = np.array(line[::2]) * ratio_width
    line_y = np.array(line[1::2]) * ratio_height
    pt0 = (int(line_x[0]),int(line_y[0]))
    if show:
        cv2.putText(im,str(idx),(int(line_x[len(line_x) // 2]),int(line_y[len(line_x) // 2]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60
        
    
    for i in range(len(line_x)-1):
        cv2.line(im,pt0,(int(line_x[i+1]),int(line_y[i+1])),(idx,),thickness = 16)
        pt0 = (int(line_x[i+1]),int(line_y[i+1]))


def spline(arr, the_anno_row_anchor, ratio_height = 1, ratio_width = 1):
    arr = np.array(arr)
    arr[1::2] = arr[1::2] * ratio_height
    arr[::2] = arr[::2] * ratio_width
    curve = np.polyfit(arr[1::2], arr[::2], min(len(arr[::2]) - 1, 3))
    _min = arr[1::2].min()
    _max = arr[1::2].max()
    valid = ~((the_anno_row_anchor <= _max) & (the_anno_row_anchor >= _min))
    new_x = np.polyval(curve, the_anno_row_anchor)
    final_anno_list = np.concatenate([new_x.reshape(-1, 1), the_anno_row_anchor.reshape(-1, 1)], -1)
    final_anno_list[valid, 0] = -99999

    return final_anno_list

def get_curvelanes_list(root, label_dir):
    '''
    Get all the files' names from the json annotation
    '''
    l = os.path.join(root, label_dir)
    line_txt = []
    names = []
    for img_name in tqdm.tqdm(os.listdir(os.path.join(l, 'images'))):
        temp_img_name = os.path.join('images', img_name)
        names.append(temp_img_name)
        label = img_name.replace('jpg', 'lines.json')
        f = open(os.path.join(l, 'labels', label), 'r')
        lines = json.load(f)['Lines']
        f.close()
        temp_lines = []
        for line in lines:
            temp_line = []
            line = sorted(line, key=lambda x: -float(x['y']))
            for point in line:
                temp_line.append(float(point['x']))
                temp_line.append(float(point['y']))
            temp_lines.append(temp_line)
        line_txt.append(temp_lines)
    
    return names,line_txt

def generate_segmentation_and_train_list(root, line_txt, names, file_name='train_gt.txt', json_name='curvelanes_anno_cache.json'):
    """
    The lane annotations of the Tusimple dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """
    assert os.path.exists(root)
    train_gt_fp = open(os.path.join(root, file_name), 'w')
    cache_dict = {}
    if not os.path.exists(os.path.join(root, 'segs')):
        os.mkdir(os.path.join(root, 'segs'))

    for i in tqdm.tqdm(range(len(line_txt))):

        lines = line_txt[i]

        width, height = imagesize.get(os.path.join(root,names[i]))

        ks = np.array([calc_k(line, height, width) for line in lines])                # get the direction of each lane
        ks_theta = np.array([calc_k(line, height, width, angle=True) for line in lines])             # get the direction of each lane

        k_neg = ks[ks_theta<0].copy()
        k_neg_theta = ks_theta[ks_theta<0].copy()
        k_pos = ks[ks_theta>0].copy()
        k_pos_theta = ks_theta[ks_theta>0].copy()
        k_neg = k_neg[k_neg_theta != -10]                                      # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos_theta != -10]
        k_neg.sort()
        k_pos.sort()
        label_path = 'segs'+names[i][6:-3]+'png'
        label = np.zeros((height,width),dtype=np.uint8)  
        bin_label = [0] * 10

        all_points = np.zeros((10,125,2), dtype=np.float)
        the_anno_row_anchor = np.array(list(range(200, 1450, 10)))
        all_points[:,:,1] = np.tile(the_anno_row_anchor, (10,1))
        all_points[:,:,0] = -99999

        rw = 2560 / width
        rh = 1440 / height
        for idx in range(len(k_neg))[:5]:
            which_lane = np.where(ks == k_neg[idx])[0][0]
            draw(label,lines[which_lane],5-idx)
            bin_label[4-idx] = 1
            all_points[4-idx] = spline(np.array(lines[which_lane]), the_anno_row_anchor, ratio_height = rh, ratio_width = rw)
        
        for idx in range(len(k_pos))[:5]:
            which_lane = np.where(ks == k_pos[-(idx+1)])[0][0]
            draw(label,lines[which_lane],6+idx)
            bin_label[5+idx] = 1
            all_points[5+idx] = spline(np.array(lines[which_lane]), the_anno_row_anchor, ratio_height = rh, ratio_width = rw)
        
        cv2.imwrite(os.path.join(root,label_path),label)
        cache_dict['train/'+names[i]] = all_points.tolist()

        train_gt_fp.write('train/'+names[i] + ' ' + 'train/' +label_path + ' '+' '.join(list(map(str,bin_label))) + '\n')
    train_gt_fp.close()
    with open(os.path.join(root, json_name), 'w') as f:
        json.dump(cache_dict, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the CurveLanes dataset')
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()


    names, line_txt = get_curvelanes_list(args.root,  'train')
    # generate training list for training
    generate_segmentation_and_train_list(os.path.join(args.root, 'train'), line_txt, names)


    # names, line_txt = get_curvelanes_list(args.root,  'valid')
    # generate_segmentation_and_train_list(os.path.join(args.root, 'valid'), line_txt, names, file_name='valid_gt.txt', json_name='culane_anno_cache_val.json')

