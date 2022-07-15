import os
import json
import tqdm
import argparse
import imagesize

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the dataset')
    return parser

def read_label(label_path, x_factor, y_factor):
    js = json.load(open(label_path,'r'))['Lines']
    all_lanes = []
    for ll in js:
        cur_lane_x = []
        cur_lane_y = []
        for pt in ll:
            loc_x = float(pt['x']) * x_factor
            loc_y = float(pt['y']) * y_factor
            cur_lane_x.append(loc_x)
            cur_lane_y.append(loc_y)
            # img = cv2.circle(img, (int(loc_x), int(loc_y)), 5, (0,0,255), -1)

        cur_lane_x_sorted = [x for _, x in sorted(zip(cur_lane_y, cur_lane_x))]
        cur_lane_y_sorted = sorted(cur_lane_y)


        all_lanes.append([str(val) for pair in zip(cur_lane_x_sorted, cur_lane_y_sorted) for val in pair])

    return all_lanes

def generate_linestxt_on_curvelane_val():
    args = get_args().parse_args()
    curvelane_val_root = os.path.join(args.root, 'valid')

    assert os.path.exists(curvelane_val_root)

    assert os.path.exists(os.path.join(curvelane_val_root, 'images'))

    list_file = os.path.join(curvelane_val_root, 'valid.txt')

    all_files = open(list_file, 'r').readlines()

    for file in tqdm.tqdm(all_files):
        file = file.strip()
        label_path = file.replace('images', 'labels')
        label_path = label_path.replace('.jpg', '.lines.json')

        label_path = os.path.join(curvelane_val_root, label_path)
        file_path = os.path.join(curvelane_val_root, file)

        width, height = imagesize.get(file_path)

        culane_style_label = read_label(label_path, x_factor = 2560 / width, y_factor = 1440 / height)
        culane_style_label_store_path = os.path.join(curvelane_val_root, file).replace('jpg','lines.txt')
        with open(culane_style_label_store_path, 'w') as f:
            for culane_style_label_i in culane_style_label:
                f.write(' '.join(culane_style_label_i)+'\n')

    fp = open(os.path.join(curvelane_val_root, 'valid.txt'), 'r')
    res = fp.readlines()
    fp.close()
    res = [os.path.join('valid', r) for r in res]
    with open(os.path.join(curvelane_val_root, 'valid_for_culane_style.txt'), 'w') as fp:
        fp.writelines(res)


if __name__ == "__main__":
    generate_linestxt_on_curvelane_val()