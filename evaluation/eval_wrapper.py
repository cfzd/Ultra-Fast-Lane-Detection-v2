
from data.dataloader import get_test_loader
from evaluation.tusimple.lane2 import LaneEval
from utils.dist_utils import is_main_process, dist_print, get_rank, get_world_size, dist_tqdm, synchronize
import os, json, torch, scipy
import numpy as np
import platform
from scipy.optimize import leastsq
from data.constant import culane_col_anchor, culane_row_anchor

def generate_lines(out, out_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):

    grid = torch.arange(out.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out.softmax(1) * grid).sum(1) 
    
    loc = loc / (out.shape[1]-1) * 1640
    # n, num_cls, num_lanes
    valid = out_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            for i in [1,2]:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            fp.write('%.3f %.3f '% ( loc[j,k,i] , culane_row_anchor[k] * 590))
                    fp.write('\n')

def generate_lines_col(out_col,out_col_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):
    
    grid = torch.arange(out_col.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out_col.softmax(1) * grid).sum(1) 
    
    loc = loc / (out_col.shape[1]-1) * 590
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            for i in [0,3]:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            fp.write('%.3f %.3f '% ( culane_col_anchor[k] * 1640, loc[j,k,i] ))
                    fp.write('\n')

def generate_lines_local(dataset, out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    max_indices = out.argmax(1).cpu()
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu()

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [1, 2]
        elif dataset == 'CurveLanes':
            # lane_list = [2, 3, 4, 5, 6, 7]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = (out[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 

                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out.shape[1]-1) * 1640
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out.shape[1]-1) * 2560
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 1440))
                            else:
                                raise Exception
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_local(dataset, out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [0, 3]
        elif dataset == 'CurveLanes':
            # lane_list = [0, 1, 8, 9]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 590
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 1440
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 2560, out_tmp ))
                            else:
                                raise Exception

                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_local_curve_combine(dataset, out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    max_indices = out.argmax(1).cpu()
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu()

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [1, 2]
        elif dataset == 'CurveLanes':
            # lane_list = [2, 3, 4, 5, 6, 7]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        # import pdb; pdb.set_trace()

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines_row.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out.shape[1]-1) * 1640
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out.shape[1]-1) * 2560
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 1440))
                            else:
                                raise Exception
                    fp.write('\n')
                else:
                    fp.write('\n')

def generate_lines_col_local_curve_combine(dataset, out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [0, 3]
        elif dataset == 'CurveLanes':
            # lane_list = [0, 1, 8, 9]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):
        # import pdb; pdb.set_trace()

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines_col.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 590
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 1440
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 2560, out_tmp ))
                            else:
                                raise Exception

                    fp.write('\n')
                # elif mode == 'all':
                #     fp.write('\n')
                else:
                    fp.write('\n')

def revise_lines_curve_combine(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        row_line_save_path = os.path.join(output_path, name[:-3] + 'lines_row.txt')
        col_line_save_path = os.path.join(output_path, name[:-3] + 'lines_col.txt')
        if not os.path.exists(row_line_save_path):
            continue
        if not os.path.exists(col_line_save_path):
            continue
        with open(row_line_save_path, 'r') as fp:
            row_lines = fp.readlines()
        with open(col_line_save_path, 'r') as fp:
            col_lines = fp.readlines()
        flag = True
        for i in range(10):
            x1, y1 = coordinate_parse(row_lines[i])
            x2, y2 = coordinate_parse(col_lines[i])
            x = x1 + x2
            y = y1 + y2
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 36)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()

def generate_lines_reg(out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu().sigmoid()

    if mode == 'normal' or mode == '2row2col':
        lane_list = [1, 2]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = out[j,0,k,i] * 1640

                            fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_reg(out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    # max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu().sigmoid()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        lane_list = [0, 3]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            # out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_col[j,0,k,i] * 590
                            fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def coordinate_parse(line):
    if line == '\n':
        return [], []

    items = line.split(' ')[:-1]
    x = [float(items[2*i]) for i in range(len(items)//2)]
    y = [float(items[2*i+1]) for i in range(len(items)//2)]

    return x, y


def func(p, x):
    f = np.poly1d(p)
    return f(x)


def resudual(p, x, y):
    error = y - func(p, x)
    return error


def revise_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for i in range(4):
            x1, y1 = coordinate_parse(lines[i])
            x2, y2 = coordinate_parse(lines[i+4])
            x = x1 + x2
            y = y1 + y2
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()
            

def rectify_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for line in lines:
            x, y = coordinate_parse(line)
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()


def run_test(dataset, net, data_root, exp_name, work_dir, distributed, crop_ratio, train_width, train_height , batch_size=8, row_anchor = None, col_anchor = None):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = net(imgs)
        
        if dataset == "CULane":
            generate_lines_local(dataset, pred['loc_row'],pred['exist_row'], names, output_path, 'normal', row_anchor=row_anchor)
            generate_lines_col_local(dataset, pred['loc_col'],pred['exist_col'], names, output_path, 'normal', col_anchor=col_anchor)
        elif dataset == 'CurveLanes':
            generate_lines_local_curve_combine(dataset, pred['loc_row'],pred['exist_row'], names, output_path, row_anchor=row_anchor)
            generate_lines_col_local_curve_combine(dataset, pred['loc_col'],pred['exist_col'], names, output_path, col_anchor=col_anchor)
            revise_lines_curve_combine(names, output_path)
        else:
            raise NotImplementedError



def generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor):

    local_width = 1

    max_indices = loc_row.argmax(1).cpu()
    valid = exist_row.argmax(1).cpu()
    loc_row = loc_row.cpu()

    max_indices_left = loc_row_left.argmax(1).cpu()
    valid_left = exist_row_left.argmax(1).cpu()
    loc_row_left = loc_row_left.cpu()

    max_indices_right = loc_row_right.argmax(1).cpu()
    valid_right = exist_row_right.argmax(1).cpu()
    loc_row_right = loc_row_right.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_row.shape

    min_lane_length = num_cls / 2

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg', 'lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [1,2]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_row[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 1640
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_left[batch_idx,cls_idx,lane_idx]:
                            all_ind_left = torch.tensor(list(range(max(0,max_indices_left[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_left[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_left = (loc_row_left[batch_idx,all_ind_left,cls_idx,lane_idx].softmax(0) * all_ind_left.float()).sum() + 0.5 
                            out_tmp_left = out_tmp_left / (num_grid-1) * 1640 + 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_left

                        if valid_right[batch_idx,cls_idx,lane_idx]:
                            all_ind_right = torch.tensor(list(range(max(0,max_indices_right[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_right[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_right = (loc_row_right[batch_idx,all_ind_right,cls_idx,lane_idx].softmax(0) * all_ind_right.float()).sum() + 0.5 
                            out_tmp_right = out_tmp_right / (num_grid-1) * 1640 - 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_right


                        if cnt >= 2:
                            pt_all.append(( out_tmp_all/cnt , row_anchor[cls_idx] * 590))
                    if len(pt_all) < min_lane_length:
                            continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor):
    local_width = 1
    
    max_indices = loc_col.argmax(1).cpu()
    valid = exist_col.argmax(1).cpu()
    loc_col = loc_col.cpu()

    max_indices_up = loc_col_up.argmax(1).cpu()
    valid_up = exist_col_up.argmax(1).cpu()
    loc_col_up = loc_col_up.cpu()

    max_indices_down = loc_col_down.argmax(1).cpu()
    valid_down = exist_col_down.argmax(1).cpu()
    loc_col_down = loc_col_down.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_col.shape

    min_lane_length = num_cls / 4

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg','lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [0,3]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_col[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_up[batch_idx,cls_idx,lane_idx]:
                            all_ind_up = torch.tensor(list(range(max(0,max_indices_up[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_up[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_up = (loc_col_up[batch_idx,all_ind_up,cls_idx,lane_idx].softmax(0) * all_ind_up.float()).sum() + 0.5 
                            out_tmp_up = out_tmp_up / (num_grid-1) * 590 + 32./534*590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_up
                        if valid_down[batch_idx,cls_idx,lane_idx]:
                            all_ind_down = torch.tensor(list(range(max(0,max_indices_down[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_down[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_down = (loc_col_down[batch_idx,all_ind_down,cls_idx,lane_idx].softmax(0) * all_ind_down.float()).sum() + 0.5 
                            out_tmp_down = out_tmp_down / (num_grid-1) * 590 - 32./534*590     
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_down

                        if cnt >= 2:
                            pt_all.append(( col_anchor[cls_idx] * 1640, out_tmp_all/cnt ))
                    if len(pt_all) < min_lane_length:
                        continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def run_test_tta(dataset, net, data_root, exp_name, work_dir,distributed, crop_ratio, train_width, train_height, batch_size=8, row_anchor = None, col_anchor = None):
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            if hasattr(net, 'module'):
                pred = net.module.forward_tta(imgs)
            else:
                pred = net.forward_tta(imgs)

            loc_row, loc_row_left, loc_row_right, _, _ = torch.chunk(pred['loc_row'], 5)
            loc_col, _, _, loc_col_up, loc_col_down = torch.chunk(pred['loc_col'], 5)

            exist_row, exist_row_left, exist_row_right, _, _ = torch.chunk(pred['exist_row'], 5)
            exist_col, _, _, exist_col_up, exist_col_down = torch.chunk(pred['exist_col'], 5)


        generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor)
        generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor)

def generate_tusimple_lines(row_out, row_ext, col_out, col_ext, row_anchor = None, col_anchor = None, mode = '2row2col'):
    tusimple_h_sample = np.linspace(160, 710, 56)
    row_num_grid, row_num_cls, row_num_lane = row_out.shape
    row_max_indices = row_out.argmax(0).cpu()
    # num_cls, num_lanes
    row_valid = row_ext.argmax(0).cpu()
    # num_cls, num_lanes
    row_out = row_out.cpu()

    col_num_grid, col_num_cls, col_num_lane = col_out.shape
    col_max_indices = col_out.argmax(0).cpu()
    # num_cls, num_lanes
    col_valid = col_ext.argmax(0).cpu()
    # num_cls, num_lanes
    col_out = col_out.cpu()

    # mode = '2row2col'

    if mode == 'normal' or mode == '2row2col':
        row_lane_list = [1, 2]
        col_lane_list = [0, 3]
    elif mode == '4row':
        row_lane_list = range(row_num_lane)
        col_lane_list = []
    elif mode == '4col':
        row_lane_list = []
        col_lane_list = range(col_num_lane)
    else:
        raise NotImplementedError

    local_width_row = 14
    local_width_col = 14
    min_lanepts_row = 3
    min_lanepts_col = 3
    
    # local_width = 2
    all_lanes = []

    for row_lane_idx in row_lane_list:
        if row_valid[ :, row_lane_idx].sum() > min_lanepts_row:
            cur_lane = []
            for row_cls_idx in range(row_num_cls):

                if row_valid[ row_cls_idx, row_lane_idx]:
                    all_ind = torch.tensor(list(
                        range(
                            max(0,row_max_indices[ row_cls_idx, row_lane_idx] - local_width_row), 
                            min(row_num_grid-1, row_max_indices[ row_cls_idx, row_lane_idx] + local_width_row) + 1)
                            )
                            )
                    coord = (row_out[all_ind,row_cls_idx,row_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_x = coord / (row_num_grid - 1) * 1280
                    coord_y = row_anchor[row_cls_idx] * 720
                    cur_lane.append(int(coord_x))
                else:
                    cur_lane.append(-2)
                    # cur_lane.append((coord_x, coord_y))
            # cur_lane = np.array(cur_lane)
            # p = np.polyfit(cur_lane[:,1], cur_lane[:,0], deg = 2)
            # top_lim = min(cur_lane[:,1])
            # # all_lane_interps.append((p, top_lim))
            # lanes_on_tusimple = np.polyval(p, tusimple_h_sample)
            # lanes_on_tusimple = np.round(lanes_on_tusimple)
            # lanes_on_tusimple = lanes_on_tusimple.astype(int)
            # lanes_on_tusimple[lanes_on_tusimple < 0] = -2
            # lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
            # lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
            # all_lanes.append(lanes_on_tusimple.tolist())
            all_lanes.append(cur_lane)
        else:
            # all_lanes.append([-2]*56)
            pass

    for col_lane_idx in col_lane_list:
        if col_valid[ :, col_lane_idx].sum() > min_lanepts_col:
            cur_lane = []
            for col_cls_idx in range(col_num_cls):
                if col_valid[ col_cls_idx, col_lane_idx]:
                    all_ind = torch.tensor(list(
                        range(
                            max(0,col_max_indices[ col_cls_idx, col_lane_idx] - local_width_col), 
                            min(col_num_grid-1, col_max_indices[ col_cls_idx, col_lane_idx] + local_width_col) + 1)
                            )
                            )
                    coord = (col_out[all_ind,col_cls_idx,col_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_y = coord / (col_num_grid - 1) * 720
                    coord_x = col_anchor[col_cls_idx] * 1280
                    cur_lane.append((coord_x, coord_y))    
            cur_lane = np.array(cur_lane)
            top_lim = min(cur_lane[:,1])
            bot_lim = max(cur_lane[:,1])
            
            p = np.polyfit(cur_lane[:,1], cur_lane[:,0], deg = 2)
            lanes_on_tusimple = np.polyval(p, tusimple_h_sample)

            # cur_lane_x = cur_lane[:,0]
            # cur_lane_y = cur_lane[:,1]
            # cur_lane_x_sorted = [x for _, x in sorted(zip(cur_lane_y, cur_lane_x))]
            # cur_lane_y_sorted = sorted(cur_lane_y)
            # p = InterpolatedUnivariateSpline(cur_lane_y_sorted, cur_lane_x_sorted, k=min(3, len(cur_lane_x_sorted) - 1))
            # lanes_on_tusimple = p(tusimple_h_sample)

            lanes_on_tusimple = np.round(lanes_on_tusimple)
            lanes_on_tusimple = lanes_on_tusimple.astype(int)
            lanes_on_tusimple[lanes_on_tusimple < 0] = -2
            lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
            lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
            lanes_on_tusimple[tusimple_h_sample > bot_lim] = -2
            all_lanes.append(lanes_on_tusimple.tolist())
        else:
            # all_lanes.append([-2]*56)
            pass
    # for (p, top_lim) in all_lane_interps:
    #     lanes_on_tusimple = np.polyval(p, tusimple_h_sample)
    #     lanes_on_tusimple = np.round(lanes_on_tusimple)
    #     lanes_on_tusimple = lanes_on_tusimple.astype(int)
    #     lanes_on_tusimple[lanes_on_tusimple < 0] = -2
    #     lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
    #     lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
    #     all_lanes.append(lanes_on_tusimple.tolist())
    return all_lanes
    
def run_test_tusimple(net,data_root,work_dir,exp_name, distributed, crop_ratio, train_width, train_height, batch_size = 8, row_anchor = None, col_anchor = None):
    output_path = os.path.join(work_dir,exp_name+'.%d.txt'% get_rank())
    fp = open(output_path,'w')
    loader = get_test_loader(batch_size,data_root,'Tusimple', distributed, crop_ratio, train_width, train_height)
    for data in dist_tqdm(loader):
        imgs,names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = net(imgs)
        for b_idx,name in enumerate(names):
            tmp_dict = {}
            tmp_dict['lanes'] = generate_tusimple_lines(pred['loc_row'][b_idx], pred['exist_row'][b_idx], pred['loc_col'][b_idx], pred['exist_col'][b_idx], row_anchor = row_anchor, col_anchor = col_anchor, mode = '4row')
            tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
             270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 
             430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 
             590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
            tmp_dict['raw_file'] = name
            tmp_dict['run_time'] = 10
            json_str = json.dumps(tmp_dict)

            fp.write(json_str+'\n')
    fp.close()

def combine_tusimple_test(work_dir,exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(work_dir,exp_name+'.%d.txt'% i)
        with open(output_path, 'r') as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find('clips')
        name = res[pos:].split('\"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir,exp_name+'.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(all_res_no_dup)
    

def eval_lane(net, cfg, ep = None, logger = None):
    net.eval()
    if cfg.dataset == 'CurveLanes':
        if not cfg.tta:
            run_test(cfg.dataset, net, cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        else:
            run_test_tta(cfg.dataset, net, cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir, cfg.distributed,  cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()   # wait for all results
        if is_main_process():
            res = call_curvelane_eval(cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir)
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
                val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
                TP += val_tp
                FP += val_fp
                FN += val_fn
                dist_print(k,val)
                if logger is not None:
                    if k == 'res_cross':
                        logger.add_scalar('CuEval_cls/'+k,val_fp,global_step = ep)
                        continue
                    logger.add_scalar('CuEval_cls/'+k,val,global_step = ep)
            if TP + FP == 0:
                P = 0
                print("nearly no results!")
            else:
                P = TP * 1.0/(TP + FP)
            if TP + FN == 0:
                R = 0
                print("nearly no results!")
            else:
                R = TP * 1.0/(TP + FN)
            if (P+R) == 0:
                F = 0
            else:
                F = 2*P*R/(P + R)
            dist_print(F)
            if logger is not None:
                logger.add_scalar('CuEval/total',F,global_step = ep)
                logger.add_scalar('CuEval/P',P,global_step = ep)
                logger.add_scalar('CuEval/R',R,global_step = ep)
              
        synchronize()
        if is_main_process():
            return F
        else:
            return None
    elif cfg.dataset == 'CULane':
        if not cfg.tta:
            run_test(cfg.dataset, net, cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        else:
            run_test_tta(cfg.dataset, net, cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()    # wait for all results
        if is_main_process():
            res = call_culane_eval(cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir)
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
                val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
                TP += val_tp
                FP += val_fp
                FN += val_fn
                dist_print(k,val)
                if logger is not None:
                    if k == 'res_cross':
                        logger.add_scalar('CuEval_cls/'+k,val_fp,global_step = ep)
                        continue
                    logger.add_scalar('CuEval_cls/'+k,val,global_step = ep)
            if TP + FP == 0:
                P = 0
                print("nearly no results!")
            else:
                P = TP * 1.0/(TP + FP)
            if TP + FN == 0:
                R = 0
                print("nearly no results!")
            else:
                R = TP * 1.0/(TP + FN)
            if (P+R) == 0:
                F = 0
            else:
                F = 2*P*R/(P + R)
            dist_print(F)
            if logger is not None:
                logger.add_scalar('CuEval/total',F,global_step = ep)
                logger.add_scalar('CuEval/P',P,global_step = ep)
                logger.add_scalar('CuEval/R',R,global_step = ep)
              
        synchronize()
        if is_main_process():
            return F
        else:
            return None
    elif cfg.dataset == 'Tusimple':
        exp_name = 'tusimple_eval_tmp'
        run_test_tusimple(net, cfg.data_root, cfg.test_work_dir, exp_name, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()  # wait for all results
        if is_main_process():
            combine_tusimple_test(cfg.test_work_dir,exp_name)
            res = LaneEval.bench_one_submit(os.path.join(cfg.test_work_dir,exp_name + '.txt'),os.path.join(cfg.data_root,'test_label.json'))
            res = json.loads(res)
            for r in res:
                dist_print(r['name'], r['value'])
                if logger is not None:
                    logger.add_scalar('TuEval/'+r['name'],r['value'],global_step = ep)
        synchronize()
        if is_main_process():
            for r in res:
                if r['name'] == 'F1':
                    return r['value']
        else:
            return None


def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res

def call_culane_eval(data_dir, exp_name,output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path,exp_name)+'/'

    w_lane=30
    iou=0.5  # Set iou to 0.3 or 0.5
    im_w=1640
    im_h=590
    frame=1
    list0 = os.path.join(data_dir,'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir,'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir,'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir,'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir,'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir,'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir,'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir,'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir,'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    out0 = os.path.join(output_path,'txt','out0_normal.txt')
    out1=os.path.join(output_path,'txt','out1_crowd.txt')
    out2=os.path.join(output_path,'txt','out2_hlight.txt')
    out3=os.path.join(output_path,'txt','out3_shadow.txt')
    out4=os.path.join(output_path,'txt','out4_noline.txt')
    out5=os.path.join(output_path,'txt','out5_arrow.txt')
    out6=os.path.join(output_path,'txt','out6_curve.txt')
    out7=os.path.join(output_path,'txt','out7_cross.txt')
    out8=os.path.join(output_path,'txt','out8_night.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    res_all = {}
    res_all['res_normal'] = read_helper(out0)
    res_all['res_crowd']= read_helper(out1)
    res_all['res_night']= read_helper(out8)
    res_all['res_noline'] = read_helper(out4)
    res_all['res_shadow'] = read_helper(out3)
    res_all['res_arrow']= read_helper(out5)
    res_all['res_hlight'] = read_helper(out2)
    res_all['res_curve']= read_helper(out6)
    res_all['res_cross']= read_helper(out7)
    return res_all

def call_curvelane_eval(data_dir, exp_name,output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path,exp_name)+'/'

    w_lane=5
    iou=0.5  # Set iou to 0.3 or 0.5
    im_w=224
    im_h=224
    x_factor = 224 / 2560
    y_factor = 224 / 1440
    frame=1
    list0 = os.path.join(data_dir, 'valid', 'valid_for_culane_style.txt')
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    out0=os.path.join(output_path,'txt','out0_curve.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    print('./evaluate -s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s -x %s -y %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0, x_factor, y_factor))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s -x %s -y %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0, x_factor, y_factor))
    res_all = {}
    res_all['res_curve'] = read_helper(out0)
    return res_all