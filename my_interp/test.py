import cv2
import torch
import numpy as np
import my_interp
import os

def draw_points(img, points, color):
    points = points.view(-1,2).cpu().numpy()
    for (x,y) in points:
        if x < 0 or y < 0:
            continue
        img = cv2.circle(img, (int(x), int(y)), 5, color, -1)
    return img


def test(culane_root):
    # points = torch.rand(32, 4, 35, 2)
    import cv2
    test_lines_txt_path = os.path.join(culane_root, '/driver_161_90frame/06031919_0929.MP4/00000.lines.txt')
    lanes = open(test_lines_txt_path, 'r').readlines()

    all_points = np.zeros((4,35,2), dtype=np.float)
    the_anno_row_anchor = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590])
    all_points[:,:,1] = np.tile(the_anno_row_anchor, (4,1))
    all_points[:,:,0] = -99999

    label_img = cv2.imread(os.path.join(culane_root, '/laneseg_label_w16/driver_161_90frame/06031919_0929.MP4/00000.png'))[:,:,0]

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
    all_points = torch.tensor(all_points).cuda().view(1,4,35,2)
    new_interp_locations = torch.linspace(0,590,30).cuda()
    new_all_points = my_interp.run(all_points.float(), new_interp_locations.float(), 0)
    # new_interp_locations = torch.linspace(0,1640,100).cuda()
    # new_all_points = my_interp.run(all_points.float(), new_interp_locations.float(), 1)
    img = cv2.imread(os.path.join(culane_root, '/laneseg_label_w16/driver_161_90frame/06031919_0929.MP4/00000.png')) * 128
    img = draw_points(img, all_points, (0,255,0))
    img = draw_points(img, new_all_points, (0,0,255))
    cv2.imwrite('test.png', img)
    torch.set_printoptions(sci_mode=False)
if __name__ == "__main__":
    test('path/to/your/culane')
