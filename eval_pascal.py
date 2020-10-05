import torch
import torch.nn.functional as F
import numpy as np
import os
import random
from custom_dataset import PF_Pascal
from geek_dataset import PairAnimeDataset
from model import SFNet
#import matplotlib.pyplot as plt
import argparse

import matplotlib.pyplot as plt
def imgshow(img):
    plt.imshow(img)
    plt.show()

parser = argparse.ArgumentParser(description="SFNet evaluation")
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loader')
parser.add_argument('--feature_h', type=int, default=48, help='height of feature volume')
parser.add_argument('--feature_w', type=int, default=32, help='width of feature volume')
parser.add_argument('--test_csv_path', type=str, default='./data/PF_Pascal/bbox_test_pairs_pf_pascal.csv', help='directory of test csv file')
parser.add_argument('--test_image_path', type=str, default='./data/PF_Pascal/', help='directory of test data')
parser.add_argument('--beta', type=float, default=50, help='inverse temperature of softmax @ kernel soft argmax')
parser.add_argument('--kernel_sigma', type=float, default=5, help='standard deviation of Gaussian kerenl @ kernel soft argmax')
parser.add_argument('--eval_type', type=str, default='image_size', choices=('bounding_box','image_size'), help='evaluation type for PCK threshold (bounding box | image size)')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Data Loader
print("Instantiate dataloader")
root_dir = "/home/kan/Desktop/cinnamon/cn/hades_painting_version_github/full_data"
w = 512
h = 768
image_size = (w, h) # (w, h)
mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]
mean = np.array(mean)[:, np.newaxis][:, np.newaxis]
std = np.array(std)[:, np.newaxis][:, np.newaxis]
test_dataset = PairAnimeDataset(root_dir, image_size, mean, std)
#test_dataset = PF_Pascal(args.test_csv_path, args.test_image_path, args.feature_h, args.feature_w, args.eval_type)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=1,
                                           shuffle=False, num_workers = args.num_workers)

# Instantiate model
print("Instantiate model")
net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma = args.kernel_sigma)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


# Load weights
print("Load pre-trained weights")
best_weights = torch.load("./weights/best_checkpoint_geek_sketch.pt")
adap3_dict = best_weights['state_dict1']
adap4_dict = best_weights['state_dict2']
net.adap_layer_feat3.load_state_dict(adap3_dict, strict=False)
net.adap_layer_feat4.load_state_dict(adap4_dict, strict=False)

import cv2
def find_corresspondence(src_image, tgt_image, output_model,
                         tgt_w, tgt_h, src_w, src_h,
                         grid_np):
    points = np.zeros(shape=(18, 18, 3))
    corr_T2S = output_model['corr_T2S'].cpu().view(20, 20, 20, 20)[1:-1, 1:-1, 1:-1, 1:-1]  # b, 18, 18, 18, 18

    for i in range(18):
        for j in range(18):
            x = grid_np[i][j][0]
            y = grid_np[i][j][1]
            x = int((x + 1) * (18 - 1) / 2)
            y = int((y + 1) * (18 - 1) / 2)
            score = corr_T2S[y, x, i, j]
            points[i][j][0] = x
            points[i][j][1] = y
            points[i][j][2] = score

    scores = points[:, :, 2]  # 18 x 18
    scores = scores.reshape(-1)
    order = scores.argsort()[::-1][:100]
    # src_image = np.array(src_image)
    # tgt_image = np.array(tgt_image)
    vis_map = np.concatenate((src_image, tgt_image), axis=1)
    for i in range(60):
        index = int(order[i])
        x = index % 18
        y = index // 18
        s_x = int(points[y][x][0])
        s_y = int(points[y][x][1])
        x = int(x / 17 * (tgt_w - 1))
        y = int(y / 17 * (tgt_h - 1))
        s_x = int(s_x / 17 * (src_w - 1))
        s_y = int(s_y / 17 * (src_h - 1))

        print (x + 320, y)
        print (s_x, s_y)

        continue

        cv2.circle(vis_map, center=(x + 320, y), radius=2, color=(255, 255, 255), thickness=-1)
        cv2.circle(vis_map, center=(s_x, s_y), radius=2, color=(255, 255, 255), thickness=-1)
        cv2.line(vis_map, (s_x, s_y), (x + 320, y), (0, 0, 255), 1)

    imgshow(vis_map)
    cv2.imshow('res1', vis_map)
    cv2.waitKey()

def revert_normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std  = torch.tensor(std).view(1,1,1,3).permute(0,3,2,1)
    mean = torch.tensor(mean).view(1,1,1,3).permute(0,3,2,1)

    return tensor_img * std + mean

def apply_grid(input, h, w, grid):
    input_resized = F.interpolate(input, (h, w), mode='bilinear')
    return F.grid_sample(input_resized, grid, mode='bilinear')


# PCK metric from 'https://github.com/ignacio-rocco/weakalign/blob/master/util/eval_util.py'
def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    # compute correct keypoints
    p_src = source_points[0,:]
    p_wrp = warped_points[0,:]

    N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
    point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
    L_pck_mat = L_pck[0].expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck


class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
per_class_pck = np.zeros(20)
num_instances = np.zeros(20)
with torch.no_grad():
    print('Computing PCK@Test set...')
    net.eval()
    total_correct_points = 0
    total_points = 0
    for i, batch in enumerate(test_loader):
        src_image = batch['image1'].to(device)
        tgt_image = batch['image2'].to(device)
        output = net(src_image, tgt_image, train=False)

        small_grid = output['grid_T2S'][:,1:-1,1:-1,:]
        small_grid[:,:,:,0] = small_grid[:,:,:,0] * (args.feature_w//2)/(args.feature_w//2 - 1)
        small_grid[:,:,:,1] = small_grid[:,:,:,1] * (args.feature_h//2)/(args.feature_h//2 - 1)
        src_image_H = 768  #int(batch['image1_size'][0][0])
        src_image_W = 512 #int(batch['image1_size'][0][1])
        tgt_image_H = 768 #int(batch['image2_size'][0][0])
        tgt_image_W = 512 #int(batch['image2_size'][0][1])
        small_grid = small_grid.permute(0,3,1,2)
        grid = F.interpolate(small_grid, size = (tgt_image_H, tgt_image_W), mode='bilinear', align_corners=True)
        grid = grid.permute(0,2,3,1)
        grid_np = grid.cpu().data.numpy()

        image1_points = np.array([[np.random.randint(100, 512), np.random.randint(100, 768)] for _ in range(10)]).transpose() #np.array([[np.random.,300], [400,400], [500, 500]]).transpose() #batch['image1_points'][0]
        image2_points = np.array([[np.random.randint(100, 512), np.random.randint(100, 768)] for _ in range(10)]).transpose()

        src_image_np = (revert_normalize(src_image.cpu())[0].permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint)
        tgt_image_np = (revert_normalize(tgt_image.cpu())[0].permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint)

        debug_image = np.concatenate([src_image_np, tgt_image_np], axis=1)
        cv2.imwrite("./tmp.png", debug_image)
        debug_image = cv2.imread("./tmp.png")

        est_image1_points = np.zeros((2, image1_points.shape[1] ))
        for j in range(image2_points.shape[1]):
            point_x = int(np.round(image2_points[0,j]))
            point_y = int(np.round(image2_points[1,j]))

            if point_x == -1 and point_y == -1:
                continue

            if point_x == tgt_image_W:
                point_x = point_x - 1

            if point_y == tgt_image_H:
                point_y = point_y - 1

            est_y = (grid_np[0,point_y,point_x,1] + 1)*(src_image_H-1)/2
            est_x = (grid_np[0,point_y,point_x,0] + 1)*(src_image_W-1)/2
            est_image1_points[:,j] = [est_x,est_y]

            print (point_x, point_y)
            print (est_x, est_y)

            cv2.circle(debug_image, (int(point_x+ 512), int(point_y)), 2, color=(255, 0, 0), thickness=3)
            cv2.circle(debug_image, (int(est_x), int(est_y)), 2, color=(255, 0, 0), thickness=3)
            cv2.line(debug_image, (int(est_x), int(est_y)), (int(point_x+ 512), int(point_y)), (0, 0, 255), 1)

        imgshow(debug_image)
        continue
        total_correct_points += correct_keypoints(batch['image1_points'], torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'], alpha=0.1)
        per_class_pck[batch['class_num']] += correct_keypoints(batch['image1_points'], torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'], alpha=0.1)
        num_instances[batch['class_num']] += 1
    PCK = total_correct_points / len(test_dataset)
    print('PCK: %5f' % PCK)

    per_class_pck = per_class_pck / num_instances
    for i in range(per_class_pck.shape[0]):
        print('%-12s' % class_names[i],': %5f' % per_class_pck[i])

