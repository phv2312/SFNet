import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import cv2
from PIL import Image
import torchvision.transforms as transforms
from geek_dataset import PairAnimeDataset
from model import SFNet
import argparse

import matplotlib.pyplot as plt
def imgshow(img):
    plt.imshow(img)
    plt.show()


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

def find_corresponding_point(point_x, point_y, tgt_image_W, tgt_image_H, grid_np):
    if point_x == -1 and point_y == -1:
        return -1, -1

    if point_x == tgt_image_W:
        point_x = point_x - 1

    if point_y == tgt_image_H:
        point_y = point_y - 1

    est_y = (grid_np[0, point_y, point_x, 1] + 1) * (tgt_image_H - 1) / 2
    est_x = (grid_np[0, point_y, point_x, 0] + 1) * (tgt_image_W - 1) / 2

    return (int(est_x), int(est_y))

from rules.component_wrapper import ComponentWrapper, resize_mask, get_component_color
component_wrapper = ComponentWrapper(min_area=10, min_size=3)
def prepare_batch(sketch_a_path, sketch_b_path, h=768, w=512, color_a_path=None, color_b_path=None):
    transform_sequences = transforms.Compose([
        transforms.Resize(size=(h, w), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sketch_a = Image.open(sketch_a_path).convert('RGB')
    color_a  = Image.open(color_a_path).convert('RGB')
    sketch_b = Image.open(sketch_b_path).convert('RGB')

    #
    mask_a, components_a = component_wrapper.extract_on_color_image(np.array(color_a))
    get_component_color(components_a, np.array(color_a), mode=ComponentWrapper.EXTRACT_COLOR)

    mask_b, components_b = component_wrapper.extract_on_sketch_v3(np.array(sketch_b))

    # resize
    mask_a_resized = resize_mask(mask_a, components_a, size=(w,h))
    mask_b_resized = resize_mask(mask_b, components_b, size=(w,h))

    #
    sketch_a_tensor = transform_sequences(sketch_a).unsqueeze(0)
    sketch_b_tensor = transform_sequences(sketch_b).unsqueeze(0)

    return (sketch_a_tensor, mask_a_resized, components_a), \
           (sketch_b_tensor, mask_b_resized, components_b)

def visualize_pair(paired_components, ref_colored_img, ref_sketch_components, sketch_components, rgb_sketch):
    """Color the sketch.
    Args:
        paired_components (dict): in the form {index of sketch component: index of corresponding reference component}
        ref_colored_img (numpy array): reference colored image to be used for coloring
        ref_sketch_components (dict): dictionary of reference components in the form {index: dict of components' properties}
        sketch_components {dict): dictionary of all sketch components in the form {index: dict of components' properties}
        rgb_sketch (numpy array): RGB image of the sketch to be colored
    Returns:
        numpy array: the colored image of the orginal_sketch
    """
    colored_sketch = rgb_sketch.copy()
    for ref_component_label, component_label in paired_components:
        ref_id = [_id for _id, c in enumerate(ref_sketch_components) if c['label'] == ref_component_label]
        tgt_id = [_id for _id, c in enumerate(sketch_components) if c['label'] == component_label]

        if len(ref_id) == 0 or len(tgt_id) == 0:
            continue

        ref_component_index = ref_id[0]
        component_index = tgt_id[0]

        component_img = sketch_components[component_index]
        ref_img = ref_sketch_components[ref_component_index]

        ref_rows, ref_cols = ref_img['coords'][:, 0], ref_img['coords'][:, 1]
        comp_rows, comp_cols = component_img['coords'][:, 0], component_img['coords'][:, 1]

        for i in range(3):
            layer = ref_colored_img[:, :, i].copy()
            color = np.max(layer[ref_rows, ref_cols])
            colored_sketch[comp_rows, comp_cols, i] = color

    return colored_sketch

def visualize(pair, ref_components, tgt_components):
    debug_im = np.ones(shape=(9999, 9999, 3), dtype=np.uint8) * 255
    max_x, max_y = -1, -1
    for ref_lbl, tgt_lbl in pair:
        ref_id = [_id for _id, c in enumerate(ref_components) if c['label'] == ref_lbl]
        tgt_id = [_id for _id, c in enumerate(tgt_components) if c['label'] == tgt_lbl]

        if len(ref_id) == 0 or len(tgt_id) == 0:
            continue

        ref_id = ref_id[0]
        tgt_id = tgt_id[0]

        ref_color  = ref_components[ref_id]['color']
        tgt_coords = tgt_components[tgt_id]['coords']

        debug_im[tgt_coords[:,0], tgt_coords[:,1]] = ref_color

        #
        _max_y = max(tgt_coords[:, 0])
        _max_x = max(tgt_coords[:, 1])

        #
        max_y = max(max_y, _max_y)
        max_x = max(max_x, _max_x)

    return debug_im[:(max_y + 1), :(max_x + 1)]

from scipy.spatial.distance import cdist
from collections import Counter
def compute_component_matrix_distance(mask_a, mask_b, grid_np, target_h, target_w):
    labels_a = np.unique(mask_a)
    labels_b = np.unique(mask_b)

    # calculate mask_b_from_a
    mask_a_tensor = torch.from_numpy(mask_a[np.newaxis, :, :]).unsqueeze(0) # (b,c,h,w)
    grid_tensor   = torch.from_numpy(grid_np) # (b,h,w,2)

    mask_b_from_a = F.grid_sample(mask_a_tensor.float(), grid_tensor, mode='nearest')
    mask_b_from_a = mask_b_from_a.squeeze().cpu().numpy().astype(np.uint8)

    print ('b_from_a')
    imgshow(mask_b_from_a)

    print ('debug_mask')
    debug_mask = np.concatenate([mask_a, mask_b_from_a, mask_b], axis=1)
    imgshow(debug_mask)

    #
    pair = []
    for label_b in labels_b:
        ys_b, xs_b = np.where(mask_b == label_b)
        points_b = [(x, y) for y, x in zip(ys_b, xs_b)]

        labels_a_in_b = [mask_b_from_a[y,x] for (x,y) in points_b]
        label_a_predict = Counter(labels_a_in_b).most_common(1)[0][0]

        pair += [(label_a_predict, label_b)]

    return pair


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

# Instantiate model
print("Instantiate model")
net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma = args.kernel_sigma)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load weights
print("Load pre-trained weights")
best_weights = torch.load("/home/kan/Desktop/cinnamon/cn/github/SFNet/weights/best_checkpoint_geek_train_full.pt")
adap3_dict = best_weights['state_dict1']
adap4_dict = best_weights['state_dict2']
net.adap_layer_feat3.load_state_dict(adap3_dict, strict=False)
net.adap_layer_feat4.load_state_dict(adap4_dict, strict=False)

sketch_a_path = "/home/kan/Desktop/cinnamon/cn/hades_painting_version_github/full_data/hor01_041_k_r_A/sketch_v3/A0001.png"
color_a_path  = "/home/kan/Desktop/cinnamon/cn/hades_painting_version_github/full_data/hor01_041_k_r_A/color/A0001.tga"

sketch_b_path = "/home/kan/Desktop/cinnamon/cn/hades_painting_version_github/full_data/hor01_041_k_r_A/sketch_v3/A0002.png"
output_path   =  "|".join(sketch_b_path.split('/')[-3:])

list_a, list_b = prepare_batch(sketch_a_path, sketch_b_path, color_a_path=color_a_path)
image_a, mask_a, components_a = list_a
image_b, mask_b, components_b = list_b

ref_colored_img = np.array(Image.open(color_a_path).convert('RGB'))
rgb_sketch = np.array(Image.open(sketch_b_path).convert('RGB'))

with torch.no_grad():
    net.eval()
    src_image = image_a.to(device)
    tgt_image = image_b.to(device)

    output = net(src_image, tgt_image, train=False)
    small_grid = output['grid_T2S'][:, 1:-1, 1:-1, :]
    small_grid[:, :, :, 0] = small_grid[:, :, :, 0] * (args.feature_w // 2) / (args.feature_w // 2 - 1)
    small_grid[:, :, :, 1] = small_grid[:, :, :, 1] * (args.feature_h // 2) / (args.feature_h // 2 - 1)
    src_image_H = 768
    src_image_W = 512
    tgt_image_H = 768
    tgt_image_W = 512
    small_grid = small_grid.permute(0, 3, 1, 2)
    grid = F.interpolate(small_grid, size=(tgt_image_H, tgt_image_W), mode='bilinear', align_corners=True)
    grid = grid.permute(0, 2, 3, 1)
    grid_np = grid.cpu().data.numpy()

    pair = compute_component_matrix_distance(mask_a, mask_b, grid_np, 768, 512)
    img = visualize_pair(pair, ref_colored_img, components_a, components_b, rgb_sketch) #visualize(pair, components_a, components_b)
    debug_img = np.concatenate([ref_colored_img, rgb_sketch, img], axis=1)
    imgshow(debug_img)

    cv2.imwrite(output_path, cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))

    exit()
    print ('finished')

    image1_points = np.array([[160, 384]]).transpose() # y,x
    image2_points = np.array([[160, 384]]).transpose()

    src_image_np = (revert_normalize(src_image.cpu())[0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint)
    tgt_image_np = (revert_normalize(tgt_image.cpu())[0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint)

    debug_image = np.concatenate([src_image_np, tgt_image_np], axis=1)
    cv2.imwrite("./tmp.png", debug_image)
    debug_image = cv2.imread("./tmp.png")

    est_image1_points = np.zeros((2, image1_points.shape[1]))
    for j in range(image2_points.shape[1]):
        point_x = int(np.round(image2_points[0, j]))
        point_y = int(np.round(image2_points[1, j]))

        print (point_x, point_y)

        if point_x == -1 and point_y == -1:
            continue

        if point_x == tgt_image_W:
            point_x = point_x - 1

        if point_y == tgt_image_H:
            point_y = point_y - 1

        est_y = (grid_np[0, point_y, point_x, 1] + 1) * (src_image_H - 1) / 2
        est_x = (grid_np[0, point_y, point_x, 0] + 1) * (src_image_W - 1) / 2
        est_image1_points[:, j] = [est_x, est_y]

        print(point_x, point_y)
        print(est_x, est_y)

        cv2.circle(debug_image, (int(point_x + 512), int(point_y)), 2, color=(255, 0, 0), thickness=3)
        cv2.circle(debug_image, (int(est_x), int(est_y)), 2, color=(255, 0, 0), thickness=3)
        cv2.line(debug_image, (int(est_x), int(est_y)), (int(point_x + 512), int(point_y)), (0, 0, 255), 1)

    imgshow(debug_image)


