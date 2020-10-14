import os
import glob
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF

from rules.color_component_matching import ComponentWrapper, ShapeMatchingWrapper, resize_mask
from rules.color_component_matching import get_component_color
from rules.component_wrapper import get_moment_features
from self_augment.gen import Generator as SelfGenerator

import matplotlib.pyplot as plt
def imgshow(im):
    plt.imshow(im)
    plt.show()

def b2label(mask_b, positive_pairs, components_a, components_b):
    mask_b_corr_a = np.zeros_like(mask_b)

    for id_a, id_b in positive_pairs:
        lbl_a = components_a[id_a]['label']
        lbl_b = components_b[id_b]['label']

        mask_b_corr_a[mask_b == lbl_b] = lbl_a

    return mask_b_corr_a

def get_image_by_index(paths, index):
    if index is None:
        return None
    path = paths[index]

    if path.endswith("tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image, path

def draw_component_image(components, mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for component in components:
        coords = component["coords"]
        image[coords[:, 0], coords[:, 1], :] = component["color"]

    cv2.imwrite("%d.png" % len(components), image)

def match_components_three_stage(components_a, components_b, matcher, is_removed):
    pairs = []

    for index_a, a in enumerate(components_a):
        matches = [(b, matcher.process(a, b)) for b in components_b]
        count_true = len([1 for match in matches if match[1][0]])
        if count_true == 0:
            continue

        distances = np.array([match[1][1] for match in matches])
        index_b = int(np.argmin(distances))
        pairs.append([index_a, index_b])

    if len(pairs) == 0:
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, threshold=0.2))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    if len(pairs) == 0 and (not is_removed):
        for index_a, a in enumerate(components_a):
            matches = [(b, matcher.process(a, b, area_filter=False, pos_filter=False, threshold=0.6))
                       for b in components_b]
            count_true = len([1 for match in matches if match[1][0]])
            if count_true == 0:
                continue

            distances = np.array([match[1][1] for match in matches])
            index_b = int(np.argmin(distances))
            pairs.append([index_a, index_b])

    return pairs

def loader_collate(batch):
    assert len(batch) == 1
    batch = batch[0]

    features_a = torch.tensor(batch[0]).unsqueeze(0).float()
    mask_a = torch.tensor(batch[1]).unsqueeze(0).float()
    features_b = torch.tensor(batch[2]).unsqueeze(0).float()
    mask_b = torch.tensor(batch[3]).unsqueeze(0).float()

    positive_pairs = torch.tensor(batch[4]).unsqueeze(0).int()
    colors_a = torch.tensor(batch[7]).unsqueeze(0).int()
    colors_b = torch.tensor(batch[8]).unsqueeze(0).int()

    color_image_a = torch.tensor(batch[9]).unsqueeze(0).float().permute(0,3,1,2) / 255.
    color_image_b = torch.tensor(batch[10]).unsqueeze(0).float().permute(0,3,1,2) / 255.

    sketch_image_a = torch.tensor(batch[11]).unsqueeze(0).float().permute(0,3,1,2) / 255.
    sketch_image_b = torch.tensor(batch[12]).unsqueeze(0).float().permute(0,3,1,2) / 255.

    return (features_a, mask_a, colors_a, batch[5], color_image_a, sketch_image_a), \
           (features_b, mask_b, colors_b, batch[6], color_image_b, sketch_image_b), positive_pairs

def random_remove_component(mask, components, max_removed=2):
    if len(components) < 10 or random.random() < 0.6:
        return mask, components, False

    index = 1
    new_components = []
    new_mask = np.zeros(mask.shape, dtype=np.int)
    removed = 0

    for component in components:
        if random.random() > 0.05 or removed >= max_removed:
            component["label"] = index
            new_components.append(component)
            new_mask[component["coords"][:, 0], component["coords"][:, 1]] = index
            index += 1
        else:
            removed += 1
    return new_mask, new_components, removed > 0

def add_random_noise(features, mask):
    noise = np.random.normal(loc=0.0, scale=0.02, size=features.shape)
    bool_mask = mask > 0
    features = features + noise * bool_mask
    return features

def resize_image(np_image):
    return cv2.resize(np_image, dsize=(512, 768))

class MultipleMaskPairAnimeDataset(data.Dataset):
    def __init__(self, root_dir, size, mean, std):
        super(MultipleMaskPairAnimeDataset, self).__init__()
        self.root_dir = root_dir
        self.size = size
        self.mean = mean
        self.std = std

        self.paths = {}
        self.lengths = {}
        dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))

        self.component_wrapper = ComponentWrapper(min_area=10, min_size=3)
        self.matcher = ShapeMatchingWrapper()

        for sub_dir in dirs:
            dir_name = os.path.basename(sub_dir)
            self.paths[dir_name] = {}

            for set_name in ["sketch_v3", "color"]:
                paths = []
                for sub_type in ["png", "jpg", "tga"]:
                    paths.extend(glob.glob(os.path.join(sub_dir, set_name, "*.%s" % sub_type)))
                self.paths[dir_name][set_name] = natsorted(paths)

            self.lengths[dir_name] = len(self.paths[dir_name]["color"])

        self.feature_H = 48  # height of feature volume
        self.feature_W = 32  # width of feature volume

        self.image_H = self.feature_H * 16
        self.image_W = self.feature_W * 16

        self.image_transform1 = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2)])

        self.image_transform2 = transforms.Compose(
            [transforms.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
             transforms.ToTensor()])

        self.mask_transform1 = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                                   transforms.ToTensor()])

        self.mask_transform2 = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((self.feature_H, self.feature_W)),
                                                   transforms.ToTensor()])

        self.to_tensor = transforms.ToTensor()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augmentor = SelfGenerator(resized_w=512, resized_h=768, min_area=10, min_size=3)

    def __len__(self):
        total = 0
        for key, count in self.lengths.items():
            total += count
        return total

    def get_component_mask(self, color_image, path, extract_prob=0.4):
        is_pd = any([(w in path) for w in ["PD09", "PD10"]])
        method = ComponentWrapper.EXTRACT_COLOR

        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(os.path.dirname(path), "%s_%s.pkl" % (name, method))

        if not os.path.exists(save_path):
            mask, components = self.component_wrapper.extract_on_color_image(color_image)
            get_component_color(components, color_image, ComponentWrapper.EXTRACT_COLOR)
            save_data = {"mask": mask, "components": components}
            pickle.dump(save_data, open(save_path, "wb+"))
        else:
            save_data = pickle.load(open(save_path, "rb"))
            mask, components = save_data["mask"], save_data["components"]

        #mask, components, is_removed = random_remove_component(mask, components)
        is_removed = False
        mask_foreground = (mask != 0).astype(np.uint) * 255
        mask = resize_mask(mask, components, self.size).astype(np.int32)

        mask_foreground = cv2.resize(mask_foreground, dsize=self.size, interpolation=cv2.INTER_NEAREST) #self.get_mask_foreground(components, color_image)

        return mask, components, is_removed, mask_foreground

    def get_mask_foreground(self, components, color_img):
        max_area_component = sorted(components, key=lambda comp: comp['area'])[-1]
        coord = max_area_component['coords'][0]
        rgb_value = color_img[coord[0], coord[1]]

        if tuple(rgb_value) == tuple([255,255,255]):
            mask = 255 - max_area_component['image']
            return mask

        return None

    def affine_transform(self, x, theta):
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def _from_real(self, index, next_index, name):
        to_tensor = transforms.ToTensor()

        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        sketch_a, _ = get_image_by_index(self.paths[name]["sketch_v3"], index)
        sketch_a = cv2.cvtColor(sketch_a, cv2.COLOR_GRAY2BGR)

        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)
        sketch_b, _ = get_image_by_index(self.paths[name]["sketch_v3"], next_index)
        sketch_b = cv2.cvtColor(sketch_b, cv2.COLOR_GRAY2BGR)

        # extract components
        mask_a, components_a, is_removed_a, mask_foreground_a = self.get_component_mask(color_a, path_a)
        mask_b, components_b, is_removed_b, mask_foreground_b = self.get_component_mask(color_b, path_b)

        # component matching
        positive_pairs = match_components_three_stage(components_a, components_b, self.matcher, False)
        positive_pairs = np.array(positive_pairs)

        # -> tensor
        sketch_a = Image.fromarray(sketch_a.astype(np.uint8))
        sketch_b = Image.fromarray(sketch_b.astype(np.uint8))

        image1 = to_tensor(self.image_transform1(sketch_a))
        image2 = to_tensor(self.image_transform1(sketch_b))

        mask1s = []
        mask2s = []
        for index_a, index_b in positive_pairs:
            lbl_a = components_a[index_a]['label']
            lbl_b = components_b[index_b]['label']

            area_a = components_a[index_a]['area']
            area_b = components_b[index_b]['area']

            min_area = 0.5 * self.feature_H * self.feature_W
            if area_a < min_area or area_b < min_area:
                continue

            if lbl_a == 0 or lbl_b == 0:
                continue

            mask1 = (mask_a == lbl_a).astype(np.uint8) * 255
            mask2 = (mask_b == lbl_b).astype(np.uint8) * 255

            # imgshow(mask1)
            # imgshow(mask2)

            mask1 = self.mask_transform2(mask1)
            mask2 = self.mask_transform2(mask2)

            mask1 = (mask1 > 0.1).float()  # binarize
            mask2 = (mask2 > 0.1).float()  # binarize

            # imgshow(tensor2image(mask1))
            # imgshow(tensor2image(mask2))

            mask1s += [mask1]
            mask2s += [mask2]

        # fore-ground vs back-ground
        mask1 = self.mask_transform2(mask_foreground_a)  # resize
        mask2 = self.mask_transform2(mask_foreground_b)  # resize
        mask1 = (mask1 > 0.1).float()  # binarize
        mask2 = (mask2 > 0.1).float()  # binarize

        mask1s += [mask1]
        mask2s += [mask2]

        return {'image1_rgb': image1.clone(), 'image2_rgb': image2.clone(), 'image1': self.normalize(image1),
                'image2': self.normalize(image2), 'mask1s': mask1s, 'mask2s': mask2s}

    def _from_augment(self, index, next_index, name):
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        sketch_a, _ = get_image_by_index(self.paths[name]["sketch_v3"], index)

        # extract components
        list_a, list_b, positive_pairs = self.augmentor.process(color_a, sketch_a, crop_bbox=False)
        colored_a, mask_a, components_a, _ = list_a
        colored_b, mask_b, components_b, sketch_b = list_b

        sketch_a = Image.fromarray(cv2.cvtColor(sketch_a.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        sketch_b = Image.fromarray(cv2.cvtColor(sketch_b.astype(np.uint8), cv2.COLOR_GRAY2BGR))

        # mask fore ground
        mask_foreground_a = (mask_a != 0).astype(np.uint8) * 255
        mask_foreground_b = (mask_b != 0).astype(np.uint8) * 255

        # -> tensor
        image1 = self.to_tensor(self.image_transform1(sketch_a))
        image2 = self.to_tensor(self.image_transform1(sketch_b))

        mask1s = []
        mask2s = []
        for index_a, index_b in positive_pairs:
            lbl_a = components_a[index_a]['label']
            lbl_b = components_b[index_b]['label']

            area_a = components_a[index_a]['area']
            area_b = components_b[index_b]['area']

            min_area = 0.5 * self.feature_H * self.feature_W
            if area_a < min_area or area_b < min_area:
                continue

            if lbl_a == 0 or lbl_b == 0:
                continue

            mask1 = (mask_a == lbl_a).astype(np.uint8) * 255
            mask2 = (mask_b == lbl_b).astype(np.uint8) * 255

            # imgshow(mask1)
            # imgshow(mask2)

            mask1 = self.mask_transform2(mask1)
            mask2 = self.mask_transform2(mask2)

            mask1 = (mask1 > 0.1).float()  # binarize
            mask2 = (mask2 > 0.1).float()  # binarize

            # imgshow(tensor2image(mask1))
            # imgshow(tensor2image(mask2))

            mask1s += [mask1]
            mask2s += [mask2]

        # fore-ground vs back-ground
        mask1 = self.mask_transform2(mask_foreground_a)  # resize
        mask2 = self.mask_transform2(mask_foreground_b)  # resize
        mask1 = (mask1 > 0.1).float()  # binarize
        mask2 = (mask2 > 0.1).float()  # binarize

        mask1s += [mask1]
        mask2s += [mask2]

        return {'image1_rgb': image1.clone(), 'image2_rgb': image2.clone(), 'image1': self.normalize(image1),
                'image2': self.normalize(image2), 'mask1s': mask1s, 'mask2s': mask2s}

    def __getitem__(self, index):
        name = None
        for key, length in self.lengths.items():
            if index < length:
                name = key
                break
            index -= length

        # next index
        length = len(self.paths[name]["color"])
        k = random.choice([1, 1, 1, 2])
        next_index = max(index - k, 0) if index == length - 1 else min(index + k, length - 1)

        if np.random.uniform(0, 1.) < -0.5:
            print ('from real')
            return self._from_real(index, next_index, name)
        else:
            print ('from augment')
            return self._from_augment(index, next_index, name)

def revert_normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std  = torch.tensor(std).view(1,1,1,3).permute(0,3,2,1)
    mean = torch.tensor(mean).view(1,1,1,3).permute(0,3,2,1)

    return tensor_img * std + mean

def tensor2image(tensor_input, revert=True):
    if revert:
        tensor_input = revert_normalize(tensor_input.cpu())

    if len(tensor_input.shape) == 3:
        return (tensor_input[0] * 255).detach().cpu().numpy().astype(np.uint)

    return (tensor_input[0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint)

if __name__ == '__main__':
    root_dir = "./../../hades_painting_version_github/full_data"
    w = 512
    h = 768
    image_size = (w, h)  # (w, h)
    mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
    std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]
    mean = np.array(mean)[:, np.newaxis][:, np.newaxis]
    std = np.array(std)[:, np.newaxis][:, np.newaxis]

    train_dataset = MultipleMaskPairAnimeDataset(root_dir, image_size, mean, std)
    train_loader  = data.DataLoader(train_dataset, shuffle=True, batch_size=1)
    print ('n_dataset:', len(train_dataset))

    train_iter = iter(train_loader)
    for batch_id, batch_data in enumerate(train_iter):
        image1, image2  = batch_data['image1'], batch_data['image2']
        mask1s, mask2s  = batch_data['mask1s'], batch_data['mask2s']

        print ('image1', image1.shape)
        print ('image2', image2.shape)
        print ('mask1s', len(mask1s))
        print ('mask2s', len(mask2s))

        image1_np   = tensor2image(image1)
        image2_np   = tensor2image(image2)

        print ('image1')
        imgshow(image1_np)

        print ('image2')
        imgshow(image2_np)

        g_count = 0
        for mask1, mask2 in zip(mask1s, mask2s):
            print ('g_count num: %d ...' % (g_count + 1))

            mask1_np = tensor2image(mask1)
            mask2_np = tensor2image(mask2)

            print ('mask1')
            imgshow(mask1_np)

            print ('mask2')
            imgshow(mask2_np)

            g_count += 1

