import os
import glob
import random
import pickle
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted
from copy import deepcopy
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from rules.color_component_matching import ComponentWrapper, ShapeMatchingWrapper, resize_mask
from rules.color_component_matching import get_component_color
from self_augment.gen import Generator as SelfGenerator


def image_show(image):
    plt.imshow(image)
    plt.show()


def get_image_by_index(paths, index):
    if index is None:
        return None
    path = paths[index]

    if path.endswith("tga"):
        image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(path)
    return image, path


def draw_component_image(components, mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for component in components:
        coords = component["coords"]
        image[coords[:, 0], coords[:, 1], :] = component["color"]

    cv2.imwrite("%d.png" % len(components), image)


def affine_transform(x, theta):
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


class RandomAugmentPairAnimeDataset(data.Dataset):
    def process_input(self):
        paths = {}
        lengths = {}
        foreground_data = {}

        for set_name in ["hor02"]:
            for cut_name in os.listdir(os.path.join(self.root_dir, set_name)):
                cut_full_path = os.path.join(self.root_dir, set_name, cut_name)
                json_path = os.path.join(self.root_dir, set_name, "foreground_data.json")
                foreground_data.update(json.load(open(json_path)))

                for root, dirs, file_paths in os.walk(cut_full_path):
                    cut_name = "_".join(root.split("/")[-2:])
                    if len(file_paths) < 1:
                        continue

                    full_path_list = [
                        os.path.join(root, file_path) for file_path in file_paths if
                        os.path.splitext(file_path)[-1] in [".tga"]]

                    paths[cut_name] = {}
                    lengths[cut_name] = {}

                    paths[cut_name]["color"] = full_path_list
                    lengths[cut_name] = len(paths[cut_name]["color"])

        return lengths, paths, foreground_data

    def __init__(self, root_dir, size, feature_h=32, feature_w=48):
        super(RandomAugmentPairAnimeDataset, self).__init__()
        self.root_dir = root_dir
        self.size = size

        self.component_wrapper = ComponentWrapper(min_area=10, min_size=3)
        self.matcher = ShapeMatchingWrapper()

        self.lengths, self.paths, self.foreground_data = self.process_input()
        self.feature_h = feature_h  # height of feature volume
        self.feature_w = feature_w  # width of feature volume

        self.image_h = self.feature_h * 16
        self.image_w = self.feature_w * 16

        self.image_transform1 = transforms.Compose(
            [transforms.Resize((self.image_h, self.image_w), interpolation=2)])

        self.image_transform2 = transforms.Compose([
            transforms.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])

        self.mask_transform1 = transforms.Compose([
            transforms.Resize((self.image_h, self.image_w), interpolation=2),
            transforms.ToTensor(),
        ])

        self.mask_transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.feature_h, self.feature_w)),
            transforms.ToTensor(),
        ])

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        total = 0
        for key, count in self.lengths.items():
            total += count
        return total

    def crop_foreground_image(self, color_image, path):
        cut_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        name = os.path.splitext(os.path.basename(path))[0]
        full_name = "%s_%s" % (cut_name, name)

        if full_name in self.foreground_data:
            box = self.foreground_data[full_name]
            color_image = color_image[box[1]:box[3], box[0]:box[2]]
        return color_image

    def get_component_mask(self, color_image, path):
        method = ComponentWrapper.EXTRACT_COLOR

        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(os.path.dirname(path), "%s_%s.pkl" % (name, method))

        if not os.path.exists(save_path):
            mask, components = self.component_wrapper.extract_on_color_image(color_image)
            get_component_color(components, color_image, method)

            mask_foreground = (mask != 0).astype(np.uint) * 255
            mask = resize_mask(mask, components, self.size).astype(np.int32)
            mask_foreground = cv2.resize(mask_foreground, dsize=self.size, interpolation=cv2.INTER_NEAREST)

            save_data = {"mask_foreground": mask_foreground, "mask": mask}
            pickle.dump(save_data, open(save_path, "wb+"))
        else:
            save_data = pickle.load(open(save_path, "rb"))
            mask_foreground = save_data["mask_foreground"]
            mask = save_data["mask"]

        return mask_foreground, mask

    def from_real(self, index, next_index, name):
        to_tensor = transforms.ToTensor()

        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_a = self.crop_foreground_image(color_a, path_a)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)
        color_b = self.crop_foreground_image(color_b, path_b)

        if len(color_a) < 3:
            color_a = cv2.cvtColor(color_a, cv2.COLOR_GRAY2BGR)
            print("color_path: %s only 1 channel?" % path_a)

        if len(color_b) < 3:
            color_b = cv2.cvtColor(color_b, cv2.COLOR_GRAY2BGR)
            print("color_path: %s only 1 channel?" % path_b)

        # extract components
        mask_foreground_a, mask_a = self.get_component_mask(color_a, path_a)
        mask_foreground_b, mask_b = self.get_component_mask(color_b, path_b)
        color_a = Image.fromarray(color_a.astype(np.uint8))
        color_b = Image.fromarray(color_b.astype(np.uint8))

        image1 = to_tensor(self.image_transform1(color_a))
        image2 = to_tensor(self.image_transform1(color_b))

        # resize
        mask1 = self.mask_transform2(mask_foreground_a)
        mask2 = self.mask_transform2(mask_foreground_b)
        # binarize
        mask1 = (mask1 > 0.01).float()
        mask2 = (mask2 > 0.01).float()

        # Return image and the label
        return {
            "image1_rgb": image1.clone(), "image2_rgb": image2.clone(),
            "image1": self.normalize(image1), "image2": self.normalize(image2),
            "mask1": mask1, "mask2": mask2,
        }

    def from_augment(self, index, next_index, name):
        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_a = self.crop_foreground_image(color_a, path_a)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)
        color_b = self.crop_foreground_image(color_b, path_b)

        # extract components
        mask_foreground_a, mask_a = self.get_component_mask(color_a, path_a)
        mask_foreground_b, mask_b = self.get_component_mask(color_b, path_b)

        if np.random.uniform(0.0, 1.0) < 0.5:
            image = Image.fromarray(color_a.astype(np.uint8))
            mask = Image.fromarray(mask_foreground_a.astype(np.uint8))
        else:
            image = Image.fromarray(color_b.astype(np.uint8))
            mask = Image.fromarray(mask_foreground_b.astype(np.uint8))

        p = np.random.uniform()
        if p < 0.5:
            # pair flip
            image, mask = TF.hflip(image), TF.hflip(mask)

        # resize
        image = self.image_transform1(image)
        # jitter -> image1
        image1 = self.image_transform2(image).unsqueeze(0)
        image2 = deepcopy(image1)
        # resize
        mask = self.mask_transform1(mask).unsqueeze(0)

        # generate source image/mask
        theta1 = np.zeros(9)
        theta1[0:6] = np.random.randn(6) * 0.1
        theta1 = theta1 + np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        affine1 = np.reshape(theta1, (3, 3))
        affine_inverse1 = np.linalg.inv(affine1)
        affine1 = np.reshape(affine1, -1)[0:6]
        affine_inverse1 = np.reshape(affine_inverse1, -1)[0:6]
        affine1 = torch.from_numpy(affine1).type(torch.FloatTensor)
        affine_inverse1 = torch.from_numpy(affine_inverse1).type(torch.FloatTensor)

        # source image
        image1 = affine_transform(image1, affine1)
        mask = affine_transform(mask, affine1)
        mask = affine_transform(mask, affine_inverse1)
        # convert truncated pixels to 0

        # generate target image/mask
        theta2 = np.zeros(9)
        theta2[0:6] = np.random.randn(6) * 0.15
        theta2 = theta2 + np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        affine2 = np.reshape(theta2, (3, 3))
        affine_inverse2 = np.linalg.inv(affine2)
        affine2 = np.reshape(affine2, -1)[0:6]
        affine_inverse2 = np.reshape(affine_inverse2, -1)[0:6]
        affine2 = torch.from_numpy(affine2).type(torch.FloatTensor)
        affine_inverse2 = torch.from_numpy(affine_inverse2).type(torch.FloatTensor)

        # target image
        image2 = affine_transform(image2, affine2)
        mask2 = affine_transform(mask, affine2)
        mask = affine_transform(mask2, affine_inverse2)
        # convert truncated pixels to 0
        # source mask
        mask1 = affine_transform(mask, affine1)

        image1, image2, mask1, mask2 = image1.squeeze(0).data, image2.squeeze(0).data, mask1.squeeze(
            0).data, mask2.squeeze(0).data

        # resize
        mask1 = self.mask_transform2(mask1)
        mask2 = self.mask_transform2(mask2)
        # binarize
        mask1 = (mask1 > 0.01).float()
        mask2 = (mask2 > 0.01).float()

        # Return image and the label
        return {
            "image1_rgb": image1.clone(), "image2_rgb": image2.clone(),
            "image1": self.normalize(image1), "image2": self.normalize(image2),
            "mask1": mask1, "mask2": mask2,
        }

    def __getitem__(self, index):
        name = None
        for key, length in self.lengths.items():
            if index < length:
                name = key
                break
            index -= length

        # next index
        length = len(self.paths[name]["color"])
        k = random.choice(list(range(length)))
        next_index = max(index - k, 0) if index == length - 1 else min(index + k, length - 1)

        # check if this sample is hard
        path = self.paths[name]["color"][index]
        is_hard = "hard_hor02" in path

        if (not is_hard) and np.random.uniform(0.0, 1.0) < 0.5:
            return self.from_real(index, next_index, name)
        return self.from_augment(index, next_index, name)


def revert_normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.tensor(std).view(1, 1, 1, 3).permute(0, 3, 2, 1)
    mean = torch.tensor(mean).view(1, 1, 1, 3).permute(0, 3, 2, 1)
    return tensor_img * std + mean


def tensor2image(tensor_input, revert=True):
    if revert:
        tensor_input = revert_normalize(tensor_input.cpu())

    if len(tensor_input.shape) == 3:
        return (tensor_input[0] * 255).detach().cpu().numpy().astype(np.uint)

    return (tensor_input[0].permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint)
