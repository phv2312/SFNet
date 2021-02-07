from loader.geek_dataset import *
from rules.component_wrapper import match_multi_color_masks


class MultiMaskAnimeDataset(data.Dataset):
    def process_input(self):
        paths = {}
        lengths = {}

        for set_name in ["hor02"]:
            for cut_name in os.listdir(os.path.join(self.root_dir, set_name)):
                cut_full_path = os.path.join(self.root_dir, set_name, cut_name)

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

        return lengths, paths

    def __init__(self, root_dir, size, feature_h=32, feature_w=48):
        super(MultiMaskAnimeDataset, self).__init__()
        self.root_dir = root_dir
        self.size = size

        self.paths = {}
        self.lengths = {}

        self.component_wrapper = ComponentWrapper(min_area=2000, min_size=5)
        self.matcher = ShapeMatchingWrapper()

        self.lengths, self.paths = self.process_input()
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

    def get_component_mask(self, color_image, path):
        method = ComponentWrapper.EXTRACT_MULTI_MASK

        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(os.path.dirname(path), "%s_%s.pkl" % (name, method))

        if not os.path.exists(save_path):
            masks, components = self.component_wrapper.extract_multi_mask_by_color(color_image)
            masks = [cv2.resize(m, dsize=self.size, interpolation=cv2.INTER_NEAREST) for m in masks]

            save_data = {"masks": masks, "components": components}
            pickle.dump(save_data, open(save_path, "wb+"))
        else:
            save_data = pickle.load(open(save_path, "rb"))
            masks = save_data["masks"]
            components = save_data["components"]

        return masks, components

    def from_real(self, index, next_index, name):
        to_tensor = transforms.ToTensor()

        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)

        if len(color_a) < 3:
            color_a = cv2.cvtColor(color_a, cv2.COLOR_GRAY2BGR)
            print("color_path: %s only 1 channel?" % path_a)

        if len(color_b) < 3:
            color_b = cv2.cvtColor(color_b, cv2.COLOR_GRAY2BGR)
            print("color_path: %s only 1 channel?" % path_b)

        # extract components
        masks_a, components_a = self.get_component_mask(color_a, path_a)
        masks_b, components_b = self.get_component_mask(color_b, path_b)
        color_a = Image.fromarray(color_a.astype(np.uint8))
        color_b = Image.fromarray(color_b.astype(np.uint8))
        masks_a, masks_b = match_multi_color_masks(masks_a, masks_b, components_a, components_b, k=9)

        image1 = to_tensor(self.image_transform1(color_a))
        image2 = to_tensor(self.image_transform1(color_b))

        # resize
        masks1 = [self.mask_transform2(m) for m in masks_a]
        masks2 = [self.mask_transform2(m) for m in masks_b]
        # binarize
        masks1 = [(m > 0.01).float() for m in masks1]
        masks2 = [(m > 0.01).float() for m in masks2]
        # make tensor
        mask1 = torch.cat(masks1, dim=0)
        mask2 = torch.cat(masks2, dim=0)

        # Return image and the label
        return {
            "image1_rgb": image1.clone(), "image2_rgb": image2.clone(),
            "image1": self.normalize(image1), "image2": self.normalize(image2),
            "mask1": mask1, "mask2": mask2,
        }

    def from_augment(self, index, next_index, name):
        # read images
        color_a, path_a = get_image_by_index(self.paths[name]["color"], index)
        color_b, path_b = get_image_by_index(self.paths[name]["color"], next_index)

        # extract components
        masks_a, components_a = self.get_component_mask(color_a, path_a)
        masks_b, components_b = self.get_component_mask(color_b, path_b)

        if np.random.uniform(0.0, 1.0) < 0.5:
            image = Image.fromarray(color_a.astype(np.uint8))
            masks = [Image.fromarray(m.astype(np.uint8)) for m in masks_a]
        else:
            image = Image.fromarray(color_b.astype(np.uint8))
            masks = [Image.fromarray(m.astype(np.uint8)) for m in masks_b]

        # resize
        image = self.image_transform1(image)
        # jitter -> image1
        image1 = self.image_transform2(image).unsqueeze(0)
        image2 = deepcopy(image1)
        # resize
        masks = masks[:9] + masks[-1:]
        masks = [self.mask_transform1(m).unsqueeze(0) for m in masks]

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
        masks = [affine_transform(m, affine1) for m in masks]
        masks = [affine_transform(m, affine_inverse1) for m in masks]
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
        masks2 = [affine_transform(m, affine2) for m in masks]
        masks = [affine_transform(m, affine_inverse2) for m in masks2]
        # convert truncated pixels to 0
        # source mask
        masks1 = [affine_transform(m, affine1) for m in masks]

        image1, image2 = image1.squeeze(0).data, image2.squeeze(0).data
        masks1 = [m.squeeze(0).data for m in masks1]
        masks2 = [m.squeeze(0).data for m in masks2]

        # resize
        masks1 = [self.mask_transform2(m) for m in masks1]
        masks2 = [self.mask_transform2(m) for m in masks2]
        # binarize
        masks1 = [(m > 0.01).float() for m in masks1]
        masks2 = [(m > 0.01).float() for m in masks2]
        # make tensor
        mask1 = torch.cat(masks1, dim=0)
        mask2 = torch.cat(masks2, dim=0)

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


def main():
    root_dir = "/home/tyler/work/data/GeekInt/data_dc"
    h = 512
    w = 768
    image_size = (w, h)

    train_dataset = MultiMaskAnimeDataset(root_dir, image_size, 32, 48)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    for i, batch in enumerate(train_loader):
        src_image = batch["image1"]
        tgt_image = batch["image2"]
        gt_src_mask = batch["mask1"]
        gt_tgt_mask = batch["mask2"]
        assert gt_src_mask.shape[1] == gt_tgt_mask.shape[1]
        print(src_image.shape, tgt_image.shape, gt_src_mask.shape, gt_tgt_mask.shape)
        print(torch.unique(gt_src_mask), torch.unique(gt_tgt_mask))

        src_image_np = tensor2image(src_image)
        tgt_image_np = tensor2image(tgt_image)
        display_image = np.concatenate([src_image_np, tgt_image_np], axis=1)

        print("input images")
        image_show(display_image)

        for j in range(0, min(gt_src_mask.shape[1], 3)):
            src_mask = gt_src_mask[:, -j - 1, ...]
            tgt_mask = gt_tgt_mask[:, -j - 1, ...]

            src_mask_np = tensor2image(src_mask, revert=False)
            tgt_mask_np = tensor2image(tgt_mask, revert=False)
            display_mask = np.concatenate([src_mask_np, tgt_mask_np], axis=1)

            print("gt mask", np.unique(src_mask_np), np.unique(tgt_mask_np))
            image_show(display_mask)
    return


if __name__ == "__main__":
    main()
