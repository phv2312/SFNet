import os
import glob
import random
import numpy as np
import cv2
import skimage.measure as measure
import skimage.feature
from math import copysign, log10
from PIL import Image
from natsort import natsorted


def get_moment_features(components, mask):
    features = np.zeros([mask.shape[0], mask.shape[1], 8])

    for component in components:
        image = component["image"]
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        moments = cv2.moments(image)
        moments = cv2.HuMoments(moments)[:, 0]
        for i in range(0, 7):
            if moments[i] == 0:
                continue
            moments[i] = -1 * copysign(1.0, moments[i]) * log10(abs(moments[i]))

        moments = np.append(moments, component["area"] / 200000.0)
        coords = np.nonzero(mask == component["label"])
        features[coords[0], coords[1], :] = moments

    features = np.transpose(features, (2, 0, 1))
    return features


def build_neighbor_graph(mask):
    max_level = mask.max() + 1
    matrix = skimage.feature.greycomatrix(mask, [1, 3], [0], levels=max_level)
    matrix = np.sum(matrix, axis=(2, 3))
    graph = np.zeros((max_level, max_level))

    for i in range(1, max_level):
        for j in range(1, max_level):
            if matrix[i, j] > 0:
                graph[i, j] = 1
                graph[j, i] = 1
    return graph


class ComponentWrapper:
    EXTRACT_COLOR = "extract_color"
    EXTRACT_SKETCH = "extract_sketch"
    EXTRACT_MULTI_MASK = "extract_multi_mask"

    def __init__(self, min_area=10, min_size=1):
        self.min_area = min_area
        self.min_size = min_size
        self.bad_values = [x + 300 * (x + 1) + 300 * 300 * (x + 1) for x in [0, 5, 10, 15, 255]]

    def extract_on_color_image(self, input_image):
        b, g, r = cv2.split(input_image)
        b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)

        index = 0
        components = {}
        mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int)

        # Pre-processing image
        processed_image = b + 300 * (g + 1) + 300 * 300 * (r + 1)
        # Get number of colors in image
        uniques = np.unique(processed_image)

        for unique in uniques:
            # Ignore sketch (ID of background is 255)
            if unique in self.bad_values:
                continue

            rows, cols = np.where(processed_image == unique)
            # Mask
            image_temp = np.zeros_like(processed_image)
            image_temp[rows, cols] = 255
            image_temp = np.array(image_temp, dtype=np.uint8)

            # Connected components
            labels = measure.label(image_temp, connectivity=1, background=0)
            regions = measure.regionprops(labels, intensity_image=processed_image)

            for region in regions:
                if region.area < self.min_area:
                    continue
                if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                    continue
                if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                    continue

                if unique == 23117055 and [0, 0] in region.coords:
                    continue

                components[index] = {
                    "centroid": np.array(region.centroid),
                    "area": region.area,
                    "image": region.image.astype(np.uint8) * 255,
                    "label": index + 1,
                    "coords": region.coords,
                    "bbox": region.bbox,
                    "min_intensity": region.min_intensity,
                    "mean_intensity": region.mean_intensity,
                    "max_intensity": region.max_intensity,
                }
                mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
                index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def extract_on_sketch_v3(self, sketch):
        binary = cv2.threshold(sketch, 100, 255, cv2.THRESH_BINARY)[1]
        labels = measure.label(binary, connectivity=1, background=0)
        regions = measure.regionprops(labels, intensity_image=sketch)

        index = 0
        mask = np.zeros((sketch.shape[0], sketch.shape[1]), dtype=np.int)
        components = dict()

        for region in regions[1:]:
            if region.area < self.min_area:
                continue
            if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                continue
            if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                continue

            components[index] = {
                "centroid": np.array(region.centroid),
                "area": region.area,
                "image": region.image.astype(np.uint8) * 255,
                "label": index + 1,
                "coords": region.coords,
                "bbox": region.bbox,
                "min_intensity": region.min_intensity,
                "mean_intensity": region.mean_intensity,
                "max_intensity": region.max_intensity,
            }
            mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
            index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def extract_multi_mask_by_color(self, input_image):
        b, g, r = cv2.split(input_image)
        b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)

        masks, components = [], []
        # Pre-processing image
        processed_image = b + 300 * (g + 1) + 300 * 300 * (r + 1)
        # Get number of colors in image
        uniques = np.unique(processed_image)
        foreground_mask = np.zeros_like(processed_image)

        for unique in uniques:
            # Ignore sketch (ID of background is 255)
            if unique in self.bad_values:
                continue

            r = unique // (300 * 300) - 1
            g = (unique % (300 * 300)) // 300 - 1
            b = unique % 300

            single_mask = (processed_image == unique).astype(np.uint8) * 255
            foreground_mask += single_mask

            area = np.count_nonzero(single_mask)
            if area < self.min_area:
                continue

            component = {"color": np.array([b, g, r]), "index": len(components)}
            masks.append(single_mask)
            components.append(component)

        masks.append(foreground_mask)
        components.append({"color": np.array([255, 255, 255]), "index": len(components)})
        return masks, components

    def process(self, input_image, sketch, method):
        assert len(cv2.split(input_image)) == 3, "Input image must be RGB, got binary"
        assert method in [self.EXTRACT_COLOR, self.EXTRACT_SKETCH, self.EXTRACT_MULTI_MASK]

        if method == self.EXTRACT_COLOR:
            mask, components = self.extract_on_color_image(input_image)
        elif method == self.EXTRACT_SKETCH:
            mask, components = self.extract_on_sketch_v3(sketch)
        else:
            mask, components = self.extract_multi_mask_by_color(input_image)
        return mask, components


def get_component_color(components, color_image, mode=ComponentWrapper.EXTRACT_SKETCH):
    if mode == ComponentWrapper.EXTRACT_COLOR:
        for component in components:
            index = len(component["coords"]) // 2
            coord = component["coords"][index]
            color = color_image[coord[0], coord[1]].tolist()
            component["color"] = color

    elif mode == ComponentWrapper.EXTRACT_SKETCH:
        for component in components:
            coords = component["coords"]
            points = color_image[coords[:, 0], coords[:, 1]]

            unique, counts = np.unique(points, return_counts=True, axis=0)
            max_index = np.argmax(counts)
            color = unique[max_index].tolist()
            component["color"] = color
    return


def rectify_mask(mask, component, ratio):
    coords = component["coords"]
    new_coords = np.array([[int(coord[0] * ratio[0]), int(coord[1] * ratio[1])] for coord in coords])
    new_coords = list(np.unique(new_coords, axis=0).tolist())

    count = 0
    mid_index = int(len(new_coords) / 2)
    new_area = {component["label"]: len(new_coords)}

    for i in range(0, mid_index + 1):
        offsets = [1] if i == 0 else [-1, 1]
        for j in offsets:
            index = mid_index + i * j
            if index >= len(new_coords):
                continue
            coord = new_coords[index]

            if mask[coord[0], coord[1]] == 0:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
                continue

            label = mask[coord[0], coord[1]]
            if label not in new_area:
                new_area[label] = np.count_nonzero(mask == label)

            if new_area[label] > new_area[component["label"]] * 5:
                mask[coord[0], coord[1]] = component["label"]
                count += 1
            elif new_area[label] > 1 and count == 0:
                mask[coord[0], coord[1]] = component["label"]
                count += 1

    return mask


def resize_mask(mask, components, size):
    new_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    ratio = (size[1] / mask.shape[0], size[0] / mask.shape[1])

    old_labels = np.unique(mask).tolist()
    new_labels = np.unique(new_mask).tolist()
    removed_labels = [i for i in old_labels if (i not in new_labels) and (i > 0)]

    for i in removed_labels:
        component = components[i - 1]
        new_mask = rectify_mask(new_mask, component, ratio)

    assert len(np.unique(mask)) == len(np.unique(new_mask))
    return new_mask


def match_multi_color_masks(masks_a, masks_b, components_a, components_b, k=None):
    matched_masks_a, matched_masks_b = [], []

    for component_a in components_a:
        color_a = component_a["color"]
        component_b = [b for b in components_b if np.all(b["color"] == color_a)]

        if len(component_b) == 0:
            continue
        else:
            component_b = component_b[0]

        matched_masks_a.append(masks_a[component_a["index"]])
        matched_masks_b.append(masks_b[component_b["index"]])

    if (k is None) or (len(matched_masks_a) <= k + 1):
        return matched_masks_a, matched_masks_b

    # k is the maximum number of paired masks
    all_indices = list(range(0, len(matched_masks_a) - 1))
    index = random.sample(all_indices, k=k) + [len(matched_masks_a) - 1]

    random_masks_a = [matched_masks_a[i] for i in index]
    random_masks_b = [matched_masks_b[i] for i in index]
    return random_masks_a, random_masks_b


def main():
    root_dir = "/home/tyler/work/data/GeekInt/real_data/test_data_for_interpolation_phase_1/boundary_data"
    output_dir = "/home/tyler/work/data/GeekInt/output/mask"

    character_dirs = natsorted(glob.glob(os.path.join(root_dir, "*")))
    component_wrapper = ComponentWrapper(min_area=400, min_size=5)

    for character_dir in character_dirs:
        character_name = os.path.basename(character_dir)
        paths = natsorted(glob.glob(os.path.join(character_dir, "*.png")))
        all_masks, all_components = [], []

        if character_name not in ["4"]:
            continue
        if not os.path.exists(os.path.join(output_dir, character_name)):
            os.mkdir(os.path.join(output_dir, character_name))

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            full_name = "%s_%s" % (character_name, name)
            print(full_name)

            if not os.path.exists(os.path.join(output_dir, full_name)):
                os.mkdir(os.path.join(output_dir, full_name))

            color_image = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            output_masks, output_components = component_wrapper.process(
                color_image, None, ComponentWrapper.EXTRACT_MULTI_MASK)

            all_masks.append(output_masks)
            all_components.append(output_components)

            for i, (output_mask, output_component) in enumerate(zip(output_masks, output_components)):
                output_mask = np.stack([output_mask] * 3, axis=-1) / 255
                component_color = output_component["color"]
                component_color = np.tile(component_color, [output_mask.shape[0], output_mask.shape[1], 1])
                color_mask = output_mask * component_color

                write_path = os.path.join(output_dir, full_name, "%03d.png" % i)
                cv2.imwrite(write_path, color_mask)

        masks_a, components_a = all_masks[0], all_components[0]
        masks_b, components_b = all_masks[1], all_components[1]
        masks_a, masks_b = match_multi_color_masks(masks_a, masks_b, components_a, components_b)

        for i, (mask_a, mask_b) in enumerate(zip(masks_a, masks_b)):
            merged_mask = np.concatenate([mask_a, mask_b], axis=1)
            write_path = os.path.join(output_dir, character_name, "%03d.png" % i)
            cv2.imwrite(write_path, merged_mask)
    return


if __name__ == "__main__":
    main()
