import gc
import os
import shutil
from typing import Dict
import cv2
import numpy as np
import torch
import glob
from PIL import Image

from sam3.visualization_utils import (
    visualize_formatted_frame_output,
    prepare_masks_for_visualization,
    plot_results,
)

from .utils import gen_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


def calculate_bbox_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1: [min_x, min_y, max_x, max_y] format
        bbox2: [min_x, min_y, max_x, max_y] format

    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate intersection area
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        inter_area = 0
    else:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    # Calculate IoU
    if union_area == 0:
        return 0
    return inter_area / union_area


def compute_mask_iou(mask1, mask2, device="cuda"):
    """
    PyTorch GPU版IOU计算，完全保留原逻辑：
    1. mask>0做二进制判断
    2. union全0时返回1.0
    3. 支持NumPy/PyTorch任意维度入参
    4. 自动GPU/CPU适配
    """
    if device == "cpu":
        intersection = np.logical_and(mask1 > 0, mask2 > 0)
        union = np.logical_or(mask1 > 0, mask2 > 0)
        sum_union = np.sum(union)
        if sum_union == 0:  # Both masks are all zeros, considered identical
            return 1.0
        iou = np.sum(intersection) / sum_union
        # diff_mask = np.logical_xor(mask1 > 0, mask2 > 0).astype(np.uint8)
        return iou
    else:
        # 步骤1：将NumPy入参转换为PyTorch张量，并移到GPU（如果是张量则直接复用）
        def to_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            return x.to(device) if x.device != device else x

        mask1 = to_torch(mask1)
        mask2 = to_torch(mask2)

        # 步骤2：完全复刻原逻辑，仅将np替换为torch
        intersection = torch.logical_and(mask1 > 0, mask2 > 0)
        union = torch.logical_or(mask1 > 0, mask2 > 0)
        sum_union = torch.sum(union).item()  # 张量转标量（CPU可计算）

        if sum_union == 0:  # 两个mask全0，返回1.0
            return 1.0

        iou = torch.sum(intersection).item() / sum_union
        return iou


def merge_masks(
    masks_dict, compare_masks_dict=None, iou_threshold=0.5, img_size=(1024, 1024)
):
    """
     Merge masks from current frame, skipping objects with high IoU in the comparison frame

    Parameters:
        masks_dict (dict): Masks from current frame {obj_id: {'mask': mask, 'prob': prob, 'box': box}}
        compare_masks_dict (dict): Masks from comparison frame {obj_id: {'mask': mask, 'prob': prob, 'box': box}} (optional)
        iou_threshold (float): IoU threshold, objects with IoU higher than this value will be skipped

    Returns:
        merged_mask (dict): Retained masks
    """
    merged_mask = {}

    # If there is no comparison frame, return masks_dict directly
    if compare_masks_dict is None:
        result = dict()
        for obj_id, item in masks_dict.items():
            mask = item.get("mask")
            result[obj_id] = mask

        return result

    # 获取masks_dict和compare_masks_dict中所有key，并保存在一个列表中，保留唯一值
    keys = list(set(masks_dict.keys()) | set(compare_masks_dict.keys()))

    # Iterate through each object in the current frame
    # for obj_id, mask_data in masks_dict.items():
    for obj_id in keys:
        mask_data = masks_dict.get(obj_id)
        compare_mask_data = compare_masks_dict.get(obj_id)

        if compare_mask_data is None:
            # If there's no corresponding mask in comparison frame, include this mask
            merged_mask[obj_id] = mask_data
            continue
        if mask_data is None:
            # If there's no corresponding mask in current frame, include this mask
            merged_mask[obj_id] = compare_mask_data
            continue

        mask = mask_data["mask"]
        box = mask_data["box"] * img_size[0]

        # Convert mask to binary image with non-zero elements as 1 and zero elements as 0
        mask_binary = (mask > 0).astype(np.uint8)

        compare_mask = compare_mask_data["mask"]
        compare_box = compare_mask_data["box"] * img_size[0]
        compare_binary = (compare_mask > 0).astype(np.uint8)

        # Extract bounding boxes to determine the region of interest
        # Box format is assumed to be [x, y, w, h] (xywh format)
        # Crop both masks to the combined bounding box region for efficient IoU calculation
        if box is not None and compare_box is not None:
            # Extract coordinates from boxes (x, y, w, h format)
            x1, y1, w1, h1 = map(int, box)
            x2, y2, w2, h2 = map(int, compare_box)

            # Crop masks using their respective boxes
            cropped_mask1 = mask_binary[y1 : y1 + h1, x1 : x1 + w1]
            cropped_mask2 = compare_binary[y2 : y2 + h2, x2 : x2 + w2]

            # Pad masks to the same size, ensuring symmetric padding
            # Determine the maximum width and height
            max_h = max(cropped_mask1.shape[0], cropped_mask2.shape[0])
            max_w = max(cropped_mask1.shape[1], cropped_mask2.shape[1])

            # Pad cropped_mask1
            pad_h1 = max_h - cropped_mask1.shape[0]
            pad_w1 = max_w - cropped_mask1.shape[1]

            pad_top1 = pad_h1 // 2
            pad_bottom1 = pad_h1 - pad_top1
            pad_left1 = pad_w1 // 2
            pad_right1 = pad_w1 - pad_left1

            padded_mask1 = np.pad(
                cropped_mask1,
                ((pad_top1, pad_bottom1), (pad_left1, pad_right1)),
                mode="constant",
                constant_values=0,
            )

            # Pad cropped_mask2
            pad_h2 = max_h - cropped_mask2.shape[0]
            pad_w2 = max_w - cropped_mask2.shape[1]

            pad_top2 = pad_h2 // 2
            pad_bottom2 = pad_h2 - pad_top2
            pad_left2 = pad_w2 // 2
            pad_right2 = pad_w2 - pad_left2

            padded_mask2 = np.pad(
                cropped_mask2,
                ((pad_top2, pad_bottom2), (pad_left2, pad_right2)),
                mode="constant",
                constant_values=0,
            )

            # Calculate IoU on the padded masks
            iou = compute_mask_iou(padded_mask1, padded_mask2)
        else:
            # If box information is not available, calculate IoU on full masks
            iou = compute_mask_iou(mask_binary, compare_binary)

        # If IoU is less than or equal to threshold, keep the mask
        if iou <= iou_threshold:
            # Only merge objects with low IoU
            # 是否应该相加？ TODO
            # 合并两个mask和它们的边界框
            # 合并mask：将两个mask叠加
            merged_mask_array = np.maximum(mask, compare_mask)

            # 合并边界框：取两个边界框的外接矩形
            if box is not None and compare_box is not None:
                # 将 xywh 格式转换为 xyxy 格式进行合并
                x1, y1, w1, h1 = box
                x2, y2, w2, h2 = compare_box
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)

                # 转换回 xywh 格式
                merged_box = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                # 如果其中一个没有边界框，则使用存在的那个或为None
                merged_box = box if box is not None else compare_box

            # 创建合并后的mask数据
            merged_mask_data = {
                "mask": merged_mask_array,
                "box": merged_box,
                # 保留其他可能的字段
                **{k: v for k, v in mask_data.items() if k not in ["mask", "box"]},
            }
            merged_mask[obj_id] = merged_mask_data

            # merged_mask[obj_id] = mask_data

    return merged_mask


def compare_masks(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
    merged_mask = {}

    if compare_masks_dict is None:
        return dict()

    keys = list(set(masks_dict.keys()) | set(compare_masks_dict.keys()))

    for obj_id in keys:
        mask_data = masks_dict.get(obj_id)
        compare_mask_data = compare_masks_dict.get(obj_id)

        if compare_mask_data is None or mask_data is None:
            continue

        mask = mask_data["mask"]
        box = mask_data["box"]

        # Convert mask to binary image with non-zero elements as 1 and zero elements as 0
        mask_binary = (mask > 0).astype(np.uint8)

        compare_mask = compare_mask_data["mask"]
        compare_box = compare_mask_data["box"]
        compare_binary = (compare_mask > 0).astype(np.uint8)

        # Extract bounding boxes to determine the region of interest
        # Box format is assumed to be [x, y, w, h] (xywh format)
        # Crop both masks to the combined bounding box region for efficient IoU calculation
        if box is not None and compare_box is not None:
            # Extract coordinates from boxes (x, y, w, h format)
            x1, y1, w1, h1 = map(int, box)
            x2, y2, w2, h2 = map(int, compare_box)

            # Crop masks using their respective boxes
            cropped_mask1 = mask_binary[y1 : y1 + h1, x1 : x1 + w1]
            cropped_mask2 = compare_binary[y2 : y2 + h2, x2 : x2 + w2]

            # Pad masks to the same size, ensuring symmetric padding
            # Determine the maximum width and height
            max_h = max(cropped_mask1.shape[0], cropped_mask2.shape[0])
            max_w = max(cropped_mask1.shape[1], cropped_mask2.shape[1])

            # Pad cropped_mask1
            pad_h1 = max_h - cropped_mask1.shape[0]
            pad_w1 = max_w - cropped_mask1.shape[1]

            pad_top1 = pad_h1 // 2
            pad_bottom1 = pad_h1 - pad_top1
            pad_left1 = pad_w1 // 2
            pad_right1 = pad_w1 - pad_left1

            padded_mask1 = np.pad(
                cropped_mask1,
                ((pad_top1, pad_bottom1), (pad_left1, pad_right1)),
                mode="constant",
                constant_values=0,
            )

            # Pad cropped_mask2
            pad_h2 = max_h - cropped_mask2.shape[0]
            pad_w2 = max_w - cropped_mask2.shape[1]

            pad_top2 = pad_h2 // 2
            pad_bottom2 = pad_h2 - pad_top2
            pad_left2 = pad_w2 // 2
            pad_right2 = pad_w2 - pad_left2

            padded_mask2 = np.pad(
                cropped_mask2,
                ((pad_top2, pad_bottom2), (pad_left2, pad_right2)),
                mode="constant",
                constant_values=0,
            )

            # Calculate IoU on the padded masks
            iou = compute_mask_iou(padded_mask1, padded_mask2)
        else:
            # If box information is not available, calculate IoU on full masks
            iou = compute_mask_iou(mask_binary, compare_binary)

        if iou > iou_threshold:
            # Only merge objects with low IoU
            merged_mask[obj_id] = mask_data

    return merged_mask


def compute_mask_iou_batch(masks1, masks2):
    """
    Compute the IoU matrix between two sets of masks.

    This function calculates the Intersection over Union (IoU) between each pair of masks
    from two batches of masks. IoU is a common metric for evaluating the similarity between
    masks, particularly in computer vision tasks.

    Args:
        masks1: First set of masks with shape (num_masks1, H, W) where num_masks1 is the
                number of masks in the first set and H, W are the height and width of each mask
        masks2: Second set of masks with shape (num_masks2, H, W) where num_masks2 is the
                number of masks in the second set and H, W are the height and width of each mask

    Returns:
        numpy.ndarray: IoU matrix of shape (num_masks1, num_masks2) where each element (i, j)
                      represents the IoU between the i-th mask in masks1 and j-th mask in masks2
    """
    # Handle edge cases for empty inputs
    if isinstance(masks1, np.ndarray) and masks1.size == 0:
        return np.zeros((0, len(masks2)))
    if isinstance(masks2, np.ndarray) and masks2.size == 0:
        return np.zeros((len(masks1), 0))
    if not isinstance(masks1, np.ndarray) and not masks1:
        return np.zeros((0, len(masks2)))
    if not isinstance(masks2, np.ndarray) and not masks2:
        return np.zeros((len(masks1), 0))

    # Flatten masks to binary vectors for efficient computation
    masks1 = masks1.astype(bool).reshape(len(masks1), -1)  # (N1, H*W)
    masks2 = masks2.astype(bool).reshape(len(masks2), -1)  # (N2, H*W)

    # Compute intersection using matrix multiplication
    intersection = masks1 @ masks2.T  # (N1, N2)

    # Compute union using inclusion-exclusion principle
    union = (
        np.sum(masks1, axis=1)[:, None] + np.sum(masks2, axis=1)[None, :] - intersection
    )

    # Calculate IoU avoiding division by zero
    iou = intersection / union
    return iou


def sum_masks_dict(masks_A, masks_B=None, iou_threshold=0.5):
    """
    Merge masks from two dictionaries, removing highly overlapping masks and performing logical OR operation

    This function processes two mask dictionaries, computes their IoU, removes highly overlapping masks,
    and returns a merged mask. When two mask dictionaries are provided, it compares their similarity,
    removes duplicate masks with IoU above the threshold, and merges the remaining masks.

    Args:
        masks_A (dict): First mask dictionary with object IDs as keys and corresponding mask arrays as values
        masks_B (dict, optional): Second mask dictionary with object IDs as keys and corresponding mask arrays as values, defaults to None
        iou_threshold (float): IoU threshold for determining mask duplicates, defaults to 0.5

    Returns:
        numpy.ndarray: Merged mask array with uint8 data type, same shape as input masks
    """
    # Handle empty inputs
    if len(masks_A) == 0 and (masks_B is None or len(masks_B) == 0):
        # Get reference shape (if unable to get, raise exception or specify default shape)
        try:
            ref_shape = next(iter(masks_A.values())).shape
        except StopIteration:
            ref_shape = (1, 1024, 1024)  # Default shape
        return np.zeros(ref_shape, dtype=np.uint8)

    try:
        merged_mask = np.zeros_like(next(iter(masks_A.values())), dtype=np.uint8)
    except StopIteration:
        ref_shape = (1, 1024, 1024)  # Default shape
        merged_mask = np.zeros(ref_shape, dtype=np.uint8)

    # No masks to compare, return merged mask directly
    if masks_B is None:
        for mask in masks_A.values():
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
        return merged_mask

    # Convert masks_A and masks_B to NumPy arrays
    mask_array_A = np.array([m > 0 for m in masks_A.values()])
    mask_array_B = np.array([m > 0 for m in masks_B.values()])

    # Compute IoU for all mask pairs
    iou_matrix = compute_mask_iou_batch(mask_array_A, mask_array_B)

    # Find keys that need to be removed
    keys_to_remove = {"A": [], "B": []}
    for idx_A, obj_id_A in enumerate(masks_A.keys()):
        for idx_B, obj_id_B in enumerate(masks_B.keys()):
            if iou_matrix[idx_A, idx_B] > iou_threshold:
                if obj_id_A not in keys_to_remove["A"]:
                    keys_to_remove["A"].append(obj_id_A)
                if obj_id_B not in keys_to_remove["B"]:
                    keys_to_remove["B"].append(obj_id_B)

    # Merge masks from the first dictionary that are not marked as duplicates
    for obj_id, mask in masks_A.items():
        if obj_id not in keys_to_remove["A"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    # Merge masks from the second dictionary that are not marked as duplicates
    for obj_id, mask in masks_B.items():
        if obj_id not in keys_to_remove["B"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    return merged_mask


class BiSAM2:
    def __init__(
        self,
        predictor=None,
        mid_frame=0,
        diff_frame_num=1,
        iou_threshold=0.5,
        max_objects_per_batch=50,
    ):
        self.predictor = predictor
        self.iou_threshold = iou_threshold
        self.mid_frame = mid_frame
        self.diff_frame_num = diff_frame_num

    def renew_session(self, video_path):
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        # note: in case you already ran one text prompt and now want to switch to another text prompt
        # it's required to reset the session first (otherwise the results would be wrong)
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        return session_id

    def generate_point_prompts_from_masks(
        self,
        diff_prompt_out: Dict,
        pos_points_per_object: int = 2,
        neg_points_per_object: int = 2,
    ) -> Dict:
        """
        根据diff_prompt_out中的mask和box信息，为每个目标生成正向和负向点提示

        Args:
            diff_prompt_out: 包含检测结果的字典，格式为：
                            {prompt_text: {"ids": [...], "masks": [...], "probs": [...], "boxes": [...]}}
            pos_points_per_object: 每个目标需要生成的正向点数量
            neg_points_per_object: 每个目标需要生成的负向点数量

        Returns:
            包含点提示信息的字典，格式为：
            {prompt_text: {obj_id: {"positive_points": [(x,y), ...], "negative_points": [(x,y), ...]}}}
        """
        point_prompts = {}

        for prompt_text, results in diff_prompt_out.items():
            prompt_point_prompts = {}

            masks = results["masks"]  # [num_objects, H, W] or [num_objects, 1, H, W]
            boxes = results["boxes"]  # [num_objects, 4] in xywh format
            obj_ids = results["ids"]  # [num_objects]

            # 确保masks是正确的形状 [num_objects, H, W]
            if len(masks.shape) == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)  # [num_objects, H, W]

            for i, obj_id in enumerate(obj_ids):
                mask = masks[i]  # [H, W]
                box = boxes[i]  # [4] in xywh format

                # 将box转换为整数坐标 [x, y, w, h]
                W, H = mask.shape
                x, y, w, h = box
                x = int(x * W)
                y = int(y * H)
                w = int(w * W)
                h = int(h * H)

                # 创建仅在当前bbox范围内的mask
                bbox_mask = mask[y : y + h, x : x + w]

                # 计算bbox的实际尺寸
                bbox_h, bbox_w = bbox_mask.shape

                # 生成正向点 (在mask上)
                positive_points = []
                if pos_points_per_object > 0:
                    # 找到mask中为True的所有像素坐标
                    pos_pixel_coords = np.column_stack(
                        np.where(bbox_mask > 0)
                    )  # [row, col] format

                    if len(pos_pixel_coords) > 0:
                        # 随机选择指定数量的正向点
                        if len(pos_pixel_coords) >= pos_points_per_object:
                            selected_indices = np.random.choice(
                                len(pos_pixel_coords),
                                size=pos_points_per_object,
                                replace=False,
                            )
                        else:
                            # 如果可用点少于所需点数，允许重复选择
                            selected_indices = np.random.choice(
                                len(pos_pixel_coords),
                                size=pos_points_per_object,
                                replace=True,
                            )

                        selected_pixels = pos_pixel_coords[selected_indices]

                        # 转换为图像坐标系 (x, y)
                        for px, py in selected_pixels:
                            positive_points.append(
                                (px + x, py + y)
                            )  # 加上bbox左上角偏移
                    else:
                        # 如果没有找到任何正向点，则随机选择bbox内的点
                        for _ in range(pos_points_per_object):
                            px = np.random.randint(x, x + w)
                            py = np.random.randint(y, y + h)
                            positive_points.append((px, py))

                # 生成负向点 (在mask外但在bbox内)
                negative_points = []
                if neg_points_per_object > 0:
                    # 获取bbox内的所有像素坐标
                    bbox_coords = []
                    for ry in range(bbox_h):
                        for rx in range(bbox_w):
                            if bbox_mask[ry, rx] == 0:  # 只包括mask为False的点
                                bbox_coords.append((ry, rx))

                    if len(bbox_coords) > 0:
                        # 随机选择指定数量的负向点
                        if len(bbox_coords) >= neg_points_per_object:
                            selected_indices = np.random.choice(
                                len(bbox_coords),
                                size=neg_points_per_object,
                                replace=False,
                            )
                        else:
                            # 如果可用点少于所需点数，允许重复选择
                            selected_indices = np.random.choice(
                                len(bbox_coords),
                                size=neg_points_per_object,
                                replace=True,
                            )

                        selected_coords = [bbox_coords[idx] for idx in selected_indices]

                        # 转换为图像坐标系 (x, y)
                        for ry, rx in selected_coords:
                            negative_points.append(
                                (rx + x, ry + y)
                            )  # 注意这里是(rx, ry)而不是(ry, rx)
                    else:
                        # 如果没有找到任何负向点，则随机选择bbox内的点
                        for _ in range(neg_points_per_object):
                            px = np.random.randint(x, x + w)
                            py = np.random.randint(y, y + h)
                            negative_points.append((px, py))

                # 存储结果
                prompt_point_prompts[obj_id] = {
                    "positive_points": positive_points,
                    "negative_points": negative_points,
                }

            point_prompts[prompt_text] = prompt_point_prompts

        return point_prompts

    def remove_overlapping_detections(
        self, all_prompt_results, iou_threshold=0.9, max_prompt=None
    ):
        """
        Remove overlapping detections across different prompts, keeping the ones with highest confidence
        Process sequentially: take first target as reference, find all overlapping targets with IoU > threshold,
        keep the one with highest confidence among overlapping ones, remove them from the list, and continue

        Args:
            all_prompt_results: Dictionary containing detection results from all prompts
            iou_threshold: Threshold for considering two detections as overlapping
            max_prompt: The prompt with the most detections (will be prioritized)

        Returns:
            final_results: Dictionary containing de-duplicated detection results
        """
        # 如果没有指定max_prompt，则找出检测结果最多的prompt
        if max_prompt is None:
            obj_nums = {}
            for prompt_text, result in all_prompt_results.items():
                ids = result["ids"]
                obj_nums[prompt_text] = ids.size if ids is not None else 0
            max_prompt = max(obj_nums, key=obj_nums.get) if obj_nums else None

        # Flatten all detection results with their prompt source and confidence
        all_detections = []

        for prompt_text, result in all_prompt_results.items():
            ids = result["ids"]
            masks = result["masks"]
            probs = result["probs"]
            boxes = result["boxes"]

            if ids is not None:
                for i in range(len(ids)):
                    detection = {
                        "prompt": prompt_text,
                        "id": ids[i],
                        "mask": (
                            masks[i] if masks is not None and i < len(masks) else None
                        ),
                        "prob": probs[i] if probs is not None and i < len(probs) else 0,
                        "box": (
                            boxes[i] if boxes is not None and i < len(boxes) else None
                        ),
                        "original_idx": i,
                    }

                    # Extract bounding box coordinates if available
                    if detection["box"] is not None:
                        x, y, w, h = detection["box"]
                        detection["bbox"] = [
                            x,
                            y,
                            x + w,
                            y + h,
                        ]  # Convert to [min_x, min_y, max_x, max_y] format
                    else:
                        # If no bounding box available, try to extract from mask
                        if detection["mask"] is not None:
                            # Find bounding box from mask
                            mask = detection["mask"]
                            pos = np.where(mask)
                            if len(pos[0]) > 0:
                                ymin, ymax = pos[0].min(), pos[0].max()
                                xmin, xmax = pos[1].min(), pos[1].max()
                                detection["bbox"] = [xmin, ymin, xmax, ymax]
                            else:
                                detection["bbox"] = [0, 0, 1, 1]  # Default small bbox
                        else:
                            detection["bbox"] = [0, 0, 1, 1]  # Default small bbox

                    all_detections.append(detection)

        # 依次处理每个目标，找到与之重叠的目标，保留置信度最高的
        remaining_detections = all_detections[:]
        final_detections = []

        while len(remaining_detections) > 0:
            # 取第一个目标作为参考（目标0）
            reference_det = remaining_detections[0]

            # 找到所有与参考目标重叠度高的目标
            overlapping_dets = [reference_det]  # 包含参考目标本身
            non_overlapping_dets = []

            for det in remaining_detections[1:]:  # 跳过参考目标本身
                iou = calculate_bbox_iou(reference_det["bbox"], det["bbox"])
                if iou > iou_threshold:
                    overlapping_dets.append(det)  # 添加到重叠目标列表
                else:
                    non_overlapping_dets.append(det)  # 添加到非重叠目标列表

            # 在重叠目标中优先选择来自max_prompt的目标，如果没有则选择置信度最高的
            max_prompt_dets = [
                det for det in overlapping_dets if det["prompt"] == max_prompt
            ]
            if max_prompt_dets:
                # 如果有来自max_prompt的目标，则在这些目标中选择置信度最高的
                highest_conf_det = max(max_prompt_dets, key=lambda x: x["prob"])
            else:
                # 如果没有来自max_prompt的目标，则在所有重叠目标中选择置信度最高的
                highest_conf_det = max(overlapping_dets, key=lambda x: x["prob"])

            final_detections.append(
                highest_conf_det
            )  # 将置信度最高的目标添加到最终结果

            # 从剩余列表中移除所有重叠的目标
            remaining_detections = non_overlapping_dets

        return final_detections

    def propagate_in_video(self, session_id, propagation_direction="both"):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    def step_one(
        self,
        img_paths: list,
        prompt_text_str=None,
    ):
        diff_mask_list = []
        ##### module 1 #####
        for sort in ["asc", "desc"]:
            ##### module 2 #####
            video_path = gen_frame(
                img_paths,
                sort=sort,
                mid_frame=self.mid_frame,
            )

            # load "video_frames_for_vis" for visualization purposes (they are not used by the model)
            if isinstance(video_path, str) and video_path.endswith(".mp4"):
                cap = cv2.VideoCapture(video_path)
                video_frames_for_vis = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
                try:
                    # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
                    video_frames_for_vis.sort(
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                    )
                except ValueError:
                    # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
                    print(
                        f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                        f"falling back to lexicographic sort."
                    )
                    video_frames_for_vis.sort()

            session_id = self.renew_session(video_path)

            ##### module 3 #####
            # 检测prompt类型
            if prompt_text_str is None:
                print("请输入prompt")
                return
            if isinstance(prompt_text_str, str):
                prompt_text_str = [prompt_text_str]
            else:
                prompt_text_str = prompt_text_str

            frame_idx = 0  # add a text prompt on frame 0
            if len(prompt_text_str) > 1:
                all_prompt_results = {}  # 存储所有提示词的结果
                obj_nums = dict()
                for prompt_text in prompt_text_str:
                    diff_prompt_out = dict()
                    response = self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=frame_idx,
                            text=prompt_text,
                        )
                    )
                    out = response["outputs"]
                    diff_prompt_out[prompt_text] = {
                        "ids": out.get("out_obj_ids"),
                        "masks": out.get("out_binary_masks"),
                        "probs": out.get("out_probs"),
                        "boxes": out.get("out_boxes_xywh"),
                    }
                    all_prompt_results.update(diff_prompt_out)
                    obj_nums[prompt_text] = out["out_obj_ids"].size

                point_prompts = self.generate_point_prompts_from_masks(
                    all_prompt_results
                )
                # 找出目标最多的提示词
                max_prompt = max(obj_nums, key=obj_nums.get)
                # 去除不同提示词检测结果中的重复框，保留置信度最高的
                all_prompt_results = self.remove_overlapping_detections(
                    all_prompt_results, max_prompt=max_prompt
                )
                # 重新设置predictor
                _ = self.predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=session_id,
                    )
                )
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        text=max_prompt,
                    )
                )
                out = response["outputs"]
                # 先运行一次
                self.propagate_in_video(session_id, "both")
                # 添加不重复的提示框
                new_obj_idx = out["out_obj_ids"].size
                for prompt_result in all_prompt_results:
                    if prompt_result["prompt"] == max_prompt:
                        continue

                    box = prompt_result["box"]
                    positive_points = point_prompts[prompt_result["prompt"]][
                        prompt_result["id"]
                    ]["positive_points"]
                    negative_points = point_prompts[prompt_result["prompt"]][
                        prompt_result["id"]
                    ]["negative_points"]

                    frame_idx = 0
                    boxes_abs = np.array(
                        [
                            box,  # positive prompt box
                        ]
                    )
                    points_abs = np.array(
                        [
                            positive_points[0],
                            positive_points[1],
                            negative_points[0],
                            negative_points[1],
                        ]
                    )
                    # positive prompt boxes have label 1, while negative prompt boxes have label 0
                    labels = np.array([1, 1, 0, 0])
                    # convert points and labels to tensors; also convert to relative coordinates
                    points_tensor = torch.tensor(
                        abs_to_rel_coords(points_abs, 1024, 1024, coord_type="point"),
                        dtype=torch.float32,
                    )
                    points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

                    # boxes_tensor = torch.tensor(
                    #     boxes_abs,
                    #     dtype=torch.float32,
                    # )
                    # boxes_labels_tensor = torch.tensor(labels, dtype=torch.int32)

                    response = self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=frame_idx,
                            # bounding_boxes=boxes_tensor,
                            # bounding_box_labels=boxes_labels_tensor,
                            points=points_tensor,
                            point_labels=points_labels_tensor,
                            obj_id=new_obj_idx,
                        )
                    )
                    new_obj_idx += 1

                out = response["outputs"]
            else:
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        text=prompt_text_str[0],
                    )
                )
                out = response["outputs"]

            ##### module 4 #####
            obj_size = out["out_obj_ids"].size

            before_masks, after_masks, first_masks = {}, {}, {}

            outputs_per_frame = self.propagate_in_video(session_id, "both")
            frame_len = len(outputs_per_frame)

            before_frame = outputs_per_frame[
                0 if self.diff_frame_num == 1 else frame_len - 2
            ]
            after_frame = outputs_per_frame[frame_len - 1]

            # 帧追踪结果可视化
            # for frame_idx, output in outputs_per_frame.items():
            #     visualize_formatted_frame_output(
            #         frame_idx,
            #         video_frames_for_vis,
            #         outputs_list=[prepare_masks_for_visualization({frame_idx: output})],
            #         titles=[f"SAM 3 Dense Tracking outputs: {sort}"],
            #         figsize=(10, 8),
            #     )

            # 旧的对象mask变化检测
            # b_ids, b_masks = before_frame.get("out_obj_ids"), before_frame.get(
            #     "out_binary_masks"
            # )
            # a_ids, a_masks = after_frame.get("out_obj_ids"), after_frame.get(
            #     "out_binary_masks"
            # )

            # for id, mask in zip(b_ids, b_masks):
            #     before_masks[id] = mask

            # for id, mask in zip(a_ids, a_masks):
            #     after_masks[id] = mask

            # 新的对象mask变化检测
            first_frame = outputs_per_frame[0]
            f_ids, f_masks, f_probs, f_boxes = (
                first_frame.get("out_obj_ids"),
                first_frame.get("out_binary_masks"),
                first_frame.get("out_probs"),
                first_frame.get("out_boxes_xywh"),
            )
            b_ids, b_masks, b_probs, b_boxes = (
                before_frame.get("out_obj_ids"),
                before_frame.get("out_binary_masks"),
                before_frame.get("out_probs"),
                before_frame.get("out_boxes_xywh"),
            )
            a_ids, a_masks, a_probs, a_boxes = (
                after_frame.get("out_obj_ids"),
                after_frame.get("out_binary_masks"),
                after_frame.get("out_probs"),
                after_frame.get("out_boxes_xywh"),
            )

            for id, mask, prob, box in zip(f_ids, f_masks, f_probs, f_boxes):
                first_masks[id] = dict(
                    mask=mask,
                    prob=prob,
                    box=box,
                )

            for id, mask, prob, box in zip(b_ids, b_masks, b_probs, b_boxes):
                before_masks[id] = dict(
                    mask=mask,
                    prob=prob,
                    box=box,
                )

            for id, mask, prob, box in zip(a_ids, a_masks, a_probs, a_boxes):
                after_masks[id] = dict(
                    mask=mask,
                    prob=prob,
                    box=box,
                )

            # compare the first and last frames to get the difference
            if self.mid_frame > 100:
                diff_mask_1 = merge_masks(
                    first_masks,
                    before_masks,
                    iou_threshold=self.iou_threshold,
                )

                diff_mask_2 = merge_masks(
                    first_masks,
                    after_masks,
                    iou_threshold=self.iou_threshold,
                )

                diff_mask = compare_masks(
                    diff_mask_1,
                    diff_mask_2,
                    iou_threshold=self.iou_threshold,
                )

                # for id, item in _diff_mask.items():
                #     diff_mask[id] = item.get("mask")
            else:
                diff_mask = merge_masks(
                    first_masks,
                    after_masks,
                    iou_threshold=self.iou_threshold,
                )

            diff_mask_list.append(diff_mask)

            _ = self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

        # after all inference is done, we can shutdown the predictor
        # to free up the multi-GPU process group
        # predictor.shutdown()
        torch.cuda.empty_cache()
        gc.collect()

        return diff_mask_list


from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt


class Baseline:
    def __init__(
        self,
        model=None,
        confidence_threshold: float = 0.25,
    ):
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    def step_one(
        self,
        img_paths: list,
        prompt_text_str=None,
    ):

        # 检测prompt类型
        if prompt_text_str is None:
            print("请输入prompt")
            return
        if isinstance(prompt_text_str, str):
            prompt_text_str = [prompt_text_str]
        else:
            prompt_text_str = prompt_text_str

        diff_mask_list = []
        for img_path in img_paths:
            image = Image.open(img_path)
            inference_state = self.processor.set_image(image)
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(
                state=inference_state, prompt=prompt_text_str[0]
            )

            masks = inference_state["masks"]

            if masks is None or len(masks) == 0:
                h, w = (
                    inference_state["original_height"],
                    inference_state["original_width"],
                )
                diff_mask_list.append(np.zeros((h, w), dtype=np.uint8))
                continue

            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()

            if len(masks.shape) == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            merged_mask = np.any(masks, axis=0).astype(np.uint8)

            diff_mask_list.append(merged_mask)

            # plt.figure(figsize=(10, 10))  # set the figure size
            # plt.subplot(1, 1, 1)
            # plt.imshow(merged_mask)
            # plt.title("T1")
            # plt.axis("off")

            # # show the plot
            # plt.tight_layout()
            # plt.show()

        if len(diff_mask_list) >= 2:
            mask1 = diff_mask_list[0]
            mask2 = diff_mask_list[1]

            # 如果是 Tensor
            if isinstance(mask1, torch.Tensor):
                diff = torch.abs(mask1.float() - mask2.float())
            # 如果是 NumPy 数组
            else:
                diff = np.abs(mask1.astype(np.float32) - mask2.astype(np.float32))

            # plt.figure(figsize=(10, 10))  # set the figure size
            # plt.subplot(1, 1, 1)
            # plt.imshow(diff)
            # plt.title("T1")
            # plt.axis("off")

            # # show the plot
            # plt.tight_layout()
            # plt.show()

            return diff
        else:
            return diff_mask_list[0]


class Baseline_Bi:
    def __init__(
        self,
        predictor=None,
        mid_frame=0,
        diff_frame_num=1,
        iou_threshold=0.5,
    ):
        self.predictor = predictor
        self.iou_threshold = iou_threshold
        self.mid_frame = mid_frame
        self.diff_frame_num = diff_frame_num

    def renew_session(self, video_path):
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        # note: in case you already ran one text prompt and now want to switch to another text prompt
        # it's required to reset the session first (otherwise the results would be wrong)
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        return session_id

    def propagate_in_video(self, session_id, propagation_direction="both"):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    def step_one(
        self,
        img_paths: list,
        prompt_text_str=None,
    ):
        diff_mask_list = []
        ##### module 1 #####
        for sort in ["asc", "desc"]:
            ##### module 2 #####
            video_path = gen_frame(
                img_paths,
                sort=sort,
                mid_frame=self.mid_frame,
            )

            if isinstance(video_path, str) and video_path.endswith(".mp4"):
                cap = cv2.VideoCapture(video_path)
                video_frames_for_vis = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
                try:
                    # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
                    video_frames_for_vis.sort(
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                    )
                except ValueError:
                    # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
                    print(
                        f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                        f"falling back to lexicographic sort."
                    )
                    video_frames_for_vis.sort()

            session_id = self.renew_session(video_path)

            if prompt_text_str is None:
                print("请输入prompt")
                return
            if isinstance(prompt_text_str, str):
                prompt_text_str = [prompt_text_str]
            else:
                prompt_text_str = prompt_text_str

            frame_idx = 0  # add a text prompt on frame 0
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt_text_str[0],
                )
            )
            # out = response["outputs"]

            outputs_per_frame = self.propagate_in_video(session_id, "both")
            frame_len = len(outputs_per_frame)

            before_frame = outputs_per_frame[
                0 if self.diff_frame_num == 1 else frame_len - 2
            ]
            after_frame = outputs_per_frame[frame_len - 1]

            before_masks = before_frame.get("out_binary_masks")
            after_masks = after_frame.get("out_binary_masks")

            ##### module n #####
            if isinstance(before_masks, torch.Tensor):
                if len(before_masks.shape) == 4 and before_masks.shape[1] == 1:
                    before_masks = before_masks.squeeze(1)  # [num_objects, H, W]
                    after_masks = after_masks.squeeze(1)

                before_mask = torch.any(before_masks, dim=0).int()  # [H, W]
                after_mask = torch.any(after_masks, dim=0).int()

                diff = torch.abs(before_mask.float() - after_mask.float())
            elif isinstance(before_masks, np.ndarray):
                if len(before_masks.shape) == 4 and before_masks.shape[1] == 1:
                    before_masks = before_masks.squeeze(1)
                    after_masks = after_masks.squeeze(1)

                before_mask = np.any(before_masks, axis=0).astype(np.uint8)  # [H, W]
                after_mask = np.any(after_masks, axis=0).astype(np.uint8)

                diff = np.abs(
                    before_mask.astype(np.float32) - after_mask.astype(np.float32)
                )

            # plt.figure(figsize=(10, 10))  # set the figure size
            # plt.subplot(1, 1, 1)
            # plt.imshow(diff)
            # plt.title("T1")
            # plt.axis("off")

            # # show the plot
            # plt.tight_layout()
            # plt.show()

            diff_mask_list.append(diff)

        _ = self.predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

        if len(diff_mask_list) >= 2:
            mask1 = diff_mask_list[0]
            mask2 = diff_mask_list[1]

            if isinstance(mask1, torch.Tensor):
                combined = mask1.float() + mask2.float()
                diff = (combined >= 2).float()
            else:
                combined = mask1.astype(np.float32) + mask2.astype(np.float32)
                diff = (combined >= 2).astype(np.float32)

            torch.cuda.empty_cache()
            gc.collect()
            return diff
        else:
            torch.cuda.empty_cache()
            gc.collect()
            return diff_mask_list[0]


def merge_masks_v1(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
    """
    Merge masks from current frame, skipping objects with high IoU in the comparison frame

    Parameters:
        masks_dict (dict): Masks from current frame {obj_id: mask}
        compare_masks_dict (dict): Masks from comparison frame {obj_id: mask} (optional)
        iou_threshold (float): IoU threshold, objects with IoU higher than this value will be skipped

    Returns:
        merged_mask (dict): Retained masks
    """
    merged_mask = {}

    # If there is no comparison frame, return masks_dict directly
    if compare_masks_dict is None:
        return masks_dict

    keys = list(set(masks_dict.keys()) | set(compare_masks_dict.keys()))

    for obj_id in keys:
        mask_data = masks_dict.get(obj_id)
        compare_mask_data = compare_masks_dict.get(obj_id)

        if compare_mask_data is None:
            # If there's no corresponding mask in comparison frame, include this mask
            merged_mask[obj_id] = mask_data
            continue
        if mask_data is None:
            # If there's no corresponding mask in current frame, include this mask
            merged_mask[obj_id] = compare_mask_data
            continue

        mask_binary = mask = mask_data["mask"]
        compare_binary = compare_mask_data["mask"]

        # Calculate IoU (ignoring cases where masks are all zeros)
        if np.any(compare_binary) or np.any(mask_binary):
            # Calculate the IoU value between two masks
            iou = compute_mask_iou(compare_binary.flatten(), mask_binary.flatten())
            # If IoU is less than or equal to threshold, keep the mask
            if iou <= iou_threshold:
                # Only merge objects with low IoU
                merged_mask[obj_id] = mask

    return merged_mask


class Baseline_Bi_SSCCE:
    def __init__(
        self,
        predictor=None,
        mid_frame=0,
        diff_frame_num=1,
        iou_threshold=0.5,
    ):
        self.predictor = predictor
        self.iou_threshold = iou_threshold
        self.mid_frame = mid_frame
        self.diff_frame_num = diff_frame_num

    def renew_session(self, video_path):
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        # note: in case you already ran one text prompt and now want to switch to another text prompt
        # it's required to reset the session first (otherwise the results would be wrong)
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        return session_id

    def propagate_in_video(self, session_id, propagation_direction="both"):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    def step_one(
        self,
        img_paths: list,
        prompt_text_str=None,
        merge_mask_func_version="v1",
    ):
        diff_mask_list = []
        ##### module 1 #####
        for sort in ["asc", "desc"]:
            ##### module 2 #####
            video_path = gen_frame(
                img_paths,
                sort=sort,
                mid_frame=self.mid_frame,
            )

            if isinstance(video_path, str) and video_path.endswith(".mp4"):
                cap = cv2.VideoCapture(video_path)
                video_frames_for_vis = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
                try:
                    # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
                    video_frames_for_vis.sort(
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                    )
                except ValueError:
                    # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
                    print(
                        f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                        f"falling back to lexicographic sort."
                    )
                    video_frames_for_vis.sort()

            session_id = self.renew_session(video_path)

            if prompt_text_str is None:
                print("请输入prompt")
                return
            if isinstance(prompt_text_str, str):
                prompt_text_str = [prompt_text_str]
            else:
                prompt_text_str = prompt_text_str

            frame_idx = 0  # add a text prompt on frame 0
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt_text_str[0],
                )
            )
            # out = response["outputs"]

            ##### module n #####
            before_masks, after_masks = {}, {}

            outputs_per_frame = self.propagate_in_video(session_id, "both")
            frame_len = len(outputs_per_frame)

            before_frame = outputs_per_frame[
                0 if self.diff_frame_num == 1 else frame_len - 2
            ]
            after_frame = outputs_per_frame[frame_len - 1]

            b_ids, b_masks, b_probs, b_boxes = (
                before_frame.get("out_obj_ids"),
                before_frame.get("out_binary_masks"),
                before_frame.get("out_probs"),
                before_frame.get("out_boxes_xywh"),
            )
            a_ids, a_masks, a_probs, a_boxes = (
                after_frame.get("out_obj_ids"),
                after_frame.get("out_binary_masks"),
                after_frame.get("out_probs"),
                after_frame.get("out_boxes_xywh"),
            )

            for id, mask, prob, box in zip(b_ids, b_masks, b_probs, b_boxes):
                before_masks[id] = dict(
                    mask=mask,
                    prob=prob,
                    box=box,
                )

            for id, mask, prob, box in zip(a_ids, a_masks, a_probs, a_boxes):
                after_masks[id] = dict(
                    mask=mask,
                    prob=prob,
                    box=box,
                )

            if merge_mask_func_version == "v1":
                diff_dict = merge_masks_v1(
                    before_masks,
                    after_masks,
                    iou_threshold=self.iou_threshold,
                )
            else:
                diff_dict = merge_masks(
                    before_masks,
                    after_masks,
                    iou_threshold=self.iou_threshold,
                )

            if diff_dict:
                # 获取第一个mask的形状作为参考
                first_mask = list(diff_dict.values())[0]
                if isinstance(first_mask, dict):
                    first_mask = first_mask.get("mask")

                h, w = first_mask.shape
                combined_mask = np.zeros((h, w), dtype=np.float32)

                # 遍历所有对象，将它们的mask叠加
                for obj_id, mask_data in diff_dict.items():
                    if isinstance(mask_data, dict):
                        mask = mask_data.get("mask")
                    else:
                        mask = mask_data

                    if mask is not None:
                        # 使用逻辑或操作合并mask
                        combined_mask = np.maximum(
                            combined_mask, (mask > 0).astype(np.float32)
                        )

                # 转换为tensor
                diff = torch.from_numpy(combined_mask)
            else:
                diff = torch.zeros((1024, 1024), dtype=torch.float32)

            # plt.figure(figsize=(10, 10))  # set the figure size
            # plt.subplot(1, 1, 1)
            # plt.imshow(diff)
            # plt.title("T1")
            # plt.axis("off")

            # # show the plot
            # plt.tight_layout()
            # plt.show()

            diff_mask_list.append(diff)

            _ = self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

        if len(diff_mask_list) >= 2:
            mask1 = diff_mask_list[0]
            mask2 = diff_mask_list[1]

            if isinstance(mask1, torch.Tensor):
                combined = mask1.float() + mask2.float()
                diff = (combined >= 2).float()
            else:
                combined = mask1.astype(np.float32) + mask2.astype(np.float32)
                diff = (combined >= 2).astype(np.float32)

            torch.cuda.empty_cache()
            gc.collect()
            return diff
        else:
            torch.cuda.empty_cache()
            gc.collect()
            return diff_mask_list[0]


class Baseline_Bi_SSCCE_CSPCF:
    def __init__(
        self,
        predictor=None,
        mid_frame=0,
        diff_frame_num=1,
        iou_threshold=0.5,
    ):
        self.predictor = predictor
        self.iou_threshold = iou_threshold
        self.mid_frame = mid_frame
        self.diff_frame_num = diff_frame_num

    def renew_session(self, video_path):
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        # note: in case you already ran one text prompt and now want to switch to another text prompt
        # it's required to reset the session first (otherwise the results would be wrong)
        _ = self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

        return session_id

    def propagate_in_video(self, session_id, propagation_direction="both"):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    def step_one(
        self,
        img_paths: list,
        prompt_text_str=None,
        merge_mask_func_version="v2",
    ):
        diff_mask_list = []
        ##### module 1 #####
        for sort in ["asc", "desc"]:
            ##### module 2 #####
            video_path = gen_frame(
                img_paths,
                sort=sort,
                mid_frame=self.mid_frame,
            )

            if isinstance(video_path, str) and video_path.endswith(".mp4"):
                cap = cv2.VideoCapture(video_path)
                video_frames_for_vis = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
                try:
                    # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
                    video_frames_for_vis.sort(
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                    )
                except ValueError:
                    # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
                    print(
                        f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                        f"falling back to lexicographic sort."
                    )
                    video_frames_for_vis.sort()

            session_id = self.renew_session(video_path)

            if prompt_text_str is None:
                print("请输入prompt")
                return
            if isinstance(prompt_text_str, str):
                prompt_text_str = [prompt_text_str]
            else:
                prompt_text_str = prompt_text_str

            frame_idx = 0  # add a text prompt on frame 0
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt_text_str[0],
                )
            )
            # out = response["outputs"]

            ##### module n #####
            before_masks, after_masks = {}, {}

            outputs_per_frame = self.propagate_in_video(session_id, "both")
            frame_len = len(outputs_per_frame)

            before_frame = outputs_per_frame[
                0 if self.diff_frame_num == 1 else frame_len - 2
            ]
            after_frame = outputs_per_frame[frame_len - 1]

            b_ids, b_masks, b_probs, b_boxes = (
                before_frame.get("out_obj_ids"),
                before_frame.get("out_binary_masks"),
                before_frame.get("out_probs"),
                before_frame.get("out_boxes_xywh"),
            )
            a_ids, a_masks, a_probs, a_boxes = (
                after_frame.get("out_obj_ids"),
                after_frame.get("out_binary_masks"),
                after_frame.get("out_probs"),
                after_frame.get("out_boxes_xywh"),
            )

            for id, mask, prob, box in zip(b_ids, b_masks, b_probs, b_boxes):
                before_masks[id] = dict(
                    mask=mask,
                    prob=prob,
                    box=box,
                )

            for id, mask, prob, box in zip(a_ids, a_masks, a_probs, a_boxes):
                after_masks[id] = dict(
                    mask=mask,
                    prob=prob,
                    box=box,
                )

            if merge_mask_func_version == "v1":
                diff_dict = merge_masks_v1(
                    before_masks,
                    after_masks,
                    iou_threshold=self.iou_threshold,
                )
            else:
                diff_dict = merge_masks(
                    before_masks,
                    after_masks,
                    iou_threshold=self.iou_threshold,
                )

            # if diff_dict:
            #     # 获取第一个mask的形状作为参考
            #     first_mask = list(diff_dict.values())[0]
            #     if isinstance(first_mask, dict):
            #         first_mask = first_mask.get("mask")

            #     h, w = first_mask.shape
            #     combined_mask = np.zeros((h, w), dtype=np.float32)

            #     # 遍历所有对象，将它们的mask叠加
            #     for obj_id, mask_data in diff_dict.items():
            #         if isinstance(mask_data, dict):
            #             mask = mask_data.get("mask")
            #         else:
            #             mask = mask_data

            #         if mask is not None:
            #             # 使用逻辑或操作合并mask
            #             combined_mask = np.maximum(
            #                 combined_mask, (mask > 0).astype(np.float32)
            #             )

            #     # 转换为tensor
            #     diff = torch.from_numpy(combined_mask)
            # else:
            #     diff = torch.zeros((1024, 1024), dtype=torch.float32)

            # plt.figure(figsize=(10, 10))  # set the figure size
            # plt.subplot(1, 1, 1)
            # plt.imshow(diff)
            # plt.title("T1")
            # plt.axis("off")

            # # show the plot
            # plt.tight_layout()
            # plt.show()

            diff_mask_list.append(diff_dict)

            _ = self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

        return diff_mask_list
