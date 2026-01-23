import gc
import os
import shutil
import cv2
import numpy as np
import torch
import glob
from PIL import Image


def linear_color_interpolation(img1, img2, alpha):
    """
    Linear color interpolation between two images

    :param img1: T1 image (can be PIL Image or numpy array)
    :param img2: T2 image (can be PIL Image or numpy array)
    :param alpha: Interpolation weight (0 for full T1, 1 for full T2)
    :return: Interpolated frame image
    """
    # Convert PIL Images to numpy arrays if necessary
    if hasattr(img1, "convert"):  # PIL Image object
        img1 = np.array(img1.convert("RGB"))
    if hasattr(img2, "convert"):  # PIL Image object
        img2 = np.array(img2.convert("RGB"))

    # Ensure images are in the correct format (RGB/BGR)
    if len(img1.shape) == 3:  # Has 3 channels
        img1_rgb = img1
    else:  # Grayscale
        img1_rgb = np.stack([img1] * 3, axis=-1)

    if len(img2.shape) == 3:  # Has 3 channels
        img2_rgb = img2
    else:  # Grayscale
        img2_rgb = np.stack([img2] * 3, axis=-1)

    # Linear interpolation
    interpolated_rgb = (1 - alpha) * img1_rgb + alpha * img2_rgb
    interpolated_rgb = interpolated_rgb.astype(np.uint8)
    return interpolated_rgb


def enhanced_color_interpolation(img1, img2, alpha, method="histogram_matching"):
    """
    Enhanced color interpolation between two images using various methods to avoid ghosting artifacts

    :param img1: T1 image (numpy array or PIL Image)
    :param img2: T2 image (numpy array or PIL Image)
    :param alpha: Interpolation weight (0 for full T1, 1 for full T2)
    :param method: Method to use ('histogram_matching', 'color_transfer', 'opt_flow', 'none')
    :return: Interpolated frame image
    """
    # Convert PIL Images to numpy arrays if necessary
    if hasattr(img1, "convert"):  # PIL Image object
        img1 = np.array(img1.convert("RGB"))
    if hasattr(img2, "convert"):  # PIL Image object
        img2 = np.array(img2.convert("RGB"))

    # Ensure images are in the correct format (RGB/BGR)
    if len(img1.shape) == 3:  # Has 3 channels
        img1_rgb = img1
    else:  # Grayscale
        img1_rgb = np.stack([img1] * 3, axis=-1)

    if len(img2.shape) == 3:  # Has 3 channels
        img2_rgb = img2
    else:  # Grayscale
        img2_rgb = np.stack([img2] * 3, axis=-1)

    # Resize images to the same size if they differ
    if img1_rgb.shape != img2_rgb.shape:
        h, w = img1_rgb.shape[:2]
        img2_rgb = cv2.resize(img2_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    if method == "histogram_matching":
        # Use histogram matching to align the color distribution
        matched_img2 = match_histograms(img1_rgb, img2_rgb)
        interpolated_rgb = matched_img2.astype(np.uint8)
        # interpolated_rgb = (1 - alpha) * img1_rgb + alpha * matched_img2
        # interpolated_rgb = interpolated_rgb.astype(np.uint8)
        return interpolated_rgb
    elif method == "color_transfer":
        # Use Reinhard color transfer to match color statistics
        transferred_img2 = color_transfer(img1_rgb, img2_rgb)
        interpolated_rgb = transferred_img2.astype(np.uint8)

        # interpolated_rgb = (1 - alpha) * img1_rgb + alpha * transferred_img2
        # interpolated_rgb = interpolated_rgb.astype(np.uint8)
        return interpolated_rgb
    elif method == "opt_flow":
        # Use optical flow to align images before interpolation
        aligned_img2 = align_images_with_optical_flow(img1_rgb, img2_rgb)
        interpolated_rgb = aligned_img2.astype(np.uint8)
        # interpolated_rgb = (1 - alpha) * img1_rgb + alpha * aligned_img2
        # interpolated_rgb = interpolated_rgb.astype(np.uint8)
        return interpolated_rgb
    else:
        # Simple linear interpolation (original method)
        interpolated_rgb = (1 - alpha) * img1_rgb + alpha * img2_rgb
        interpolated_rgb = interpolated_rgb.astype(np.uint8)
        return interpolated_rgb


def match_histograms(img1, img2):
    """
    Match the histogram of img2 to img1 using OpenCV
    """
    # Convert to LAB color space for histogram matching
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Match histograms for each channel separately
    matched_lab = np.zeros_like(lab2)
    for i in range(3):  # L, A, B channels
        matched_lab[:, :, i] = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(lab2[:, :, i].astype(np.uint8))
        # Alternative: match histograms directly
        # matched_lab[:, :, i] = cv2.equalizeHist(lab2[:, :, i].astype(np.uint8))

    # Convert back to RGB
    matched_img = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return matched_img


def color_transfer(img1, img2):
    """
    Transfer color statistics from img1 to img2 using Reinhard color transfer
    """
    # Convert images to float for processing
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Convert to LAB color space
    lab1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2LAB)

    # Compute mean and std of each channel in LAB for img1
    l1_mean, a1_mean, b1_mean = np.mean(lab1, axis=(0, 1))
    l1_std, a1_std, b1_std = np.std(lab1, axis=(0, 1))

    # Compute mean and std of each channel in LAB for img2
    l2_mean, a2_mean, b2_mean = np.mean(lab2, axis=(0, 1))
    l2_std, a2_std, b2_std = np.std(lab2, axis=(0, 1))

    # Ensure the variables are arrays to allow indexing
    if np.isscalar(l1_std):
        l1_std = np.array([l1_std, a1_std, b1_std])
        l2_std = np.array([l2_std, a2_std, b2_std])
        l1_mean = np.array([l1_mean, a1_mean, b1_mean])
        l2_mean = np.array([l2_mean, a2_mean, b2_mean])
    else:
        l1_std = np.array([l1_std[0], a1_std[0], b1_std[0]])
        l2_std = np.array([l2_std[0], a2_std[0], b2_std[0]])
        l1_mean = np.array([l1_mean[0], a1_mean[0], b1_mean[0]])
        l2_mean = np.array([l2_mean[0], a2_mean[0], b2_mean[0]])

    # Normalize img2's LAB channels
    lab2_norm = np.zeros_like(lab2, dtype=np.float32)
    lab2_norm[:, :, 0] = (lab2[:, :, 0] - l2_mean[0]) * (
        l1_std[0] / l2_std[0]
    ) + l1_mean[0]
    lab2_norm[:, :, 1] = (lab2[:, :, 1] - l2_mean[1]) * (
        l1_std[1] / l2_std[1]
    ) + l1_mean[1]
    lab2_norm[:, :, 2] = (lab2[:, :, 2] - l2_mean[2]) * (
        l1_std[2] / l2_std[2]
    ) + l1_mean[2]

    # Clip values to valid range
    lab2_norm = np.clip(lab2_norm, 0, 255)

    # Convert back to RGB
    transferred_img = cv2.cvtColor(lab2_norm.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return transferred_img


def align_images_with_optical_flow(img1, img2):
    """
    Align img2 to img1 using optical flow
    """
    # Convert to grayscale for optical flow computation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Get image dimensions
    h, w = gray1.shape

    # Generate coordinate grids
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(w)
    flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]
    flow_map = flow_map + flow

    # Remap img2 to align with img1
    aligned_img2 = cv2.remap(
        img2,
        flow_map[:, :, 0],
        flow_map[:, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return aligned_img2


def gen_frame(folder_paths, output_dir="output_jpg", sort="asc", mid_frame=0):
    """
    Convert PNG format image files to JPEG format and optionally generate intermediate interpolated frames

    This function iterates through the input folder path list, converts PNG images to JPEG format,
    and generates intermediate interpolated frames as needed. The main processing includes
    image format conversion (RGBA/LA to RGB), file renaming, and color interpolation.

    Parameters:
        folder_paths (list): List of paths to PNG image files
        output_dir (str): Directory path for output JPEG images, defaults to "output_jpg"
        sort (str): File processing order, "asc" for ascending order, other values for descending, defaults to "asc"
        mid_frame (int): Number of intermediate frames to generate, defaults to 0 (no intermediate frames)

    Returns:
        str: Output directory path
    """
    # Determine traversal order based on sorting method
    paths_to_process = folder_paths if sort == "asc" else list(reversed(folder_paths))

    # Clear folder contents
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Ensure output folder exists (only need to check once)
    os.makedirs(output_dir, exist_ok=True)

    # Process all input image files
    for idx, folder_path in enumerate(paths_to_process):
        # Construct input and output paths
        input_path = folder_path
        output_filename = f"{idx + 1}.jpg" if idx == 0 else f"{idx + mid_frame + 1}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Open PNG image and convert to RGB mode (JPEG does not support PNG's RGBA transparency)
        filename = os.path.basename(folder_path)
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # Create an RGB image with white background
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(
                        img, mask=img.split()[-1]
                    )  # Use alpha channel as mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Save as JPEG
                img.save(output_path, "JPEG", quality=100)
                print(
                    f"Conversion successful: {filename} -> {os.path.basename(output_path)}"
                )
        except Exception as e:
            print(f"Conversion failed {filename}: {str(e)}")

    def generate_uniform_alphas(num_frames):
        """Generate uniformly spaced alpha values"""
        return [i / (num_frames + 1) for i in range(1, num_frames + 1)]

    # Generate intermediate frames
    alphas = generate_uniform_alphas(mid_frame)
    for idx, alpha in enumerate(alphas):
        # Construct input and output paths
        input_path = paths_to_process[0]
        output_filename = f"{idx + 2}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Open PNG image and convert to RGB mode (JPEG does not support PNG's RGBA transparency)
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # Create an RGB image with white background
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(
                        img, mask=img.split()[-1]
                    )  # Use alpha channel as mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Perform linear color interpolation to generate intermediate frames
                first_frame = paths_to_process[0]
                final_frame = paths_to_process[-1]
                img = enhanced_color_interpolation(
                    # cv2.imread(first_frame, cv2.IMREAD_UNCHANGED),
                    # cv2.imread(final_frame, cv2.IMREAD_UNCHANGED),
                    Image.open(first_frame),
                    Image.open(final_frame),
                    alpha=alpha,
                    method="color_transfer",
                )
                # Save as JPEG
                cv2.imwrite(output_path, img)
                # img.save(output_path, "JPEG", quality=100)
                print(
                    f"Intermediate frame generated: {alpha} -> {os.path.basename(output_path)}"
                )
        except Exception as e:
            print(f"Intermediate frame generation failed {alpha}: {str(e)}")

    return output_dir


def compute_mask_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) between two masks

    This function measures the similarity between two binary masks by computing
    the ratio of their intersection area to their union area. The IoU value
    ranges from 0 to 1, where 1 indicates identical masks and 0 indicates
    no overlap.

    Args:
        mask1 (numpy.ndarray): First mask array where non-zero values
                              represent foreground regions
        mask2 (numpy.ndarray): Second mask array where non-zero values
                              represent foreground regions

    Returns:
        float: IoU value between the two masks in range [0, 1]
               Returns 1.0 when both masks are all zeros (considered identical)
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    sum_union = np.sum(union)
    if sum_union == 0:  # Both masks are all zeros, considered identical
        return 1.0
    iou = np.sum(intersection) / sum_union
    # diff_mask = np.logical_xor(mask1 > 0, mask2 > 0).astype(np.uint8)
    return iou


def merge_masks_old(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
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

    # Iterate through each object in the current frame
    for obj_id, mask in masks_dict.items():
        # Convert mask to binary image with non-zero elements as 1 and zero elements as 0
        mask_binary = (mask > 0).astype(np.uint8)

        # Check if there is an object with the same ID in the comparison frame
        compare_mask = compare_masks_dict.get(obj_id)
        # Also convert the mask in the comparison frame to binary image
        # Handle case where compare_mask is None
        if compare_mask is None:
            # If there's no corresponding mask in comparison frame, include this mask
            merged_mask[obj_id] = mask
            continue

        compare_binary = (compare_mask > 0).astype(np.uint8)

        # Calculate IoU (ignoring cases where masks are all zeros)
        if np.any(compare_binary) or np.any(mask_binary):
            # Calculate the IoU value between two masks
            iou = compute_mask_iou(compare_binary.flatten(), mask_binary.flatten())
            # If IoU is less than or equal to threshold, keep the mask
            if iou <= iou_threshold:
                # Only merge objects with low IoU
                merged_mask[obj_id] = mask

    return merged_mask


def merge_masks(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
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

    # Iterate through each object in the current frame
    for obj_id, mask_data in masks_dict.items():
        mask = mask_data["mask"]
        box = mask_data["box"]

        # Convert mask to binary image with non-zero elements as 1 and zero elements as 0
        mask_binary = (mask > 0).astype(np.uint8)

        # Check if there is an object with the same ID in the comparison frame
        compare_mask_data = compare_masks_dict.get(obj_id)
        # Also convert the mask in the comparison frame to binary image
        # Handle case where compare_mask_data is None
        if compare_mask_data is None:
            # If there's no corresponding mask in comparison frame, include this mask
            merged_mask[obj_id] = mask
            continue

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

        # If IoU is less than or equal to threshold, keep the mask
        if iou <= iou_threshold:
            # Only merge objects with low IoU
            merged_mask[obj_id] = mask

    return merged_mask


def propagate_in_video(predictor, session_id, propagation_direction="both"):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=propagation_direction,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def remove_overlapping_detections(
    all_prompt_results, iou_threshold=0.9, max_prompt=None
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
                    "mask": masks[i] if masks is not None and i < len(masks) else None,
                    "prob": probs[i] if probs is not None and i < len(probs) else 0,
                    "box": boxes[i] if boxes is not None and i < len(boxes) else None,
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

        final_detections.append(highest_conf_det)  # 将置信度最高的目标添加到最终结果

        # 从剩余列表中移除所有重叠的目标
        remaining_detections = non_overlapping_dets

    return final_detections


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


from typing import Dict


def generate_point_prompts_from_masks(
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
                        positive_points.append((px + x, py + y))  # 加上bbox左上角偏移
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
                            len(bbox_coords), size=neg_points_per_object, replace=False
                        )
                    else:
                        # 如果可用点少于所需点数，允许重复选择
                        selected_indices = np.random.choice(
                            len(bbox_coords), size=neg_points_per_object, replace=True
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


from sam3.visualization_utils import (
    visualize_formatted_frame_output,
    prepare_masks_for_visualization,
    plot_results,
)


def step_one(
    img_paths: list,
    predictor=None,
    mid_frame=0,
    diff_frame_num=1,
    iou_threshold=0.5,
    prompt_text_str=None,
    max_objects_per_batch=50,
):
    diff_mask_list = []

    for sort in ["asc", "desc"]:
        video_path = gen_frame(
            img_paths,
            sort=sort,
            mid_frame=mid_frame,
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

        def renew_session():
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            session_id = response["session_id"]

            # note: in case you already ran one text prompt and now want to switch to another text prompt
            # it's required to reset the session first (otherwise the results would be wrong)
            _ = predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )

            return session_id

        session_id = renew_session()

        # prompt_text_str = "person"
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
                response = predictor.handle_request(
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

            point_prompts = generate_point_prompts_from_masks(all_prompt_results)
            # 找出目标最多的提示词
            max_prompt = max(obj_nums, key=obj_nums.get)
            # 去除不同提示词检测结果中的重复框，保留置信度最高的
            all_prompt_results = remove_overlapping_detections(
                all_prompt_results, max_prompt=max_prompt
            )
            # 重新设置predictor
            _ = predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=max_prompt,
                )
            )
            out = response["outputs"]
            # 先运行一次
            propagate_in_video(predictor, session_id)
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

                boxes_tensor = torch.tensor(
                    boxes_abs,
                    dtype=torch.float32,
                )
                boxes_labels_tensor = torch.tensor(labels, dtype=torch.int32)

                response = predictor.handle_request(
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
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt_text_str[0],
                )
            )
            out = response["outputs"]

        obj_size = out["out_obj_ids"].size

        before_masks, after_masks = {}, {}

        if obj_size > max_objects_per_batch:
            out_obj_ids = out["out_obj_ids"]
            for i in range(0, obj_size, max_objects_per_batch):
                save_arr = out_obj_ids[i : min(i + max_objects_per_batch, obj_size)]
                print("切片数组save_arr：", save_arr)

                if len(prompt_text_str) > 1 and i > 0:
                    if any(num + 1 > obj_nums[max_prompt] for num in save_arr):
                        # 添加不重复的提示框
                        new_obj_idx = response["outputs"]["out_obj_ids"].size
                        # for prompt_result in all_prompt_results:
                        #     if prompt_result["prompt"] == max_prompt:
                        #         continue

                        #     box = prompt_result["box"]
                        #     frame_idx = 0
                        #     boxes_abs = np.array(
                        #         [
                        #             box,  # positive prompt box
                        #         ]
                        #     )
                        #     # positive prompt boxes have label 1, while negative prompt boxes have label 0
                        #     labels = np.array([1])
                        #     boxes_tensor = torch.tensor(
                        #         boxes_abs,
                        #         dtype=torch.float32,
                        #     )
                        #     boxes_labels_tensor = torch.tensor(
                        #         labels, dtype=torch.int32
                        #     )

                        #     response = predictor.handle_request(
                        #         request=dict(
                        #             type="add_prompt",
                        #             session_id=session_id,
                        #             frame_index=frame_idx,
                        #             bounding_boxes=boxes_tensor,
                        #             bounding_box_labels=boxes_labels_tensor,
                        #             obj_id=new_obj_idx,
                        #         )
                        #     )
                        #     new_obj_idx += 1

                # # 生成索引掩码：标记需要保留的元素（剔除save_arr对应的索引）
                # # 步骤1：创建全True的掩码（默认所有元素都保留）
                # mask = np.ones(len(out_obj_ids), dtype=bool)
                # # 步骤2：将save_arr对应的索引标记为False（剔除这些位置的元素）
                # mask[i : i + len(save_arr)] = False
                # # 步骤3：通过掩码提取剩余元素（即remove_arr）
                # remove_arr = out_obj_ids[mask]

                # for obj_id in remove_arr:
                #     if len(prompt_text_str) > 1 and obj_id + 1 > obj_nums[max_prompt]:
                #         continue
                #     response = predictor.handle_request(
                #         request=dict(
                #             type="remove_object",
                #             session_id=session_id,
                #             obj_id=obj_id,
                #         )
                #     )

                outputs_per_frame = propagate_in_video(predictor, session_id)
                frame_len = len(outputs_per_frame)
                # # 帧追踪结果可视化
                for frame_idx, output in outputs_per_frame.items():
                    visualize_formatted_frame_output(
                        frame_idx,
                        video_frames_for_vis,
                        outputs_list=[
                            prepare_masks_for_visualization({frame_idx: output})
                        ],
                        titles=[f"SAM 3 Dense Tracking outputs: {sort}"],
                        figsize=(6, 4),
                    )

                before_frame = outputs_per_frame[
                    0 if diff_frame_num == 1 else frame_len - 2
                ]
                after_frame = outputs_per_frame[frame_len - 1]

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

                _ = predictor.handle_request(
                    request=dict(
                        type="reset_session",
                        session_id=session_id,
                    )
                )

                frame_idx = 0  # add a text prompt on frame 0
                response = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        text=(
                            max_prompt
                            if len(prompt_text_str) > 1
                            else prompt_text_str[0]
                        ),
                    )
                )

        else:
            outputs_per_frame = propagate_in_video(predictor, session_id)
            frame_len = len(outputs_per_frame)

            before_frame = outputs_per_frame[
                0 if diff_frame_num == 1 else frame_len - 2
            ]
            after_frame = outputs_per_frame[frame_len - 1]

            # 帧追踪结果可视化
            # for frame_idx, output in outputs_per_frame.items():
            #     visualize_formatted_frame_output(
            #         frame_idx,
            #         video_frames_for_vis,
            #         outputs_list=[prepare_masks_for_visualization({frame_idx: output})],
            #         titles=[f"SAM 3 Dense Tracking outputs: {sort}"],
            #         figsize=(6, 4),
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

        # merge masks
        if obj_size == 0:
            diff_mask = {}
        else:

            # compare the first and last frames to get the difference
            diff_mask = merge_masks(
                before_masks,
                after_masks,
                iou_threshold=iou_threshold,
            )

        diff_mask_list.append(diff_mask)

        _ = predictor.handle_request(
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


if __name__ == "__main__":
    arr = np.arange(0, 190)
    print(arr)
    for i in range(0, 190, 50):
        print(f"循环外层: {i}")
        save_arr = arr[i : min(i + 50, 190)]
        print("切片数组save_arr：", save_arr)

        # 生成索引掩码：标记需要保留的元素（剔除save_arr对应的索引）
        # 步骤1：创建全True的掩码（默认所有元素都保留）
        mask = np.ones(len(arr), dtype=bool)
        # 步骤2：将save_arr对应的索引标记为False（剔除这些位置的元素）
        mask[i : i + len(save_arr)] = False
        # 步骤3：通过掩码提取剩余元素（即remove_arr）
        remove_arr = arr[mask]

        print("移除save_arr后的剩余数组remove_arr：", remove_arr)
