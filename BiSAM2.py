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


def enhanced_color_interpolation(img1,
                                 img2,
                                 alpha,
                                 method="histogram_matching"):
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
        matched_lab[:, :, i] = cv2.createCLAHE(clipLimit=2.0,
                                               tileGridSize=(8, 8)).apply(
                                                   lab2[:, :,
                                                        i].astype(np.uint8))
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
    lab2_norm[:, :, 0] = (lab2[:, :, 0] -
                          l2_mean[0]) * (l1_std[0] / l2_std[0]) + l1_mean[0]
    lab2_norm[:, :, 1] = (lab2[:, :, 1] -
                          l2_mean[1]) * (l1_std[1] / l2_std[1]) + l1_mean[1]
    lab2_norm[:, :, 2] = (lab2[:, :, 2] -
                          l2_mean[2]) * (l1_std[2] / l2_std[2]) + l1_mean[2]

    # Clip values to valid range
    lab2_norm = np.clip(lab2_norm, 0, 255)

    # Convert back to RGB
    transferred_img = cv2.cvtColor(lab2_norm.astype(np.uint8),
                                   cv2.COLOR_LAB2RGB)
    return transferred_img


def align_images_with_optical_flow(img1, img2):
    """
    Align img2 to img1 using optical flow
    """
    # Convert to grayscale for optical flow computation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5,
                                        1.2, 0)

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
    paths_to_process = folder_paths if sort == "asc" else list(
        reversed(folder_paths))

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
                        img, mask=img.split()[-1])  # Use alpha channel as mask
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
                        img, mask=img.split()[-1])  # Use alpha channel as mask
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


def merge_masks(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
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
            iou = compute_mask_iou(compare_binary.flatten(),
                                   mask_binary.flatten())
            # If IoU is less than or equal to threshold, keep the mask
            if iou <= iou_threshold:
                # Only merge objects with low IoU
                merged_mask[obj_id] = mask

    return merged_mask


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(request=dict(
            type="propagate_in_video",
            session_id=session_id,
    )):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


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
                video_frames_for_vis.append(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        else:
            video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
            try:
                # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
                video_frames_for_vis.sort(key=lambda p: int(
                    os.path.splitext(os.path.basename(p))[0]))
            except ValueError:
                # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
                print(
                    f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                    f"falling back to lexicographic sort.")
                video_frames_for_vis.sort()

        response = predictor.handle_request(request=dict(
            type="start_session",
            resource_path=video_path,
        ))
        session_id = response["session_id"]

        # note: in case you already ran one text prompt and now want to switch to another text prompt
        # it's required to reset the session first (otherwise the results would be wrong)
        _ = predictor.handle_request(request=dict(
            type="reset_session",
            session_id=session_id,
        ))

        # prompt_text_str = "person"
        frame_idx = 0  # add a text prompt on frame 0
        response = predictor.handle_request(request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        ))
        out = response["outputs"]

        obj_size = out["out_obj_ids"].size

        before_masks, after_masks = {}, {}

        if obj_size > max_objects_per_batch:
            out_obj_ids = out["out_obj_ids"]
            for i in range(0, obj_size, max_objects_per_batch):
                save_arr = out_obj_ids[i:min(i +
                                             max_objects_per_batch, obj_size)]
                print("切片数组save_arr：", save_arr)

                # 生成索引掩码：标记需要保留的元素（剔除save_arr对应的索引）
                # 步骤1：创建全True的掩码（默认所有元素都保留）
                mask = np.ones(len(out_obj_ids), dtype=bool)
                # 步骤2：将save_arr对应的索引标记为False（剔除这些位置的元素）
                mask[i:i + len(save_arr)] = False
                # 步骤3：通过掩码提取剩余元素（即remove_arr）
                remove_arr = out_obj_ids[mask]

                for obj_id in remove_arr:
                    response = predictor.handle_request(request=dict(
                        type="remove_object",
                        session_id=session_id,
                        obj_id=obj_id,
                    ))

                outputs_per_frame = propagate_in_video(predictor, session_id)
                frame_len = len(outputs_per_frame)

                before_frame = outputs_per_frame[0 if diff_frame_num ==
                                                 1 else frame_len - 2]
                after_frame = outputs_per_frame[frame_len - 1]

                b_ids, b_masks = before_frame.get(
                    "out_obj_ids"), before_frame.get("out_binary_masks")
                a_ids, a_masks = after_frame.get(
                    "out_obj_ids"), after_frame.get("out_binary_masks")

                for id, mask in zip(b_ids, b_masks):
                    before_masks[id] = mask

                for id, mask in zip(a_ids, a_masks):
                    after_masks[id] = mask

                _ = predictor.handle_request(request=dict(
                    type="reset_session",
                    session_id=session_id,
                ))

                # prompt_text_str = "person"
                frame_idx = 0  # add a text prompt on frame 0
                response = predictor.handle_request(request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt_text_str,
                ))
        else:
            outputs_per_frame = propagate_in_video(predictor, session_id)
            frame_len = len(outputs_per_frame)

            before_frame = outputs_per_frame[0 if diff_frame_num ==
                                             1 else frame_len - 2]
            after_frame = outputs_per_frame[frame_len - 1]

            b_ids, b_masks = before_frame.get("out_obj_ids"), before_frame.get(
                "out_binary_masks")
            a_ids, a_masks = after_frame.get("out_obj_ids"), after_frame.get(
                "out_binary_masks")

            for id, mask in zip(b_ids, b_masks):
                before_masks[id] = mask

            for id, mask in zip(a_ids, a_masks):
                after_masks[id] = mask

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

        _ = predictor.handle_request(request=dict(
            type="close_session",
            session_id=session_id,
        ))

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
        save_arr = arr[i:min(i + 50, 190)]
        print("切片数组save_arr：", save_arr)

        # 生成索引掩码：标记需要保留的元素（剔除save_arr对应的索引）
        # 步骤1：创建全True的掩码（默认所有元素都保留）
        mask = np.ones(len(arr), dtype=bool)
        # 步骤2：将save_arr对应的索引标记为False（剔除这些位置的元素）
        mask[i:i + len(save_arr)] = False
        # 步骤3：通过掩码提取剩余元素（即remove_arr）
        remove_arr = arr[mask]

        print("移除save_arr后的剩余数组remove_arr：", remove_arr)
