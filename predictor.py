import gc
import os
import statistics
import sam3
import torch

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# use all available GPUs on the machine
gpus_to_use = range(torch.cuda.device_count())
# # use only a single GPU
# gpus_to_use = [torch.cuda.current_device()]

from sam3.model_builder import build_sam3_video_predictor

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

from BiSAM2 import step_one


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
    union = (np.sum(masks1, axis=1)[:, None] +
             np.sum(masks2, axis=1)[None, :] - intersection)

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
    if not masks_A and (masks_B is None or not masks_B):
        # Get reference shape (if unable to get, raise exception or specify default shape)
        try:
            ref_shape = next(iter(masks_A.values())).shape
        except StopIteration:
            ref_shape = (1, 1024, 1024)  # Default shape
        return np.zeros(ref_shape, dtype=np.uint8)

    try:
        merged_mask = np.zeros_like(next(iter(masks_A.values())),
                                    dtype=np.uint8)
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


def predict(
    img_paths: list,
    prompt_text_str: str,
    mid_frame=0,
    diff_frame_num=-1,
    iou_threshold=0.5,
    max_objects_per_batch=50,
    **kwargs,
):

    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path="/home/yyyjvm/Checkpoints/sam3-model/sam3.pt")

    diff_mask_list = step_one(
        img_paths,
        predictor,
        mid_frame=mid_frame,
        diff_frame_num=diff_frame_num,
        iou_threshold=iou_threshold,
        prompt_text_str=prompt_text_str,
        max_objects_per_batch=max_objects_per_batch,
    )

    diff_mask = sum_masks_dict(*diff_mask_list, iou_threshold=iou_threshold)

    h, w = diff_mask.shape[-2:]
    mask = diff_mask.reshape(h, w, 1)

    if "predictor" in locals():
        del predictor
    torch.cuda.empty_cache()
    gc.collect()

    return mask


from utils.metrics import binary_accuracy, binary_accuracy_sklearn, AverageMeter


def inference(
    before_img_dir: str,
    after_img_dir: str,
    label_img_dir: str,
    prompt_text_str: str,
    mid_frame=0,
    diff_frame_num=-1,
    iou_threshold=0.5,
):
    if None in [before_img_dir, after_img_dir, label_img_dir]:
        print("请输入前后时相图片路径和标签路径")
        return
    # 如果prompt_text_str不是str且为None则返回
    if not isinstance(prompt_text_str, str) and prompt_text_str is None:
        print("请输入prompt")
        return

    # 读取前后时相路径中的所有文件名
    img_names = [
        p for p in os.listdir(before_img_dir)
        if os.path.splitext(p)[-1] in [".png"]
    ]

    output_dir = f"./logs/WHU-CD/generate_mid{mid_frame}_{diff_frame_num}_iou{iou_threshold}_[{prompt_text_str}]/automatic"

    # 存在的文件夹则读取已完成文件
    if os.path.isdir(output_dir):
        print(f"{output_dir} 已存在")
        # return
        exist_files = os.listdir(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
        exist_files = []

    with open(os.path.join(output_dir, "log.txt"), "a", encoding="utf-8") as f:
        F1_meter = AverageMeter()
        IoU_meter = AverageMeter()
        Acc_meter = AverageMeter()
        Pre_meter = AverageMeter()
        Rec_meter = AverageMeter()

        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path="/home/yyyjvm/Checkpoints/sam3-model/sam3.pt",
        )

        for idx, img_name in enumerate(img_names):
            # 跳过已存在的文件
            if img_name in exist_files:
                print(f"Skipping image {idx+1}/{len(img_names)}: {img_name}")
                continue
            else:
                print(f"Processing image {idx+1}/{len(img_names)}: {img_name}")

            img_paths = [
                os.path.join(before_img_dir, img_name),
                os.path.join(after_img_dir, img_name),
            ]

            diff_mask_list = step_one(
                img_paths,
                predictor,
                mid_frame=mid_frame,
                diff_frame_num=diff_frame_num,
                iou_threshold=iou_threshold,
                prompt_text_str=prompt_text_str,
                max_objects_per_batch=500,
            )

            diff_mask = sum_masks_dict(*diff_mask_list,
                                       iou_threshold=iou_threshold)

            # 读取标签图（单通道）
            label_path = os.path.join(label_img_dir, img_name)
            # 如果标签图不存在则返回
            if not os.path.exists(label_path):
                print(f"{label_path} 不存在")

            # label_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label_mask = Image.open(label_path)
            # 将PIL Image转换为numpy数组
            label_mask_np = np.array(label_mask)

            # 如果需要灰度图像，可以先转换为灰度再转为numpy数组
            if len(label_mask_np.shape) == 3:  # 如果是RGB图像
                label_mask = np.array(label_mask.convert("L"))
            else:
                label_mask = label_mask_np

            # iou = compute_mask_iou(diff_mask, label_mask)
            acc, precision, recall, f1, iou = binary_accuracy(
                diff_mask, label_mask)
            # acc, precision, recall, f1, iou = binary_accuracy_sklearn(
            #     diff_mask, label_mask
            # )

            F1_meter.update(f1)
            Acc_meter.update(acc)
            IoU_meter.update(iou)
            Pre_meter.update(precision)
            Rec_meter.update(recall)

            print(
                f"{idx+1}/{len(img_names)} iou: {iou} f1: {f1} pre: {precision} rec: {recall} acc:{format(acc*100,'.2f')}"
            )
            f.write(
                f"{idx+1}/{len(img_names)} f1: {format(f1*100,'.2f')} iou: {format(iou*100,'.2f')} pre: {format(precision*100,'.2f')} rec: {format(recall*100,'.2f')} acc:{format(acc*100,'.2f')} name: {img_name}\n"
            )

            # 保存mask
            h, w = diff_mask.shape[-2:]
            mask_image = diff_mask.reshape(h, w, 1)
            cv2.imwrite(os.path.join(output_dir, img_name), mask_image * 255)

            # if "predictor" in locals():
            #     del predictor
            torch.cuda.empty_cache()
            gc.collect()

        try:
            print(
                f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
            )
            f.write(
                f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
            )
        except statistics.StatisticsError:
            print("列表为空，无法计算平均值")


if __name__ == "__main__":
    # img_name = "tile_7168_8192.png"
    # img_dirs = [
    #     "/home/yyyjvm/CD_datasets/WHU-CD/test/A",
    #     "/home/yyyjvm/CD_datasets/WHU-CD/test/B",
    # ]
    # img_paths = []
    # for img_dir in img_dirs:
    #     img_paths.append(os.path.join(img_dir, img_name))

    # mask = predict(
    #     img_paths=img_paths,
    #     prompt_text_str="building",
    #     mid_frame=1,
    #     max_objects_per_batch=500,
    # )
    # # create a figure that can hold three subplots
    # plt.figure(figsize=(15, 5))  # set the figure size

    # # drawing img_A
    # # img_A = cv2.imread(img_paths[0])
    # img_A = Image.open(img_paths[0])
    # plt.subplot(1, 3, 1)
    # plt.imshow(img_A)
    # plt.title("T1")
    # plt.axis("off")

    # # drawing img_B
    # # img_B = cv2.imread(img_paths[1])
    # img_B = Image.open(img_paths[1])
    # plt.subplot(1, 3, 2)
    # plt.imshow(img_B)
    # plt.title("T2")
    # plt.axis("off")

    # # drawing mask
    # plt.subplot(1, 3, 3)
    # plt.imshow(mask, cmap="gray")
    # plt.title("mask")
    # plt.axis("off")

    # # show the plot
    # plt.tight_layout()
    # plt.show()

    ### 批量推理 ###
    inference(
        before_img_dir="/home/yyyjvm/CD_datasets/WHU-CD/test/A",
        after_img_dir="/home/yyyjvm/CD_datasets/WHU-CD/test/B",
        label_img_dir="/home/yyyjvm/CD_datasets/WHU-CD/test/label",
        prompt_text_str="building",
        mid_frame=1,
    )
