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
    if not masks_A and (masks_B is None or not masks_B):
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


from BiSAM2 import calculate_bbox_iou, compute_mask_iou


def merge_overlapping_masks(masks_A, masks_B, iou_threshold=0.5):
    ref_shape = (1, 1024, 1024)  # Default shape
    merged_mask = np.zeros(ref_shape, dtype=np.uint8)

    # 找到高重叠度的mask对
    matched_masks = []
    for idx_A, obj_id_A in enumerate(masks_A.keys()):
        mask_A = masks_A[obj_id_A].get("mask")
        # box_A = masks_A[obj_id_A].get("box")
        mask_binary = (mask_A > 0).astype(np.uint8)
        for idx_B, obj_id_B in enumerate(masks_B.keys()):
            mask_B = masks_B[obj_id_B].get("mask")
            # box_B = masks_B[obj_id_B].get("box")
            compare_binary = (mask_B > 0).astype(np.uint8)

            iou = compute_mask_iou(mask_binary, compare_binary)

            if iou > iou_threshold:
                # 记录匹配的mask对
                matched_masks.append((mask_A, mask_B))

    # 将所有匹配的mask对进行叠加
    for mask_a, mask_b in matched_masks:
        combined_mask = np.logical_or(mask_a > 0, mask_b > 0).astype(np.uint8)
        merged_mask = np.logical_or(merged_mask, combined_mask).astype(np.uint8)

    return merged_mask


from multiprocessing import Pool, cpu_count


def match_single_A_global(mask_A_data, B_masks, iou_threshold, device):
    obj_id_A, mask_A = mask_A_data
    mask_A_bin = (mask_A > 0).astype(np.uint8)
    matched_pairs = []
    for obj_id_B, mask_B_bin in B_masks:
        iou = compute_mask_iou(mask_A_bin, mask_B_bin, device=device)
        if iou > iou_threshold:
            matched_pairs.append((mask_A_bin, mask_B_bin))
    return matched_pairs


def merge_overlapping_masks_multiprocess(masks_A, masks_B, iou_threshold=0.5):
    ref_shape = (1, 1024, 1024)
    merged_mask = np.zeros(ref_shape, dtype=np.uint8)

    # 预处理B-mask：转换为二进制，整理为列表（子进程直接使用）
    B_masks = [
        (obj_id, (mask["mask"] > 0).astype(np.uint8))
        for obj_id, mask in masks_B.items()
    ]
    if not B_masks:
        return merged_mask

    # 预处理A-mask：构造待处理数据
    A_masks = [(obj_id, mask["mask"]) for obj_id, mask in masks_A.items()]
    if not A_masks:
        return merged_mask

    # 多进程执行：用starmap传递多参数（解决多个参数传递问题）
    cpu_cores = max(1, cpu_count() - 1)  # 预留1核给系统，避免卡顿
    with Pool(processes=cpu_cores) as pool:
        # 用starmap替代map，支持传递多参数（每个A-mask对应一组参数）
        # 格式：(mask_A_data, B_masks, iou_threshold)，B_masks和阈值所有子进程共享
        task_args = [(a_data, B_masks, iou_threshold, "cpu") for a_data in A_masks]
        all_matched = pool.starmap(match_single_A_global, task_args)

    # 展平匹配对并叠加mask，和原逻辑一致
    for matched_pairs in all_matched:
        for mask_a, mask_b in matched_pairs:
            combined = np.logical_or(mask_a, mask_b)
            merged_mask = np.logical_or(merged_mask[0], combined).astype(np.uint8)[
                None, ...
            ]

    return merged_mask


# 全局配置：自动检测GPU，设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备：{DEVICE}")


def merge_overlapping_masks_gpu(masks_A, masks_B, iou_threshold=0.5):
    ref_shape = (1, 1024, 1024)
    merged_mask = np.zeros(ref_shape, dtype=np.uint8)

    # 步骤1：预处理mask，转换为PyTorch GPU张量（一次性拷贝到GPU，避免多次传输）
    def preprocess_masks(masks_dict):
        mask_list = []
        for mask in masks_dict.values():
            # NumPy→Tensor，二进制转换，移到GPU
            mask_np = (mask["mask"] > 0).astype(np.uint8)
            mask_torch = torch.from_numpy(mask_np).to(DEVICE).squeeze()  # (1024,1024)
            mask_list.append(mask_torch)
        return mask_list

    masks_A_gpu = preprocess_masks(masks_A)
    masks_B_gpu = preprocess_masks(masks_B)

    if not masks_A_gpu or not masks_B_gpu:
        return merged_mask

    # 步骤2：GPU上完成所有IOU计算和mask叠加（无Python循环，CUDA内核并行）
    # 提前创建GPU版的merged_mask，全程在GPU运算，最后只拷贝一次回CPU
    merged_mask_gpu = torch.zeros(ref_shape, dtype=torch.uint8).to(DEVICE).squeeze()
    for mask_A in masks_A_gpu:
        for mask_B in masks_B_gpu:
            iou = compute_mask_iou(mask_A, mask_B, device=DEVICE)
            if iou > iou_threshold:
                # GPU上叠加mask，无数据传输
                combined = torch.logical_or(mask_A, mask_B)
                merged_mask_gpu = torch.logical_or(merged_mask_gpu, combined)

    # 步骤3：GPU→CPU，转换为原NumPy格式（仅一次数据拷贝）
    merged_mask = merged_mask_gpu.cpu().numpy().astype(np.uint8)[None, ...]
    return merged_mask


def predict(
    img_paths: list,
    prompt_text_str: str | list,
    mid_frame=0,
    diff_frame_num=-1,
    iou_threshold=0.5,
    max_objects_per_batch=50,
    score_threshold_detection: float = 0.5,
    new_det_thresh: float = 0.7,
    **kwargs,
):

    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path="/home/qy/weights/sam3-model/sam3.pt",
        score_threshold_detection=score_threshold_detection,
        new_det_thresh=new_det_thresh,
        # apply_temporal_disambiguation=False,
    )

    diff_mask_list = step_one(
        img_paths,
        predictor,
        mid_frame=mid_frame,
        diff_frame_num=diff_frame_num,
        iou_threshold=iou_threshold,
        prompt_text_str=prompt_text_str,
        max_objects_per_batch=max_objects_per_batch,
    )

    # create a figure that can hold three subplots
    plt.figure(figsize=(15, 10))  # set the figure size

    b_mask_list = {}
    for id, item in diff_mask_list[0].items():
        b_mask_list[id] = item.get("mask")
    mask_before = sum_masks_dict(b_mask_list, iou_threshold=iou_threshold)
    h, w = mask_before.shape[-2:]
    mask = mask_before.reshape(h, w, 1)
    # drawing img_A
    plt.subplot(1, 1, 1)
    plt.imshow(mask)
    plt.title("T1")
    plt.axis("off")

    # show the plot
    plt.tight_layout()
    plt.show()

    a_mask_list = {}
    for id, item in diff_mask_list[1].items():
        a_mask_list[id] = item.get("mask")
    mask_before = sum_masks_dict(a_mask_list, iou_threshold=iou_threshold)
    h, w = mask_before.shape[-2:]
    mask = mask_before.reshape(h, w, 1)
    # drawing img_A
    plt.subplot(1, 1, 1)
    plt.imshow(mask)
    plt.title("T1")
    plt.axis("off")

    # show the plot
    plt.tight_layout()
    plt.show()

    import time

    print("计时开始: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if DEVICE == torch.device("cuda"):
        diff_mask = merge_overlapping_masks_gpu(
            *diff_mask_list, iou_threshold=iou_threshold
        )
    else:
        diff_mask = merge_overlapping_masks_multiprocess(
            *diff_mask_list, iou_threshold=iou_threshold
        )

    print("计时结束: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    h, w = diff_mask.shape[-2:]
    mask = diff_mask.reshape(h, w, 1)

    if "predictor" in locals():
        del predictor
    torch.cuda.empty_cache()
    gc.collect()

    return mask


from utils.metrics import (
    binary_accuracy,
    binary_accuracy_sklearn,
    binary_accuracy_torchmetrics,
    AverageMeter,
)


def inference(
    dataset_name: str,
    prompt_text_str: str | list,
    mid_frame=0,
    diff_frame_num=-1,
    iou_threshold=0.5,
    max_objects_per_batch=50,
    score_threshold_detection: float = 0.5,
    new_det_thresh: float = 0.7,
):
    if dataset_name is None:
        print("请输入数据集名称")
        return

    if dataset_name in ["WHU-CD", "LEVIR-CD"]:
        before_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/A"
        after_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/B"
        label_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/label"
    else:
        print("数据集不支持")
        return

    # 如果prompt_text_str不是str且为None则返回
    if not isinstance(prompt_text_str, str) and prompt_text_str is None:
        print("请输入prompt")
        return

    # 读取前后时相路径中的所有文件名
    img_names = [
        p for p in os.listdir(before_img_dir) if os.path.splitext(p)[-1] in [".png"]
    ]

    output_dir = f"./logs/{dataset_name}/generate_mid{mid_frame}_{diff_frame_num}_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"

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
            checkpoint_path="/home/qy/weights/sam3-model/sam3.pt",
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
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
                max_objects_per_batch=max_objects_per_batch,
            )

            if DEVICE == torch.device("cuda"):
                diff_mask = merge_overlapping_masks_gpu(
                    *diff_mask_list, iou_threshold=iou_threshold
                )
            else:
                diff_mask = merge_overlapping_masks_multiprocess(
                    *diff_mask_list, iou_threshold=iou_threshold
                )

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

            acc, precision, recall, f1, iou = binary_accuracy(diff_mask, label_mask)
            # acc, precision, recall, f1, iou = binary_accuracy_sklearn(
            #     diff_mask, label_mask
            # )
            # acc, precision, recall, f1, iou = binary_accuracy_torchmetrics(
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
    img_name = "tile_9216_11264.png"
    img_dirs = [
        "/home/qy/CD_datasets/WHU-CD/test/A",
        "/home/qy/CD_datasets/WHU-CD/test/B",
        "/home/qy/CD_datasets/WHU-CD/test/label",
    ]
    # img_name = "test_118.png"
    # img_dirs = [
    #     "/home/qy/CD_datasets/LEVIR-CD/test/A",
    #     "/home/qy/CD_datasets/LEVIR-CD/test/B",
    #     "/home/qy/CD_datasets/LEVIR-CD/test/label",
    # ]
    img_paths = []
    for img_dir in img_dirs:
        img_paths.append(os.path.join(img_dir, img_name))

    mask = predict(
        img_paths=img_paths[:2],
        prompt_text_str=["building or roof or house"],
        mid_frame=0,
        diff_frame_num=-1,
        max_objects_per_batch=500,
        score_threshold_detection=0.3,
        new_det_thresh=0.3,
    )
    # create a figure that can hold three subplots
    plt.figure(figsize=(15, 10))  # set the figure size

    # drawing img_A
    # img_A = cv2.imread(img_paths[0])
    img_A = Image.open(img_paths[0])
    plt.subplot(2, 2, 1)
    plt.imshow(img_A)
    plt.title("T1")
    plt.axis("off")

    # drawing img_B
    # img_B = cv2.imread(img_paths[1])
    img_B = Image.open(img_paths[1])
    plt.subplot(2, 2, 2)
    plt.imshow(img_B)
    plt.title("T2")
    plt.axis("off")

    # drawing mask
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("mask")
    plt.axis("off")

    # drawing label
    plt.subplot(2, 2, 4)
    plt.imshow(Image.open(img_paths[2]), cmap="gray")
    plt.title("label")
    plt.axis("off")

    # show the plot
    plt.tight_layout()
    plt.show()

    ### 批量推理 ###
    # inference(
    #     dataset_name="WHU-CD",
    #     prompt_text_str=["building or roof or house"],
    #     mid_frame=0,
    #     diff_frame_num=-1,
    #     max_objects_per_batch=500,
    #     score_threshold_detection=0.25,
    #     new_det_thresh=0.25,
    # )

    # ### 测试插值方法 ###
    # from BiSAM2 import color_transfer, match_histograms, align_images_with_optical_flow

    # img1 = Image.open(img_paths[0])
    # img2 = Image.open(img_paths[1])

    # # # Convert PIL Images to numpy arrays if necessary
    # # if hasattr(img1, "convert"):  # PIL Image object
    # #     img1 = np.array(img1.convert("RGB"))
    # # if hasattr(img2, "convert"):  # PIL Image object
    # #     img2 = np.array(img2.convert("RGB"))

    # transferred_img1 = color_transfer(
    #     np.array(img2.convert("RGB")), np.array(img1.convert("RGB"))
    # )
    # transferred_img2 = color_transfer(
    #     np.array(img1.convert("RGB")), np.array(img2.convert("RGB"))
    # )
    # interpolated_rgb = transferred_img2.astype(np.uint8)

    # # create a figure that can hold three subplots
    # plt.figure(figsize=(15, 10))  # set the figure size

    # # drawing img_A
    # # img_A = cv2.imread(img_paths[0])
    # img_A = Image.open(img_paths[0])
    # plt.subplot(2, 2, 1)
    # plt.imshow(img_A)
    # plt.title("T1")
    # plt.axis("off")

    # # drawing img_B
    # # img_B = cv2.imread(img_paths[1])
    # img_B = Image.open(img_paths[1])
    # plt.subplot(2, 2, 2)
    # plt.imshow(img_B)
    # plt.title("T2")
    # plt.axis("off")

    # # drawing mask
    # plt.subplot(2, 2, 3)
    # plt.imshow(transferred_img1, cmap="gray")
    # plt.title("mask")
    # plt.axis("off")

    # # drawing label
    # plt.subplot(2, 2, 4)
    # plt.imshow(transferred_img2, cmap="gray")
    # plt.title("label")
    # plt.axis("off")

    # # show the plot
    # plt.tight_layout()
    # plt.show()
