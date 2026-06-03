import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch

from mmseg.evaluation.metrics import IoUMetric
from mmengine.logging import MMLogger


def evaluate_binary_cd_folder(pred_dir, gt_dir, img_suffix=".png", gt_suffix=".png"):
    """
    使用 MMSegmentation 1.x 的 IoUMetric 评估二值变化检测结果

    Args:
        pred_dir: 预测结果文件夹路径
        gt_dir:  ground truth 文件夹路径
        img_suffix: 预测文件后缀
        gt_suffix: ground truth 文件后缀

    Returns:
        dict: 包含 mIoU, mFscore, aAcc, mAcc 等指标
    """
    # 初始化 logger
    logger = MMLogger.get_instance("evaluator")

    # 【新版变更 1】：必须显式传入 dataset_meta（包含类名和调色板），否则 1.x 无法打印和计算多类别的平均值
    # 对于二分类，classes 分别是背景和变化区域
    dataset_meta = {
        "classes": ("background", "change"),
        "palette": [[0, 0, 0], [255, 255, 255]],
    }

    # 初始化 IoUMetric
    metric = IoUMetric(
        iou_metrics=["mIoU", "mFscore"],
        collect_device="cpu",
        output_class_wise=True,  # 允许输出每个类别的独立得分
    )

    metric.dataset_meta = dataset_meta

    # 获取所有预测文件
    pred_files = sorted(Path(pred_dir).glob(f"*{img_suffix}"))

    if not pred_files:
        raise ValueError(f"在 {pred_dir} 中未找到预测文件")

    logger.info(f"找到 {len(pred_files)} 个预测文件")

    # 记录有效处理的样本数
    processed_count = 0

    # 逐对加载并处理
    for pred_path in pred_files:
        # 构建对应的 GT 路径
        gt_filename = pred_path.name.replace(img_suffix, gt_suffix)
        gt_path = Path(gt_dir) / gt_filename

        if not gt_path.exists():
            logger.warning(f"GT 文件不存在: {gt_path}，跳过")
            continue

        # 读取图像
        pred_img = np.array(Image.open(pred_path).convert("L"))
        gt_img = np.array(Image.open(gt_path).convert("L"))

        # 确保是二值图像 (0 和 1)
        pred_img = (pred_img > 0).astype(np.int64)
        gt_img = (gt_img > 0).astype(np.int64)

        # 检查尺寸是否匹配
        if pred_img.shape != gt_img.shape:
            logger.warning(
                f"尺寸不匹配: {pred_path.name}, pred={pred_img.shape}, gt={gt_img.shape}，跳过"
            )
            continue

        # 转换为 MMSegmentation 1.x 需要的格式
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample
        import torch

        data_sample = SegDataSample()

        # 【新版变更 2】：1.x 的 pred_sem_seg 的 data 形状必须与 gt_sem_seg 一致
        # 直接使用类别索引矩阵 (1, H, W) 或 (H, W)，不能再用双通道的概率图
        # 我们这里统一使用 (1, H, W) 形状，最符合 OpenMMLab 官方格式
        # 【关键修复】：将 numpy 数组转换为 PyTorch Tensor
        pred_label = torch.from_numpy(
            pred_img[np.newaxis, ...]
        )  # 形状变为 (1, H, W)，转换为 Tensor
        gt_label = torch.from_numpy(
            gt_img[np.newaxis, ...]
        )  # 形状变为 (1, H, W)，转换为 Tensor

        data_sample.pred_sem_seg = PixelData(data=pred_label)
        data_sample.gt_sem_seg = PixelData(data=gt_label)

        # 添加到 metric，【新版变更 3】：将 SegDataSample 转换为字典格式
        # 注意：需要确保字典中的 Tensor 保持为 Tensor 格式，不要转成 numpy
        data_sample_dict = {
            "pred_sem_seg": {"data": pred_label},
            "gt_sem_seg": {"data": gt_label},
        }
        metric.process([None], [data_sample_dict])
        processed_count += 1

    # 【新版变更 4】：1.x 的 evaluate 已经不再需要传入总样本数参数，它会自动计算内部缓存
    results = metric.evaluate(processed_count)

    return results


def evaluate_binary_cd_folder_fast(
    pred_dir, gt_dir, img_suffix=".png", gt_suffix=".png", device="cuda"
):
    """
    High-speed binary change detection evaluation using PyTorch GPU acceleration.
    Includes IoU, F1-Score, Precision, Recall, and Accuracy.
    """
    pred_files = sorted(Path(pred_dir).glob(f"*{img_suffix}"))
    if not pred_files:
        raise ValueError(f"No prediction files found in {pred_dir}")

    # Initialize a global confusion matrix on the selected device (GPU)
    # [[TN, FP],
    #  [FN, TP]]
    total_cm = torch.zeros((2, 2), dtype=torch.int64, device=device)

    # Use torch.no_grad() to save memory and prevent gradient tracking
    with torch.no_grad():
        for pred_path in pred_files:
            gt_filename = pred_path.name.replace(img_suffix, gt_suffix)
            gt_path = Path(gt_dir) / gt_filename

            if not gt_path.exists():
                continue

            # 1. Load images using PIL and convert to numpy
            pred_np = np.array(Image.open(pred_path).convert("L"))
            gt_np = np.array(Image.open(gt_path).convert("L"))

            if pred_np.shape != gt_np.shape:
                continue

            # 2. Push to GPU immediately and binarize
            pred_tensor = (torch.from_numpy(pred_np).to(device) > 0).long()
            gt_tensor = (torch.from_numpy(gt_np).to(device) > 0).long()

            # 3. Fast vectorized calculation of True/False Positives/Negatives
            tp = torch.sum((pred_tensor == 1) & (gt_tensor == 1))
            fp = torch.sum((pred_tensor == 1) & (gt_tensor == 0))
            fn = torch.sum((pred_tensor == 0) & (gt_tensor == 1))
            tn = torch.sum((pred_tensor == 0) & (gt_tensor == 0))

            # Accumulate into the total confusion matrix
            total_cm[0, 0] += tn
            total_cm[0, 1] += fp
            total_cm[1, 0] += fn
            total_cm[1, 1] += tp

    # Extract final scalar counts back to CPU for metrics generation
    tn, fp, fn, tp = total_cm.flatten().cpu().numpy().astype(float)

    # 4. Calculate all requested metrics for the 'change' class (Class 1)
    eps = 1e-7  # Prevent division by zero

    iou_change = tp / (tp + fp + fn + eps)
    precision_change = tp / (tp + fp + eps)
    recall_change = tp / (tp + fn + eps)
    fscore_change = (2 * precision_change * recall_change) / (
        precision_change + recall_change + eps
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "IoU.change": float(iou_change),
        "Fscore.change": float(fscore_change),
        "Precision.change": float(precision_change),
        "Recall.change": float(recall_change),
        "Accuracy.global": float(accuracy),
    }


def main():
    import argparse

    score_threshold_detections = [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
    ]
    new_det_threshs = [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
    ]
    for score_threshold_detection in score_threshold_detections:
        for new_det_thresh in new_det_threshs:
            if new_det_thresh < score_threshold_detection:
                continue
            # pred_dir = rf"logs/yyyjvm/SECOND/baseline_bi_ssccev2/generate_mixed[color_transfer]_iou0.5_thresh({score_threshold_detection},{new_det_thresh})_[['playground', 'football court', 'basketball court', 'baseball court']]/automatic"

            # pred_dir = r"logs/LEVIR-CD_256/baseline_bi_ssccev2/generate_mixed[color_transfer]_iou0.5_thresh(0.3,0.5)_[['roof']]/automatic"
            # pred_dir = r"logs/LEVIR-CD/baseline_bi_ssccev2/generate_mid1_-1_iou0.5_thresh(0.25,0.25)_[['roof']]/automatic"

            pred_dir = rf"logs/WHU-CD/baseline_bi_ssccev2/generate_mixed[color_transfer]_iou0.5_thresh({score_threshold_detection},{new_det_thresh})_[['roof']]/automatic"

            # pred_dir = r"logs/SECOND/baseline_bi_ssccev2/generate_mixed[color_transfer]_iou0.5_thresh(0.3,0.3)_[['football court', 'basketball court', 'baseball court']]/automatic"
            # pred_dir = r"logs/SECOND/baseline_bi_ssccev2/generate_mid1_-1_iou0.5_thresh(0.3,0.3)_[['non-vegetated ground', 'surface', 'tree', 'low vegetation', 'water', 'building', 'playground']]/automatic"
            # pred_dir = r"logs/LEVIR-CD/OmniOVCD/automatic"
            # pred_dir = r"logs/WHU-CD_256/OmniOVCD/automatic"
            # pred_dir = r"logs/SECOND/OmniOVCD_water/automatic"

            # SCM 的预测结果路径示例
            # pred_dir = r"/home/qy/CD_projects/UCD-SCM/results/samples_WHU-CD/dis"

            # gt_dir = r"/home/qy/CD_datasets/WHU-CD/256/test/label"
            gt_dir = r"/home/qy/CD_datasets/WHU-CD/test/label"
            # gt_dir = r"/home/qy/CD_datasets/SECOND/test/playground_label"
            # gt_dir = r"/home/qy/CD_datasets/SECOND/test/label"

            if not os.path.exists(pred_dir):
                print(f"{pred_dir} 不存在")
                continue

            parser = argparse.ArgumentParser(description="评估二值变化检测结果")
            parser.add_argument(
                "--pred_dir",
                type=str,
                default=pred_dir,
                help="预测结果文件夹路径",
            )
            parser.add_argument(
                "--gt_dir",
                type=str,
                default=gt_dir,
                help="Ground truth 文件夹路径",
            )
            parser.add_argument(
                "--pred_suffix",
                type=str,
                default=".png",
                help="预测文件后缀 (默认: .png)",
            )
            parser.add_argument(
                "--gt_suffix", type=str, default=".png", help="GT 文件后缀 (默认: .png)"
            )
            parser.add_argument(
                "--output",
                type=str,
                default="eval_results.txt",
                help="结果输出文件 (默认: eval_results.txt)",
            )

            args = parser.parse_args()

            # 执行评估
            print(f"开始评估...")
            print(f"预测目录: {args.pred_dir}")
            print(f"GT 目录: {args.gt_dir}")

            # results = evaluate_binary_cd_folder(
            #     pred_dir=args.pred_dir,
            #     gt_dir=args.gt_dir,
            #     img_suffix=args.pred_suffix,
            #     gt_suffix=args.gt_suffix,
            # )

            # results = evaluate_binary_cd_folder_torchmetrics(
            #     pred_dir=args.pred_dir,
            #     gt_dir=args.gt_dir,
            #     img_suffix=args.pred_suffix,
            #     gt_suffix=args.gt_suffix,
            # )

            results = evaluate_binary_cd_folder_fast(
                pred_dir=args.pred_dir,
                gt_dir=args.gt_dir,
                img_suffix=args.pred_suffix,
                gt_suffix=args.gt_suffix,
            )

            # 打印结果
            print("\n" + "=" * 50)
            print("评估结果:")
            print("=" * 50)
            for key, value in results.items():
                print(f"{key}: {value:.4f}")
            print("=" * 50)

            # 保存到文件
            with open(args.output, "a") as f:
                f.write("变化检测评估结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"预测目录: {args.pred_dir}\n")
                f.write(f"GT 目录: {args.gt_dir}\n")
                f.write("=" * 50 + "\n")
                for key, value in results.items():
                    f.write(f"{key}: {value:.4f}\n")

            print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
