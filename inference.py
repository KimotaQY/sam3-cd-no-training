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

# 全局配置：自动检测GPU，设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备：{DEVICE}")

from sam3.model_builder import build_sam3_video_predictor
from sam3 import build_sam3_image_model

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from BiSAM import (
    Baseline,
    BiSAM2,
    Baseline_Bi,
    Baseline_Bi_SSCCE,
    Baseline_Bi_SSCCE_CSPCF,
)

from utils.metrics import (
    binary_accuracy,
    binary_accuracy_sklearn,
    binary_accuracy_torchmetrics,
    AverageMeter,
)

from predict import merge_overlapping_masks_gpu, merge_overlapping_masks_multiprocess


def inference(
    dataset_name: str,
    prompt_text_str: str | list,
    iou_threshold=0.5,
    score_threshold_detection: float = 0.5,
    new_det_thresh: float = 0.7,
    model_type: str = "baseline",
    mixed_methods: str = "color_transfer",
    **kwargs,
):
    if dataset_name is None:
        print("请输入数据集名称")
        return

    if dataset_name in ["WHU-CD", "LEVIR-CD"]:
        before_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/A"
        after_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/B"
        label_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/label"
    elif dataset_name == "SECOND":
        before_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/im1"
        after_img_dir = f"/home/qy/CD_datasets/{dataset_name}/test/im2"
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

    if model_type == "baseline":
        bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(
            bpe_path=bpe_path, checkpoint_path="/home/qy/weights/sam3-model/sam3.pt"
        )
        baseline = Baseline(
            model,
            confidence_threshold=score_threshold_detection,
        )

        output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection})_[{prompt_text_str}]/automatic"
    elif model_type == "baseline_bi":
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path="/home/qy/weights/sam3-model/sam3.pt",
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            # apply_temporal_disambiguation=False,
            use_decoupled_selection=kwargs.get("use_decoupled_selection", False),
        )

        baseline_bi = Baseline_Bi(predictor)
        if kwargs.get("use_decoupled_selection", False):
            output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic_use_decoupled_selection"
        else:
            output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"
    elif "baseline_bi_ssccev" in model_type:
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path="/home/qy/weights/sam3-model/sam3.pt",
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            # apply_temporal_disambiguation=False,
            use_decoupled_selection=kwargs.get("use_decoupled_selection", False),
        )

        baseline_bi_sscce = Baseline_Bi_SSCCE(
            predictor, iou_threshold=iou_threshold, mixed_methods=mixed_methods
        )

        if mixed_methods is None:
            if kwargs.get("use_decoupled_selection", False):
                output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic_use_decoupled_selection"
            else:
                output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"
        else:
            output_dir = f"./logs/{dataset_name}/{model_type}/generate_mixed[{mixed_methods}]_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"

    elif model_type == "baseline_bi_sscce_cspcf":
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path="/home/qy/weights/sam3-model/sam3.pt",
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            # apply_temporal_disambiguation=False,
            use_decoupled_selection=kwargs.get("use_decoupled_selection", False),
        )

        baseline_bi_sscce_cspcf = Baseline_Bi_SSCCE_CSPCF(
            predictor, iou_threshold=iou_threshold, mixed_methods=mixed_methods
        )

        if mixed_methods is None:
            if kwargs.get("use_decoupled_selection", False):
                output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic_use_decoupled_selection"
            else:
                output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"
        else:
            output_dir = f"./logs/{dataset_name}/{model_type}/generate_mixed[{mixed_methods}]_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"

    elif model_type == "bisam2":
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path="/home/qy/weights/sam3-model/sam3.pt",
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            use_decoupled_selection=kwargs.get("use_decoupled_selection", False),
        )
        bisam2 = BiSAM2(
            predictor=predictor,
            iou_threshold=iou_threshold,
            mixed_methods=mixed_methods,
        )

        if mixed_methods is None:
            output_dir = f"./logs/{dataset_name}/{model_type}/generate_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"
        else:
            output_dir = f"./logs/{dataset_name}/{model_type}/generate_mixed[{mixed_methods}]_iou{iou_threshold}_thresh({score_threshold_detection},{new_det_thresh})_[{prompt_text_str}]/automatic"

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

            if model_type == "baseline":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = baseline.step_one(img_paths, prompt_text_str)
            elif model_type == "baseline_bi":
                pred = baseline_bi.step_one(
                    img_paths,
                    prompt_text_str=prompt_text_str,
                )
            elif "baseline_bi_ssccev" in model_type:
                pred = baseline_bi_sscce.step_one(
                    img_paths,
                    prompt_text=prompt_text_str,
                    merge_mask_func_version=(
                        "v1" if model_type == "baseline_bi_ssccev1" else "v2"
                    ),
                )
                # pred转为numpy数组
                pred = pred.cpu().numpy().astype(np.uint8)

            elif model_type == "baseline_bi_sscce_cspcf":
                diff_mask_list = baseline_bi_sscce_cspcf.step_one(
                    img_paths,
                    prompt_text_str=prompt_text_str,
                )

                if DEVICE == torch.device("cuda"):
                    pred = merge_overlapping_masks_gpu(
                        *diff_mask_list, iou_threshold=iou_threshold
                    )
                else:
                    pred = merge_overlapping_masks_multiprocess(
                        *diff_mask_list, iou_threshold=iou_threshold
                    )
            elif model_type == "bisam2":
                diff_mask_list = bisam2.step_one(
                    img_paths,
                    prompt_text_str=prompt_text_str,
                )

                if DEVICE == torch.device("cuda"):
                    pred = merge_overlapping_masks_gpu(
                        *diff_mask_list, iou_threshold=iou_threshold
                    )
                else:
                    pred = merge_overlapping_masks_multiprocess(
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

            # iou = compute_mask_iou(pred, label_mask)

            acc, precision, recall, f1, iou = binary_accuracy(pred, label_mask)
            # acc, precision, recall, f1, iou = binary_accuracy_sklearn(
            #     pred, label_mask
            # )
            # acc, precision, recall, f1, iou = binary_accuracy_torchmetrics(
            #     pred, label_mask
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
            h, w = pred.shape[-2:]
            mask_image = pred.reshape(h, w, 1)
            cv2.imwrite(os.path.join(output_dir, img_name), mask_image * 255)

            # if "predictor" in locals():
            #     del predictor
            torch.cuda.empty_cache()
            gc.collect()

        try:
            print(
                f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
            )
            if (
                IoU_meter.avg is not None
                and F1_meter.avg is not None
                and Pre_meter.avg is not None
                and Rec_meter.avg is not None
                and Acc_meter.avg is not None
            ):
                f.write(
                    f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
                )
        except statistics.StatisticsError:
            print("列表为空，无法计算平均值")


def main(
    dataset_name: str,
    prompt_text_str: str | list,
    iou_threshold=0.5,
    score_threshold_detection: float = 0.5,
    new_det_thresh: float = 0.7,
    model_type: str = "baseline",
    mixed_methods: str = "color_transfer",
    use_decoupled_selection: bool = False,
):

    inference(
        dataset_name,
        prompt_text_str,
        iou_threshold=iou_threshold,
        score_threshold_detection=score_threshold_detection,
        new_det_thresh=new_det_thresh,
        model_type=model_type,
        mixed_methods=mixed_methods,
        use_decoupled_selection=use_decoupled_selection,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="WHU-CD",
        help="Name of the dataset to inference",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        help="Type of the model to inference",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="IoU threshold for inference"
    )
    parser.add_argument(
        "--score-threshold-detection",
        type=float,
        default=0.5,
        help="Score threshold for detection",
    )
    parser.add_argument(
        "--new-det-thresh",
        type=float,
        default=0.7,
        help="New detection threshold for inference",
    )
    parser.add_argument(
        "--mixed-methods",
        type=str,
        default="color_transfer",
        help="Mixed methods for inference",
    )
    parser.add_argument(
        "--use-decoupled-selection",
        type=bool,
        default=False,
        help="decoupled selection for inference",
    )

    args = parser.parse_args()

    # 如果提供了模型名称参数，使用它；否则使用默认值
    main(
        args.dataset_name,
        prompt_text_str=["roof"],
        iou_threshold=args.iou_threshold,
        score_threshold_detection=args.score_threshold_detection,
        new_det_thresh=args.new_det_thresh,
        model_type=args.model_type,
        mixed_methods=args.mixed_methods,
        use_decoupled_selection=args.use_decoupled_selection,
    )
