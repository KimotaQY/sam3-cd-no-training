#!/usr/bin/env python
"""批量运行多个模型的训练"""

import subprocess
import sys
import os

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_model_training(
    dataset_name: str,
    iou_threshold=0.5,
    score_threshold_detection: float = 0.5,
    new_det_thresh: float = 0.7,
    model_type: str = "baseline",
    mixed_methods: str = "color_transfer",
    use_decoupled_selection: bool = False,
):
    """运行单个模型的训练"""
    cmd = [
        "torchrun",
        "inference.py",
        "--dataset-name",
        dataset_name,
        "--iou-threshold",
        str(iou_threshold),
        "--score-threshold-detection",
        str(score_threshold_detection),
        "--new-det-thresh",
        str(new_det_thresh),
        "--model-type",
        model_type,
        "--mixed-methods",
        mixed_methods,
        "--use-decoupled-selection",
        str(use_decoupled_selection),
    ]

    print(f"开始训练模型: {model_type}")
    print(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"模型 {model_type} 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"模型 {model_type} 训练失败，错误码: {e.returncode}")
        return False


if __name__ == "__main__":
    model_type = "baseline_bi_ssccev2"
    dataset_name = "WHU-CD"
    iou_thresholds = [0.5]
    score_threshold_detections = [
        # 0.1,
        # 0.15,
        # 0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        # 0.55,
        # 0.6,
        # 0.65,
        # 0.7,
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
    for iou_threshold in iou_thresholds:
        for score_threshold_detection in score_threshold_detections:
            for new_det_thresh in new_det_threshs:
                if new_det_thresh < score_threshold_detection:
                    continue
                run_model_training(
                    dataset_name=dataset_name,
                    iou_threshold=iou_threshold,
                    score_threshold_detection=score_threshold_detection,
                    new_det_thresh=new_det_thresh,
                    model_type=model_type,
                    mixed_methods="color_transfer",
                    use_decoupled_selection=False,
                )
