def print_last_line_scores(txt_file_path):
    """
    读取txt文件最后一行的得分并以百分数形式打印(保留两位小数)

    Args:
        txt_file_path: txt文件路径
    """
    try:
        # 读取文件所有行
        with open(txt_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            print("文件为空")
            return

        # 获取最后一行
        last_line = lines[-1].strip()

        # 检查是否包含"平均值"关键字
        if "平均值" not in last_line:
            print(f"最后一行不包含得分信息: {last_line}")
            return

        # 提取各个指标的值
        import re

        iou_match = re.search(r"iou:\s*([\d.]+)", last_line)
        f1_match = re.search(r"f1:\s*([\d.]+)", last_line)
        pre_match = re.search(r"pre:\s*([\d.]+)", last_line)
        rec_match = re.search(r"rec:\s*([\d.]+)", last_line)
        acc_match = re.search(r"acc:\s*([\d.]+)", last_line)

        # 转换为百分数并格式化输出
        if all([iou_match, f1_match, pre_match, rec_match, acc_match]):
            iou = float(iou_match.group(1)) * 100
            f1 = float(f1_match.group(1)) * 100
            pre = float(pre_match.group(1)) * 100
            rec = float(rec_match.group(1)) * 100
            acc = float(acc_match.group(1)) * 100

            print(f"IoU: {iou:.2f}%")
            print(f"F1: {f1:.2f}%")
            print(f"Precision: {pre:.2f}%")
            print(f"Recall: {rec:.2f}%")
            print(f"Accuracy: {acc:.2f}%")
        else:
            print("无法解析得分信息")

    except FileNotFoundError:
        print(f"文件不存在: {txt_file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")


# 使用示例
if __name__ == "__main__":
    score_threshold_detections = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    new_det_threshs = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    for score_threshold_detection in score_threshold_detections:
        for new_det_thresh in new_det_threshs:
            if new_det_thresh < score_threshold_detection:
                continue

            print(
                f"===== score_threshold_detection: {score_threshold_detection}, new_det_thresh: {new_det_thresh} ====="
            )
            txt_path = f"./logs/WHU-CD/baseline_bi_ssccev2/generate_mixed[color_transfer]_iou0.5_thresh({score_threshold_detection},{new_det_thresh})_[['roof']]/automatic/log.txt"
            print_last_line_scores(txt_path)
