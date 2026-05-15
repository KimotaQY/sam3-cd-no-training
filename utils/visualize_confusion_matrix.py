import os

import cv2
import numpy as np


def visualize_confusion_matrix(pred_dir: str, label_dir: str, output_dir: str):
    """
    Visualize confusion matrix (TP, FP, FN, TN) from prediction and label masks.

    Color scheme:
    - White: True Positives (TP)
    - Black: True Negatives (TN)
    - Red: False Positives (FP)
    - Green: False Negatives (FN)

    Args:
        pred_dir: Directory containing prediction mask images
        label_dir: Directory containing ground truth label images
        output_dir: Directory to save visualization results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    pred_files = [
        f
        for f in os.listdir(pred_dir)
        if os.path.splitext(f)[-1] in [".png", ".jpg", ".jpeg"]
    ]

    print(f"Found {len(pred_files)} images to process")

    for idx, filename in enumerate(pred_files):
        print(f"Processing {idx+1}/{len(pred_files)}: {filename}")

        # Load prediction mask
        pred_path = os.path.join(pred_dir, filename)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if pred_mask is None:
            print(f"Warning: Could not read {pred_path}, skipping...")
            continue

        # Load label mask
        label_path = os.path.join(label_dir, filename)
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found {label_path}, skipping...")
            continue

        label_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if label_mask is None:
            print(f"Warning: Could not read {label_path}, skipping...")
            continue

        # Ensure same shape
        if pred_mask.shape != label_mask.shape:
            print(f"Warning: Shape mismatch for {filename}, resizing...")
            pred_mask = cv2.resize(
                pred_mask, (label_mask.shape[1], label_mask.shape[0])
            )

        # Binarize masks (threshold at 127)
        pred_binary = (pred_mask > 127).astype(np.uint8)
        label_binary = (label_mask > 127).astype(np.uint8)

        # Calculate confusion matrix components
        TP = (pred_binary == 1) & (label_binary == 1)  # True Positives
        TN = (pred_binary == 0) & (label_binary == 0)  # True Negatives
        FP = (pred_binary == 1) & (label_binary == 0)  # False Positives
        FN = (pred_binary == 0) & (label_binary == 1)  # False Negatives

        # Create RGB visualization image
        h, w = pred_binary.shape
        viz_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Apply colors according to specification
        # White for TP (255, 255, 255)
        viz_image[TP] = [255, 255, 255]

        # Black for TN (0, 0, 0) - already initialized as zeros

        # Red for FP (0, 0, 255) in BGR format for OpenCV
        viz_image[FP] = [0, 0, 255]

        # Green for FN (0, 255, 0) in BGR format for OpenCV
        viz_image[FN] = [0, 255, 0]

        # Save visualization
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, viz_image)

        # Print statistics
        total_pixels = h * w
        tp_count = np.sum(TP)
        tn_count = np.sum(TN)
        fp_count = np.sum(FP)
        fn_count = np.sum(FN)

        print(f"  TP: {tp_count} ({tp_count/total_pixels*100:.2f}%)")
        print(f"  TN: {tn_count} ({tn_count/total_pixels*100:.2f}%)")
        print(f"  FP: {fp_count} ({fp_count/total_pixels*100:.2f}%)")
        print(f"  FN: {fn_count} ({fn_count/total_pixels*100:.2f}%)")

    print(f"\nVisualization completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    pred_dir = "logs/LEVIR-CD/baseline_bi_ssccev2/generate_mid1_-1_iou0.5_thresh(0.25,0.25)_[['roof']]/automatic"
    label_dir = "/home/qy/CD_datasets/LEVIR-CD/test/label"
    output_dir = pred_dir + "_confusion_matrix"
    visualize_confusion_matrix(pred_dir, label_dir, output_dir)
