import os
import shutil
import cv2
import numpy as np
from PIL import Image


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


def gen_frame(folder_paths, output_dir="_tmp_jpg", sort="asc", mid_frame=0):
    """
    Convert PNG format image files to JPEG format and optionally generate intermediate interpolated frames

    This function iterates through the input folder path list, converts PNG images to JPEG format,
    and generates intermediate interpolated frames as needed. The main processing includes
    image format conversion (RGBA/LA to RGB), file renaming, and color interpolation.

    Parameters:
        folder_paths (list): List of paths to PNG image files
        output_dir (str): Directory path for output JPEG images, defaults to "_tmp_jpg"
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
        if mid_frame > 0 and idx > 0:
            continue
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

                # import matplotlib.pyplot as plt

                # # create a figure that can hold three subplots
                # plt.figure(figsize=(15, 10))  # set the figure size

                # # drawing img_A
                # # img_A = cv2.imread(img_paths[0])
                # img_A = Image.open(first_frame)
                # plt.subplot(2, 2, 1)
                # plt.imshow(img_A)
                # plt.title("T1")
                # plt.axis("off")

                # # drawing img_B
                # # img_B = cv2.imread(img_paths[1])
                # img_B = Image.open(final_frame)
                # plt.subplot(2, 2, 2)
                # plt.imshow(img_B)
                # plt.title("T2")
                # plt.axis("off")

                # # drawing mask
                # plt.subplot(2, 2, 3)
                # plt.imshow(img, cmap="gray")
                # plt.title("mask")
                # plt.axis("off")

                # # drawing label
                # plt.subplot(2, 2, 4)
                # plt.imshow(img, cmap="gray")
                # plt.title("label")
                # plt.axis("off")

                # # show the plot
                # plt.tight_layout()
                # plt.show()

                # Save as JPEG
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, img_bgr)
                # img.save(output_path, "JPEG", quality=100)
                print(
                    f"Intermediate frame generated: {alpha} -> {os.path.basename(output_path)}"
                )
        except Exception as e:
            print(f"Intermediate frame generation failed {alpha}: {str(e)}")

    return output_dir
