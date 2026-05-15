from PIL import Image
import os

# pro_folders = ['train', 'val', 'test']
# sub_folders = ['A', 'B', 'label']

# # 源、目标文件夹名称
# src_folder = "/home/yyyjvm/CD_datasets/LEVIR-CD"
# dst_folder = "/home/yyyjvm/CD_datasets/LEVIR-CD/256"

# # 列出目录中的所有项，包括文件和子目录
# items = os.listdir(src_folder)

# # 读出源文件夹子目录
# subdirs = [d for d in items if os.path.isdir(os.path.join(src_folder, d)) and d in pro_folders]

# print(subdirs)

# for subdir in subdirs:
#     # 读出子目录中的数据集文件夹
#     dir = os.path.join(src_folder, subdir)
#     items = os.listdir(dir)
#     data_dirs = [d for d in items if os.path.isdir(os.path.join(dir, d)) and d in sub_folders]
#     print(data_dirs)
#     for data_dir in data_dirs:
#         s_folder = os.path.join(src_folder, f"{subdir}/{data_dir}")
#         d_folder = os.path.join(dst_folder, f"{subdir}/{data_dir}")

#         # 如果目标文件夹不存在则创建
#         if not os.path.exists(d_folder):
#             os.makedirs(d_folder)

#         for img_file in os.listdir(s_folder):
#             if img_file.endswith(".png"):
#                 img_path = os.path.join(s_folder, img_file)
#                 print(f"正在处理：{img_file}")
#                 img = Image.open(img_path)
#                 # 开始分割图片
#                 for i in range(0, img.width, 256):
#                     for j in range(0, img.height, 256):
#                         box = (i, j, i + 256, j + 256)
#                         cropped_img = img.crop(box)
#                         filename, ext = os.path.splitext(img_file)
#                         new_file = (
#                             filename + "_{0}_{1}".format(int(i / 256), int(j / 256)) + ext
#                         )
#                         save_path = os.path.join(d_folder, new_file)
#                         cropped_img.save(save_path)


def split_img(src_folder, dst_folder):
    # 如果目标文件夹不存在则创建
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for img_file in os.listdir(src_folder):
        if img_file.endswith(".png"):
            img_path = os.path.join(src_folder, img_file)
            print(f"正在处理：{img_file}")
            img = Image.open(img_path)
            # 开始分割图片
            for i in range(0, img.width, 256):
                for j in range(0, img.height, 256):
                    box = (i, j, i + 256, j + 256)
                    cropped_img = img.crop(box)
                    filename, ext = os.path.splitext(img_file)
                    new_file = (
                        filename + "_{0}_{1}".format(int(i / 256), int(j / 256)) + ext
                    )
                    save_path = os.path.join(dst_folder, new_file)
                    cropped_img.save(save_path)


if __name__ == "__main__":
    src_folder = "logs/LEVIR-CD/baseline_bi_ssccev2/generate_mid1_-1_iou0.5_thresh(0.25,0.25)_[['roof']]/automatic_confusion_matrix"
    dst_folder = src_folder + "_256"
    split_img(src_folder, dst_folder)
