import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(
    bpe_path=bpe_path, checkpoint_path="/home/qy/weights/sam3-model/sam3.pt"
)

# print(model)

# image_path = f"{sam3_root}/assets/images/test_image.jpg"
image_path = f"/home/qy/CD_datasets/LEVIR-CD/test/B/test_43.png"
image = Image.open(image_path)
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)

processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="house")
inference_state = processor.set_text_prompt(state=inference_state, prompt="roof")
inference_state = processor.set_text_prompt(state=inference_state, prompt="building")

img0 = Image.open(image_path)
plot_results(img0, inference_state)
plt.imshow(img0)
plt.axis("off")  # Hide the axis
plt.show()

# # Here the box is in  (x,y,w,h) format, where (x,y) is the top left corner.
# box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
# box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

# norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
# print("Normalized box input:", norm_box_cxcywh)

# processor.reset_all_prompts(inference_state)
# inference_state = processor.add_geometric_prompt(
#     state=inference_state, box=norm_box_cxcywh, label=True
# )

# img0 = Image.open(image_path)
# image_with_box = draw_box_on_image(img0, box_input_xywh.flatten().tolist())
# plt.imshow(image_with_box)
# plt.axis("off")  # Hide the axis
# plt.show()
