from transformers import (
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
)
import torch
from PIL import Image
import requests
import os
import imghdr

checkpoint = "omlab/omchat-v2.0-13B-single-beta_hf"
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=[
        # "multi_modal_projector",
        "vision_tower",
    ],
)
model = AutoModel.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    # torch_dtype=torch.float16,
).eval()
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

# Write the model structure to a file
model_structure_file = f"/data/debug/{checkpoint}/model_structure.txt"
os.makedirs(os.path.dirname(model_structure_file), exist_ok=True)
with open(model_structure_file, "w") as file:
    file.write(str(model))

# ---
# load query string from the text file /data/query.txt
with open("/data/query.txt", "r") as f:
    query = f"{f.read()}"

print("Query:", query)

# iterate over the files in "/data/images" directory, and add detected image paths to img_path_list
img_path_list = []
for root, dirs, files in os.walk("/data/images"):
    for file in files:
        file_path = os.path.join(root, file)
        if imghdr.what(file_path) is not None:
            img_path_list.append(file_path)

output_dir = f"/data/output/{checkpoint}"
# Create the output directory if required
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for img_path in img_path_list:
    # Get name of the output file, excluding the extension
    img_file_name = os.path.basename(img_path)
    output_path = f"{output_dir}/{os.path.splitext(img_file_name)[0]}.txt"

    # If the output file already exists, skip the image
    if os.path.exists(output_path):
        continue

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        error_dir = f"/data/error/"
        if not os.path.exists(error_dir):
            os.makedirs(error_dir)
        os.rename(img_path, f"{error_dir}/{img_file_name}")
        continue
    inputs = processor(text=query, images=image, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    response = processor.tokenizer.decode(
        output_ids[0, inputs.input_ids.shape[1] :]
    ).strip()

    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)
