from PIL import Image
import requests
from typing import Dict
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import os
import imghdr

checkpoint = "Qwen/Qwen2-VL-7B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=[
        "visual",
    ],
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
).eval()
processor = AutoProcessor.from_pretrained(checkpoint)

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

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": query},
        ],
    }
]
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


# iterate over the files in "/data/images" directory, and add detected image paths to img_path_list
img_path_list = []
for root, dirs, files in os.walk("/data/images"):
    for file in files:
        file_path = os.path.join(root, file)
        if imghdr.what(file_path) is not None:
            img_path_list.append(file_path)
print(f"processing {len(img_path_list)} images")

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

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)
