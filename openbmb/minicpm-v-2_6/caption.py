from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import os
import imghdr

checkpoint = "openbmb/MiniCPM-V-2_6"
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=[
        "vpm",
        "resampler",
    ],
)
model = AutoModel.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

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

    msgs = [{'role': 'user', 'content': [image, query]}]
    response = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)
