from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    BitsAndBytesConfig,
)
import torch
from PIL import Image
import requests
import os
import imghdr

checkpoint = "Salesforce/instructblip-vicuna-7b"
# checkpoint = "Salesforce/instructblip-vicuna-13b" # seems to hallucinate extra people into the scene
# checkpoint = "Salesforce/instructblip-flan-t5-xl" # seems to hallucinate handbags into the scene
# checkpoint = "Salesforce/instructblip-flan-t5-xxl" # gibberish
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = InstructBlipForConditionalGeneration.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    device_map="auto",
)
processor = InstructBlipProcessor.from_pretrained(checkpoint)

device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
model.eval()

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
    output_path = f"{output_dir}/{img_path.split('/')[-1].split('.')[0]}.txt"

    # If the output file already exists, skip the image
    if os.path.exists(output_path):
        continue

    # url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)
