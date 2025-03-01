import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)
import os
import imghdr

# checkpoint = "Salesforce/blip2-opt-6.7b-coco" # mostly blank output
# checkpoint = "Salesforce/blip2-opt-2.7b" # mostly blank output
# checkpoint = "Salesforce/blip2-opt-6.7b"  # mostly blank output
checkpoint = "Salesforce/blip2-flan-t5-xl"  # very short descriptions
# checkpoint = "Salesforce/blip2-flan-t5-xxl" # gibberish
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=[
        "vision_model",
        "qformer",
        "language_projection",
    ],
)
processor = Blip2Processor.from_pretrained(checkpoint)
model = Blip2ForConditionalGeneration.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    device_map="auto",
)

# Write the model structure to a file
model_structure_file = f"/data/debug/{checkpoint}/model_structure.txt"
os.makedirs(os.path.dirname(model_structure_file), exist_ok=True)
with open(model_structure_file, "w") as file:
    file.write(str(model))

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

    raw_image = Image.open(img_path).convert("RGB")
    inputs = processor(raw_image, query, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(
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
    response = processor.decode(out[0], skip_special_tokens=True)

    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)
