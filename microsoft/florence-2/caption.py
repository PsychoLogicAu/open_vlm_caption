import os
import imghdr
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

checkpoint = "microsoft/Florence-2-large"
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
).to(device)
processor = AutoProcessor.from_pretrained(
    checkpoint,
    trust_remote_code=True,
)

# Write the model structure to a file
model_structure_file = f"/data/debug/{checkpoint}/model_structure.txt"
os.makedirs(os.path.dirname(model_structure_file), exist_ok=True)
with open(model_structure_file, "w") as file:
    file.write(str(model))

# prompt = "<CAPTION>" # Captioning
# prompt = "<DETAILED_CAPTION>" 
prompt = "<MORE_DETAILED_CAPTION>"

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
        print(f"Caption file {output_path} already exists, skipping")
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

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        device, torch_dtype
    )

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    response = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )[prompt]

    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)
