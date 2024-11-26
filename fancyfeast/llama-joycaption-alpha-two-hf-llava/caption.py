import torch
import torch.amp
import torchvision.transforms.functional as TVF
from PIL import Image
from transformers import (AutoTokenizer, BitsAndBytesConfig,LlavaForConditionalGeneration)
import os
import imghdr

checkpoint = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=[
        # "vpm",
        # "resampler",
    ],
)

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    use_fast=True,
)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype="bfloat16",
    device_map="auto"
).eval()

# Write the model structure to a file
model_structure_file = f"/data/debug/{checkpoint}/model_structure.txt"
os.makedirs(os.path.dirname(model_structure_file), exist_ok=True)
with open(model_structure_file, "w") as file:
    file.write(str(llava_model))

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

    with torch.no_grad():
        if image.size != (384, 384):
            image = image.resize((384, 384), Image.LANCZOS)

        pixel_values = TVF.pil_to_tensor(image)

        # Normalize the image
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(torch.bfloat16).unsqueeze(0)

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        # Format the conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        # Tokenize the conversation
        convo_tokens = tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

        # Repeat the image tokens
        input_tokens = []
        for token in convo_tokens:
            if token == llava_model.config.image_token_index:
                input_tokens.extend([llava_model.config.image_token_index] * llava_model.config.image_seq_length)
            else:
                input_tokens.append(token)

        input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        # Generate the caption
        generate_ids = llava_model.generate(input_ids=input_ids.to('cuda'), pixel_values=pixel_values.to('cuda'), attention_mask=attention_mask.to('cuda'), max_new_tokens=300, do_sample=True, suppress_tokens=None, use_cache=True)[0]

        # Trim off the prompt
        generate_ids = generate_ids[input_ids.shape[1]:]

        # Decode the caption
        caption = tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = caption.strip()
        print(f"{img_path}: {response}")

        # Write the response to the output file
        with open(output_path, "w") as f:
            f.write(response)
