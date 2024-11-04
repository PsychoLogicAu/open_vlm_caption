import torch
from PIL import Image
import os
import imghdr
from ovis.serve.runner import RunnerArguments, OvisRunner

checkpoint = "AIDC-AI/Ovis1.6-Llama3.2-3B"
runner_args = RunnerArguments(model_path=checkpoint)
runner = OvisRunner(runner_args)

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

    image = Image.open(img_path)

    generation = runner.run([image, query])
    response = generation["output"]

    print(f"{img_path}: {response}")

    # Write the response to the output file
    with open(output_path, "w") as f:
        f.write(response)