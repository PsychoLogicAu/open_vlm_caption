import os

import logging
import vlm_models

import imghdr
import huggingface_hub


def load_images(image_dir):
    img_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if imghdr.what(file_path) is not None:
                img_paths.append(file_path)
    logging.info(f"Found {len(img_paths)} images")
    return img_paths


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def main(args):
    logging.basicConfig(level=logging.INFO)

    query_path = "/data/query.txt"
    image_dir = "/data/images/"

    # Process images
    img_paths = load_images(image_dir)

    # Load query
    with open(query_path, "r") as f:
        query = f.read()
    logging.info(f"Query: {query}")

    if "HF_TOKEN" in os.environ:
        logging.info("Logging in to the Hugging Face Hub")
        huggingface_hub.login(token=os.environ["HF_TOKEN"])

    # Initialize model
    if args.model == "minicpm-v-2_6":
        model = vlm_models.MiniCPM_V_2_6(query=query, quantize=args.quantize)
    elif args.model.startswith("internvl2"):
        model = vlm_models.InternVL2Model(checkpoint=args.model, query=query, quantize=args.quantize)
    elif args.model.startswith("ovis1.6"):
        model = vlm_models.Ovis1_6Model(checkpoint=args.model, query=query, quantize=args.quantize)
    elif args.model.startswith("instructblip"):
        model = vlm_models.InstructBlipModel(checkpoint=args.model, query=query, quantize=args.quantize)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    output_base_dir = "/data/output/"
    output_dir = output_base_dir + model.checkpoint_name()
    create_output_dir(output_dir)

    for img_path in img_paths:
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        if os.path.exists(output_path):
            logging.info(f"Skipping {img_path}, output already exists")
            continue

        try:
            response = model.caption_image(img_path)
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            continue

        logging.info(f"Processed {img_path}: {response}")
        
        with open(output_path, "w") as f:
            f.write(response)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model when loading"
    )
    parser.add_argument(
        "--model", type=str, default="minicpm-v-2_6", help="Model type to use"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
