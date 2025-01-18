import os

import logging
import vlm_models
from datetime import datetime
import traceback

import imghdr
import huggingface_hub


def load_images(image_dir):
    img_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if imghdr.what(file_path) is not None:
                img_paths.append(file_path)
    img_paths.sort()
    logging.info(f"Found {len(img_paths)} images")
    return img_paths


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def main(args):
    logging.basicConfig(level=logging.INFO)

    system_prompt_path = "/data/system_prompt.txt"
    query_path = "/data/query.txt"
    image_dir = "/data/images/"

    # Process images
    img_paths = load_images(image_dir)

    # Load prompts
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(query_path, "r") as f:
        query = f.read()
    logging.info(f"Query: {query}")

    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
        logging.info("Logging in to the Hugging Face Hub")
        huggingface_hub.login(token=os.environ["HF_TOKEN"])

    # Initialize model
    if args.model in ["minicpm-v-2_6", "minicpm-o-2_6"]:
        model = vlm_models.MiniCPM_V_2_6(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    elif args.model.startswith("internvl2"):
        model = vlm_models.InternVL2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    elif args.model.startswith("ovis1.6"):
        model = vlm_models.Ovis1_6Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    elif args.model.startswith("instructblip"):
        model = vlm_models.InstructBlipModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    elif args.model.startswith("blip2"):
        model = vlm_models.Blip2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    elif args.model.startswith("paligemma2"):
        model = vlm_models.PaliGemma2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    elif args.model.startswith("wepoints"):
        model = vlm_models.WePOINTSModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    elif args.model.startswith("joycaption"):
        model = vlm_models.JoyCaptionModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=query,
            quantize=args.quantize,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    output_base_dir = "/data/output/"
    output_dir = output_base_dir + model.checkpoint_name()
    create_output_dir(output_dir)

    batch = []
    batch_output_paths = []
    for img_path in img_paths:
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        if os.path.exists(output_path):
            logging.info(f"Skipping {img_path}, output already exists")
            continue

        if args.batch_size > 1:
            batch.append(img_path)
            batch_output_paths.append(output_path)
            # TODO: This will miss the last batch if the number of images is not a multiple of the batch size
            if len(batch) < args.batch_size:
                continue

        start_time = datetime.now()
        if args.batch_size > 1:
            try:
                responses = model.batch_caption_images(batch)
                for i, response in enumerate(responses):
                    logging.info(f"Processed {batch[i]} : {response}")
                    with open(batch_output_paths[i], "w") as f:
                        f.write(response)
            except Exception as e:
                logging.error(f"Error processing batch {batch}: {e}")
                traceback.print_exc()
        else:
            try:
                response = model.caption_image(img_path)
                with open(output_path, "w") as f:
                    logging.info(f"Processed {img_path} : {response}")
                    f.write(response)
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                traceback.print_exc()

        end_time = datetime.now()
        execution_time = end_time - start_time
        if args.batch_size > 1:
            logging.info(
                f"Processed batch in {execution_time.total_seconds()}s: {batch}"
            )
        else:
            logging.info(
                f"Processed image in {execution_time.total_seconds()}s: {img_path}"
            )

        batch = []
        batch_output_paths = []


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="minicpm-v-2_6", help="Model type to use"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model when loading"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
