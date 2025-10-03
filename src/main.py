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

    system_prompt_path = f"/data/{args.system_prompt}"
    user_prompt_path = f"/data/{args.user_prompt}"
    image_dir = "/data/images/"

    # Process images
    img_paths = load_images(image_dir)

    # Load prompts
    with open(system_prompt_path, "r") as f:
        system_prompt_template = f.read()

    with open(user_prompt_path, "r") as f:
        user_prompt_template = f.read()

    if args.subject_name:
        subject_name_clause = f' using the name "{args.subject_name}"'
        subject_name_pose_clause = f' using the name "{args.subject_name}" where appropriate'
        subject_name_pronoun_clause = (
            f'When referring to the subject, always use the name "{args.subject_name}". '
            'Avoid the use of pronouns like "he", "she", or "they" where appropriate. Repeat the name when needed.'
        )
    else:
        subject_name_clause = ''
        subject_name_pose_clause = ''
        subject_name_pronoun_clause = ''

    system_prompt = system_prompt_template.format(
        subject_name_pronoun_clause=subject_name_pronoun_clause,
    )
    user_prompt = user_prompt_template.format(
        subject_name_clause=subject_name_clause,
        subject_name_pose_clause=subject_name_pose_clause,
    )
    if args.content_hint:
        user_prompt = f"{user_prompt}\nContent hint: {args.content_hint}"

    logging.info(f"System prompt: {system_prompt}")
    logging.info(f"User prompt: {user_prompt}")

    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
        logging.info("Logging in to the Hugging Face Hub")
        huggingface_hub.login(token=os.environ["HF_TOKEN"])

    # Initialize model
    if args.model.startswith("blip2"):
        model = vlm_models.Blip2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("deepseek-vl2"):
        from vlm_models import deepseekvl2

        model = deepseekvl2.DeepSeekVL2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("instructblip"):
        model = vlm_models.InstructBlipModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("internvl2"):
        model = vlm_models.InternVL2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("internvl3"):
        model = vlm_models.InternVL3Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
            thinking=args.thinking,
        )
    elif args.model.startswith("joycaption"):
        model = vlm_models.JoyCaptionModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("kimi-vl"):
        # model = vlm_models.Kimi_VL_Model(
        #     checkpoint=args.model,
        #     system_prompt=system_prompt,
        #     prompt=user_prompt,
        #     quantize=args.quantize,
        # )
        raise Exception("computer says no")
    elif args.model.startswith("minicpm"):
        model = vlm_models.MiniCPM_V_2_6(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("ovis1.6"):
        from vlm_models import Ovis1_6Model
        
        model = vlm_models.Ovis1_6Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("ovis2"):
        '''
        model = vlm_models.Ovis2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
        '''
    elif args.model.startswith("paligemma2"):
        from vlm_models import pali_gemma2

        model = pali_gemma2.PaliGemma2Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("phi"):
        model = vlm_models.PhiModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("qwen2.5"):
        model = vlm_models.Qwen2_5VLModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("revisual-r1"):
        model = vlm_models.RevisualR1Model(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("wepoints"):
        from vlm_models import wepoints

        model = wepoints.WePOINTSModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
        )
    elif args.model.startswith("yannqi-r"):
        model = vlm_models.YannQiRModel(
            checkpoint=args.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            quantize=args.quantize,
            thinking=args.thinking,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    output_base_dir = "/data/output/"
    output_dir = output_base_dir + model.checkpoint_name()
    create_output_dir(output_dir)

    batch = []
    batch_output_paths = []
    batch_processing = args.batch_size > 1

    i = 0
    for img_path in img_paths:
        i = i + 1
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        if os.path.exists(output_path):
            logging.info(f"Skipping {img_path}, output already exists")
            continue

        if batch_processing:
            batch.append(img_path)
            batch_output_paths.append(output_path)
            # TODO: This will miss the last batch if the number of images is not a multiple of the batch size
            if len(batch) < args.batch_size:
                continue

        start_time = datetime.now()
        if batch_processing:
            try:
                responses = model.batch_caption_images(batch)
                for i, response in enumerate(responses):
                    response = response.replace("\n", " ") if args.replace_newlines else response
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
                    response = response.replace("\n", " ") if args.replace_newlines else response
                    logging.info(f"Processed {img_path} : {response}")
                    f.write(response)
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                traceback.print_exc()

        end_time = datetime.now()
        execution_time = end_time - start_time
        progress = (100 * i / float(len(img_paths)))
        if batch_processing:
            logging.info(
                f"[{progress}%] Processed batch in {execution_time.total_seconds()}s: {batch}"
            )
        else:
            logging.info(
                f"[{progress}%] Processed image in {execution_time.total_seconds()}s: {img_path}"
            )

        batch = []
        batch_output_paths = []


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="internvl3", help="Model type to use"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model when loading"
    )
    # Note: paths relative to data/
    parser.add_argument(
        "--system_prompt", type=str, default="system_prompt.txt", help="System prompt"
    )
    parser.add_argument(
        "--user_prompt", type=str, default="query.txt", help="User prompt"
    )
    parser.add_argument(
        "--subject_name", type=str, default="", help="Subject name"
    )
    parser.add_argument(
        "--content_hint", type=str, default="", help="Content hint for the captioner"
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Use thinking (InternVL3.5, YannQi/R)"
    )
    parser.add_argument(
        "--replace_newlines", action="store_true", help="Replace newline characters in output captions with ' '"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
