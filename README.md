# open_vlm_caption

Docker compose project for captioning images with VLMs

## How to use (internvl2-8b example)
```
# Build the container
docker compose build internvl2-8b

# Run 
docker compose run --rm --remove-orphans internvl2-8b
```

## More examples
```
docker compose run --rm --remove-orphans vlm-caption --model=yannqi-r --thinking --content_hint="selfie photo" --system_prompt=system_prompt_example.txt --user_prompt=user_prompt_example.txt --replace_newlines

docker compose run --rm --remove-orphans vlm-caption --model=joycaption --quantize --system_prompt=system_prompt_basic.txt --user_prompt=user_prompt_image_quality.txt --replace_newlines
```

## Huggingface tokens
If the model is gated and a token is required to download or access the git repository, this can be specified
- at build time as a build argument, e.g. `--build-arg HF_TOKEN=<insert token here>`
- at run time `export HF_TOKEN=<insert token here> && docker compose run --rm --remove-orphans internvl2-8b`

## Notes
I originally had a lowly 16GB GPU, so models were selected and adjusted to fit into this, e.g. by loading with 8 bit quantization. I now have a 5090 so larger models are being added.

## TODO:
- refactor, hide model implementations behind an interface
- reduce common code, e.g. image & caption loading
- parameters for quantization
- ComfyUI node to wrap
- more VLMs
- buy a bigger GPU!

If you can help with any of those, please submit a PR, or https://buymeacoffee.com/psychologic
