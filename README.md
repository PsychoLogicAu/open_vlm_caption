# open_vlm_caption

Docker compose project for captioning images with VLMs

## How to use (internvl2-8b example)
```
# Build the container
docker compose build internvl2-8b

# Run 
docker compose run --rm --remove-orphans internvl2-8b
```

## Huggingface tokens
If the model is gated and a token is required to download or access the git repository, this can be specified
- at build time as a build argument, e.g. `--build-arg HF_TOKEN=<insert token here>`
- at run time `export HF_TOKEN=<insert token here> && docker compose run --rm --remove-orphans internvl2-8b`

## Notes
I have a lowly 16GB GPU, so examples have been adjusted to fit into this, e.g. by loading with 8 bit quantization.

## TODO:
- refactor, hide model implementations behind an interface
- reduce common code, e.g. image & caption loading
- parameters for quantization
- ComfyUI node to wrap
- more VLMs
- buy a bigger GPU!

If you can help with any of those, please submit a PR, or https://buymeacoffee.com/psychologic
