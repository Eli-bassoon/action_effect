# action_effect

## Generating Captions

Use `./captioning/captioning.py` to generate captions. Change your directory into the same directory as the python file, then run the program to generate captions for all images using a specific model and prompt. The arguments are below:

- `--version` (default: "blip") the version of the captioning model used, either "blip" or "blip2"
- `--function_to_use` (default: 0) which captioning function to use, between 0-2
- `--save_file` (default: "captions.txt") where to save the captions
- `--debug` (default: false) use debug mode which tries out all possible captioning models and functions on a small subset of the images

## Prompting and Finetuning the LLaVa model

- To see the prompting capabilities of the LLaVa model run the llava_action_effect.ipnyb file. The initial part of the notebook contains the setup, zero-shot and few-shot prompting code.
- Following the zero-shot and few-shot prompting code, we have the finetuning code for the LLaVa model.
