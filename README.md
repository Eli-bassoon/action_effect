# action_effect

## Generating Captions

Use `./captioning/captioning.py` to generate captions. Change your directory into the same directory as the python file, then run the program to generate captions for all images using a specific model and prompt. This saves the captions to the `./captioning/saved_captions` folder.

## Condensing Captions

Use `./llm_finetuning/train_compress_captions.py` to train Mistral on a list of captions. Change your directory into the same directory as the python file. This saves the lora into the `./compressed_captions` folder.

Use `./llm_finetuning/compress_captions.py` to evaluate Mistral on the test set of a list of captions. Change your directory into the same directory as the python file. This saves the output into the `./compressed_captions` folder.

## Evaluation

Use `./evaluation/eval_all.py` to evaluate all of the condensed captions. This can be run from the top-level directory.

## Prompting and Finetuning the LLaVa model

- To see the prompting capabilities of the LLaVa model run the `llava_action_effect.ipnyb file`. The initial part of the notebook contains the setup, zero-shot and few-shot prompting code.
- Following the zero-shot and few-shot prompting code, we have the finetuning code for the LLaVa model.

## Training the ViT model

- See the `vision_transformer_action_effect.ipynb` notebook. That contains a step-by-step process to train the vision transformer on the data.