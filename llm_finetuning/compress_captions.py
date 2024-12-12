# For Programming Problem 23
#
# Please implement the functions below, and feel free to use the any library.

from datasets import load_dataset, Dataset, load_from_disk
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, LoraModel, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import random
import argparse
import sys, os
import tqdm
import re

def placeholder():
    raise NotImplementedError("This function has not been implemented yet.")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_model(
    model_name="mistralai/Mistral-7B-v0.3",
    use_cache=False,
    load_in_4bit=True,
    quant_type="nf4",
    compute_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
):
    """
    Loads the LLaMA model and tokenizer with configurable options for quantization and memory usage.

    Params:
        model_name (str, optional): The name or path of the model to load. Default is "mistralai/Mistral-7B-v0.3".
        use_cache (bool, optional): Whether to use cache for generation. Default is False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit precision. Default is True.
        quant_type (str, optional): The quantization type, either 'nf4' or other types. Default is 'nf4'.
        compute_dtype (torch.dtype, optional): The computation dtype, such as torch.float16. Default is torch.float16.
        low_cpu_mem_usage (bool, optional): Whether to use less CPU memory. Default is True.
        trust_remote_code (bool, optional): Whether to trust remote code from the model hub. Default is True.

    Returns:
        model (PreTrainedModel): The loaded LLaMA model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Configure quantization options for the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load the model with quantization and other configurations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage
    )

    # Set caching behavior based on input parameter
    model.config.use_cache = use_cache

    return model, tokenizer


def get_response_template():
    """
    Returns a response template for formatting answers in a multiple-choice task.

    Returns:
        str: A string template for the answer section.
    """
    ###############################################################
    #########         TODO: Your code starts here        ##########
    ###############################################################
    return f"### Answer:"
    ###############################################################
    #########          TODO: Your code ends here         ##########
    ###############################################################


# A prompting formatting function
def create_prompt_instruction(caption, label=None):
    """
    Formats a prompt instruction for a multiple-choice question about a goal with two solutions.

    This function generates a prompt string based on the given goal and two solutions. If a label is provided,
    the prompt will include the correct answer. If no label is provided, the answer section will be left blank
    for model completion.

    Params:
        goal (str): The goal or question prompt.
        sol1 (str): The first possible solution to the goal.
        sol2 (str): The second possible solution to the goal.
        label (int, optional): The index (0 or 1) indicating the preferred solution. If None, the answer section
                               is left incomplete.

    Returns:
        str: The formatted prompt string ready for input into a language model.
    """
    ###############################################################
    #########         TODO: Your code starts here        ##########
    ###############################################################
    
    s = f'''
Identify the most likely action that would result in the state described by the caption. Respond in two words. The first word should be a verb, and the second a noun.

Ex. cut grass, blow bubble, water plant, slice cabbage

### Caption: {caption}

 {get_response_template()} '''
    
    if label is not None:
        s += label
    
    return s
    
    ###############################################################
    #########          TODO: Your code ends here         ##########
    ###############################################################


def formatting_prompts_func(example):
    """
    Prepares a batch of examples by formatting each into a prompt string suitable for input to a model.

    This function takes in a batch of examples containing goals, two possible solutions, and labels, and uses
    `create_prompt_instruction` to generate formatted strings for each example. These strings are later used
    for input into a language model for fine-tuning or testing.

    Params:
        example (dict): A dictionary containing the batch of examples, with keys 'goal', 'sol1', 'sol2', and 'label'.
                        Each key maps to a list of values.

    Returns:
        list of str: A list of formatted prompt strings for each example.
    """
    inputs = []
    ###############################################################
    #########         TODO: Your code starts here        ##########
    ###############################################################
    
    n = len(example['caption'])
    inputs = [
        create_prompt_instruction(
            example['caption'][i],
            example['action'][i])
        for i in range(n)]

    ###############################################################
    #########          TODO: Your code ends here         ##########
    ###############################################################
    return inputs


def load_data_with_collator(
    split="train",
    sample_size=None,
    tokenizer=None,
    response_template="### Answer:",
    mlm=False
):
    """
    Loads a sample of the PIQA dataset and sets up a DataCollator for completion-only language modeling.

    Params:
        dataset_name (str, optional): The name of the dataset to load. Default is 'piqa'.
        split (str, optional): The dataset split to load (e.g., 'train', 'validation', 'test'). Default is 'train'.
        sample_size (int, optional): The number of examples to select from the dataset. Default is 1000.
        tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the data. Must be provided.
        response_template (str, optional): The template used for formatting model responses. Default is '### Answer:'.
        mlm (bool, optional): Whether to use masked language modeling. Default is False (for completion tasks).

    Returns:
        tuple: A tuple containing the loaded dataset sample and the data collator.
    """

    # Load the dataset split and select a sample of the data
    dataset = load_from_disk('./flat_dataset_captions')[split]
    if sample_size is not None:
        dataset = dataset.select(range(sample_size))

    # Initialize the data collator for completion-only language modeling
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=mlm
    )

    return dataset, collator


def load_lora_model(base_model, peft_dir="compressor_lora"):
    """
    Saves the trained model, handles distributed/parallel training, and reloads the model with LoRA configuration.

    Params:
        trainer (Trainer): The trainer object used during model training.
        model (PreTrainedModel): The model to be saved and reloaded with LoRA configuration.
        output_dir (str, optional): The directory to save the model. Default is 'outputs'.

    Returns:
        model (PreTrainedModel): The model reloaded with LoRA configuration.
    """

    # Load the LoRA configuration from the saved model directory
    model = PeftModel.from_pretrained(base_model, peft_dir)
    model = model.merge_and_unload()
    
    return model


def test(model, tokenizer, dataset, create_prompt_func, prediction_save='compressed_captions_zsh.torch'):
    # metric = evaluate.load("accuracy")
    outputs, predictions, labels = [], [], []

    for example in dataset:
        # Create the prompt for the current example
        prompt = create_prompt_func(
            example['caption'],
            None  # No label for inference
        )

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate the answer (1 or 2)
        output = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated output to get the predicted answer
        predicted_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        predicted_answer = re.search(r'### Answer:.*?([a-z]+ [a-z]+)', predicted_output)
        if predicted_answer != None:
            predicted_answer = predicted_answer.groups()[0].strip()
        else:
            predicted_answer = ''

        print()
        print(predicted_output)
        print()
        print('=' + '-'*50 + '=')
        
        outputs.append(predicted_output)
        predictions.append(predicted_answer)
        labels.append(example['action'])

    # Calculate accuracy
    # metric.add_batch(predictions=predictions, references=dataset["label"])

    # Print test accuracy
    # score = metric.compute()
    # print(f'Test Accuracy: {score["accuracy"]}')
    
    save_data = dict(
        outputs=outputs,
        pred=predictions,
        gt=labels
    )
    
    # Save predictions to a file
    torch.save(save_data, prediction_save)
    

def main(params):
    """
    Main function that handles loading the model, preparing the dataset, training, saving, and testing.

    Params:
        params (argparse.Namespace): Command-line arguments passed to configure the training pipeline.
    """

    # Get the response template for the task
    response_template = get_response_template()

    # Save the trained model and load with LoRA configuration
    model, tokenizer = load_model(params.model)
    
    # Load a lora unless told not to
    if not params.no_lora:
        model = load_lora_model(model, params.lora_dir)

    # Load test data
    dataset_test, _ = load_data_with_collator(split="test", tokenizer=tokenizer, sample_size=None, response_template=response_template)
    
    # Test the model on the test set
    test(model, tokenizer, dataset_test, create_prompt_instruction)

    return model


if __name__ == "__main__":
    lora_dir = 'compressor_lora'
    
    parser = argparse.ArgumentParser(description="Evaluate LLaMA-based model for PIQA Task")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3", help="Pretrained model name")
    parser.add_argument("--no_lora", action='store_true', help='Use the base model without a lora')
    parser.add_argument('--lora_dir', type=str, default=lora_dir, help='Lora directory location')
    
    # Parse arguments and run the main function
    params, unknown = parser.parse_known_args()
    model = main(params)
