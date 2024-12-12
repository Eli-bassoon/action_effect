# For Programming Problem 23
#
# Please implement the functions below, and feel free to use the any library.

from datasets import load_dataset, Dataset, load_from_disk
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import random
import argparse
import re
import os

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

SAVE_DIR = './compressed_captions'

from dotenv import load_dotenv
import wandb

# Load environment variables from the .env file
load_dotenv()

# Log in to W&B using the API key from the environment
wandb.login()

# Initialize a W&B run
run = wandb.init(
    project='Compress Captions',
    job_type="training",
    anonymous="allow"
)


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
    dataset='./datasets/flat_dataset_captions',
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
    dataset = load_from_disk(dataset)[split]
    if sample_size is not None:
        dataset = dataset.select(range(sample_size))

    # Initialize the data collator for completion-only language modeling
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=mlm
    )

    return dataset, collator


def setup_and_train_model(
    model,
    dataset,
    tokenizer,
    collator,
    formatting_prompts_func,
    run_name="qlora-piqa",
    output_dir="./results",
    num_train_epochs=2,
    max_seq_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=5,
    learning_rate=5e-5,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    report_to="wandb",
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    seed=595
):
    """
    Sets up LoRA configuration, defines training arguments, and trains the model using a custom Trainer.

    Params:
        model (PreTrainedModel): The pre-trained model to fine-tune.
        dataset (Dataset): The dataset to use for training.
        tokenizer (AutoTokenizer): The tokenizer for preprocessing data.
        collator (DataCollator): The data collator used for batching data.
        formatting_prompts_func (function): The function to format prompts for the model.
        run_name (str, optional): The name of the run for logging. Default is 'qlora-piqa'.
        output_dir (str, optional): The directory to save the model outputs. Default is './results'.
        num_train_epochs (int, optional): Number of training epochs. Default is 2.
        max_seq_length (int, optional): Maximum sequence length for input data. Default is 512.
        per_device_train_batch_size (int, optional): Batch size per device. Default is 4.
        gradient_accumulation_steps (int, optional): Steps for gradient accumulation. Default is 4.
        optim (str, optional): The optimizer to use. Default is 'paged_adamw_32bit'.
        save_steps (int, optional): Number of steps between model saving. Default is 200.
        logging_steps (int, optional): Number of steps between logging. Default is 5.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 5e-5.
        max_grad_norm (float, optional): Maximum gradient norm. Default is 0.3.
        warmup_ratio (float, optional): Warmup ratio for learning rate schedule. Default is 0.03.
        lr_scheduler_type (str, optional): Type of learning rate scheduler. Default is 'linear'.
        report_to (str, optional): Reporting platform, such as 'wandb'. Default is 'wandb'.
        r (int, optional): LoRA rank parameter. Default is 8.
        lora_alpha (int, optional): Alpha parameter for LoRA. Default is 32.
        target_modules (list, optional): List of target modules for LoRA. Default is ["q_proj", "v_proj"].
        lora_dropout (float, optional): Dropout rate for LoRA. Default is 0.1.
        seed (int, optional): Random seed for reproducibility. Default is 595.

    Returns:
        None
    """

    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout
    )

    # Define training arguments
    training_arguments = SFTConfig(
        run_name=run_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        do_eval=False,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        report_to=report_to,
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # Ensure 'norm' layers are in float32 precision
    # This stablize the training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    # Train the model
    trainer.train()

    return trainer, model


def save_and_load_lora_model(trainer, model, output_dir="outputs"):
    """
    Saves the trained model, handles distributed/parallel training, and reloads the model with LoRA configuration.

    Params:
        trainer (Trainer): The trainer object used during model training.
        model (PreTrainedModel): The model to be saved and reloaded with LoRA configuration.
        output_dir (str, optional): The directory to save the model. Default is 'outputs'.

    Returns:
        model (PreTrainedModel): The model reloaded with LoRA configuration.
    """

    # Handle saving model in case of distributed/parallel training
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(output_dir)

    # Load the LoRA configuration from the saved model directory
    # lora_config = LoraConfig.from_pretrained(output_dir)

    # Reapply the LoRA configuration to the model
    # model = get_peft_model(model, lora_config)


def test(model, tokenizer, dataset, create_prompt_func, prediction_save='compressed_captions.torch'):
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
        output = model.generate(**inputs, max_new_tokens=6, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated output to get the predicted answer
        predicted_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        predicted_answer = re.search(r'### Answer:.*?([a-z]+ [a-z]+)', predicted_output).groups()[0].strip()

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
        predicted_labels=predictions,
        labels=labels
    )
    
    # Save predictions to a file
    torch.save(save_data, prediction_save)


def main(params):
    """
    Main function that handles loading the model, preparing the dataset, training, saving, and testing.

    Params:
        params (argparse.Namespace): Command-line arguments passed to configure the training pipeline.
    """

    # Load the model and tokenizer
    model, tokenizer = load_model(params.model)

    # Get the response template for the task
    response_template = get_response_template()

    # Load training data and the associated collator
    dataset_train, collator = load_data_with_collator(dataset=params.dataset, split="train", tokenizer=tokenizer, sample_size=None, response_template=response_template)

    # Set up and train the model with LoRA
    trainer, model = setup_and_train_model(
        model,
        dataset_train,
        tokenizer,
        collator,
        formatting_prompts_func=formatting_prompts_func,
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.per_device_train_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        optim=params.optim,
        save_steps=params.save_steps,
        logging_steps=params.logging_steps,
        learning_rate=params.learning_rate,
        max_grad_norm=params.max_grad_norm,
        warmup_ratio=params.warmup_ratio,
        lr_scheduler_type=params.lr_scheduler_type,
        report_to=params.report_to,
        r=params.r,
        lora_alpha=params.lora_alpha,
        target_modules=params.target_modules,
        lora_dropout=params.lora_dropout,
        seed=params.seed
    )

    os.mkdir(os.path.join(SAVE_DIR, params.run_name))
    
    # Save the trained model with LoRA configuration
    save_and_load_lora_model(trainer, model, output_dir=os.path.join('./compressed_captions/', params.run_name, 'lora'))

    # Load test data
    dataset_test, _ = load_data_with_collator(dataset=params.dataset, split="test", tokenizer=tokenizer, sample_size=None, response_template=response_template)

    # Test the model on the test set
    test(model, tokenizer, dataset_test, create_prompt_instruction, prediction_save=os.path.join('./compressed_captions', params.run_name, 'compressed_captions.pt'))

    return model


if __name__ == "__main__":
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    num_train_epochs = 8
    max_seq_length = 2048
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    save_steps = 200
    logging_steps = 5
    learning_rate = 1e-5
    max_grad_norm = 1
    warmup_ratio = 0.2
    r = 64
    lora_alpha = 64
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    lora_dropout = 0.1

    optim = "paged_adamw_32bit"
    lr_scheduler_type = "linear"
    report_to = "wandb"
    seed = SEED
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    parser = argparse.ArgumentParser(description="Finetune LLaMA-based model for PIQA Task")
    parser.add_argument("--dataset", type=str, default="./datasets/flat_dataset_captions", help="Dataset name")
    parser.add_argument("--run_name", type=str, default="base", help="The name of the run to save the files")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3", help="Pretrained model name")
    parser.add_argument("--num_train_epochs", type=int, default=num_train_epochs, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=max_seq_length, help="Maximum sequence length for inputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=per_device_train_batch_size, help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=gradient_accumulation_steps, help="Number of gradient accumulation steps")
    parser.add_argument("--optim", type=str, default=optim, help="Optimizer type")
    parser.add_argument("--save_steps", type=int, default=save_steps, help="Number of steps between saves")
    parser.add_argument("--logging_steps", type=int, default=logging_steps, help="Number of steps between logging updates")
    parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Learning rate for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=max_grad_norm, help="Max gradient norm for clipping")
    parser.add_argument("--warmup_ratio", type=float, default=warmup_ratio, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default=lr_scheduler_type, help="Learning rate scheduler type")
    parser.add_argument("--report_to", type=str, default=report_to, help="Reporting platform (e.g., wandb)")
    parser.add_argument("--r", type=int, default=r, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=lora_alpha, help="Alpha parameter for LoRA")
    parser.add_argument("--target_modules", type=list, default=target_modules, help="Target modules for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=lora_dropout, help="Dropout rate for LoRA")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed for reproducibility")

    # Parse arguments and run the main function
    params, unknown = parser.parse_known_args()
    model = main(params)
