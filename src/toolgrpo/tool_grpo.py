from unsloth import FastLanguageModel
from trl import GRPOConfig
import traceback
import torch
from toolgrpo.grpo_trainer import GRPOTrainer
import os 
from typing import Any, Union
from dotenv import load_dotenv
from toolgrpo.python_interpreter import PythonInterpreter
import re
from datasets import load_dataset, Dataset

# Load environment variables from .env file
load_dotenv()

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 128 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.5B-Instruct",
    # model_name = "Qwen/Qwen3-0.6B",
    # model_name="google/gemma-3-1b-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none", # Supports any, but = "none" is optimized
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


class CustomTrainer(GRPOTrainer):
    def compute(self, script):
        result = PythonInterpreter(script)
        return result

    def extract_python(self, text: str) -> list:
        matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        pythons = []

        for match in matches:
            pythons.append(match)

        return pythons

    def pythons(self, prompt, pythons):
        if pythons:
            try:
                outputs = []
                for python in pythons:
                    result = self.compute(python)
                    outputs.append(result)

                prompt.append(
                    {"role": "tool", "content": "\n".join([f"```output\n{s}\n```" for s in outputs])}
                )
            except Exception as e:
                prompt.append({"role": "tool", "content": f"```output\n{traceback.format_exc().splitlines()[-1]}\n```"})
            return True
        return False
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        prompts, prompts_text, prompt_ids, prompt_mask, completion_ids, prompt_completion_ids, completions_text, completions = self.get_completions(inputs)
        
        for index, completion in enumerate(completions):
            text = completion[0]['content']
            messages = inputs[index]['messages']
            messages.append(
                {
                    "role": "assistant",
                    "content": text,
                }
            )

            try:
                pythons = self.extract_python(text)
                self.pythons(messages, pythons)
            except Exception as e: 
                print(e)
                pass

        return super()._prepare_inputs(inputs)
    
SYSTEM_PROMPT = """
Use Python code blocks for computations. Use imports (math, numpy, sympy) if needed. Example:
```python
print(4 * 5 / 4)
```
Think step by step. Put final answer in \\boxed{}
"""

def get_aime_questions(split="train") -> Dataset:
    data = load_dataset("gneubig/aime-1983-2024")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["Question"]},
            ],
            "answer": str(x["Answer"]),
        }
    )  # type: ignore
    return data  # type: ignore

# Load AIME dataset for training
train_dataset = get_aime_questions("train").shuffle()

# Load AIME2025 subset for evaluation
eval_dataset = load_dataset("opencompass/AIME2025", "AIME2025-I")  # type: ignore
eval_dataset = eval_dataset.map(
    lambda x: {  # type: ignore
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": str(x["answer"]),
    }
)  # type: ignore

def extract_answer(text: str) -> str:
    match = re.search(r"\\boxed{(.*?)}", text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return ""
    
def extract_output(text: str) -> str:
    match = re.search(r"```output\n(.*?)```", text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return ""

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][1]["content"]
    extracted_responses = [extract_answer(r) for r in responses]
    return [3.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def count_structure(prompts, completion) -> float:
    text = completion
    if len(prompts) >= 4:
        text += prompts[2]['content']

    count = 0.0
    if text.count("Let's solve this step by step:") >= 1:
        count += 0.125
    if text.count("\\boxed{") == 1:
        count += 0.125
    if text.count("```python") >= 1:
        count += 0.125
    return count

def structure_reward_func(prompts, completions, **kwargs) -> list[float]:
    return [count_structure(prompt, completion[0]['content']) for prompt, completion in zip(prompts, completions)]

def python_reward(prompt, completion) -> float:
    if len(prompt) < 4:
        return 0

    count = 0.0
    assistant_message = prompt[2]['content']
    if assistant_message.count("```python") >= 1:
        count += .1
        
        output = prompt[3]['content']
        if output.count("```output") >= 1:
            count += .5

            if extract_output(output):
                count += .5
    return count

def all_python_reward(prompts, completions, **kwargs) -> list[float]:
    return [python_reward(prompt, completion[0]['content']) for prompt, completion in zip(prompts, completions)]

training_args = GRPOConfig(
    max_completion_length=1024,
    max_prompt_length=1024,
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_generations = 2, # Decrease if out of memory
    num_train_epochs = 2, 
    # max_steps = 250,
    save_steps = 50,
    report_to = "wandb",
    output_dir = "outputs",
    eval_strategy = "steps",
    eval_steps = 50,
    learning_rate = 2e-5,             
    warmup_ratio = 0.1,              
    weight_decay = 0.01,              
    max_grad_norm = 1.0,              
    eval_on_start=True,
)
    
trainer = CustomTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        structure_reward_func,
        int_reward_func,
        correctness_reward_func,
        all_python_reward
    ],
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)
trainer.train()

model.save_pretrained("qwen-3b")
tokenizer.save_pretrained("qwen-3b")