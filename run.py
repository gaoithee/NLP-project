from datasets import load_dataset
import torch
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


torch.random.manual_seed(0)

dataset = load_dataset('openai/gsm8k', 'main')
train_dataset = dataset['train']


model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "do_sample": False,
}


def estrai_numero(input_string):
    # Usa una regex per trovare il numero dopo ###
    match = re.search(r'###\s*(\d+)', input_string)
    if match:
        return match.group(1)
    else:
        return None


def create_message_baseline(question):
    content = f"""
    Question: "{question}".
    """
    messages = [
        {"role": "system",
        "content": """
        You are a helpful AI assistant asked to solve mathematical problems. Output your numerical answer only after these symbols ###.
        """},
        {"role": "user",
        "content": content},
    ]
    return messages

def create_message_baseline2(question):
    content = f"""
    Question: "{question}".
    """
    messages = [
        {"role": "system",
        "content": """
        You are a helpful AI assistant asked to solve mathematical problems. Output your numerical answer only after these symbols ###.
        Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
        """},
        {"role": "assistant",
        "content": """
        Assistant: ### 624
        """},
        {"role": "user",
        "content": content},
    ]
    return messages

def create_message_cot(question):
    content = f"""
    Question: "{question}".
    """
    messages = [
        {"role": "system",
        "content": """
        You are a helpful AI assistant asked to solve mathematical problems.
        Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
        """},
        {"role": "assistant",
        "content": """
        Assistant: He writes each friend 3*2=6 pages a week So he writes 6*2=12 pages every week That means he writes 12*52=624 pages a year. ### 624
        """},
        {"role": "user",
        "content": content},
    ]
    return messages

correct_answers = []
for i in range(len(train_dataset)):
  correct_answers.append(estrai_numero(train_dataset['answer'][i]))

answers_baseline = []
for i in range(len(train_dataset)):
  messages = create_message_baseline(train_dataset['question'][i])
  output = pipe(messages, **generation_args)
  temp = output[0]['generated_text']
  answers_baseline.append(estrai_numero(temp))

answers_baseline2 = []
for i in range(len(train_dataset)):
  messages = create_message_baseline2(train_dataset['question'][i])
  output = pipe(messages, **generation_args)
  temp = output[0]['generated_text']
  answers_baseline2.append(estrai_numero(temp))

answers_cot = []
for i in range(len(train_dataset)):
  messages = create_message_cot(train_dataset['question'][i])
  output = pipe(messages, **generation_args)
  temp = output[0]['generated_text']
  answers_cot.append(estrai_numero(temp))


dati = {
    'domanda': train_dataset['question'],
    'corretta': correct_answers,
    'baseline 1': answers_baseline,
    'baseline 2': answers_baseline2,
    'one-shot CoT': answers_cot
}

df = pd.DataFrame(dati)
df.head()
df.to_csv('full-test.csv')