from datasets import load_dataset
import torch
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

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

dataset = load_dataset('openai/gsm8k', 'main')
train_dataset = dataset['train']
test_dataset = dataset['test']

def estrai_numero(input_string):
    # Usa una regex per trovare il numero dopo ###
    match = re.search(r'###\s*(\d+)', input_string)
    if match:
        return match.group(1)
    else:
        return None

# N_rows = 2000

correct_answers = []
# for i in range(N_rows):
#   correct_answers.append(estrai_numero(train_dataset['answer'][i]))
for i in range(len(test_dataset)):
   correct_answers.append(estrai_numero(test_dataset['answer'][i]))

def create_message_cot_zeroshot(question):
    content = f"""
    Question: "{question}" Let's think step by step.
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


answers_cot_0s = []

for i in range(len(test_dataset)):
    messages = create_message_cot_zeroshot(test_dataset['question'][i])
    output = pipe(messages, **generation_args)
    answers_cot_0s.append(output[0]['generated_text'])

# queries = train_dataset['question']
queries = test_dataset['queries']

df = {
    'query': queries,
    'correct': correct_answers,
    'answer': answers_cot_0s
}

df = pd.DataFrame.from_dict(df)
df.to_csv('testset-cot-zeroshot.csv')
