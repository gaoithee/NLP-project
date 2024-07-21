from datasets import load_dataset
import torch
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-small-8k-instruct",
    device_map = "cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-small-8k-instruct")

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
#  correct_answers.append(estrai_numero(train_dataset['answer'][i]))
for i in range(len(test_dataset)):
  correct_answers.append(estrai_numero(test_dataset['answer'][i]))

def create_message_cot_oneshot(question):
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


answers_cot_1s = []
answers_cot_1s_def = []

for i in range(len(test_dataset)):
    messages = create_message_cot_oneshot(test_dataset['question'][i])
    output = pipe(messages, **generation_args)
    answers_cot_1s.append(output[0]['generated_text'])
    answers_cot_1s_def.append(estrai_numero(answers_cot_1s[i]))

# queries = train_dataset['question']
queries = test_dataset['question']

df = {
    'query': queries,
    'correct': correct_answers,
    'answer': answers_cot_1s_def
}

df = pd.DataFrame.from_dict(df)
df.to_csv('testset-cot-oneshot-SMALL.csv')
