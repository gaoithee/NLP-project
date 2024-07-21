from datasets import load_dataset
import torch
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

torch.random.manual_seed(0)

##############################################################################

# upload the model and the tokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

##############################################################################

# construct the pipeline and fix the generation arguments
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

##############################################################################

# load the dataset and select the test set 
dataset = load_dataset('openai/gsm8k', 'main')
test_dataset = dataset['test']

# function to extract the answer from the golden answers
def estrai_numero(input_string):
    # Usa una regex per trovare il numero dopo ###
    match = re.search(r'###\s*(\d+)', input_string)
    if match:
        return match.group(1)
    else:
        return None

# extract the correct answers
correct_answers = []
for i in range(len(test_dataset)):
  correct_answers.append(estrai_numero(test_dataset['answer'][i]))

##############################################################################

loader = HuggingFaceDatasetLoader('saracandu/references-gsm8k', 'docs')
documents = loader.load()

# create an instance of the RecursiveCharacterTextSplitter class with specific parameters
# (it splits text into chunks of 50 characters each with a 20-character overlap)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)

# 'documents' holds the text you want to split, split the text into documents using the text splitter
docs = text_splitter.split_documents(documents)

# choose an embedding method
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# embed the documents 'docs' into vectors using the embedding method specified by 'embedding'
# the result is stored in a FAISS index:
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(
    search_kwargs={'k': 5,}
)

##############################################################################

# function to format the retrieved documents in a proper way 

def format_page_content(documents):
    """
    Formats the list of retrieved documents such that 'page_content', 'Documents', 'metadata' 
    words are removed and just the true content is kept.
    """
    formatted_output = ""
    for i, doc in enumerate(documents, start=1):
        content = doc.page_content.strip(" ")
        formatted_output += f"[{i}]: {content}\n"
    return formatted_output

##############################################################################

# function to create the prompt for each question

def create_message_baseline_RAG(question, context):
    content = f"""
    Question: "{question}";
    Context: "{context}". 
    """

    messages = [
        {"role": "system",
        "content": """
        You are a helpful AI assistant asked to solve mathematical problems using similar problems already solved as context. 
        Output your numerical answer only after these symbols ###.
        """},
        {"role": "user",
        "content": content},
    ]

    return messages

##############################################################################

answers_baseline_RAG = []
answers_baseline_RAG_def = []

for i in range(len(test_dataset)):
    relevant_passages = format_page_content(retriever.invoke(test_dataset['question'][i]))
    messages = create_message_baseline_RAG(test_dataset['question'][i], relevant_passages)
    output = pipe(messages, **generation_args)
    answers_baseline_RAG.append(output[0]['generated_text'])
    answers_baseline_RAG_def.append(estrai_numero(answers_baseline_RAG[i]))

##############################################################################

df = {
    'query': test_dataset['question'],
    'correct': correct_answers,
    'long answer': answers_baseline_RAG,
    'answer': answers_baseline_RAG_def
}

df = pd.DataFrame.from_dict(df)
df.to_csv('testset-baseline-RAG-zeroshot.csv')
