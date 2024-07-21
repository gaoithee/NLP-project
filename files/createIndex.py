from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# to avoid computing it each time (since the docs won't change), save the result in the storage
db.save_local(folder_path="faiss_db", index_name="GSM8K_FaissIndex_MiniLM")