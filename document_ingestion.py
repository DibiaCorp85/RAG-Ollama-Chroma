# Import Required Libraries, Define OllamaClient, and Load Documents
import os
import json
import requests
import magic  # python-magic for file type detection
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define OllamaClient
class OllamaClient:
    """
    Client for interacting with the Ollama API.
    """
    def __init__(self, model, base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate_response(self, prompt):
        data = {"model": self.model, "prompt": prompt}
        response = requests.post(f"{self.base_url}/api/generate", json=data)
        
        if response.status_code != 200:
            return {"error": f"API request failed: {response.status_code}"}
        
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}

# Initialize Ollama model
ollama_model = "wizardlm2:latest"
llm = OllamaClient(model=ollama_model)

# Load documents
def load_documents(directory):
    txt_loader = DirectoryLoader(directory, glob=["*.txt"], show_progress=True, recursive=True)
    pdf_loader = DirectoryLoader(directory, glob=["*.pdf"], loader_cls=PyPDFLoader, show_progress=True, recursive=True)
    docx_loader = DirectoryLoader(directory, glob=["*.docx"], loader_cls=Docx2txtLoader, show_progress=True, recursive=True)
    return txt_loader.load() + pdf_loader.load() + docx_loader.load()

documents = load_documents("./sample_files")

# Detect file types and join contents
file_type_detector = magic.Magic(mime=True)
combined_content = "\n".join(doc.page_content for doc in documents if doc.page_content)

# Save combined content
processed_dir = "./processed"
os.makedirs(processed_dir, exist_ok=True)
combined_output_path = os.path.join(processed_dir, "combined_output.txt")
with open(combined_output_path, "w", encoding="utf-8") as output_file:
    output_file.write(combined_content)


# Textwrap the combined_output.txt file
# Text wrapping function
def wrap_text(text, width=150):
    from textwrap import fill
    return "\n\n".join(fill(line, width) for line in text.split('\n\n'))

# Reload and wrap text
processed_loader = DirectoryLoader(processed_dir, glob=["combined_output.txt"], show_progress=True)
processed_documents = processed_loader.load()

if processed_documents:
    print("\n--- Wrapped Combined Output ---\n")
    print(wrap_text(processed_documents[0].page_content))
else:
    print("No processed documents loaded.")

# Chunk and store in ChromaDB

# Chunking for vector search
chunk_size = 2000
chunk_overlap = 300
splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                          chunk_overlap = chunk_overlap,
                                          separators=["\n\n", "\n", " ", ""])
document_chunks = splitter.split_documents(processed_documents)

# Embedding the chunks into ChromaDB
embedding_model = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model, 
                                  model_kwargs={"device": "cuda"}, 
                                  encode_kwargs={"batch_size": 32, 
                                                 "normalize_embeddings": False})

vector_store = Chroma.from_documents(document_chunks, 
                                     embedding=embedding, 
                                     persist_directory=os.getcwd())

# Print Success Message
print("Document ingestion, embedding, and storage in ChromaDB completed successfully!")
