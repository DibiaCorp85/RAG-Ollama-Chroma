import os
import json
import requests
import gradio as gr
from gradio.themes.base import Base
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


class OllamaClient:
    """
    Client for interacting with the Ollama API.
    """
    def __init__(self, model, base_url="http://localhost:11434"):
        """
        Initialize the OllamaClient.

        Args:
            model (str): The name of the model to use (e.g., "wizardlm2:latest").
            base_url (str): The base URL of the Ollama API (e.g., "http://localhost:11434").
        """
        self.model = model
        self.base_url = base_url
    
    def generate_response(self, prompt):
        """
        Generate a response from the Ollama model based on the given prompt.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            dict: A dictionary containing the response or an error message.
        """
        data = {"model": self.model,
                "prompt": prompt, 
                "stream": False}  # Non-streaming response
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

# Load ChromaDB stored embeddings
persist_directory = os.getcwd()

# Load the same embedding model used for indexing
embedding_model = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model, 
                                  model_kwargs={"device": "cuda"}, 
                                  encode_kwargs={"batch_size": 32, 
                                                "normalize_embeddings": False})

# Load the ChromaDB vector store
vector_store = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)

# Define Prompt Template
template = """
You are a helpful assistant that provides accurate answers based on the retrieved context. 
If the context does not contain the answer, say, "I do not have answers to your query." 
Keep the response within three sentences.

Question: {question}  
Context: {context}  
Answer:
"""

#Hybrid Retrieval Function
def query_data(query):
    """
    Query the ChromaDB vector store and generate a response.
    """
    docs = vector_store.similarity_search(query, k = 4)
    
    if not docs:
        return "No relevant document found.", "No relevant context."
    
    # Only retrieve document content
    retrieved_text = "\n\n".join([doc.page_content for doc in docs])

    # Format the query using the template
    prompt = template.format(question=query,
                            context=retrieved_text)

    # Generate response
    try:
        retriever_output = llm.generate_response(prompt)  # Send formatted prompt to Ollama
        return retrieved_text, retriever_output.get("response", "No response generated.")
    except Exception as e:
        return retrieved_text, f"Error generating response: {str(e)}"

# Web Interface using Gradio
with gr.Blocks(theme=Base(), title="RAG Question Answering") as demo:
    gr.Markdown("""# RAG Question Answering System""")
    
    textbox = gr.Textbox(label="Enter your question here")
    button = gr.Button("Submit", variant="primary")
    
    with gr.Column():
        output1 = gr.Textbox(lines=5, label="Retrieved Context")
        output2 = gr.Textbox(lines=5, label="Generated Response")
    
    button.click(query_data, inputs=[textbox], outputs=[output1, output2])

demo.launch()
