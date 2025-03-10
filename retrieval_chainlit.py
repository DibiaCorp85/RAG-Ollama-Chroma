import aiohttp  # Asynchronous HTTP requests for streaming
import json  # JSON handling
import os  # File operations
import chainlit as cl  # Chainlit for chat UI
from langchain_huggingface import HuggingFaceEmbeddings  # Hugging Face embeddings
from langchain_chroma import Chroma  # ChromaDB for retrieval

# Define Ollama Client for Streaming Responses
class OllamaClient:
    """
    A client for interacting with the Ollama API asynchronously.
    Handles streaming responses efficiently.
    """
    def __init__(self, model, base_url):
        self.model = model
        self.base_url = base_url

    async def generate_response(self, prompt):
        """
        Generates a streaming response from the Ollama model based on the given prompt.
        """
        data = {"model": self.model,
                "prompt": prompt,
                "stream": True}

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/generate", json=data) as response:
                if response.status != 200:
                    yield f"Error: API request failed with status {response.status}"
                    return
                
                # Stream response line by line
                async for line in response.content:
                    if line:
                        try:
                            parsed = json.loads(line.decode("utf-8"))
                            yield parsed.get("response", "")
                        except json.JSONDecodeError:
                            yield "[Error: Invalid JSON response]"

# Initialize Ollama model
Base_URL = "http://localhost:11434"
llm = OllamaClient(model="wizardlm2:latest", base_url=Base_URL)

# Load ChromaDB stored embeddings
persist_directory = os.getcwd()  # Use current directory for storage
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cuda"},  # Use GPU for fast embeddings
    encode_kwargs={"batch_size": 32, "normalize_embeddings": False}
)

# Load ChromaDB vector store
vectorStore = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)

# Define the RAG Prompt Template
template = """
You are a helpful assistant that provides accurate answers based on the retrieved context. 
If the context does not contain the answer, say, "I do not have answers to your query." 
Keep the response within three sentences.

Question: {question}  
Context: {context}  
Answer:
"""

# Chainlit Application for Interactive Retrieval-Augmented Generation (RAG)
@cl.on_message
async def main(message: cl.Message):
    """
    Handles incoming user messages, retrieves relevant context, and generates a response.
    """
    query = message.content.strip()  # Extract user input
    num_retrieved_chunks = 4  # Retrieve top 4 chunks for better context

    # Perform similarity search (without scores)
    docs = vectorStore.similarity_search(query, k=num_retrieved_chunks)

    # Prepare retrieved content
    if not docs:
        retrieved_text = "No relevant documents found."
    else:
        retrieved_text = "\n\n".join([doc.page_content for doc in docs])

    # Format the prompt using the template
    formatted_prompt = template.format(question=query, context=retrieved_text)

    # Send an initial "Processing..." message
    response = cl.Message(content="Processing...")
    await response.send()

    # Update response with retrieved context
    response.content = f"### Retrieved Context\n{retrieved_text}\n\n### AI-Generated Response\n"

    # Stream Ollama response
    async for chunk in llm.generate_response(formatted_prompt):
        response.content += chunk
        await response.update()

    # Final response update
    await response.update()
