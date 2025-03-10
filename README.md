📚 RAG-Ollama-Chroma
A modern Retrieval-Augmented Generation (RAG) system powered by ChromaDB, Ollama LLM, and LangChain for efficient document retrieval and AI-generated responses. Supports multi-format document ingestion (.txt, .pdf, .docx) and provides two retrieval interfaces:

✅ Gradio Web UI – for easy question-answering
✅ Chainlit Chatbot – for interactive, streaming responses with multi-turn memory

🚀 Features
✅ Multi-format document ingestion (.txt, .pdf, .docx)
✅ Fast document embeddings with HuggingFace
✅ Efficient vector search using ChromaDB
✅ AI-powered retrieval using Ollama's wizardlm2 model
✅ Web-based question-answering interface with Gradio
✅ Interactive chatbot with streaming responses via Chainlit
✅ Multi-turn memory support (Chainlit)


🛠️ Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/YOUR_GITHUB_USERNAME/RAG-Ollama-Chroma.git
cd RAG-Ollama-Chroma

2️⃣ Set Up a Python Virtual Environment
  
venv\Scripts\activate  # Windows  
3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Start Ollama (if not running)
If you haven't installed Ollama, follow the installation guide and start the local server:

ollama serve

📥 Document Ingestion
To process and index documents in ChromaDB, run:

python document_ingestion.py
Supported formats: TXT, PDF, DOCX
This will store vector embeddings for retrieval.

🔍 Retrieval Interfaces
1️⃣ Gradio Web UI
Run the Gradio interface for document retrieval:

python retrieval_gradio.py
This starts a web-based question-answering UI at:

2️⃣ Chainlit Chatbot (Streaming)
Initialize chainlit, run "chainlit init" on the terminal (for windows)
To enable conversational AI retrieval, run:

chainlit run retrieval_chainlit.py
This launches a chatbot with real-time streaming responses.

📄 Example Usage
1️⃣ Run document_ingestion.py to index files.
2️⃣ Start Gradio (retrieval_gradio.py) or Chainlit (retrieval_chainlit.py).
3️⃣ Ask a question in Gradio or Chainlit:

"What are separation processes?"
4️⃣ Get a context-aware AI response.

⚡ TODO (Upcoming Features)
 Add hybrid search (keyword + semantic)
 Implement RAG caching for faster responses
 Enhance multi-turn chat memory
 Deploy as a FastAPI or Flask app

👥 Contributing
Contributions are welcome!

Fork the repo
Create a feature branch (git checkout -b feature-name)
Commit changes (git commit -m "Add feature")
Push and create a Pull Request (PR)
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgements
💡 Built using:

LangChain – For document processing
ChromaDB – Vector storage
Ollama – Local LLM for response generation
Gradio – Web UI for retrieval
Chainlit – Chatbot UI with streaming

📩 Questions or Issues?
Open an issue on GitHub or reach out for support. 🚀
Happy coding! 🎯