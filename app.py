from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

import os

app = Flask(__name__)
CORS(app)


# Configuration
GROQ_API_KEY = "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" # Set as environment variable or hardcode (not recommended)
OLLAMA_MODEL = "gemma:2b"
GROQ_MODEL = "llama3-70b-8192"
FAISS_DIR = "FAISS_DataBase"

# Store for chat sessions
store = {}

# === Function: Load and index documents if FAISS DB is missing ===
def load_and_index_documents():
    document = []

    # Load local text file (optional)
    text_path = "sample.txt"  # <- Set correct path if needed
    if os.path.exists(text_path):
        document.extend(TextLoader(text_path).load())

    # Load PDF file
    pdf_path = "attention.pdf"
    if os.path.exists(pdf_path):
        document.extend(PyPDFLoader(pdf_path).load())

    # Load webpages
    web_loader = WebBaseLoader(web_paths=[
        "https://en.wikipedia.org/wiki/Cardamom",
        "https://jntuh.org/home"
    ])
    document.extend(web_loader.load())

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = splitter.split_documents(document)

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)

# Build FAISS DB on first run
if not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
    print("No FAISS DB found. Creating one...")
    load_and_index_documents()

# === RAG Chain Setup ===
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

system_prompt = (
    "Answer the question with respect to the context given. "
    "If the answer is not in the context, respond with 'sorry, the question you are asking is irrelevant or we don't have the answer.' "
    "Given a chat history and the latest user question, formulate a standalone question if needed and answer it.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat history"),
    ("human", "{input}")
])

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model=GROQ_MODEL)

history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# === Session History Management ===
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat history",
    output_messages_key="answer",
)

def ask_question(question: str, chat_id: str) -> str:
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": chat_id}}
    )
    return response["answer"]

# === Flask Routes ===

@app.route('/')
def home():
    return jsonify({"status": "RAG ChatBot backend is running."})

@app.route('/ask', methods=['POST'])
def handle_question():
    data = request.json
    question = data.get("question")
    chat_id = data.get("chat_id", "default_session")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        answer = ask_question(question, chat_id)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    chat_id = request.json.get("chat_id", "default_session")
    if chat_id in store:
        del store[chat_id]
    return jsonify({"message": f"Chat history for '{chat_id}' cleared."})

if __name__ == '__main__':
    app.run(debug=True)
