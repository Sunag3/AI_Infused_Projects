import os
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Suppress minor community warning logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# Global variable to hold our active vector database session
current_db = None

# Initialize LLM and Prompt template
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful PDF assistant. Answer the question using only the context below.
If the context doesn't contain the answer, say "I could not find that in the document."
After your answer, list the page numbers you used as: Source: page X, page Y.

Context: {context}
Question: {question}
"""
)

def build_index(pdf_path):
    """Loads a PDF file, splits it into chunks, and builds a local Chroma index."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedder)
    return db

def chat_with_pdf(message, history):
    """Gradio ChatInterface handler supporting multimodal PDF uploads."""
    global current_db
    
    # 1. Check if the user uploaded a new file in this turn
    files = message.get("files")
    if files:
        # Take the first uploaded PDF file path
        pdf_path = files[0] if isinstance(files[0], str) else files[0].get("path")
        try:
            current_db = build_index(pdf_path)
            return "📄 **PDF uploaded and indexed successfully!** You can now ask questions about its contents."
        except Exception as e:
            return f"⚠️ Error processing PDF: {str(e)}"

    # 2. Check if a database exists before answering questions
    if current_db is None:
        return "⚠️ Please upload a PDF file using the attachment icon (+) in the chat box before asking questions."

    user_query = message.get("text", "").strip()
    if not user_query:
        return "Please provide a question."

    try:
        # 3. Retrieve relevant chunks and format context with page numbers
        chunks = current_db.similarity_search(user_query, k=4)
        if not chunks:
            return "I could not find that in the document."

        context = "\n\n".join(
            f"[page {c.metadata.get('page', 0) + 1}] {c.page_content}" for c in chunks
        )

        # 4. Generate response via LangChain
        chain = prompt | model
        response = chain.invoke({"context": context, "question": user_query})
        return response.content

    except Exception as e:
        return f"⚠️ An error occurred: {str(e)}"

# 5. Build Gradio Multimodal UI Layout
demo = gr.ChatInterface(
    fn=chat_with_pdf,
    multimodal=True,
    title="💬 Chat with your PDF Assistant",
    description="Upload a PDF document using the **+** button in the chat box, index it automatically, and ask questions with exact page citations.",
    textbox=gr.MultimodalTextbox(
        file_count="single",
        file_types=[".pdf"],
        placeholder="Upload a PDF, then type your question here...",
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)