import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
import gradio as gr

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# 1. Initialize RAG Components
print("Initializing RAG models and database...")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedder)

# Cross-Encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM and Prompt
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the content doesn't contain the answer, say "I don't know".
Be concise and quote facts directly.

Context:
{context}

Question: {question}
"""
)


def retrieve_with_rerank(question, top_k=3):
    """Two-stage retrieval: Broad search via Chroma (Bi-Encoder) + Reranking (Cross-Encoder)."""
    candidates = db.similarity_search(question, k=25)
    if not candidates:
        return [], []

    pairs = [(question, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)

    scored_candidates = sorted(
        zip(scores, candidates), key=lambda x: x[0], reverse=True
    )

    top_docs = [c for _, c in scored_candidates[:top_k]]
    top_scores = [float(s) for s, _ in scored_candidates[:top_k]]

    return top_docs, top_scores


def chat_with_rag(message, history):
    """Handles chat interaction, queries RAG pipeline, and formats sources."""
    if not message.strip():
        return "", history, "No query provided."

    try:
        best_chunks, scores = retrieve_with_rerank(message, top_k=3)

        if not best_chunks:
            answer = "I don't know. No relevant documents were found in the database."
            sources_md = "*No sources retrieved.*"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return "", history, sources_md

        context = "\n\n".join(c.page_content for c in best_chunks)

        chain = prompt | model
        response = chain.invoke({"context": context, "question": message})
        answer = response.content

        sources_md = "### 🔍 Retrieved & Reranked Context Chunks\n\n"
        for i, (chunk, score) in enumerate(zip(best_chunks, scores)):
            sources_md += f"**Rank {i+1}** *(Relevance Score: `{score:.4f}`)*\n"
            sources_md += f"> {chunk.page_content.replace(chr(10), ' ')}\n\n---\n"

    except Exception as e:
        answer = f"⚠️ An error occurred while processing your request: {str(e)}"
        sources_md = "### Error\n" + str(e)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, sources_md


# 2. Professional Custom CSS UI Styling
custom_css = """
body {
    background-color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.gradio-container {
    max-width: 1250px !important;
    margin: auto;
    padding: 20px;
}
header h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}
"""

# 3. Build Gradio UI Layout (Cleaned of legacy parameters)
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🏢 Enterprise RAG Assistant
        *Advanced Retrieval-Augmented Generation powered by **ChromaDB**, **HuggingFace Embeddings**, **Cross-Encoder Reranking**, and **GPT-4o-mini**.*
        """
    )

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="AI Policy Assistant",
                height=520,
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question about the return policy (e.g., 'What is the refund timeline?')...",
                    show_label=False,
                    scale=4,
                    container=False,
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")

            clear = gr.Button("Clear Conversation 🗑️", variant="secondary")

        with gr.Column(scale=3):
            sources_output = gr.Markdown(
                value="*Ask a question to inspect the retrieved chunks and cross-encoder scores.*",
                label="RAG Inspector",
            )

    msg.submit(
        chat_with_rag, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_output]
    )
    submit_btn.click(
        chat_with_rag, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_output]
    )
    clear.click(lambda: ([], "*Conversation cleared.*"), None, [chatbot, sources_output])

# Launch app locally
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=custom_css,
    )