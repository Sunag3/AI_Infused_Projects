from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder  # 1. Import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Create embedder and load database
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory='./chroma_db', embedding_function=embedder)

# Initialize the Cross-Encoder reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Fixed model name spelling

# Initialize LLM and prompt
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
    """Retrieves 25 candidates via Chroma, then reranks them using a Cross-Encoder."""
    # Step 1: Broad retrieval (Bi-Encoder)
    candidates = db.similarity_search(question, k=25)
    
    # Step 2: Create pairs of (question, chunk_text) for the Cross-Encoder
    pairs = [(question, c.page_content) for c in candidates]
    
    # Step 3: Score all pairs simultaneously
    scores = reranker.predict(pairs)
    
    # Step 4: Sort by score descending and pick the top_k best documents
    ranked_docs = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
    return ranked_docs[:top_k]

def rag_answer(question):
    # Use our reranking function instead of raw similarity search
    best_chunks = retrieve_with_rerank(question, top_k=3)
    
    # Build context from the reranked chunks
    context = "\n\n".join(c.page_content for c in best_chunks)
    
    chain = prompt | model
    response = chain.invoke({"context": context, "question": question})
    return response.content

print(rag_answer("how many days for the credit to officially reflect on your bank or credit card?"))