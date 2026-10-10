from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedder)

reranker = CrossEncoder("cross-encoder/ms-macro-MiniLM-L6-2")

def retrieve_with_rerank(question, top_k=3):
    candidates = db.similarity_search(question, k=25)
    pairs = [(question, c.page_content) for c in candidates]
    scores = reranker.predict(pairs)
    return [c for _,c in sorted(zip(scores, candidates), reverse=True)][:top_k]