# Retrieval-Augmented Generation (RAG) - Study Notes

## 1. What is RAG?
**Retrieval-Augmented Generation (RAG)** is an AI technique used to improve the accuracy and reliability of Large Language Models (LLMs) by fetching relevant facts from external sources before generating a response.

### The 3-Step RAG Process:
1. **Retrieval:** When a query is made, the system searches an external database or document repository for text chunks relevant to the query.
2. **Augmentation:** The original query is combined with the retrieved documents into a single prompt for the LLM.
3. **Generation:** The LLM reads the context and facts provided in the prompt to generate an accurate, verified answer.

### Why Use RAG?
* **Reduces Hallucinations:** Relies on verified source documents instead of guessing.
* **Access to Private Data:** Enables querying internal company docs, personal notes, or private databases without retraining models.
* **Up-to-Date Knowledge:** Bypasses LLM training cutoffs by fetching real-time data.
* **Traceability:** Allows systems to cite their source documents.

---

## 2. Embeddings
**Embeddings** translate text, images, or audio into lists of numbers (**vectors**) so computers can understand semantic meaning.

* **Vector Space:** Words or concepts with similar meanings are placed close to each other in a multi-dimensional mathematical space.
* **Why They Matter:** They capture context, synonymy, and enable semantic search (matching by meaning rather than exact keywords).

---

## 3. Cosine Similarity
**Cosine Similarity** is a mathematical metric used to measure how similar two vectors are, regardless of their magnitude, by calculating the cosine of the angle between them.

* **Score of 1.0:** The texts share identical meanings or themes.
* **Score of 0.0:** The texts are completely unrelated.
* **Role in RAG:** Used by vector databases to quickly compare a query vector against millions of stored document vectors to find the highest matches.

---

## 4. Chunking
**Chunking** is the process of breaking large documents into smaller, manageable pieces before converting them into embeddings and storing them in a vector database.

### Why Chunking is Necessary:
* **Context Window Limits:** Keeps text within LLM token limits.
* **Retrieval Precision:** Prevents large document meanings from getting "averaged out," ensuring exact answers are found.

### Key Chunking Strategies:
1. **Fixed-Size Chunking (Character/Token):** Splits text by a set character length using tools like LangChain's `CharacterTextSplitter`.
2. **Recursive Chunking:** Splits text hierarchically (paragraphs, then sentences, then words) using `RecursiveCharacterTextSplitter` to preserve semantic boundaries.
3. **Structure-Aware Chunking:** Breaks text based on layout—such as Markdown headers (`MarkdownHeaderTextSplitter`) or programming syntax (`RecursiveCharacterTextSplitter.from_language()`).

---

## 5. Vector Databases
A **Vector Database** is specialized to store, manage, and query high-dimensional embedding vectors efficiently for similarity search.

* **How They Work:** Uses spatial indexing algorithms (like HNSW) to avoid brute-force searches, returning relevant matches in milliseconds.
* **Popular Options:** 
  * **Chroma / pgvector:** Great for local development and lightweight setups.
  * **Pinecone / Qdrant / Milvus:** Powerful for production, scaling, and enterprise workloads.