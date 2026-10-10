from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

large_data = """
Chunking is the process of breaking large documents (like PDFs, long articles, or entire books) into smaller, manageable pieces (or "chunks") before converting them into embeddings and storing them in a vector database.

It is a crucial step in building effective RAG pipelines.

Why Chunking is Necessary
Context Window Limits: LLMs can only process a limited amount of text at once (their context window). Feeding an entire 50-page document into a prompt is inefficient and expensive.

Precision in Retrieval: If you store an entire document as a single embedding, its meaning gets "averaged out." If a user asks a specific question, a massive document vector won't match as closely as a small, focused chunk containing the exact answer.

Token Efficiency: Smaller chunks mean you only retrieve and send the exact paragraphs needed to answer a query, saving tokens and reducing noise for the LLM.

Common Chunking Strategies
Fixed-Size Chunking: Splitting text strictly by a set number of characters or tokens (e.g., every 500 characters). This is easy to implement, but it risks cutting sentences or paragraphs in half.

Recursive Chunking: A smarter approach where the system tries to split text using natural boundaries (like paragraphs \n\n, then sentences ., then words) until each chunk fits the desired size limit. This preserves semantic meaning much better.

Semantic Chunking: Using an embedding model to evaluate where the topic of a text naturally shifts, creating chunks based on changes in meaning rather than fixed lengths.

Key Concepts: Chunk Size & Overlap
When configuring chunking, developers typically adjust two main parameters:

Chunk Size: How large each piece is (e.g., 250 to 500 tokens is a common sweet spot for balance).

Chunk Overlap: Having consecutive chunks share a small amount of text (e.g., 50 tokens) so that if an important idea or sentence falls right on a split boundary, it isn't lost or separated from its context.
"""

chunks = splitter.split_text(large_data)
print(chunks)
print(len(chunks))


# Fixed Chunking

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=30,
    chunk_overlap=10,
    length_function=len
)

chunks = text_splitter.split_text(large_data)


for i , chunk in enumerate(chunks):
    print(f"chunk {i+1}, {repr( chunk)}")



# Structure Aware Chunking
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """
# Introduction to RAG
Retrieval-Augmented Generation improves LLM accuracy.

## Embeddings
Embeddings convert text into vectors.

### Cosine Similarity
Cosine similarity measures the angle between vectors.
"""


# Define the headers you want to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_document)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content}")
    print(f"Chunk Metadata: {chunk.metadata}")