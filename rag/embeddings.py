from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2') # has 384 dimensions

vec = model.encode("A cat is sleeping on couch")
print(vec.shape)
print(vec[:5])
print(len(vec))


# to compare 2 pieces of text - encode both - take a cosine similarity

from numpy import dot
from numpy.linalg import norm

vec1 = model.encode("A cat is sleeping on couch")
vec2 = model.encode("A kitten is napping on the sofa")

similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))

print(f"Cosine Similarity {similarity}")

