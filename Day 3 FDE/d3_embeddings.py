import os, math
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════
# WHAT IS AN EMBEDDING?
# Text converted to a list of numbers (a vector) that
# captures semantic meaning. Similar meaning = similar vectors.
#
# "dog"  → [0.21, -0.43, 0.87, ...]  (1536 numbers)
# "cat"  → [0.19, -0.41, 0.85, ...]  (close to dog!)
# "car"  → [-0.62, 0.33, -0.11, ...] (far from dog)
# ══════════════════════════════════════════════════════

# We'll use the voyage-3-lite model via Anthropic for embeddings
# Install: pip install anthropic[bedrock] voyageai
# OR use openai embeddings — both work the same way

# For this demo we use a pure-Python cosine similarity
# so no extra library is needed beyond anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Get an embedding via Voyage AI (via Anthropic) ─────
# NOTE: If you don't have Voyage access, use the OpenAI
# approach in the comment below — identical concept

# Option A: Voyage (best quality, use in production)
# pip install voyageai
import voyageai
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))  # get free key at voyageai.com

def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """
    Embed a list of texts. Returns list of vectors.
    input_type: "document" for chunks, "query" for user questions
    """
    result = vo.embed(texts, model="voyage-3-lite", input_type=input_type)
    return result.embeddings   # list of float lists

# Option B: OpenAI (if you have OpenAI key)
# from openai import OpenAI
# oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# def embed(texts, input_type="document"):
#     resp = oai.embeddings.create(input=texts, model="text-embedding-3-small")
#     return [item.embedding for item in resp.data]


# ══════════════════════════════════════════════════════
# COSINE SIMILARITY — The math behind vector search
# Measures the angle between two vectors
# 1.0 = identical direction (same meaning)
# 0.0 = perpendicular (unrelated)
# -1.0 = opposite directions (opposite meaning)
# ══════════════════════════════════════════════════════

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    cos(θ) = (A · B) / (|A| × |B|)
    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1  = math.sqrt(sum(a * a for a in v1))
    magnitude2  = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


# ── Experiment 1: Semantic similarity ─────────────────
print("=== Semantic Similarity Experiment ===")

sentences = [
    "How do I reset my password?",          # query
    "Steps to change your account password", # very similar
    "I forgot my login credentials",         # somewhat similar
    "How do I export data to CSV?",          # different topic
    "What is the refund policy?",            # unrelated
]

# Embed all sentences
vectors = embed(sentences)

query_vec = vectors[0]
print(f"Query: '{sentences[0]}'\n")

for i in range(1, len(sentences)):
    score = cosine_similarity(query_vec, vectors[i])
    bar   = "█" * int(score * 30)
    print(f"  [{score:.3f}] {bar}")
    print(f"          '{sentences[i]}'")

# Expected output (roughly):
# [0.92] ████████████████████████████  'Steps to change your account password'
# [0.78] ████████████████████          'I forgot my login credentials'
# [0.41] ████████████                  'How do I export data to CSV?'
# [0.31] █████████                     'What is the refund policy?'


# ── Experiment 2: Cross-domain similarity ─────────────
print("\n=== Cross-domain Similarity ===")

pairs = [
    ("dog", "puppy"),           # very similar
    ("dog", "cat"),             # somewhat similar (both animals)
    ("dog", "car"),             # unrelated
    ("king", "queen"),          # semantic relationship
    ("Python", "programming"),  # related concepts
    ("Python", "snake"),        # ambiguous — watch what embedding picks!
]

for word1, word2 in pairs:
    v1, v2 = embed([word1, word2])
    score  = cosine_similarity(v1, v2)
    print(f"  '{word1}' vs '{word2}': {score:.3f}")


# ── Experiment 3: Visualise a mini vector space ────────
print("\n=== Mini Vector Space (2D projection via PCA) ===")
# We can't visualise 1536-dim vectors directly,
# but we can project to 2D using simple PCA

def simple_2d_projection(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Very simple 2-component projection for visualisation."""
    # Use first 2 principal components (simplified)
    n = len(vectors)
    # Center the data
    means = [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]
    centered = [[v[i] - means[i] for i in range(len(v))] for v in vectors]
    # Project onto dims 0 and 1 of centered data (simplified PCA)
    return [(c[0] * 10, c[1] * 10) for c in centered]

words = ["dog", "cat", "fish", "car", "truck", "bus", "apple", "banana"]
word_vecs = embed(words)
points = simple_2d_projection(word_vecs)

print("2D projection (approximate clustering):")
for word, (x, y) in zip(words, points):
    print(f"  {word:10s} x={x:+.3f}  y={y:+.3f}")
# Animals should cluster together, vehicles together, fruits together
