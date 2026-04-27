# pip install chromadb voyageai
import chromadb
import voyageai
import os
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════
# CHROMADB — Local vector database (no server needed)
# Great for development, prototyping, and small-scale prod
# Data stored locally in a folder on disk
# ══════════════════════════════════════════════════════

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    return vo.embed(texts, model="voyage-3-lite", input_type=input_type).embeddings


# ── 1. Create a persistent ChromaDB client ─────────────
# persistent_client: saves to disk → survives restarts
client = chromadb.PersistentClient(path="./chroma_db")

# ── 2. Create (or get existing) a collection ──────────
# A collection = a table in a regular database
# Each collection has its own vector space
collection = client.get_or_create_collection(
    name="product_docs",
    metadata={"description": "DataSync Pro product documentation"}
)
print(f"Collection ready: {collection.name} ({collection.count()} docs)")


# ── 3. Add documents to the collection ────────────────
# Each document needs: id, text (document), embedding, metadata
documents = [
    "To install DataSync Pro, download the installer from our portal and run with admin privileges.",
    "For SAML configuration, upload your IDP metadata XML file in Settings > Security > SSO.",
    "OAuth 2.0 requires registering a redirect URI. Format: https://your-domain.datasync.io/auth/callback",
    "DataSync supports PostgreSQL, MySQL, MongoDB, Snowflake, BigQuery, Salesforce, and HubSpot.",
    "Error DS-401 means your authentication token expired. Re-authenticate via Settings > Connections.",
    "Error DS-500 indicates a server-side issue. Check status.datasync.io and contact support.",
    "All credentials are encrypted using AES-256. Test your connection before saving.",
    "Installation typically takes 5-10 minutes. Administrator access is required.",
]

ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [
    {"section": "installation",     "version": "3.2"},
    {"section": "authentication",   "version": "3.2"},
    {"section": "authentication",   "version": "3.2"},
    {"section": "connectors",       "version": "3.2"},
    {"section": "troubleshooting",  "version": "3.2"},
    {"section": "troubleshooting",  "version": "3.2"},
    {"section": "connectors",       "version": "3.2"},
    {"section": "installation",     "version": "3.2"},
]

# Generate embeddings
embeddings = embed_texts(documents, input_type="document")

# Add to collection (upsert = update if exists, insert if not)
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)
print(f"Added {len(documents)} documents. Total: {collection.count()}")


# ── 4. Basic similarity search ─────────────────────────
print("\n=== Basic Semantic Search ===")

def search(query: str, n_results: int = 3) -> list[dict]:
    """Search the collection for the most relevant documents."""
    query_embedding = embed_texts([query], input_type="query")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "section":  results["metadatas"][0][i]["section"],
            "distance": round(results["distances"][0][i], 4),
            # distance: lower = more similar (ChromaDB uses L2 by default)
            # convert to similarity: 1 - distance/2 (rough approximation)
            "similarity": round(1 - results["distances"][0][i] / 2, 4)
        })
    return output

queries = [
    "How do I fix error 401?",
    "Set up SSO with Okta",
    "What databases are supported?",
    "How long does installation take?",
]

for q in queries:
    print(f"\nQuery: '{q}'")
    hits = search(q, n_results=2)
    for h in hits:
        print(f"  [{h['similarity']:.3f}] ({h['section']}) {h['text'][:70]}...")


# ── 5. Metadata filtering — critical for enterprise use ─
print("\n=== Filtered Search (section=troubleshooting only) ===")

query_emb = embed_texts(["something isn't working"], input_type="query")[0]
filtered = collection.query(
    query_embeddings=[query_emb],
    n_results=3,
    where={"section": "troubleshooting"},   # only search this section
    include=["documents", "distances"]
)
for doc, dist in zip(filtered["documents"][0], filtered["distances"][0]):
    print(f"  [{1-dist/2:.3f}] {doc[:80]}...")


# ── 6. Inspect and manage the collection ───────────────
print("\n=== Collection Management ===")

# Get specific document by ID
item = collection.get(ids=["doc_0"])
print(f"doc_0: {item['documents'][0][:60]}...")

# Get all documents in a section
auth_docs = collection.get(where={"section": "authentication"})
print(f"\nAuthentication docs: {len(auth_docs['ids'])}")

# Count
print(f"Total documents: {collection.count()}")

# Delete a document
# collection.delete(ids=["doc_0"])

# List all collections
all_collections = client.list_collections()
print(f"\nAll collections: {[c.name for c in all_collections]}")
