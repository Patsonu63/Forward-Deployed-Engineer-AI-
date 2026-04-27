# pip install chromadb voyageai anthropic python-dotenv pypdf2
import os, re, json, time
import chromadb
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════
# COMPLETE RAG PIPELINE
# This is what you'll actually build for enterprise clients
# Handles: PDF loading, chunking, embedding, storage,
#          retrieval, grounded generation, evaluation
# ══════════════════════════════════════════════════════

claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
vo     = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
chroma = chromadb.PersistentClient(path="./rag_db")


# ════════════════════════════════
# STEP 1: DOCUMENT LOADING
# ════════════════════════════════

def load_pdf(filepath: str) -> str:
    """Load text from a PDF file."""
    import PyPDF2
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n\n".join(
            page.extract_text() for page in reader.pages
            if page.extract_text()
        )

def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# For this demo, we'll use an inline document
DEMO_DOCUMENT = """
# DataSync Pro — Complete User Guide

## Installation
Download the installer from portal.datasync.io. Run as administrator.
Installation requires Windows 10+ or Ubuntu 20.04+. Takes 5-10 minutes.
After installation, launch DataSync Pro and complete the setup wizard.
You will need your license key from your purchase confirmation email.

## Authentication
DataSync Pro supports three authentication methods:
1. Local accounts: username and password stored in DataSync
2. SAML 2.0: integrate with Okta, Azure AD, or any SAML provider
3. OAuth 2.0: connect via Google, GitHub, or custom OAuth server

To configure SAML, go to Settings > Security > SSO Configuration.
Upload your IDP metadata XML. Set the NameID format to email address.
Test the configuration using the built-in SAML debugger before going live.

## Data Connectors
Supported databases: PostgreSQL, MySQL, MariaDB, SQLite, MongoDB, DynamoDB.
Cloud warehouses: Snowflake, BigQuery, Redshift, Databricks.
CRM: Salesforce, HubSpot, Zoho CRM.
File storage: S3, Google Cloud Storage, Azure Blob Storage.

To add a connector: Connections > Add New > Select type > Enter credentials.
All credentials encrypted with AES-256. DataSync never stores plaintext passwords.
Run a connectivity test before saving any new connection.

## Sync Configuration
Create a sync job from the Jobs panel. Specify source, destination, and schedule.
Sync modes: Full refresh (replace all data), Incremental (new rows only), CDC (change data capture).
For CDC mode, ensure your source database has binary logging enabled.
Schedule options: every minute, hourly, daily, weekly, or custom cron expression.

## Monitoring & Alerts
View sync job status on the Dashboard. Each job shows: last run time, rows synced, duration, status.
Set up alerts via Settings > Notifications. Supports email, Slack, and PagerDuty.
Error logs available at Logs > Job Logs. Filter by date range, job name, or error type.

## Troubleshooting
DS-401: Token expired. Go to Settings > Connections > Refresh Token.
DS-403: Permission denied. Ensure service account has read access to source tables.
DS-404: Table or schema not found. Verify table names are case-sensitive and correct.
DS-500: Internal server error. Check status.datasync.io. Contact support if persists over 30 mins.
DS-503: Source database unreachable. Check network connectivity and firewall rules.

## API Reference
REST API available at https://api.datasync.io/v2
Authentication: Bearer token from Settings > API Keys
Rate limits: 100 requests/minute on Standard, 1000/minute on Enterprise.
Webhooks: configure at Settings > Webhooks to receive real-time sync events.
"""


# ════════════════════════════════
# STEP 2: SEMANTIC CHUNKING
# ════════════════════════════════

def chunk_document(text: str, doc_id: str, source: str) -> list[dict]:
    """Chunk a document into semantic sections with metadata."""
    chunks = []
    # Split on markdown headings
    sections = re.split(r'\n(?=##\s)', text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines   = section.split('\n')
        heading = lines[0].replace('#', '').strip()
        content = section.strip()

        # Split large sections into sub-chunks (500 token limit)
        max_chars = 2000
        if len(content) > max_chars:
            # Paragraph-level splitting for large sections
            paragraphs = content.split('\n\n')
            current, current_len = [], 0
            sub_idx = 0

            for para in paragraphs:
                if current_len + len(para) > max_chars and current:
                    chunks.append({
                        "id":    f"{doc_id}_{heading.lower().replace(' ','_')}_{sub_idx}",
                        "text":  '\n\n'.join(current),
                        "metadata": {
                            "doc_id":  doc_id,
                            "source":  source,
                            "heading": heading,
                            "section": heading.lower().replace(' ', '_')
                        }
                    })
                    sub_idx += 1
                    current, current_len = [], 0
                current.append(para)
                current_len += len(para)

            if current:
                chunks.append({
                    "id":    f"{doc_id}_{heading.lower().replace(' ','_')}_{sub_idx}",
                    "text":  '\n\n'.join(current),
                    "metadata": {
                        "doc_id":  doc_id,
                        "source":  source,
                        "heading": heading,
                        "section": heading.lower().replace(' ', '_')
                    }
                })
        else:
            chunks.append({
                "id":    f"{doc_id}_{heading.lower().replace(' ','_')}",
                "text":  content,
                "metadata": {
                    "doc_id":  doc_id,
                    "source":  source,
                    "heading": heading,
                    "section": heading.lower().replace(' ', '_')
                }
            })

    print(f"Chunked '{source}' into {len(chunks)} chunks")
    return chunks


# ════════════════════════════════
# STEP 3: EMBED & INDEX
# ════════════════════════════════

def index_document(text: str, doc_id: str, source: str, collection_name: str = "docs") -> chromadb.Collection:
    """Full pipeline: chunk → embed → store."""
    col    = chroma.get_or_create_collection(collection_name)
    chunks = chunk_document(text, doc_id, source)

    # Embed in batches (API has limits)
    BATCH = 32
    for i in range(0, len(chunks), BATCH):
        batch  = chunks[i:i+BATCH]
        texts  = [c["text"] for c in batch]
        embeds = vo.embed(texts, model="voyage-3-lite", input_type="document").embeddings

        col.upsert(
            ids        = [c["id"]       for c in batch],
            documents  = [c["text"]     for c in batch],
            embeddings = embeds,
            metadatas  = [c["metadata"] for c in batch]
        )
        print(f"  Indexed batch {i//BATCH + 1}: {len(batch)} chunks")

    print(f"Total in collection: {col.count()}")
    return col


# ════════════════════════════════
# STEP 4: RETRIEVAL
# ════════════════════════════════

def retrieve(query: str, collection: chromadb.Collection,
             n_results: int = 4, section_filter: str = None) -> list[dict]:
    """Retrieve the most relevant chunks for a query."""
    q_emb = vo.embed([query], model="voyage-3-lite", input_type="query").embeddings[0]

    where = {"section": section_filter} if section_filter else None

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "text":       results["documents"][0][i],
            "heading":    results["metadatas"][0][i]["heading"],
            "source":     results["metadatas"][0][i]["source"],
            "similarity": round(1 - results["distances"][0][i] / 2, 3)
        }
        for i in range(len(results["ids"][0]))
    ]


# ════════════════════════════════
# STEP 5: GROUNDED GENERATION
# ════════════════════════════════

SYSTEM_PROMPT = """You are a helpful technical support assistant for DataSync Pro.

Rules:
1. Answer ONLY using the provided context. Never use your own training data.
2. If the answer is not in the context, say exactly: "I don't have information about that in the documentation. Please contact support@datasync.io."
3. Always cite which section your answer comes from.
4. Be concise — 2-4 sentences unless the question requires more detail.
5. For error codes, always include the recommended fix steps."""

def rag_answer(query: str, collection: chromadb.Collection,
               n_results: int = 4, show_sources: bool = True) -> dict:
    """Full RAG pipeline: retrieve → build context → generate."""
    t0 = time.time()

    # Retrieve relevant chunks
    chunks = retrieve(query, collection, n_results=n_results)

    # Build context block
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {c['heading']} (relevance: {c['similarity']})]:\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Generate grounded answer
    prompt = f"""<context>
{context}
</context>

<question>{query}</question>

Answer the question using ONLY the context above. Cite the source section."""

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.content[0].text
    elapsed = round(time.time() - t0, 2)

    result = {
        "query":   query,
        "answer":  answer,
        "latency": f"{elapsed}s",
        "tokens":  response.usage.input_tokens + response.usage.output_tokens,
    }
    if show_sources:
        result["sources"] = [
            {"heading": c["heading"], "similarity": c["similarity"], "preview": c["text"][:80]+"..."}
            for c in chunks
        ]
    return result


# ════════════════════════════════
# STEP 6: EVALUATION
# ════════════════════════════════

def evaluate_rag(collection: chromadb.Collection) -> dict:
    """
    Run a test suite and score the RAG system.
    In production use RAGAS library for automated evaluation.
    """
    test_cases = [
        {
            "query":    "How do I fix DS-401 error?",
            "expected": "token expired",  # must appear in answer
        },
        {
            "query":    "What databases does DataSync support?",
            "expected": "postgresql",
        },
        {
            "query":    "How do I set up SAML SSO?",
            "expected": "saml",
        },
        {
            "query":    "What is the rate limit for the API?",
            "expected": "100 requests",
        },
        {
            "query":    "What is the CEO's name?",   # should be unanswerable
            "expected": "don't have information",
        }
    ]

    passed = 0
    print("=== RAG Evaluation ===\n")

    for tc in test_cases:
        result  = rag_answer(tc["query"], collection, show_sources=False)
        answer  = result["answer"].lower()
        correct = tc["expected"].lower() in answer

        status = "PASS" if correct else "FAIL"
        if correct:
            passed += 1

        print(f"[{status}] {tc['query']}")
        print(f"       Expected to contain: '{tc['expected']}'")
        print(f"       Answer: {result['answer'][:100]}...")
        print(f"       Latency: {result['latency']} | Tokens: {result['tokens']}\n")

    score = passed / len(test_cases)
    print(f"Score: {passed}/{len(test_cases)} = {score:.0%}")
    return {"score": score, "passed": passed, "total": len(test_cases)}


# ════════════════════════════════
# RUN THE FULL PIPELINE
# ════════════════════════════════

if __name__ == "__main__":
    print("=== Step 1: Index document ===")
    col = index_document(DEMO_DOCUMENT, "datasync_v3", "DataSync Pro User Guide v3")

    print("\n=== Step 2: Test queries ===")
    test_queries = [
        "How do I set up SAML with Okta?",
        "What does error DS-500 mean?",
        "How often can I schedule a sync job?",
        "How do I monitor job failures?",
    ]

    for q in test_queries:
        print(f"\nQ: {q}")
        result = rag_answer(q, col, show_sources=True)
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['heading'] for s in result.get('sources', [])]}")
        print(f"Latency: {result['latency']}")

    print("\n=== Step 3: Evaluation ===")
    evaluate_rag(col)
