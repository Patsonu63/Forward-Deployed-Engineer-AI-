import re

# ══════════════════════════════════════════════════════
# WHY CHUNKING MATTERS
# Too large  → retrieval returns entire chapters, wastes context
# Too small  → retrieval returns sentence fragments, loses context
# Just right → retrieves the exact paragraph that answers the question
#
# Chunk size: 200-500 tokens is the sweet spot for most use cases
# Overlap: 10-20% overlap prevents cutting answers at boundaries
# ══════════════════════════════════════════════════════

SAMPLE_DOC = """
# Product Manual: DataSync Pro v3.2

## Getting Started

DataSync Pro is an enterprise data synchronization platform that connects
your databases, APIs, and cloud storage in real-time. Before you begin,
ensure you have administrator access to your account.

To install DataSync Pro, download the installer from our portal and run it
with administrator privileges. The installation wizard will guide you through
the setup process. Installation typically takes 5-10 minutes.

## Authentication & SSO Setup

DataSync Pro supports SAML 2.0, OAuth 2.0, and LDAP for enterprise
authentication. To configure SSO, navigate to Settings > Security > SSO.

For SAML configuration, you will need your Identity Provider's metadata XML
file. Upload this file in the SAML Configuration section. Ensure your IDP
is configured to send the email attribute as the NameID.

OAuth 2.0 requires registering a redirect URI in your OAuth provider.
The callback URL format is: https://your-domain.datasync.io/auth/callback

## Data Connectors

DataSync Pro currently supports over 200 data connectors including:
PostgreSQL, MySQL, MongoDB, Snowflake, BigQuery, Salesforce, and HubSpot.

To add a connector, click the + button in the Connections panel. Enter your
credentials — DataSync encrypts all credentials using AES-256. Test the
connection before saving to verify your credentials are correct.

## Troubleshooting

If the sync fails with error code DS-401, your authentication token has
expired. Re-authenticate by navigating to Settings > Connections and
clicking Refresh Token next to the affected connector.

Error code DS-500 indicates a server-side issue. Check our status page at
status.datasync.io and contact support if the issue persists for more than
30 minutes.
"""


# ══════════════════════════════════════════════════════
# STRATEGY 1: FIXED SIZE CHUNKING
# Simplest — split every N characters with overlap
# Problem: often cuts mid-sentence
# ══════════════════════════════════════════════════════

def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "id":    idx,
            "text":  chunk,
            "start": start,
            "end":   min(end, len(text)),
            "chars": len(chunk)
        })
        start += chunk_size - overlap
        idx   += 1

    return chunks

chunks = fixed_size_chunks(SAMPLE_DOC, chunk_size=300, overlap=50)
print(f"=== Fixed Size Chunking (300 chars, 50 overlap) ===")
print(f"Total chunks: {len(chunks)}")
print(f"\nChunk 0:\n{chunks[0]['text'][:200]}...")
print(f"\nChunk 1 (note overlap with chunk 0):\n{chunks[1]['text'][:200]}...")


# ══════════════════════════════════════════════════════
# STRATEGY 2: SENTENCE-AWARE CHUNKING
# Better — respects sentence boundaries
# Never cuts a sentence in half
# ══════════════════════════════════════════════════════

def sentence_chunks(text: str, max_tokens: int = 150, overlap_sentences: int = 1) -> list[dict]:
    """
    Split into chunks that always end on sentence boundaries.
    Approximation: 1 token ≈ 4 characters
    """
    max_chars = max_tokens * 4

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks  = []
    current = []
    current_len = 0
    idx = 0

    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)

        if current_len + sentence_len > max_chars and current:
            # Save current chunk
            chunks.append({
                "id":    idx,
                "text":  " ".join(current),
                "tokens_approx": current_len // 4
            })
            idx += 1
            # Overlap: keep last N sentences
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += sentence_len

    # Don't forget last chunk
    if current:
        chunks.append({
            "id":    idx,
            "text":  " ".join(current),
            "tokens_approx": current_len // 4
        })

    return chunks

s_chunks = sentence_chunks(SAMPLE_DOC, max_tokens=120)
print(f"\n=== Sentence-Aware Chunking (120 tokens max) ===")
print(f"Total chunks: {len(s_chunks)}")
for c in s_chunks[:3]:
    print(f"\nChunk {c['id']} (~{c['tokens_approx']} tokens):")
    print(f"  {c['text'][:180]}...")


# ══════════════════════════════════════════════════════
# STRATEGY 3: SEMANTIC / HEADING-BASED CHUNKING (BEST)
# Split on document structure: headers, sections, paragraphs
# Each chunk = one complete idea/section
# ══════════════════════════════════════════════════════

def semantic_chunks(text: str, max_chars: int = 1000) -> list[dict]:
    """
    Split on markdown headings and paragraph breaks.
    This is the best strategy for structured documents.
    """
    chunks = []
    # Split on ## headings or double newlines
    sections = re.split(r'\n(?=##|\n)', text)
    idx = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading if present
        lines   = section.split('\n')
        heading = lines[0].replace('#', '').strip() if lines[0].startswith('#') else "section"
        content = '\n'.join(lines).strip()

        # If section is too large, split into sub-chunks
        if len(content) > max_chars:
            sub = fixed_size_chunks(content, chunk_size=max_chars, overlap=100)
            for s in sub:
                chunks.append({
                    "id":      idx,
                    "heading": heading,
                    "text":    s['text'],
                    "tokens_approx": len(s['text']) // 4
                })
                idx += 1
        else:
            chunks.append({
                "id":      idx,
                "heading": heading,
                "text":    content,
                "tokens_approx": len(content) // 4
            })
            idx += 1

    return chunks

sem_chunks = semantic_chunks(SAMPLE_DOC)
print(f"\n=== Semantic / Heading-Based Chunking ===")
print(f"Total chunks: {len(sem_chunks)}")
for c in sem_chunks:
    print(f"\nChunk {c['id']} — [{c['heading']}] (~{c['tokens_approx']} tokens)")
    print(f"  {c['text'][:120].strip()}...")


# ══════════════════════════════════════════════════════
# STRATEGY 4: ADD METADATA TO EVERY CHUNK
# Metadata makes retrieval dramatically better
# Always attach: source, page, heading, chunk_id
# ══════════════════════════════════════════════════════

def chunks_with_metadata(text: str, source: str, doc_id: str) -> list[dict]:
    """Production-grade chunking with full metadata."""
    raw_chunks = semantic_chunks(text)
    enriched   = []

    for c in raw_chunks:
        enriched.append({
            # Content
            "text": c["text"],
            # Metadata for retrieval filtering
            "metadata": {
                "source":    source,
                "doc_id":    doc_id,
                "chunk_id":  f"{doc_id}_chunk_{c['id']}",
                "heading":   c["heading"],
                "tokens":    c["tokens_approx"],
                # Useful for filtering: "only search the troubleshooting section"
                "section":   c["heading"].lower().replace(" ", "_"),
            }
        })

    return enriched

production_chunks = chunks_with_metadata(
    SAMPLE_DOC,
    source="DataSync Pro Manual v3.2",
    doc_id="datasync_manual_v32"
)

print(f"\n=== Production Chunks with Metadata ===")
for c in production_chunks[:2]:
    print(f"\nText: {c['text'][:80]}...")
    print(f"Metadata: {c['metadata']}")


# ── Chunking strategy comparison ─────────────────────
print("\n\n=== Strategy Comparison ===")
print(f"Fixed size:       {len(fixed_size_chunks(SAMPLE_DOC))} chunks — fast, may cut sentences")
print(f"Sentence-aware:   {len(sentence_chunks(SAMPLE_DOC))} chunks — clean sentences, good default")
print(f"Semantic/heading: {len(semantic_chunks(SAMPLE_DOC))} chunks — best for structured docs")
print("\nRecommendation: Use semantic chunking for manuals/docs, sentence chunking for free-form text")
