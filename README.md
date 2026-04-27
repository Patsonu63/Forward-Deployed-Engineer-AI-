# Forward-Deployed-Engineer-AI-
how to study and actual work FDE

## Day 1: Python, Git & API Basics
Day 1 — Complete Checklist
Here's exactly what you should do today, in order:
Setup (30 min)

Install Python 3.11+ and VS Code
Run all commands in the Setup artifact
Get your free API key from console.anthropic.com
Add it to your .env file

Code practice (2 hrs)
5. Run the OOP file — modify the Customer class to add an urgency_score() method
6. Run the JSON file — add a function that appends a new customer and saves back to the file
7. Run the requests file — try fetching your own GitHub profile
8. Run the Claude API file — experiment with changing the system prompt and see how the persona changes
Git (30 min)
9. Initialize a repo in your fde-learning folder
10. Commit each file separately with meaningful commit messages
11. Push to a new GitHub repo
Experiment (1 hr)
12. Change temperature in the Claude call — try 0 vs 1 and compare outputs on the same question
13. Try making Claude return different JSON structures — this is core FDE prompt engineering
#### DAY2
# Topic 1: How LLMs Work — The Mental Model You Need
  
<img width="1198" height="600" alt="image" src="https://github.com/user-attachments/assets/6038473e-1220-4304-8854-ef5a4281f5a3" />

This is the full pipeline every time you call an LLM API. Your text gets tokenized into integer IDs, converted to vectors, processed through stacked transformer layers (where attention and temperature live), and decoded back to text.
Day 2 Complete Checklist
Here's exactly what to do today, step by step:
Understand the concepts (1 hr) — re-read the LLM pipeline diagram until it's clear. Understand what happens at each stage before touching code. Pay special attention to why temperature=0 gives the same answer every time (it always picks the highest probability token).
Run every code file in order (2 hrs):

d2_tokens.py — run the temperature experiment 3 times at temp=1.0 and see different outputs each time
d2_api.py — read every field of the response object in the terminal
d2_prompts.py — run all 6 techniques, then modify the few-shot examples and see how the output changes
d2_hallucination.py — deliberately try to make Claude hallucinate by asking about very recent events without grounding



Experiments to try (1 hr):

Change the system prompt persona in technique 5 to something completely different (e.g., a pirate) and see the same question answered differently
Add a 4th example to the few-shot prompt — see if it improves consistency
In the grounding example, ask a question that's NOT in the knowledge base — Claude should say it doesn't know

# Build the capstone (1 hr): Run d2_capstone.py with all 4 test tickets. Then add your own ticket and see how it classifies. Try to break it — write an ambiguous ticket and see how it handles confidence: low.
# Commit everything to GitHub with message: feat: day 2 — LLM fundamentals and prompt engineering lab




## Day 3: RAG & Vector Databases — Complete Deep Dive
Let's build everything from scratch — embeddings, vector search, chunking, and a full RAG pipeline.
Topic 1: What is RAG & Why It Exists
<img width="1198" height="740" alt="image" src="https://github.com/user-attachments/assets/7b1e8121-3448-4a3e-882e-81dc68bc7a2e" />
The key insight: RAG has two separate flows. The offline flow runs once — it processes your documents, splits them, embeds them, and stores the vectors. The online flow runs every time a user asks a question — it embeds the query, finds the most similar chunks, and sends them to the LLM as context. Click any node to dive deeper.

# Day 3: RAG & Vector Databases — Complete Deep Dive

Let's build everything from scratch — embeddings, vector search, chunking, and a full RAG pipeline.

---

## Topic 1: What is RAG & Why It ExistsThe key insight: RAG has two separate flows. The offline flow runs once — it processes your documents, splits them, embeds them, and stores the vectors. The online flow runs every time a user asks a question — it embeds the query, finds the most similar chunks, and sends them to the LLM as context. Click any node to dive deeper.

---

## Topic 2: Embeddings From Scratch---

## Topic 3: Text Chunking — The Most Underrated RAG Skill---

## Topic 4: ChromaDB — Your First Vector Database---

## Topic 5: Build the Complete RAG Pipeline---

## Topic 6: RAG Evaluation with RAGAS---

## Day 3 — Complete Checklist

Here's exactly what to do today, in order:

**Setup (15 min)**
Run `pip install chromadb voyageai anthropic pypdf2 python-dotenv` and get a free Voyage AI key from `voyageai.com` — it gives you 200M free tokens, more than enough for all of Day 3 and Day 7.

**Run every file in order (3 hrs):**
1. `d3_embeddings.py` — run the semantic similarity experiment. Change the sentences and watch the scores shift. This is the most important intuition to build on Day 3.
2. `d3_chunking.py` — run all 4 chunking strategies on the sample doc. Then paste in a real document (any `.txt` file) and compare the chunk quality visually.
3. `d3_chromadb.py` — store documents, run all 4 queries, try the metadata filter. Check the `./chroma_db` folder that gets created on disk.
4. `d3_rag_pipeline.py` — this is the full system. Run it, read every print statement carefully. Change one of the test queries to something NOT in the document and verify it returns "I don't have information about that."
5. `d3_eval.py` — look at test case 3 carefully. The answer is hallucinated, and the faithfulness score should catch it. This is exactly how you'd audit a client's RAG system.

**Key experiments to try:**
- In `d3_rag_pipeline.py`, remove the system prompt grounding rules and re-run — watch Claude start hallucinating answers
- Change `n_results` from 4 to 1 and see how answer quality drops
- Add a completely off-topic chunk to ChromaDB (e.g., a recipe) and verify it doesn't get retrieved for technical questions

**Commit to GitHub:** `feat: day 3 — embeddings, chunking, chromadb, full RAG pipeline with evaluation`








