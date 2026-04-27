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







