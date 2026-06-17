import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=30.0)

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")

def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return f.read()

def get_relevant_chunks(query, knowledge_base, top_k=3):
    chunks = knowledge_base.split("\n\n")
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_k]]

def get_rag_response(user_query):
    knowledge_base = load_knowledge_base()
    relevant_chunks = get_relevant_chunks(user_query, knowledge_base)

    if not relevant_chunks:
        context = "No specific information found in knowledge base."
    else:
        context = "\n\n".join(relevant_chunks)

    prompt = f"""You are Nidra AI, a sleep health and stress management assistant.
Use the following context from the knowledge base to answer the user's question.
If the context does not fully answer the question, use your general knowledge but stay focused on sleep and stress topics.

Context:
{context}

User Question: {user_query}

Respond in 2-3 sentences only. Be direct and friendly. No bullet points."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content


