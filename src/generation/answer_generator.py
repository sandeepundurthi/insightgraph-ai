import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.retrieval.hybrid_retriever import hybrid_retrieve

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/free")


def generate_answer(question, entity_hint):
    context = hybrid_retrieve(question, entity_hint)

    prompt = f"""
You are an expert GraphRAG research assistant.

Answer the question using BOTH:
1. Graph relationships
2. Retrieved text evidence

Question:
{question}

Vector Context:
{context["vector_context"]}

Graph Context:
{context["graph_context"]}

Instructions:
- Use specific facts from vector context (numbers, results, comparisons)
- Explicitly mention graph relationships like:
  "GraphRAG INTEGRATES Knowledge Graphs"
- Explain WHY it reduces hallucinations
- Include measurable improvements if available
- Do NOT be generic
- If evidence is weak, say so
- Keep answer structured and clear
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You answer using graph and retrieved document evidence only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    answer = generate_answer(
        question="How does GraphRAG reduce hallucinations?",
        entity_hint="GraphRAG"
    )

    print("\n===== ANSWER =====\n")
    print(answer)
