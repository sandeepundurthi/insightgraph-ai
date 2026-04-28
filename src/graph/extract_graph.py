import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")


def extract_entities_relationships(text: str):
    prompt = f"""
You are an expert knowledge graph extraction system.

Extract important entities and relationships from the text.

Return ONLY valid JSON with this structure:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "Concept/Method/Model/Dataset/Metric/Problem/Tool"
    }}
  ],
  "relationships": [
    {{
      "source": "entity 1",
      "relationship": "relationship type",
      "target": "entity 2",
      "evidence": "short evidence from text"
    }}
  ]
}}

Text:
{text}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You extract structured knowledge graphs from text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}


def build_graph_json(chunks_file="data/chunks/chunks.json", output_file="data/graphs/kg.json"):
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    graph_data = []

    for chunk in chunks[:20]:
        print(f"Processing {chunk['chunk_id']}")

        extracted = extract_entities_relationships(chunk["text"])
        extracted["chunk_id"] = chunk["chunk_id"]
        extracted["source"] = chunk["source"]

        graph_data.append(extracted)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Saved graph data to {output_file}")


if __name__ == "__main__":
    build_graph_json()
