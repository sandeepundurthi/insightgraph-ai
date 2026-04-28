from pathlib import Path
import re
import json


def clean_text(text: str) -> str:
    import re

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Fix PDF line-break hyphenation: "halluci-\nnations" -> "hallucinations"
    text = re.sub(r"-\s+", "", text)

    # Add spaces between glued words caused by PDF extraction
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove references section if present
    text = re.sub(r"References\s+.*", "", text, flags=re.IGNORECASE)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def process_documents(input_dir="data/processed", output_file="data/chunks/chunks.json"):
    input_path = Path(input_dir)
    all_chunks = []

    for txt_file in input_path.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{txt_file.stem}_chunk_{i}",
                "source": txt_file.name,
                "text": chunk
            })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {output_file}")


if __name__ == "__main__":
    process_documents()
