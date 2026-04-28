from pathlib import Path
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


def load_all_pdfs(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pdf_file in raw_path.glob("*.pdf"):
        print(f"Extracting: {pdf_file.name}")

        text = extract_text_from_pdf(str(pdf_file))

        output_file = output_path / f"{pdf_file.stem}.txt"
        output_file.write_text(text, encoding="utf-8")

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    load_all_pdfs()
