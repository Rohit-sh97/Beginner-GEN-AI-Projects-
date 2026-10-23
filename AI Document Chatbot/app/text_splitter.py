

from app.console import print_status


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    print_status(
        f"✂️ [SPLITTER] Starting: {len(text)} characters, "
        f"chunk size {chunk_size}, overlap {overlap}"
    )
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap values")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    print_status(f"✅ [SPLITTER] Finished: produced {len(chunks)} non-empty chunk(s)")
    return chunks

            