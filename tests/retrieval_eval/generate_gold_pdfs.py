"""Generate real PDF fixture files for the gold evaluation benchmark."""

import io
from pathlib import Path

from gold_corpus import GOLD_PAPERS


def generate_pdf_from_pages(pages_text: list[str]) -> bytes:
    """Generate a valid multi-page PDF 1.4 document containing selectable text per page."""
    num_pages = len(pages_text)
    page_obj_ids = [3 + i * 2 for i in range(num_pages)]
    font_id = 3 + num_pages * 2

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}

    def write_obj(num: int, data: bytes):
        offsets[num] = out.tell()
        out.write(f"{num} 0 obj\n".encode())
        out.write(data)
        out.write(b"\nendobj\n")

    write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    write_obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode())

    for i, text in enumerate(pages_text):
        pid = page_obj_ids[i]
        cid = pid + 1
        # Escape parenthesis in PDF text string
        escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT\n/F1 12 Tf\n72 700 Td\n({escaped_text}) Tj\nET\n".encode("utf-8")
        write_obj(
            pid,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode(),
        )
        write_obj(
            cid,
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        )

    write_obj(font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_pos = out.tell()
    total_objs = font_id + 1
    out.write(f"xref\n0 {total_objs}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for i in range(1, total_objs):
        out.write(f"{offsets[i]:010d} 00000 n \n".encode())
    out.write(
        f"trailer << /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    return out.getvalue()


def main():
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    for paper in GOLD_PAPERS:
        pages_text = [page["text"] for page in paper["pages"]]
        pdf_bytes = generate_pdf_from_pages(pages_text)
        pdf_path = fixtures_dir / paper["filename"]
        pdf_path.write_bytes(pdf_bytes)
        print(f"Generated {pdf_path.name} ({len(pages_text)} pages, {len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    main()
