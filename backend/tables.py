import logging

import pdfplumber

logger = logging.getLogger(__name__)


def _table_to_markdown(rows: list[list]) -> str:
    """Convert one pdfplumber-extracted table (list of rows) into a Markdown table.

    Cells can be None when a table has ragged/merged cells — rendered as an
    empty string rather than the literal "None".
    """
    cleaned_rows = [
        [(cell or "").strip().replace("\n", " ") for cell in row] for row in rows
    ]

    header, *body = cleaned_rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_tables_as_markdown(file_path: str, page_number: int) -> list[str]:
    """Return one Markdown table string per table detected on the page.

    Pure structural parsing, not OCR — no LLM call, so this is cheap enough
    to run on every upload automatically (unlike the image-description
    feature, which is opt-in because it costs a vision LLM call per page).
    """
    with pdfplumber.open(file_path) as pdf:
        page = pdf.pages[page_number]
        raw_tables = page.extract_tables()

    tables = [_table_to_markdown(rows) for rows in raw_tables if rows]

    logger.debug(
        "extract_tables_as_markdown(%s, page=%d): found %d table(s)",
        file_path, page_number, len(tables),
    )
    return tables
