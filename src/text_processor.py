import re
from collections import Counter
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from src.document_loader import (
    load_all_documents,
    show_loading_summary,
)


REPEATED_CORPORATE_LINES = {
    "NexoData Consulting",
    "Sol AI · Documento corporativo interno",
    "Documento corporativo interno",
    "Datos que conectan. Decisiones que transforman.",
}


COVER_NOISE_PATTERNS = [
    r"(?i)^nexodata$",
    r"(?i)^c\s+o\s+n\s+s\s+u\s+l\s+t\s+i\s+n\s+g$",
    r"(?i)^sol ai$",
    r"(?i)^inteligencia que ilumina decisiones$",
    r"(?i)^documento corporativo.*$",
    r"(?i)^versión[:\s].*$",
    r"(?i)^julio de 2026$",
]


def normalize_line(
    line: str,
) -> str:
    """
    Normaliza una línea para comparar encabezados,
    pies de página y contenido repetido.
    """
    normalized = line.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def remove_markdown_noise(
    text: str,
) -> str:
    """
    Elimina marcas visuales de Markdown sin borrar
    el contenido útil.
    """
    text = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s+",
        "",
        text,
    )

    text = re.sub(
        r"(\*\*|__|~~)",
        "",
        text,
    )
    text = re.sub(
        r"(?<!\*)\*(?!\*)",
        "",
        text,
    )
    text = re.sub(
        r"(?<!_)_(?!_)",
        "",
        text,
    )

    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r"\1",
        text,
    )

    text = re.sub(
        r"(?m)^\s*[-*_]{3,}\s*$",
        "",
        text,
    )

    text = re.sub(
        r"(?m)^\s*[-+*]\s+",
        "",
        text,
    )

    return text


def remove_pdf_noise(
    text: str,
    metadata: dict,
) -> str:
    """
    Elimina encabezados, pies y números de página
    extraídos desde documentos PDF.
    """
    if metadata.get("document_type") != "pdf":
        return text

    official_title = str(
        metadata.get(
            "official_title",
            "",
        )
    ).strip()

    cleaned_lines = []

    for raw_line in text.split("\n"):
        line = normalize_line(raw_line)

        if not line:
            cleaned_lines.append("")
            continue

        if re.fullmatch(
            r"(?i)página\s+\d+(?:\s+de\s+\d+)?",
            line,
        ):
            continue

        if re.fullmatch(
            r"\d{1,3}",
            line,
        ):
            continue

        if line in REPEATED_CORPORATE_LINES:
            continue

        if (
            official_title
            and line.casefold() == official_title.casefold()
        ):
            continue

        if re.fullmatch(
            r"(?i)sol ai\s*[·|-]\s*documento corporativo interno",
            line,
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def is_pdf_cover_noise(
    document: Document,
) -> bool:
    """
    Detecta portadas PDF con muy poco contenido informativo.

    Solo evalúa la primera página. No elimina automáticamente
    páginas que contengan párrafos sustanciales.
    """
    metadata = document.metadata

    if metadata.get("document_type") != "pdf":
        return False

    if metadata.get("page") != 1:
        return False

    text = document.page_content.strip()

    if not text:
        return True

    lines = [
        normalize_line(line)
        for line in text.split("\n")
        if normalize_line(line)
    ]

    if not lines:
        return True

    words = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE,
    )

    long_sentences = [
        line
        for line in lines
        if len(line) >= 120
    ]

    matched_noise_lines = 0

    for line in lines:
        if any(
            re.fullmatch(pattern, line)
            for pattern in COVER_NOISE_PATTERNS
        ):
            matched_noise_lines += 1

    noise_ratio = (
        matched_noise_lines / len(lines)
        if lines
        else 0
    )

    # Regla conservadora:
    # - primera página
    # - pocas palabras
    # - sin párrafos largos
    # - alta proporción de líneas corporativas o decorativas
    return (
        len(words) <= 45
        and not long_sentences
        and noise_ratio >= 0.35
    )


def filter_non_informative_pages(
    documents: List[Document],
) -> List[Document]:
    """
    Excluye páginas de portada detectadas como poco informativas.
    """
    filtered_documents = []

    for document in documents:
        if is_pdf_cover_noise(document):
            metadata = document.metadata

            print(
                "[OMITIDO] Portada sin contenido sustancial: "
                f"{metadata.get('file_name')} "
                f"(página {metadata.get('page')})"
            )
            continue

        filtered_documents.append(document)

    return filtered_documents


def detect_repeated_pdf_lines(
    documents: List[Document],
    minimum_occurrences: int = 4,
) -> dict[str, set[str]]:
    """
    Detecta líneas cortas repetidas en varias páginas
    del mismo PDF.
    """
    lines_by_file: dict[str, Counter] = {}

    for document in documents:
        if document.metadata.get("document_type") != "pdf":
            continue

        file_name = str(
            document.metadata.get(
                "file_name",
                "documento.pdf",
            )
        )

        lines_by_file.setdefault(
            file_name,
            Counter(),
        )

        unique_lines = {
            normalize_line(line)
            for line in document.page_content.split("\n")
            if normalize_line(line)
        }

        lines_by_file[file_name].update(
            unique_lines
        )

    repeated_by_file: dict[str, set[str]] = {}

    for file_name, counter in lines_by_file.items():
        repeated_by_file[file_name] = {
            line
            for line, count in counter.items()
            if (
                count >= minimum_occurrences
                and 3 <= len(line) <= 90
                and not line.endswith((".", ":", ";", "?"))
                and not line.startswith(("•", "-", "*"))
            )
        }

    return repeated_by_file


def remove_repeated_pdf_lines(
    documents: List[Document],
) -> List[Document]:
    """
    Elimina encabezados y pies repetidos detectados
    entre páginas del mismo PDF.
    """
    repeated_by_file = detect_repeated_pdf_lines(
        documents
    )

    processed_documents = []

    for document in documents:
        if document.metadata.get("document_type") != "pdf":
            processed_documents.append(document)
            continue

        file_name = str(
            document.metadata.get(
                "file_name",
                "documento.pdf",
            )
        )

        repeated_lines = repeated_by_file.get(
            file_name,
            set(),
        )

        cleaned_lines = []

        for raw_line in document.page_content.split("\n"):
            normalized = normalize_line(raw_line)

            if normalized and normalized in repeated_lines:
                continue

            cleaned_lines.append(raw_line)

        processed_documents.append(
            Document(
                page_content="\n".join(cleaned_lines),
                metadata=document.metadata.copy(),
            )
        )

    return processed_documents


def clean_text(
    text: str,
    metadata: dict | None = None,
) -> str:
    """
    Limpia espacios, saltos de línea, caracteres de control,
    ruido de PDF y marcas visuales de Markdown.
    """
    metadata = metadata or {}

    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = remove_pdf_noise(
        text=text,
        metadata=metadata,
    )

    if metadata.get("document_type") == "markdown":
        text = remove_markdown_noise(text)

    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        "",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    text = re.sub(
        r"(?m)^\s*[─━═—–-]{3,}\s*$",
        "",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def clean_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Limpia todos los documentos y conserva sus metadatos.
    """
    filtered_documents = filter_non_informative_pages(
        documents
    )

    documents_without_repeated_lines = (
        remove_repeated_pdf_lines(
            filtered_documents
        )
    )

    cleaned_documents = []

    for document in documents_without_repeated_lines:
        original_length = len(
            document.page_content
        )

        cleaned_content = clean_text(
            text=document.page_content,
            metadata=document.metadata,
        )

        if not cleaned_content:
            continue

        metadata = document.metadata.copy()

        metadata["original_character_count"] = (
            original_length
        )
        metadata["clean_character_count"] = len(
            cleaned_content
        )
        metadata["characters_removed"] = (
            original_length
            - len(cleaned_content)
        )

        cleaned_documents.append(
            Document(
                page_content=cleaned_content,
                metadata=metadata,
            )
        )

    return cleaned_documents


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Crea el divisor de texto utilizado por Sol AI.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def split_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Divide los documentos en fragmentos conservando
    sus metadatos originales.
    """
    splitter = create_text_splitter()

    chunks = splitter.split_documents(
        documents
    )

    processed_chunks = []

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = chunk.metadata.copy()

        metadata["chunk_id"] = chunk_index
        metadata["chunk_size"] = len(
            chunk.page_content
        )

        processed_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    return processed_chunks


def add_chunk_identifiers(
    chunks: List[Document],
) -> List[Document]:
    """
    Crea identificadores estables y legibles
    para cada fragmento.
    """
    counters_by_file = {}

    for chunk in chunks:
        file_name = chunk.metadata.get(
            "file_name",
            "documento_desconocido",
        )

        counters_by_file[file_name] = (
            counters_by_file.get(
                file_name,
                0,
            )
            + 1
        )

        chunk_number = counters_by_file[file_name]

        file_stem = chunk.metadata.get(
            "file_stem",
            "documento",
        )

        chunk.metadata["document_chunk_id"] = (
            f"{file_stem}_chunk_{chunk_number:04d}"
        )

        chunk.metadata[
            "chunk_number_in_document"
        ] = chunk_number

    return chunks


def process_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Ejecuta todo el procesamiento documental.
    """
    cleaned_documents = clean_documents(
        documents
    )

    chunks = split_documents(
        cleaned_documents
    )

    chunks = add_chunk_identifiers(
        chunks
    )

    return chunks


def show_processing_summary(
    original_documents: List[Document],
    cleaned_documents: List[Document],
    chunks: List[Document],
) -> None:
    """
    Muestra un resumen del procesamiento.
    """
    original_characters = sum(
        len(document.page_content)
        for document in original_documents
    )

    cleaned_characters = sum(
        len(document.page_content)
        for document in cleaned_documents
    )

    removed_characters = (
        original_characters
        - cleaned_characters
    )

    total_chunk_characters = sum(
        len(chunk.page_content)
        for chunk in chunks
    )

    average_chunk_size = (
        total_chunk_characters / len(chunks)
        if chunks
        else 0
    )

    reduction_percentage = (
        removed_characters
        / original_characters
        * 100
        if original_characters
        else 0
    )

    print("\nResumen del procesamiento")
    print("-" * 50)
    print(
        f"Unidades originales: "
        f"{len(original_documents)}"
    )
    print(
        f"Unidades limpias: "
        f"{len(cleaned_documents)}"
    )
    print(
        f"Unidades omitidas: "
        f"{len(original_documents) - len(cleaned_documents)}"
    )
    print(
        f"Fragmentos generados: "
        f"{len(chunks)}"
    )
    print(
        f"Caracteres originales: "
        f"{original_characters:,}"
    )
    print(
        f"Caracteres después de limpiar: "
        f"{cleaned_characters:,}"
    )
    print(
        f"Caracteres eliminados: "
        f"{removed_characters:,}"
    )
    print(
        f"Reducción por limpieza: "
        f"{reduction_percentage:.2f}%"
    )
    print(
        f"Tamaño promedio por fragmento: "
        f"{average_chunk_size:.2f}"
    )
    print("-" * 50)


if __name__ == "__main__":
    documents = load_all_documents()

    show_loading_summary(
        documents
    )

    cleaned_documents = clean_documents(
        documents
    )

    chunks = split_documents(
        cleaned_documents
    )

    chunks = add_chunk_identifiers(
        chunks
    )

    show_processing_summary(
        original_documents=documents,
        cleaned_documents=cleaned_documents,
        chunks=chunks,
    )

    if chunks:
        sample_chunk = chunks[0]

        print("\nEjemplo del primer fragmento")
        print("-" * 50)
        print("Metadatos:")
        print(sample_chunk.metadata)
        print("\nContenido:")
        print(sample_chunk.page_content)
