import re
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


def clean_text(text: str) -> str:
    """
    Limpia espacios, saltos de línea y caracteres innecesarios
    sin eliminar información importante.
    """

    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Reduce espacios repetidos
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce saltos de línea excesivos
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Elimina espacios al inicio y final de cada línea
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    cleaned_text = "\n".join(lines)

    return cleaned_text.strip()


def clean_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Limpia el contenido de todos los documentos.
    """

    cleaned_documents = []

    for document in documents:
        cleaned_content = clean_text(
            document.page_content
        )

        if not cleaned_content:
            continue

        cleaned_documents.append(
            Document(
                page_content=cleaned_content,
                metadata=document.metadata.copy(),
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
            "\n## ",
            "\n### ",
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
    Crea un identificador legible para cada fragmento.
    """

    counters_by_file = {}

    for chunk in chunks:
        file_name = chunk.metadata.get(
            "file_name",
            "documento_desconocido",
        )

        counters_by_file[file_name] = (
            counters_by_file.get(file_name, 0) + 1
        )

        chunk_number = counters_by_file[file_name]

        file_stem = chunk.metadata.get(
            "file_stem",
            "documento",
        )

        chunk.metadata["document_chunk_id"] = (
            f"{file_stem}_chunk_{chunk_number:04d}"
        )

        chunk.metadata["chunk_number_in_document"] = (
            chunk_number
        )

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
    chunks: List[Document],
) -> None:
    """
    Muestra un resumen del procesamiento.
    """

    total_characters = sum(
        len(document.page_content)
        for document in original_documents
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

    print("\nResumen del procesamiento")
    print("-" * 45)
    print(
        f"Unidades originales: "
        f"{len(original_documents)}"
    )
    print(
        f"Fragmentos generados: "
        f"{len(chunks)}"
    )
    print(
        f"Caracteres originales: "
        f"{total_characters:,}"
    )
    print(
        f"Tamaño promedio por fragmento: "
        f"{average_chunk_size:.2f}"
    )
    print("-" * 45)


if __name__ == "__main__":
    documents = load_all_documents()

    show_loading_summary(
        documents
    )

    chunks = process_documents(
        documents
    )

    show_processing_summary(
        original_documents=documents,
        chunks=chunks,
    )

    if chunks:
        sample_chunk = chunks[0]

        print("\nEjemplo del primer fragmento")
        print("-" * 45)

        print("Metadatos:")
        print(sample_chunk.metadata)

        print("\nContenido:")
        print(sample_chunk.page_content)