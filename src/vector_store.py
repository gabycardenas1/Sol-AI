import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_TOP_K,
)
from src.document_loader import load_all_documents
from src.text_processor import process_documents


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Crea el modelo de embeddings multilingüe de Sol AI.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


def remove_existing_vector_store() -> None:
    """
    Elimina la base vectorial anterior para evitar
    fragmentos duplicados al reconstruirla.
    """

    vectorstore_path = Path(
        CHROMA_PERSIST_DIRECTORY
    )

    if vectorstore_path.exists():
        shutil.rmtree(
            vectorstore_path
        )

    vectorstore_path.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_vector_store(
    chunks: List[Document],
    recreate: bool = True,
) -> Chroma:
    """
    Genera los embeddings y almacena los fragmentos
    en una colección persistente de ChromaDB.
    """

    if not chunks:
        raise ValueError(
            "No existen fragmentos para indexar."
        )

    if recreate:
        remove_existing_vector_store()

    embedding_model = create_embedding_model()

    chunk_ids = [
        chunk.metadata["document_chunk_id"]
        for chunk in chunks
    ]

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        ids=chunk_ids,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(
            CHROMA_PERSIST_DIRECTORY
        ),
        collection_metadata={
            "description": (
                "Base documental interna de "
                "NexoData Consulting para Sol AI"
            ),
            "hnsw:space": "cosine",
        },
    )

    return vector_store


def load_vector_store() -> Chroma:
    """
    Abre una base vectorial existente.
    """

    vectorstore_path = Path(
        CHROMA_PERSIST_DIRECTORY
    )

    if not vectorstore_path.exists():
        raise FileNotFoundError(
            "La base vectorial todavía no existe. "
            "Ejecuta primero la creación del índice."
        )

    embedding_model = create_embedding_model()

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(
            CHROMA_PERSIST_DIRECTORY
        ),
    )


def get_vector_count(
    vector_store: Chroma,
) -> int:
    """
    Devuelve la cantidad de fragmentos almacenados.
    """

    return vector_store._collection.count()


def search_documents(
    vector_store: Chroma,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[tuple[Document, float]]:
    """
    Ejecuta una búsqueda semántica.

    El puntaje devuelto por Chroma representa distancia:
    mientras menor sea, mayor es la similitud.
    """

    if not query.strip():
        raise ValueError(
            "La consulta no puede estar vacía."
        )

    return vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )


def show_search_results(
    results: list[tuple[Document, float]],
) -> None:
    """
    Muestra los resultados recuperados.
    """

    if not results:
        print(
            "No se encontraron resultados."
        )
        return

    print("\nResultados de búsqueda")
    print("=" * 60)

    for position, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        metadata = document.metadata

        print(
            f"\nResultado {position}"
        )
        print("-" * 60)

        print(
            f"Archivo: "
            f"{metadata.get('file_name')}"
        )

        if metadata.get("page"):
            print(
                f"Página: "
                f"{metadata.get('page')}"
            )

        if metadata.get("sheet_name"):
            print(
                f"Hoja: "
                f"{metadata.get('sheet_name')}"
            )

        print(
            f"Fragmento: "
            f"{metadata.get('document_chunk_id')}"
        )

        print(
            f"Distancia: {score:.4f}"
        )

        print("\nContenido:")

        print(
            document.page_content[:600]
        )


def build_vector_store() -> Chroma:
    """
    Ejecuta el pipeline documental completo.
    """

    print(
        "1. Cargando documentos..."
    )

    documents = load_all_documents()

    print(
        "\n2. Procesando documentos..."
    )

    chunks = process_documents(
        documents
    )

    print(
        f"Fragmentos preparados: {len(chunks)}"
    )

    print(
        "\n3. Generando embeddings..."
    )

    vector_store = create_vector_store(
        chunks=chunks,
        recreate=True,
    )

    print(
        "\n4. Base vectorial creada."
    )

    print(
        f"Vectores almacenados: "
        f"{get_vector_count(vector_store)}"
    )

    return vector_store


if __name__ == "__main__":
    vector_store = build_vector_store()

    test_question = (
        "¿Qué debo hacer si detecto "
        "un incidente de seguridad?"
    )

    print(
        f"\nConsulta de prueba: "
        f"{test_question}"
    )

    results = search_documents(
        vector_store=vector_store,
        query=test_question,
    )

    show_search_results(
        results
    )