import shutil
from pathlib import Path
from typing import Any, List

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


ALLOWED_FILTER_FIELDS = {
    "document_id",
    "official_title",
    "category",
    "version",
    "last_updated",
    "document_status",
    "responsible_area",
    "access_level",
    "official_source",
    "ingestion_method",
    "next_review",
    "file_name",
    "file_extension",
    "document_type",
    "page",
    "sheet_name",
    "row",

    # Metadatos específicos de filas del CSV
    "row_category",
    "row_responsible_area",
    "related_document",
}


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


def validate_filters(
    filters: dict[str, Any] | None,
) -> None:
    """
    Verifica que los filtros utilicen campos permitidos
    y valores compatibles con ChromaDB.
    """

    if filters is None:
        return

    if not isinstance(filters, dict):
        raise TypeError(
            "Los filtros deben enviarse como un diccionario."
        )

    for field, value in filters.items():
        if field not in ALLOWED_FILTER_FIELDS:
            allowed_fields = ", ".join(
                sorted(ALLOWED_FILTER_FIELDS)
            )

            raise ValueError(
                f"Filtro no permitido: {field}. "
                f"Campos disponibles: {allowed_fields}"
            )

        if value is None:
            raise ValueError(
                f"El filtro '{field}' no puede tener valor None."
            )

        if not isinstance(
            value,
            (str, int, float, bool, list, tuple, set),
        ):
            raise TypeError(
                f"El valor del filtro '{field}' "
                "debe ser texto, número, booleano o una colección."
            )

        if isinstance(value, (list, tuple, set)) and not value:
            raise ValueError(
                f"El filtro '{field}' no puede estar vacío."
            )


def build_chroma_filter(
    filters: dict[str, Any] | None,
) -> dict | None:
    """
    Convierte filtros simples en la sintaxis esperada por ChromaDB.

    Ejemplos:
        {"category": "Seguridad de la Información"}

        {
            "category": "Seguridad de la Información",
            "document_status": "Aprobado y vigente",
        }

        {
            "document_type": ["pdf", "csv"],
        }
    """

    if not filters:
        return None

    validate_filters(
        filters
    )

    conditions = []

    for field, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            normalized_values = list(
                value
            )

            conditions.append(
                {
                    field: {
                        "$in": normalized_values,
                    }
                }
            )
        else:
            conditions.append(
                {
                    field: {
                        "$eq": value,
                    }
                }
            )

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions,
    }


def search_documents(
    vector_store: Chroma,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """
    Ejecuta una búsqueda semántica con filtros opcionales
    sobre los metadatos.

    El puntaje devuelto por Chroma representa distancia:
    mientras menor sea, mayor es la similitud.
    """

    if not query.strip():
        raise ValueError(
            "La consulta no puede estar vacía."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k debe ser mayor que cero."
        )

    chroma_filter = build_chroma_filter(
        filters
    )

    return vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter=chroma_filter,
    )


def show_search_results(
    results: list[tuple[Document, float]],
    filters: dict[str, Any] | None = None,
) -> None:
    """
    Muestra los resultados recuperados.
    """

    if filters:
        print(
            f"Filtros aplicados: {filters}"
        )

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

        print(
            f"Categoría: "
            f"{metadata.get('category')}"
        )

        print(
            f"Estado: "
            f"{metadata.get('document_status')}"
        )

        print(
            f"Área responsable: "
            f"{metadata.get('responsible_area')}"
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

        if metadata.get("row"):
            print(
                f"Fila: "
                f"{metadata.get('row')}"
            )

        if metadata.get("row_category"):
            print(
                "Categoría de la fila: "
                f"{metadata.get('row_category')}"
            )

        if metadata.get("row_responsible_area"):
            print(
                "Área responsable de la fila: "
                f"{metadata.get('row_responsible_area')}"
            )

        if metadata.get("related_document"):
            print(
                "Documento relacionado: "
                f"{metadata.get('related_document')}"
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


def run_search_tests(
    vector_store: Chroma,
) -> None:
    """
    Ejecuta una búsqueda general y tres búsquedas filtradas.
    """

    test_question = (
        "¿Qué debo hacer si detecto "
        "un incidente de seguridad?"
    )

    print(
        "\n\nPRUEBA 1: BÚSQUEDA GENERAL"
    )
    print(
        f"Consulta: {test_question}"
    )

    general_results = search_documents(
        vector_store=vector_store,
        query=test_question,
    )

    show_search_results(
        general_results
    )

    security_filters = {
        "responsible_area": (
            "Seguridad de la Información"
        ),
        "document_status": (
            "Aprobado y vigente"
        ),
    }

    print(
        "\n\nPRUEBA 2: BÚSQUEDA FILTRADA "
        "POR ÁREA Y ESTADO"
    )
    print(
        f"Consulta: {test_question}"
    )

    filtered_results = search_documents(
        vector_store=vector_store,
        query=test_question,
        filters=security_filters,
    )

    show_search_results(
        filtered_results,
        filters=security_filters,
    )

    format_filters = {
        "document_type": [
            "pdf",
            "csv",
        ],
    }

    print(
        "\n\nPRUEBA 3: BÚSQUEDA FILTRADA "
        "POR FORMATOS"
    )
    print(
        f"Consulta: {test_question}"
    )

    format_results = search_documents(
        vector_store=vector_store,
        query=test_question,
        filters=format_filters,
    )

    show_search_results(
        format_results,
        filters=format_filters,
    )

    csv_filters = {
        "row_responsible_area": (
            "Seguridad de la Información"
        ),
        "document_type": "csv",
    }

    print(
        "\n\nPRUEBA 4: BÚSQUEDA FILTRADA "
        "POR METADATOS DE FILA CSV"
    )
    print(
        f"Consulta: {test_question}"
    )

    csv_results = search_documents(
        vector_store=vector_store,
        query=test_question,
        filters=csv_filters,
    )

    show_search_results(
        csv_results,
        filters=csv_filters,
    )


if __name__ == "__main__":
    vector_store = build_vector_store()

    run_search_tests(
        vector_store
    )
