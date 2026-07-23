from functools import lru_cache
from pathlib import Path
from typing import Any, List

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from src.config import (
    DOCUMENT_INVENTORY_PATH,
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
    validate_configuration,
)


def normalize_metadata_value(
    value: Any,
) -> str | int | float | bool | None:
    """
    Convierte valores de pandas y Excel a tipos simples
    compatibles con los metadatos de LangChain y ChromaDB.
    """
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


@lru_cache(maxsize=1)
def load_document_inventory() -> dict[str, dict]:
    """
    Lee el inventario documental y crea un diccionario
    indexado por nombre de archivo.
    """
    if not DOCUMENT_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el inventario documental en: "
            f"{DOCUMENT_INVENTORY_PATH}"
        )

    dataframe = pd.read_excel(
        DOCUMENT_INVENTORY_PATH,
        sheet_name="Inventario documental",
        header=3,
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    required_columns = {
        "ID",
        "Nombre del archivo",
        "Título",
        "Categoría",
        "Versión",
        "Fecha de actualización",
        "Estado",
        "Área responsable",
        "Nivel de acceso",
        "Fuente oficial",
        "Método de ingesta",
        "Próxima revisión",
        "Observaciones",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            "El inventario documental no contiene "
            f"las columnas requeridas: {missing_text}"
        )

    inventory: dict[str, dict] = {}

    for _, row in dataframe.iterrows():
        file_name = normalize_metadata_value(
            row.get("Nombre del archivo")
        )

        if not file_name:
            continue

        normalized_file_name = str(file_name).strip().lower()

        inventory[normalized_file_name] = {
            "document_id": normalize_metadata_value(row.get("ID")),
            "official_title": normalize_metadata_value(row.get("Título")),
            "category": normalize_metadata_value(row.get("Categoría")),
            "version": normalize_metadata_value(row.get("Versión")),
            "last_updated": normalize_metadata_value(
                row.get("Fecha de actualización")
            ),
            "document_status": normalize_metadata_value(row.get("Estado")),
            "responsible_area": normalize_metadata_value(
                row.get("Área responsable")
            ),
            "access_level": normalize_metadata_value(
                row.get("Nivel de acceso")
            ),
            "official_source": normalize_metadata_value(
                row.get("Fuente oficial")
            ),
            "ingestion_method": normalize_metadata_value(
                row.get("Método de ingesta")
            ),
            "next_review": normalize_metadata_value(
                row.get("Próxima revisión")
            ),
            "inventory_observations": normalize_metadata_value(
                row.get("Observaciones")
            ),
        }

    if not inventory:
        raise ValueError("El inventario documental está vacío.")

    return inventory


def get_inventory_metadata(
    file_path: Path,
) -> dict:
    """
    Obtiene los metadatos oficiales de un archivo.
    """
    inventory = load_document_inventory()
    file_metadata = inventory.get(file_path.name.lower())

    if file_metadata is None:
        raise ValueError(
            "El archivo no está registrado en el "
            f"inventario documental: {file_path.name}"
        )

    return {
        key: value
        for key, value in file_metadata.items()
        if value is not None
    }


def is_document_approved(
    file_path: Path,
) -> bool:
    """
    Verifica si el documento está aprobado y vigente.
    """
    metadata = get_inventory_metadata(file_path)

    status = str(
        metadata.get("document_status", "")
    ).strip().lower()

    return status == "aprobado y vigente"


def build_metadata(
    file_path: Path,
    document_type: str,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Combina los metadatos técnicos del archivo
    con la información oficial del inventario.
    """
    inventory_metadata = get_inventory_metadata(file_path)

    metadata = {
        "source": str(file_path),
        "file_name": file_path.name,
        "file_stem": file_path.stem,
        "file_extension": file_path.suffix.lower(),
        "document_type": document_type,
        **inventory_metadata,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata


def load_pdf(file_path: Path) -> List[Document]:
    """
    Lee un archivo PDF y devuelve un documento por página.
    """
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    documents = []

    for page_number, page in enumerate(pages, start=1):
        content = page.page_content.strip()

        if not content:
            continue

        metadata = build_metadata(
            file_path=file_path,
            document_type="pdf",
            extra_metadata={"page": page_number},
        )

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def load_markdown(file_path: Path) -> List[Document]:
    """
    Lee un archivo Markdown como texto.
    """
    loader = TextLoader(
        str(file_path),
        encoding="utf-8",
        autodetect_encoding=True,
    )

    loaded_documents = loader.load()
    documents = []

    for document in loaded_documents:
        content = document.page_content.strip()

        if not content:
            continue

        metadata = build_metadata(
            file_path=file_path,
            document_type="markdown",
        )

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def load_csv(file_path: Path) -> List[Document]:
    """
    Lee un archivo CSV y convierte cada fila
    en un documento independiente.
    """
    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    documents = []

    for row_index, row in dataframe.iterrows():
        clean_values = {
            str(column): normalize_metadata_value(value)
            for column, value in row.items()
            if pd.notna(value)
        }

        if not clean_values:
            continue

        content_lines = [
            f"{column}: {value}"
            for column, value in clean_values.items()
        ]

        content = "\n".join(content_lines).strip()

        if not content:
            continue

        metadata = build_metadata(
            file_path=file_path,
            document_type="csv",
            extra_metadata={"row": int(row_index) + 2},
        )

        if "categoria" in clean_values:
            metadata["row_category"] = str(
                clean_values["categoria"]
            )

        if "area_responsable" in clean_values:
            metadata["row_responsible_area"] = str(
                clean_values["area_responsable"]
            )

        if "documento_relacionado" in clean_values:
            metadata["related_document"] = str(
                clean_values["documento_relacionado"]
            )

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


def load_excel(file_path: Path) -> List[Document]:
    """
    Lee todas las hojas de un archivo Excel.
    Cada fila se convierte en un documento independiente.
    """
    excel_file = pd.ExcelFile(file_path)
    documents = []

    for sheet_name in excel_file.sheet_names:
        dataframe = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
        )

        dataframe = dataframe.dropna(how="all")

        if dataframe.empty:
            continue

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        for row_index, row in dataframe.iterrows():
            clean_values = {
                str(column): normalize_metadata_value(value)
                for column, value in row.items()
                if pd.notna(value)
            }

            if not clean_values:
                continue

            content_lines = [
                f"{column}: {value}"
                for column, value in clean_values.items()
            ]

            content = "\n".join(content_lines).strip()

            if not content:
                continue

            metadata = build_metadata(
                file_path=file_path,
                document_type="excel",
                extra_metadata={
                    "sheet_name": sheet_name,
                    "row": int(row_index) + 2,
                },
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    return documents


def load_document(file_path: Path) -> List[Document]:
    """
    Selecciona el cargador correcto según la extensión.
    """
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".md":
        return load_markdown(file_path)

    if extension == ".csv":
        return load_csv(file_path)

    if extension == ".xlsx":
        return load_excel(file_path)

    raise ValueError(
        f"Formato no compatible: {extension}"
    )


def find_supported_files(
    documents_directory: Path = DOCUMENTS_DIR,
) -> List[Path]:
    """
    Encuentra archivos compatibles, registrados
    y aprobados dentro de documents/.
    """
    supported_files = []

    for file_path in documents_directory.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.name.startswith("~$"):
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        try:
            if not is_document_approved(file_path):
                print(
                    f"[OMITIDO] {file_path.name}: "
                    "no está aprobado y vigente"
                )
                continue

        except ValueError as error:
            print(
                f"[OMITIDO] {file_path.name}: {error}"
            )
            continue

        supported_files.append(file_path)

    return sorted(supported_files)


def load_all_documents(
    documents_directory: Path = DOCUMENTS_DIR,
) -> List[Document]:
    """
    Carga todos los documentos compatibles,
    registrados y aprobados.
    """
    files = find_supported_files(documents_directory)

    if not files:
        raise FileNotFoundError(
            "No se encontraron documentos compatibles, "
            "registrados y aprobados en: "
            f"{documents_directory}"
        )

    all_documents = []

    for file_path in files:
        try:
            loaded_documents = load_document(file_path)
            all_documents.extend(loaded_documents)

            print(
                f"[OK] {file_path.name}: "
                f"{len(loaded_documents)} unidades cargadas"
            )

        except Exception as error:
            print(
                f"[ERROR] No se pudo cargar "
                f"{file_path.name}: {error}"
            )

    return all_documents


def show_loading_summary(
    documents: List[Document],
) -> None:
    """
    Muestra un resumen de los documentos cargados.
    """
    counts_by_type: dict[str, int] = {}
    document_ids = set()

    for document in documents:
        document_type = document.metadata.get(
            "document_type",
            "desconocido",
        )

        counts_by_type[document_type] = (
            counts_by_type.get(document_type, 0) + 1
        )

        document_id = document.metadata.get("document_id")

        if document_id:
            document_ids.add(str(document_id))

    print("\nResumen de carga")
    print("-" * 45)

    for document_type, count in counts_by_type.items():
        print(f"{document_type}: {count}")

    print("-" * 45)
    print(f"Total de unidades cargadas: {len(documents)}")
    print(
        "Documentos oficiales representados: "
        f"{len(document_ids)}"
    )


def show_sample_document(
    document: Document,
) -> None:
    """
    Muestra un ejemplo de contenido y metadatos.
    """
    print("\nEjemplo del primer documento")
    print("-" * 45)

    print("Metadatos completos:")
    print(document.metadata)

    print("\nMetadatos oficiales verificados:")
    print(f"ID: {document.metadata.get('document_id')}")
    print(f"Título: {document.metadata.get('official_title')}")
    print(f"Categoría: {document.metadata.get('category')}")
    print(f"Versión: {document.metadata.get('version')}")
    print(f"Estado: {document.metadata.get('document_status')}")
    print(
        "Área responsable: "
        f"{document.metadata.get('responsible_area')}"
    )
    print(
        "Nivel de acceso: "
        f"{document.metadata.get('access_level')}"
    )
    print(
        "Última actualización: "
        f"{document.metadata.get('last_updated')}"
    )
    print(
        "Próxima revisión: "
        f"{document.metadata.get('next_review')}"
    )

    print("\nContenido:")
    print(document.page_content[:500])


if __name__ == "__main__":
    validate_configuration()

    inventory = load_document_inventory()

    print(
        "Documentos registrados en el inventario: "
        f"{len(inventory)}"
    )

    loaded_documents = load_all_documents()

    show_loading_summary(loaded_documents)

    if loaded_documents:
        show_sample_document(loaded_documents[0])
