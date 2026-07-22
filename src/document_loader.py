from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from src.config import (
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
    validate_configuration,
)


def build_metadata(
    file_path: Path,
    document_type: str,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Construye los metadatos básicos de cada documento.
    """

    metadata = {
        "source": str(file_path),
        "file_name": file_path.name,
        "file_stem": file_path.stem,
        "file_extension": file_path.suffix.lower(),
        "document_type": document_type,
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
            extra_metadata={
                "page": page_number,
            },
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
    Lee un archivo CSV y convierte cada fila en un documento.
    """

    dataframe = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
    )

    documents = []

    for row_index, row in dataframe.iterrows():
        clean_values = {
            column: value
            for column, value in row.items()
            if pd.notna(value)
        }

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
            extra_metadata={
                "row": int(row_index) + 2,
            },
        )

        if "categoria" in clean_values:
            metadata["category"] = str(clean_values["categoria"])

        if "area_responsable" in clean_values:
            metadata["responsible_area"] = str(
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

        dataframe = dataframe.dropna(
            how="all",
        )

        if dataframe.empty:
            continue

        for row_index, row in dataframe.iterrows():
            clean_values = {
                column: value
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
    Encuentra todos los archivos compatibles dentro de documents/.
    """

    supported_files = []

    for file_path in documents_directory.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not file_path.name.startswith("~$")
        ):
            supported_files.append(file_path)

    return sorted(supported_files)


def load_all_documents(
    documents_directory: Path = DOCUMENTS_DIR,
) -> List[Document]:
    """
    Carga todos los documentos compatibles.
    """

    files = find_supported_files(documents_directory)

    if not files:
        raise FileNotFoundError(
            f"No se encontraron documentos compatibles en: "
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

    counts_by_type = {}

    for document in documents:
        document_type = document.metadata.get(
            "document_type",
            "desconocido",
        )

        counts_by_type[document_type] = (
            counts_by_type.get(document_type, 0) + 1
        )

    print("\nResumen de carga")
    print("-" * 40)

    for document_type, count in counts_by_type.items():
        print(f"{document_type}: {count}")

    print("-" * 40)
    print(f"Total de unidades cargadas: {len(documents)}")


if __name__ == "__main__":
    validate_configuration()

    loaded_documents = load_all_documents()

    show_loading_summary(
        loaded_documents
    )

    if loaded_documents:
        sample = loaded_documents[0]

        print("\nEjemplo del primer documento")
        print("-" * 40)
        print("Metadatos:")
        print(sample.metadata)

        print("\nContenido:")
        print(
            sample.page_content[:500]
        )