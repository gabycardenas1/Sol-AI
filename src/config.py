from pathlib import Path
import os

from dotenv import load_dotenv


# Carga las variables guardadas en el archivo .env
load_dotenv()


# Ruta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent


# Carpetas principales
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
TESTS_DIR = BASE_DIR / "tests"


# Inventario documental
DOCUMENT_INVENTORY_PATH = (
    DATA_DIR / "inventario_documental_nexodata.xlsx"
)


# Extensiones que Sol AI podrá procesar
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".xlsx",
    ".csv",
}


# Configuración para dividir los documentos
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# Modelo de embeddings de Hugging Face
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# Configuración de ChromaDB
CHROMA_COLLECTION_NAME = "sol_ai_nexodata"
CHROMA_PERSIST_DIRECTORY = VECTORSTORE_DIR


# Configuración de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv(
    "GEMINI_MODEL_NAME",
    "gemini-2.5-flash",
)


# Número de fragmentos que se recuperarán por consulta
RETRIEVAL_TOP_K = 5


# Temperatura baja para evitar respuestas inventadas
MODEL_TEMPERATURE = 0.2


# Mensaje cuando no existe suficiente información
NO_INFORMATION_MESSAGE = (
    "No encontré información suficiente en los documentos internos "
    "de NexoData Consulting para responder con seguridad. "
    "Te recomiendo consultar al área responsable."
)


def validate_directories() -> None:
    """
    Crea las carpetas necesarias si todavía no existen.
    """

    required_directories = [
        DOCUMENTS_DIR,
        DATA_DIR,
        VECTORSTORE_DIR,
        TESTS_DIR,
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_configuration() -> None:
    """
    Verifica que la configuración mínima del proyecto sea válida.
    """

    validate_directories()

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta de documentos: {DOCUMENTS_DIR}"
        )

    if not DOCUMENT_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el inventario documental en: "
            f"{DOCUMENT_INVENTORY_PATH}"
        )


if __name__ == "__main__":
    validate_configuration()

    print("Configuración validada correctamente.")
    print(f"Proyecto: {BASE_DIR}")
    print(f"Documentos: {DOCUMENTS_DIR}")
    print(f"Inventario: {DOCUMENT_INVENTORY_PATH}")
    print(f"Vectorstore: {VECTORSTORE_DIR}")