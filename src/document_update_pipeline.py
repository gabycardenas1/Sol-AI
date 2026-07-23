import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import (
    DOCUMENT_INVENTORY_PATH,
    DOCUMENTS_DIR,
)
from src.document_loader import (
    find_supported_files,
)
from src.vector_store import (
    build_vector_store,
    get_vector_count,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "document_manifest.json"
UPDATE_LOG_PATH = DATA_DIR / "index_update_logs.csv"


def calculate_file_hash(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """
    Calcula el hash SHA-256 de un archivo.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            block = file.read(block_size)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def build_file_record(
    file_path: Path,
) -> dict[str, Any]:
    """
    Construye la información de control de un archivo.
    """

    stat = file_path.stat()

    try:
        relative_path = file_path.relative_to(
            PROJECT_ROOT
        )
    except ValueError:
        relative_path = file_path

    return {
        "path": relative_path.as_posix(),
        "file_name": file_path.name,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(
            timespec="seconds"
        ),
        "sha256": calculate_file_hash(
            file_path
        ),
    }


def get_monitored_files() -> list[Path]:
    """
    Obtiene los documentos aprobados y el inventario documental.

    El inventario también se monitorea porque una modificación
    de versión, estado, categoría o responsable cambia los
    metadatos que deben quedar guardados en ChromaDB.
    """

    files = find_supported_files(
        DOCUMENTS_DIR
    )

    if DOCUMENT_INVENTORY_PATH.exists():
        files.append(
            DOCUMENT_INVENTORY_PATH
        )

    unique_files = {
        file_path.resolve(): file_path
        for file_path in files
    }

    return sorted(
        unique_files.values(),
        key=lambda path: str(path).lower(),
    )


def create_current_manifest() -> dict[str, Any]:
    """
    Crea una fotografía actual de los archivos monitoreados.
    """

    files = get_monitored_files()

    records = {
        record["path"]: record
        for record in (
            build_file_record(file_path)
            for file_path in files
        )
    }

    return {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "documents_directory": str(
            DOCUMENTS_DIR
        ),
        "inventory_path": str(
            DOCUMENT_INVENTORY_PATH
        ),
        "file_count": len(records),
        "files": records,
    }


def load_previous_manifest() -> dict[str, Any] | None:
    """
    Lee el manifiesto anterior cuando existe.
    """

    if not MANIFEST_PATH.exists():
        return None

    try:
        with MANIFEST_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ) as error:
        raise RuntimeError(
            "No se pudo leer el manifiesto anterior: "
            f"{error}"
        ) from error


def save_manifest(
    manifest: dict[str, Any],
) -> None:
    """
    Guarda el manifiesto después de una actualización exitosa.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = MANIFEST_PATH.with_suffix(
        ".json.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(
        MANIFEST_PATH
    )


def compare_manifests(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Detecta archivos añadidos, modificados y eliminados.
    """

    current_files = current.get(
        "files",
        {},
    )

    if previous is None:
        return {
            "added": sorted(
                current_files.keys()
            ),
            "modified": [],
            "deleted": [],
        }

    previous_files = previous.get(
        "files",
        {},
    )

    previous_paths = set(
        previous_files.keys()
    )
    current_paths = set(
        current_files.keys()
    )

    added = sorted(
        current_paths - previous_paths
    )

    deleted = sorted(
        previous_paths - current_paths
    )

    modified = sorted(
        path
        for path in (
            previous_paths
            & current_paths
        )
        if (
            previous_files[path].get("sha256")
            != current_files[path].get("sha256")
        )
    )

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
    }


def has_changes(
    changes: dict[str, list[str]],
) -> bool:
    """
    Indica si existe al menos un cambio documental.
    """

    return any(
        changes.get(change_type)
        for change_type in (
            "added",
            "modified",
            "deleted",
        )
    )


def show_changes(
    changes: dict[str, list[str]],
) -> None:
    """
    Muestra un resumen legible de los cambios.
    """

    labels = {
        "added": "Archivos añadidos",
        "modified": "Archivos modificados",
        "deleted": "Archivos eliminados",
    }

    print("\nCambios detectados")
    print("=" * 60)

    for change_type, label in labels.items():
        files = changes.get(
            change_type,
            [],
        )

        print(
            f"{label}: {len(files)}"
        )

        for file_path in files:
            print(
                f"  - {file_path}"
            )


def append_update_log(
    status: str,
    trigger: str,
    changes: dict[str, list[str]],
    vector_count: int | None = None,
    error_message: str = "",
) -> None:
    """
    Registra cada revisión o actualización del índice.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "timestamp",
        "status",
        "trigger",
        "added_count",
        "modified_count",
        "deleted_count",
        "vector_count",
        "added_files",
        "modified_files",
        "deleted_files",
        "error_message",
    ]

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "status": status,
        "trigger": trigger,
        "added_count": len(
            changes.get("added", [])
        ),
        "modified_count": len(
            changes.get("modified", [])
        ),
        "deleted_count": len(
            changes.get("deleted", [])
        ),
        "vector_count": (
            vector_count
            if vector_count is not None
            else ""
        ),
        "added_files": json.dumps(
            changes.get("added", []),
            ensure_ascii=False,
        ),
        "modified_files": json.dumps(
            changes.get("modified", []),
            ensure_ascii=False,
        ),
        "deleted_files": json.dumps(
            changes.get("deleted", []),
            ensure_ascii=False,
        ),
        "error_message": error_message,
    }

    file_exists = UPDATE_LOG_PATH.exists()

    with UPDATE_LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


def initialize_manifest(
    current_manifest: dict[str, Any],
) -> None:
    """
    Inicializa el control sin reconstruir el índice.

    Se usa cuando el índice ya existe y solo queremos crear
    la línea base para detectar cambios futuros.
    """

    save_manifest(
        current_manifest
    )

    empty_changes = {
        "added": [],
        "modified": [],
        "deleted": [],
    }

    append_update_log(
        status="manifest_initialized",
        trigger="initialization",
        changes=empty_changes,
    )

    print(
        "\nManifiesto inicial creado correctamente."
    )
    print(
        f"Archivos monitoreados: "
        f"{current_manifest['file_count']}"
    )
    print(
        "No se reconstruyó ChromaDB porque esta fue "
        "solo la inicialización del control."
    )


def rebuild_index(
    current_manifest: dict[str, Any],
    changes: dict[str, list[str]],
    trigger: str,
) -> int:
    """
    Reconstruye ChromaDB y actualiza el manifiesto
    únicamente si todo termina correctamente.
    """

    print(
        "\nReconstruyendo el índice vectorial..."
    )

    try:
        vector_store = build_vector_store()

        vector_count = get_vector_count(
            vector_store
        )

        save_manifest(
            current_manifest
        )

        append_update_log(
            status="success",
            trigger=trigger,
            changes=changes,
            vector_count=vector_count,
        )

        print(
            "\nActualización completada."
        )
        print(
            f"Vectores almacenados: "
            f"{vector_count}"
        )
        print(
            f"Manifiesto actualizado: "
            f"{MANIFEST_PATH}"
        )

        return vector_count

    except PermissionError as error:
        message = (
            "No se pudo reconstruir el índice porque ChromaDB "
            "está siendo utilizado por otro proceso. Cierra "
            "Streamlit y cualquier ejecución de rag_agent.py."
        )

        append_update_log(
            status="failed",
            trigger=trigger,
            changes=changes,
            error_message=f"{message} Detalle: {error}",
        )

        raise RuntimeError(
            message
        ) from error

    except Exception as error:
        append_update_log(
            status="failed",
            trigger=trigger,
            changes=changes,
            error_message=str(error),
        )

        raise


def check_and_update(
    force: bool = False,
    initialize_only: bool = False,
) -> None:
    """
    Revisa cambios documentales y actualiza el índice
    cuando sea necesario.
    """

    print(
        "Revisando documentos de NexoData..."
    )

    current_manifest = create_current_manifest()
    previous_manifest = load_previous_manifest()

    if (
        previous_manifest is None
        and initialize_only
        and not force
    ):
        initialize_manifest(
            current_manifest
        )
        return

    changes = compare_manifests(
        previous=previous_manifest,
        current=current_manifest,
    )

    show_changes(
        changes
    )

    if force:
        print(
            "\nReconstrucción forzada solicitada."
        )

        rebuild_index(
            current_manifest=current_manifest,
            changes=changes,
            trigger="force",
        )
        return

    if previous_manifest is None:
        print(
            "\nNo existe un manifiesto anterior."
        )
        print(
            "Se realizará la primera reconstrucción para "
            "sincronizar el índice y guardar la línea base."
        )

        rebuild_index(
            current_manifest=current_manifest,
            changes=changes,
            trigger="first_run",
        )
        return

    if not has_changes(
        changes
    ):
        append_update_log(
            status="no_changes",
            trigger="scheduled_check",
            changes=changes,
        )

        print(
            "\nNo se detectaron cambios."
        )
        print(
            "La base vectorial no necesita reconstruirse."
        )
        return

    rebuild_index(
        current_manifest=current_manifest,
        changes=changes,
        trigger="document_change",
    )


def parse_arguments() -> argparse.Namespace:
    """
    Lee los argumentos de línea de comandos.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Detecta cambios en los documentos de Sol AI "
            "y actualiza ChromaDB cuando corresponde."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reconstruye el índice aunque no existan cambios."
        ),
    )

    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help=(
            "Crea el manifiesto inicial sin reconstruir ChromaDB."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Punto de entrada del pipeline.
    """

    arguments = parse_arguments()

    try:
        check_and_update(
            force=arguments.force,
            initialize_only=arguments.initialize_only,
        )

    except Exception as error:
        print(
            f"\n[ERROR] {error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
