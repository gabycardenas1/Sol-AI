import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

INTERACTION_LOG_PATH = DATA_DIR / "interaction_logs.csv"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_logs.csv"

SUMMARY_JSON_PATH = DATA_DIR / "monitoring_summary.json"
REPORT_MARKDOWN_PATH = REPORTS_DIR / "monitoring_report.md"


def load_csv(
    file_path: Path,
) -> pd.DataFrame:
    """
    Lee un CSV de monitoreo.

    Si el archivo no existe o está vacío, devuelve un DataFrame vacío.
    """

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            file_path,
            encoding="utf-8-sig",
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_percentage(
    numerator: int,
    denominator: int,
) -> float:
    """
    Calcula un porcentaje evitando divisiones para cero.
    """

    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100,
        2,
    )


def calculate_monitoring_metrics(
    interactions: pd.DataFrame,
    feedback: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calcula las métricas principales de calidad y uso.
    """

    total_questions = len(
        interactions
    )

    sufficient_context = 0
    insufficient_context = 0
    unknown_context = 0

    if (
        not interactions.empty
        and "confidence_status" in interactions.columns
    ):
        statuses = (
            interactions["confidence_status"]
            .fillna("unknown")
            .astype(str)
        )

        sufficient_context = int(
            (
                statuses
                == "sufficient_context"
            ).sum()
        )

        insufficient_context = int(
            (
                statuses
                == "insufficient_context"
            ).sum()
        )

        unknown_context = int(
            total_questions
            - sufficient_context
            - insufficient_context
        )

    average_response_time = 0.0
    median_response_time = 0.0
    maximum_response_time = 0.0

    if (
        not interactions.empty
        and "response_time_seconds" in interactions.columns
    ):
        response_times = pd.to_numeric(
            interactions["response_time_seconds"],
            errors="coerce",
        ).dropna()

        if not response_times.empty:
            average_response_time = round(
                float(response_times.mean()),
                3,
            )
            median_response_time = round(
                float(response_times.median()),
                3,
            )
            maximum_response_time = round(
                float(response_times.max()),
                3,
            )

    average_sources = 0.0
    responses_without_sources = 0

    if (
        not interactions.empty
        and "source_count" in interactions.columns
    ):
        source_counts = pd.to_numeric(
            interactions["source_count"],
            errors="coerce",
        ).fillna(0)

        average_sources = round(
            float(source_counts.mean()),
            2,
        )

        responses_without_sources = int(
            (
                source_counts
                == 0
            ).sum()
        )

    positive_feedback = 0
    negative_feedback = 0
    total_feedback = 0
    unique_feedback_interactions = 0

    if (
        not feedback.empty
        and "feedback" in feedback.columns
    ):
        feedback_values = (
            feedback["feedback"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        positive_feedback = int(
            (
                feedback_values
                == "positive"
            ).sum()
        )

        negative_feedback = int(
            (
                feedback_values
                == "negative"
            ).sum()
        )

        total_feedback = (
            positive_feedback
            + negative_feedback
        )

        if "interaction_id" in feedback.columns:
            unique_feedback_interactions = int(
                feedback["interaction_id"]
                .dropna()
                .astype(str)
                .nunique()
            )
        else:
            unique_feedback_interactions = total_feedback

    feedback_coverage = safe_percentage(
        unique_feedback_interactions,
        total_questions,
    )

    positive_feedback_rate = safe_percentage(
        positive_feedback,
        total_feedback,
    )

    negative_feedback_rate = safe_percentage(
        negative_feedback,
        total_feedback,
    )

    return {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "total_questions": total_questions,
        "sufficient_context_count": sufficient_context,
        "sufficient_context_rate": safe_percentage(
            sufficient_context,
            total_questions,
        ),
        "insufficient_context_count": insufficient_context,
        "insufficient_context_rate": safe_percentage(
            insufficient_context,
            total_questions,
        ),
        "unknown_context_count": unknown_context,
        "responses_without_sources": responses_without_sources,
        "responses_without_sources_rate": safe_percentage(
            responses_without_sources,
            total_questions,
        ),
        "average_response_time_seconds": average_response_time,
        "median_response_time_seconds": median_response_time,
        "maximum_response_time_seconds": maximum_response_time,
        "average_sources_per_response": average_sources,
        "total_feedback": total_feedback,
        "positive_feedback_count": positive_feedback,
        "negative_feedback_count": negative_feedback,
        "positive_feedback_rate": positive_feedback_rate,
        "negative_feedback_rate": negative_feedback_rate,
        "feedback_coverage_rate": feedback_coverage,
    }


def get_quality_status(
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Evalúa métricas simples para orientar el mantenimiento.
    """

    checks = []

    sufficient_rate = metrics[
        "sufficient_context_rate"
    ]

    if sufficient_rate >= 85:
        checks.append(
            {
                "metric": "Cobertura de contexto",
                "status": "Adecuado",
                "detail": (
                    f"{sufficient_rate}% de las respuestas "
                    "tienen contexto suficiente."
                ),
            }
        )
    elif sufficient_rate >= 70:
        checks.append(
            {
                "metric": "Cobertura de contexto",
                "status": "Revisar",
                "detail": (
                    f"{sufficient_rate}% de las respuestas "
                    "tienen contexto suficiente."
                ),
            }
        )
    else:
        checks.append(
            {
                "metric": "Cobertura de contexto",
                "status": "Crítico",
                "detail": (
                    f"{sufficient_rate}% de las respuestas "
                    "tienen contexto suficiente."
                ),
            }
        )

    positive_rate = metrics[
        "positive_feedback_rate"
    ]

    if metrics["total_feedback"] == 0:
        checks.append(
            {
                "metric": "Satisfacción",
                "status": "Sin datos",
                "detail": (
                    "Aún no existe feedback suficiente "
                    "para evaluar satisfacción."
                ),
            }
        )
    elif positive_rate >= 80:
        checks.append(
            {
                "metric": "Satisfacción",
                "status": "Adecuado",
                "detail": (
                    f"{positive_rate}% del feedback es positivo."
                ),
            }
        )
    elif positive_rate >= 60:
        checks.append(
            {
                "metric": "Satisfacción",
                "status": "Revisar",
                "detail": (
                    f"{positive_rate}% del feedback es positivo."
                ),
            }
        )
    else:
        checks.append(
            {
                "metric": "Satisfacción",
                "status": "Crítico",
                "detail": (
                    f"{positive_rate}% del feedback es positivo."
                ),
            }
        )

    average_time = metrics[
        "average_response_time_seconds"
    ]

    if average_time == 0:
        time_status = "Sin datos"
    elif average_time <= 8:
        time_status = "Adecuado"
    elif average_time <= 15:
        time_status = "Revisar"
    else:
        time_status = "Crítico"

    checks.append(
        {
            "metric": "Tiempo de respuesta",
            "status": time_status,
            "detail": (
                f"Promedio de {average_time} segundos."
            ),
        }
    )

    return checks


def save_summary_json(
    metrics: dict[str, Any],
    quality_checks: list[dict[str, str]],
) -> None:
    """
    Guarda las métricas en JSON.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metrics": metrics,
        "quality_checks": quality_checks,
    }

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_markdown_report(
    metrics: dict[str, Any],
    quality_checks: list[dict[str, str]],
) -> str:
    """
    Construye el informe de monitoreo en Markdown.
    """

    quality_rows = "\n".join(
        (
            f"| {check['metric']} | "
            f"{check['status']} | "
            f"{check['detail']} |"
        )
        for check in quality_checks
    )

    return f"""# Reporte de Monitoreo de Sol AI

**Fecha de generación:** {metrics["generated_at"]}

## 1. Resumen general

| Indicador | Resultado |
|---|---:|
| Total de preguntas | {metrics["total_questions"]} |
| Respuestas con contexto suficiente | {metrics["sufficient_context_count"]} |
| Tasa de contexto suficiente | {metrics["sufficient_context_rate"]}% |
| Respuestas con contexto insuficiente | {metrics["insufficient_context_count"]} |
| Tasa de contexto insuficiente | {metrics["insufficient_context_rate"]}% |
| Respuestas sin fuentes | {metrics["responses_without_sources"]} |
| Promedio de fuentes por respuesta | {metrics["average_sources_per_response"]} |

## 2. Rendimiento

| Indicador | Resultado |
|---|---:|
| Tiempo promedio de respuesta | {metrics["average_response_time_seconds"]} s |
| Mediana del tiempo de respuesta | {metrics["median_response_time_seconds"]} s |
| Tiempo máximo de respuesta | {metrics["maximum_response_time_seconds"]} s |

## 3. Feedback de usuarios

| Indicador | Resultado |
|---|---:|
| Total de evaluaciones | {metrics["total_feedback"]} |
| Feedback positivo | {metrics["positive_feedback_count"]} |
| Feedback negativo | {metrics["negative_feedback_count"]} |
| Tasa de feedback positivo | {metrics["positive_feedback_rate"]}% |
| Tasa de feedback negativo | {metrics["negative_feedback_rate"]}% |
| Cobertura de feedback | {metrics["feedback_coverage_rate"]}% |

## 4. Estado de calidad

| Dimensión | Estado | Interpretación |
|---|---|---|
{quality_rows}

## 5. Acciones recomendadas

- Revisar las interacciones marcadas con feedback negativo.
- Analizar preguntas con `insufficient_context`.
- Incorporar nuevos documentos solo después de su aprobación en el inventario.
- Ejecutar el pipeline documental después de cada actualización.
- Revisar este reporte periódicamente para detectar cambios en calidad y rendimiento.

## 6. Archivos de origen

- `data/interaction_logs.csv`
- `data/feedback_logs.csv`

Este reporte se genera automáticamente a partir de las interacciones reales de Sol AI.
"""


def save_markdown_report(
    report_content: str,
) -> None:
    """
    Guarda el reporte en la carpeta reports.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_MARKDOWN_PATH.write_text(
        report_content,
        encoding="utf-8",
    )


def print_summary(
    metrics: dict[str, Any],
) -> None:
    """
    Imprime un resumen corto en la terminal.
    """

    print("\nReporte de monitoreo generado")
    print("=" * 55)
    print(
        f"Preguntas registradas: "
        f"{metrics['total_questions']}"
    )
    print(
        f"Contexto suficiente: "
        f"{metrics['sufficient_context_rate']}%"
    )
    print(
        f"Feedback positivo: "
        f"{metrics['positive_feedback_rate']}%"
    )
    print(
        f"Tiempo promedio: "
        f"{metrics['average_response_time_seconds']} s"
    )
    print(
        f"Fuentes promedio: "
        f"{metrics['average_sources_per_response']}"
    )
    print(
        f"\nReporte: {REPORT_MARKDOWN_PATH}"
    )
    print(
        f"Resumen JSON: {SUMMARY_JSON_PATH}"
    )


def generate_report() -> None:
    """
    Ejecuta el proceso completo.
    """

    interactions = load_csv(
        INTERACTION_LOG_PATH
    )

    feedback = load_csv(
        FEEDBACK_LOG_PATH
    )

    metrics = calculate_monitoring_metrics(
        interactions=interactions,
        feedback=feedback,
    )

    quality_checks = get_quality_status(
        metrics
    )

    save_summary_json(
        metrics=metrics,
        quality_checks=quality_checks,
    )

    report_content = build_markdown_report(
        metrics=metrics,
        quality_checks=quality_checks,
    )

    save_markdown_report(
        report_content
    )

    print_summary(
        metrics
    )


def parse_arguments() -> argparse.Namespace:
    """
    Mantiene una interfaz de línea de comandos extensible.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Genera el reporte de monitoreo de Sol AI "
            "a partir de los logs de interacción y feedback."
        )
    )

    return parser.parse_args()


def main() -> None:
    parse_arguments()
    generate_report()


if __name__ == "__main__":
    main()
