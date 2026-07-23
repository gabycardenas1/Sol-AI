from dataclasses import dataclass
from typing import Any, List

from google import genai
from google.genai import types
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MODEL_TEMPERATURE,
    NO_INFORMATION_MESSAGE,
    RETRIEVAL_TOP_K,
)
from src.vector_store import (
    load_vector_store,
    search_documents,
)


RERANKER_MODEL_NAME = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

RERANK_CANDIDATES = max(
    RETRIEVAL_TOP_K * 3,
    15,
)

RERANK_BATCH_SIZE = 8

# Umbrales iniciales de confianza.
# Se validarán con preguntas conocidas y preguntas fuera de alcance.
MAX_ACCEPTABLE_VECTOR_DISTANCE = 0.75
MIN_ACCEPTABLE_RERANK_SCORE = -2.50
MIN_RELEVANT_RESULTS = 1


@dataclass
class RAGResponse:
    """
    Estructura de respuesta generada por Sol AI.
    """

    answer: str
    sources: List[dict]
    retrieved_documents: List[Document]
    confidence_status: str = "unknown"


SYSTEM_INSTRUCTIONS = """
Eres Sol AI, el asistente interno de NexoData Consulting.

Tu propósito es ayudar a los colaboradores a encontrar información
dentro de los documentos corporativos.

Personalidad:
- Cercana, amable y profesional.
- Clara y fácil de entender.
- Cálida, pero sin perder precisión.
- Puedes usar un tono natural, como una compañera confiable.

Reglas obligatorias:
1. Responde únicamente con base en el contexto documental proporcionado.
2. No inventes políticas, procedimientos, nombres, fechas ni responsables.
3. Si el contexto no contiene información suficiente, dilo claramente.
4. No afirmes que realizaste acciones que no puedes ejecutar.
5. No reveles claves, credenciales ni información sensible.
6. Cuando sea útil, menciona naturalmente el documento consultado.
7. Prioriza la información más específica sobre la información general.
8. Si los documentos parecen contradecirse, explica la diferencia.
9. No menciones fragmentos, embeddings, vectores ni procesos técnicos
   al colaborador.
10. Responde en español.
"""


def create_gemini_client() -> genai.Client:
    """
    Crea el cliente oficial de Gemini.
    """

    if not GEMINI_API_KEY:
        raise ValueError(
            "No se encontró GEMINI_API_KEY. "
            "Agrégala al archivo .env."
        )

    return genai.Client(
        api_key=GEMINI_API_KEY,
    )


def create_reranker_model() -> CrossEncoder:
    """
    Crea el CrossEncoder multilingüe utilizado
    para reclasificar los candidatos recuperados.
    """

    return CrossEncoder(
        RERANKER_MODEL_NAME,
        max_length=512,
        device="cpu",
    )


def format_document_source(
    document: Document,
) -> str:
    """
    Convierte los metadatos en una referencia legible.
    """

    metadata = document.metadata

    source_parts = [
        metadata.get(
            "file_name",
            "Documento interno",
        )
    ]

    if metadata.get("page"):
        source_parts.append(
            f"página {metadata['page']}"
        )

    if metadata.get("sheet_name"):
        source_parts.append(
            f"hoja {metadata['sheet_name']}"
        )

    if metadata.get("row"):
        source_parts.append(
            f"fila {metadata['row']}"
        )

    return " - ".join(source_parts)


def rerank_results(
    reranker: CrossEncoder,
    question: str,
    candidates: list[tuple[Document, float]],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[tuple[Document, float]]:
    """
    Reclasifica los candidatos usando un CrossEncoder.
    """

    if not candidates:
        return []

    if top_k <= 0:
        raise ValueError(
            "top_k debe ser mayor que cero."
        )

    pairs = [
        (
            question,
            document.page_content,
        )
        for document, _ in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    scored_candidates = []

    for (
        document,
        vector_distance,
    ), rerank_score in zip(
        candidates,
        scores,
    ):
        metadata = document.metadata.copy()

        metadata["vector_distance"] = round(
            float(vector_distance),
            6,
        )
        metadata["rerank_score"] = round(
            float(rerank_score),
            6,
        )

        scored_candidates.append(
            (
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                ),
                float(vector_distance),
                float(rerank_score),
            )
        )

    scored_candidates.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    final_results = []

    for position, (
        document,
        vector_distance,
        _,
    ) in enumerate(
        scored_candidates[:top_k],
        start=1,
    ):
        document.metadata[
            "rerank_position"
        ] = position

        final_results.append(
            (
                document,
                vector_distance,
            )
        )

    return final_results


def has_sufficient_context(
    results: list[tuple[Document, float]],
) -> bool:
    """
    Evalúa si los resultados recuperados tienen una
    relevancia mínima para permitir que Gemini responda.

    La decisión usa:
    - distancia vectorial;
    - puntuación del reranker;
    - cantidad mínima de resultados relevantes.
    """

    if not results:
        return False

    relevant_results = 0

    for document, vector_distance in results:
        rerank_score = document.metadata.get(
            "rerank_score"
        )

        vector_ok = (
            float(vector_distance)
            <= MAX_ACCEPTABLE_VECTOR_DISTANCE
        )

        rerank_ok = (
            rerank_score is not None
            and float(rerank_score)
            >= MIN_ACCEPTABLE_RERANK_SCORE
        )

        # Si el reranker no está disponible, se utiliza
        # únicamente la distancia vectorial.
        if rerank_score is None:
            is_relevant = vector_ok
        else:
            is_relevant = vector_ok and rerank_ok

        document.metadata[
            "passes_confidence_threshold"
        ] = is_relevant

        if is_relevant:
            relevant_results += 1

    return relevant_results >= MIN_RELEVANT_RESULTS


def extract_responsible_areas(
    results: list[tuple[Document, float]],
) -> List[str]:
    """
    Obtiene áreas responsables sin duplicados.

    No inventa correos, teléfonos ni canales que no estén
    presentes explícitamente en los documentos.
    """

    areas = []
    seen = set()

    for document, _ in results:
        metadata = document.metadata

        candidates = [
            metadata.get(
                "row_responsible_area"
            ),
            metadata.get(
                "responsible_area"
            ),
        ]

        for area in candidates:
            if not area:
                continue

            normalized_area = str(
                area
            ).strip()

            if (
                normalized_area
                and normalized_area not in seen
            ):
                seen.add(
                    normalized_area
                )
                areas.append(
                    normalized_area
                )

    return areas


def build_fallback_message(
    results: list[tuple[Document, float]] | None = None,
) -> str:
    """
    Construye una respuesta segura cuando el contexto
    no alcanza el nivel mínimo de confianza.
    """

    base_message = (
        "No encontré información suficiente en los "
        "documentos internos de NexoData Consulting "
        "para responder con seguridad."
    )

    if not results:
        return (
            f"{base_message} Te recomiendo consultar "
            "con el área responsable correspondiente."
        )

    areas = extract_responsible_areas(
        results
    )

    if not areas:
        return (
            f"{base_message} Te recomiendo consultar "
            "con el área responsable correspondiente."
        )

    if len(areas) == 1:
        return (
            f"{base_message} Te recomiendo consultar "
            f"con el área de {areas[0]}."
        )

    areas_text = " o ".join(
        areas[:2]
    )

    return (
        f"{base_message} Te recomiendo consultar "
        f"con {areas_text}."
    )

def build_context(
    results: list[tuple[Document, float]],
) -> str:
    """
    Organiza los fragmentos recuperados y reclasificados
    para enviarlos a Gemini.
    """

    context_blocks = []

    for position, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        source = format_document_source(
            document
        )

        metadata = document.metadata

        metadata_lines = [
            f"Fuente: {source}",
        ]

        if metadata.get("official_title"):
            metadata_lines.append(
                "Título oficial: "
                f"{metadata['official_title']}"
            )

        if metadata.get("category"):
            metadata_lines.append(
                f"Categoría: {metadata['category']}"
            )

        if metadata.get("responsible_area"):
            metadata_lines.append(
                "Área responsable: "
                f"{metadata['responsible_area']}"
            )

        if metadata.get("last_updated"):
            metadata_lines.append(
                "Fecha de actualización: "
                f"{metadata['last_updated']}"
            )

        if metadata.get("row_category"):
            metadata_lines.append(
                "Categoría de la fila: "
                f"{metadata['row_category']}"
            )

        if metadata.get("row_responsible_area"):
            metadata_lines.append(
                "Área responsable de la fila: "
                f"{metadata['row_responsible_area']}"
            )

        if metadata.get("related_document"):
            metadata_lines.append(
                "Documento relacionado: "
                f"{metadata['related_document']}"
            )

        metadata_lines.append(
            f"Distancia vectorial: {score:.4f}"
        )

        if metadata.get("rerank_score") is not None:
            metadata_lines.append(
                "Puntuación de reclasificación: "
                f"{metadata['rerank_score']:.4f}"
            )

        context_blocks.append(
            f"""
DOCUMENTO {position}
{chr(10).join(metadata_lines)}

Contenido:
{document.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_blocks
    )


def build_prompt(
    question: str,
    context: str,
    conversation_history: List[dict] | None = None,
) -> str:
    """
    Construye el prompt final del agente.
    """

    history_text = ""

    if conversation_history:
        recent_messages = conversation_history[-6:]

        formatted_messages = []

        for message in recent_messages:
            role = message.get(
                "role",
                "usuario",
            )

            content = message.get(
                "content",
                "",
            )

            formatted_messages.append(
                f"{role.upper()}: {content}"
            )

        history_text = "\n".join(
            formatted_messages
        )

    return f"""
{SYSTEM_INSTRUCTIONS}

HISTORIAL RECIENTE:
{history_text if history_text else "No existe historial previo."}

CONTEXTO DOCUMENTAL:
{context}

PREGUNTA DEL COLABORADOR:
{question}

INSTRUCCIONES PARA LA RESPUESTA:
- Contesta directamente la pregunta.
- Usa solamente el contexto documental.
- Explica los pasos en orden cuando corresponda.
- Menciona la fuente de forma natural.
- No agregues una bibliografía extensa.
- Si falta información, responde exactamente con este sentido:
  "{NO_INFORMATION_MESSAGE}"

RESPUESTA DE SOL AI:
""".strip()


def extract_sources(
    results: list[tuple[Document, float]],
) -> List[dict]:
    """
    Genera una lista de fuentes sin duplicados.
    """

    sources = []
    seen_sources = set()

    for document, score in results:
        source_name = format_document_source(
            document
        )

        if source_name in seen_sources:
            continue

        seen_sources.add(
            source_name
        )

        sources.append(
            {
                "source": source_name,
                "file_name": document.metadata.get(
                    "file_name"
                ),
                "page": document.metadata.get(
                    "page"
                ),
                "sheet_name": document.metadata.get(
                    "sheet_name"
                ),
                "row": document.metadata.get(
                    "row"
                ),
                "category": document.metadata.get(
                    "category"
                ),
                "responsible_area": (
                    document.metadata.get(
                        "responsible_area"
                    )
                ),
                "last_updated": document.metadata.get(
                    "last_updated"
                ),
                "distance": round(
                    float(score),
                    4,
                ),
                "rerank_score": round(
                    float(
                        document.metadata.get(
                            "rerank_score",
                            0.0,
                        )
                    ),
                    4,
                ),
            }
        )

    return sources


class SolAIAgent:
    """
    Agente RAG interno de NexoData Consulting.
    """

    def __init__(self) -> None:
        self.client = create_gemini_client()
        self.vector_store = load_vector_store()

        self.reranker: CrossEncoder | None = None
        self.reranker_available = False

        try:
            print(
                "Cargando modelo de reranking..."
            )

            self.reranker = create_reranker_model()
            self.reranker_available = True

            print(
                "Modelo de reranking cargado."
            )

        except Exception as error:
            print(
                "[ADVERTENCIA] No se pudo cargar el reranker. "
                "Se conservará el orden de ChromaDB. "
                f"Detalle: {error}"
            )

    def retrieve(
        self,
        question: str,
        top_k: int = RETRIEVAL_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Recupera una lista amplia de candidatos y luego
        los reclasifica.
        """

        candidate_k = max(
            RERANK_CANDIDATES,
            top_k,
        )

        candidates = search_documents(
            vector_store=self.vector_store,
            query=question,
            top_k=candidate_k,
            filters=filters,
        )

        if (
            not self.reranker_available
            or self.reranker is None
        ):
            return candidates[:top_k]

        try:
            return rerank_results(
                reranker=self.reranker,
                question=question,
                candidates=candidates,
                top_k=top_k,
            )

        except Exception as error:
            print(
                "[ADVERTENCIA] Falló el reranking. "
                "Se utilizará el orden vectorial original. "
                f"Detalle: {error}"
            )

            return candidates[:top_k]

    def answer(
        self,
        question: str,
        conversation_history: List[dict] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Recupera, evalúa la confianza y genera una respuesta.
        """

        clean_question = question.strip()

        if not clean_question:
            raise ValueError(
                "La pregunta no puede estar vacía."
            )

        results = self.retrieve(
            question=clean_question,
            filters=filters,
        )

        if not results:
            return RAGResponse(
                answer=build_fallback_message(),
                sources=[],
                retrieved_documents=[],
                confidence_status="no_results",
            )

        if not has_sufficient_context(
            results
        ):
            return RAGResponse(
                answer=build_fallback_message(),
                sources=[],
                retrieved_documents=[],
                confidence_status="insufficient_context",
            )

        context = build_context(
            results
        )

        prompt = build_prompt(
            question=clean_question,
            context=context,
            conversation_history=conversation_history,
        )

        response = self.client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=MODEL_TEMPERATURE,
                max_output_tokens=700,
            ),
        )

        answer_text = (
            response.text.strip()
            if response.text
            else build_fallback_message(
                results
            )
        )

        sources = extract_sources(
            results
        )

        retrieved_documents = [
            document
            for document, _ in results
        ]

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            retrieved_documents=retrieved_documents,
            confidence_status="sufficient_context",
        )


def show_response(
    response: RAGResponse,
) -> None:
    """
    Muestra una respuesta de prueba en la terminal.
    """

    print("\nRespuesta de Sol AI")
    print("=" * 60)
    print(response.answer)

    print(
        "\nEstado de confianza: "
        f"{response.confidence_status}"
    )

    print("\nFuentes recuperadas")
    print("=" * 60)

    if not response.sources:
        print("No se recuperaron fuentes.")
        return

    for position, source in enumerate(
        response.sources,
        start=1,
    ):
        print(
            f"{position}. {source['source']} "
            f"(distancia: {source['distance']}, "
            f"rerank: {source['rerank_score']})"
        )


def run_confidence_tests(
    agent: SolAIAgent,
) -> None:
    """
    Prueba una pregunta cubierta y otra fuera del alcance.
    """

    test_questions = [
        (
            "PREGUNTA CON RESPALDO",
            (
                "¿Qué debo hacer si detecto "
                "un incidente de seguridad?"
            ),
        ),
        (
            "PREGUNTA FUERA DEL ALCANCE",
            (
                "¿Cuál es la receta oficial de NexoData "
                "para preparar una pizza napolitana?"
            ),
        ),
    ]

    for title, question in test_questions:
        print(
            f"\n\n{title}"
        )
        print("-" * 60)
        print(
            f"Pregunta: {question}"
        )

        result = agent.answer(
            question=question,
        )

        show_response(
            result
        )


if __name__ == "__main__":
    print("Iniciando Sol AI...")

    agent = SolAIAgent()

    run_confidence_tests(
        agent
    )
