import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.rag_agent import (
    RAGResponse,
    SolAIAgent,
)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Sol AI | NexoData Consulting",
    page_icon="🔷",
    layout="centered",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INTERACTION_LOG_PATH = DATA_DIR / "interaction_logs.csv"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_logs.csv"


# =========================================================
# ESTADO DE SESIÓN
# =========================================================

def initialize_session_state() -> None:
    """
    Inicializa las variables de la sesión.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}


def clear_conversation() -> None:
    """
    Limpia el historial de conversación.
    """

    st.session_state.messages = []
    st.session_state.feedback_given = {}
    st.rerun()


initialize_session_state()


# =========================================================
# ESTILOS
# =========================================================

def apply_styles() -> None:
    """
    Aplica un tema azul corporativo sin usar HTML complejo.
    """

    if st.session_state.dark_mode:
        background = "#071426"
        surface = "#0D1D33"
        surface_alt = "#122944"
        text = "#EAF2FF"
        muted = "#A9BCD4"
        border = "#284564"
        sidebar = "#09182A"
        input_background = "#0A1B30"
        shadow = "rgba(0, 0, 0, 0.28)"
    else:
        background = "#F3F7FD"
        surface = "#FFFFFF"
        surface_alt = "#EEF4FF"
        text = "#17233A"
        muted = "#66758C"
        border = "#D9E4F2"
        sidebar = "#FFFFFF"
        input_background = "#FFFFFF"
        shadow = "rgba(18, 47, 91, 0.09)"

    st.markdown(
        f"""
<style>
    .stApp {{
        background-color: {background};
        color: {text};
    }}

    .main .block-container {{
        max-width: 900px;
        padding-top: 1.4rem;
        padding-bottom: 6rem;
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar};
        border-right: 1px solid {border};
    }}

    [data-testid="stSidebar"] * {{
        color: {text};
    }}

    [data-testid="stChatMessage"] {{
        background-color: {surface};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 14px {shadow};
    }}

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em {{
        color: {text};
    }}

    [data-testid="stChatInput"] textarea {{
        background-color: {input_background};
        color: {text};
    }}

    [data-testid="stExpander"] {{
        background-color: {surface};
        border: 1px solid {border};
        border-radius: 12px;
    }}

    div.stButton > button {{
        border-radius: 10px;
        border: 1px solid {border};
        background-color: {surface_alt};
        color: {text};
    }}

    div.stButton > button:hover {{
        border-color: #1D5FEA;
        color: #1D5FEA;
    }}

    [data-testid="stMetric"] {{
        background-color: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 0.7rem;
    }}

    h1, h2, h3, h4, p, li, label {{
        color: {text};
    }}

    small {{
        color: {muted};
    }}

    footer {{
        visibility: hidden;
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


apply_styles()


# =========================================================
# REGISTRO DE INTERACCIONES Y FEEDBACK
# =========================================================

def ensure_data_directory() -> None:
    """
    Crea la carpeta data si no existe.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def append_csv_row(
    file_path: Path,
    fieldnames: list[str],
    row: dict,
) -> None:
    """
    Agrega una fila a un archivo CSV.
    """

    ensure_data_directory()
    file_exists = file_path.exists()

    with file_path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


def log_interaction(
    interaction_id: str,
    question: str,
    response: RAGResponse,
    response_time_seconds: float,
) -> None:
    """
    Guarda información básica para monitoreo.
    """

    append_csv_row(
        file_path=INTERACTION_LOG_PATH,
        fieldnames=[
            "timestamp",
            "interaction_id",
            "question",
            "answer",
            "confidence_status",
            "response_time_seconds",
            "source_count",
            "sources_json",
        ],
        row={
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "interaction_id": interaction_id,
            "question": question,
            "answer": response.answer,
            "confidence_status": response.confidence_status,
            "response_time_seconds": round(
                response_time_seconds,
                3,
            ),
            "source_count": len(
                response.sources
            ),
            "sources_json": json.dumps(
                response.sources,
                ensure_ascii=False,
            ),
        },
    )


def save_feedback(
    interaction_id: str,
    feedback: str,
) -> None:
    """
    Guarda feedback positivo o negativo.
    """

    append_csv_row(
        file_path=FEEDBACK_LOG_PATH,
        fieldnames=[
            "timestamp",
            "interaction_id",
            "feedback",
        ],
        row={
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "interaction_id": interaction_id,
            "feedback": feedback,
        },
    )

    st.session_state.feedback_given[
        interaction_id
    ] = feedback

    st.toast(
        "Gracias. Tu opinión quedó registrada.",
        icon="✅",
    )


# =========================================================
# AGENTE
# =========================================================

@st.cache_resource(
    show_spinner=False,
)
def load_agent() -> SolAIAgent:
    """
    Carga una sola instancia del agente.
    """

    return SolAIAgent()


# =========================================================
# FUENTES
# =========================================================

def format_source_location(
    source: dict,
) -> str:
    """
    Devuelve la ubicación de una fuente.
    """

    location_parts = []

    if source.get("page"):
        location_parts.append(
            f"Página {source['page']}"
        )

    if source.get("sheet_name"):
        location_parts.append(
            f"Hoja {source['sheet_name']}"
        )

    if source.get("row"):
        location_parts.append(
            f"Fila {source['row']}"
        )

    return (
        " · ".join(location_parts)
        if location_parts
        else "Documento completo"
    )


def show_sources(
    sources: list[dict],
) -> None:
    """
    Muestra las fuentes sin usar HTML personalizado.
    """

    if not sources:
        return

    with st.expander(
        f"Fuentes consultadas ({len(sources)})",
        expanded=False,
    ):
        for source in sources:
            file_name = source.get(
                "file_name",
                "Documento interno",
            )

            st.markdown(
                f"**📄 {file_name}**"
            )

            details = [
                format_source_location(
                    source
                )
            ]

            if source.get("category"):
                details.append(
                    f"Categoría: {source['category']}"
                )

            if source.get("responsible_area"):
                details.append(
                    f"Área: {source['responsible_area']}"
                )

            if source.get("last_updated"):
                details.append(
                    f"Actualizado: {source['last_updated']}"
                )

            st.caption(
                " · ".join(details)
            )

            st.divider()


# =========================================================
# HISTORIAL Y FEEDBACK
# =========================================================

def build_conversation_history() -> list[dict]:
    """
    Convierte el historial al formato del agente.
    """

    return [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages
    ]


def show_feedback_controls(
    interaction_id: str | None,
) -> None:
    """
    Muestra botones de feedback para una respuesta.
    """

    if not interaction_id:
        return

    saved_feedback = (
        st.session_state.feedback_given.get(
            interaction_id
        )
    )

    if saved_feedback:
        if saved_feedback == "positive":
            st.caption(
                "¡Marcaste esta respuesta como útil!"
            )
        else:
            st.caption(
                "Marcaste esta respuesta para revisión."
            )

        return

    st.caption(
        "¿Esta respuesta te resultó útil?"
    )

    positive_column, negative_column, _ = st.columns(
        [1, 1, 4]
    )

    with positive_column:
        if st.button(
            "Sí",
            key=f"positive_{interaction_id}",
            use_container_width=True,
        ):
            save_feedback(
                interaction_id=interaction_id,
                feedback="positive",
            )
            st.rerun()

    with negative_column:
        if st.button(
            "No",
            key=f"negative_{interaction_id}",
            use_container_width=True,
        ):
            save_feedback(
                interaction_id=interaction_id,
                feedback="negative",
            )
            st.rerun()


# =========================================================
# INTERFAZ
# =========================================================

def show_sidebar() -> None:
    """
    Construye una barra lateral simple y funcional.
    """

    with st.sidebar:
        st.markdown("# 🔷 NexoData")
        st.caption(
            "Datos que conectan. Decisiones que transforman."
        )

        st.toggle(
            "🌙 Modo oscuro",
            key="dark_mode",
            help=(
                "Cambia entre modo claro y oscuro."
            ),
        )

        st.info(
            "Estás conversando con un agente de inteligencia "
            "artificial que responde usando documentos internos."
        )

        st.markdown("### Base de conocimiento")

        st.markdown(
            """
- 8 documentos oficiales
- 314 fragmentos indexados
- PDF, Markdown, Excel y CSV
- Búsqueda semántica y reranking
            """
        )

        st.divider()

        if st.button(
            "Limpiar conversación",
            use_container_width=True,
        ):
            clear_conversation()

        st.warning(
            "Uso interno: no ingreses contraseñas, claves API "
            "ni información sensible."
        )


def show_header() -> None:
    """
    Muestra un encabezado simple.
    """

    st.title(
        "🔷 Sol AI"
    )

    st.subheader(
        "Asistente documental de NexoData Consulting"
    )

    st.caption(
        "Agente de inteligencia artificial · Base documental disponible"
    )

    st.divider()


def show_welcome_message() -> None:
    """
    Muestra la bienvenida inicial usando componentes nativos.
    """

    st.info(
        "Hola, soy Sol AI. Puedo ayudarte a consultar los "
        "documentos internos de NexoData Consulting."
    )

    st.markdown(
        "### Puedes preguntarme, por ejemplo:"
    )

    suggestion_columns = st.columns(
        2
    )

    suggestions = [
        "¿Cómo debo reportar un incidente de seguridad?",
        "¿Qué herramientas usamos para crear dashboards?",
        "¿Cuándo se considera aprobado un entregable?",
        "¿Qué documentos debo leer durante el onboarding?",
    ]

    for index, suggestion in enumerate(
        suggestions
    ):
        with suggestion_columns[index % 2]:
            st.markdown(
                f"🔹 {suggestion}"
            )


def render_chat_history() -> None:
    """
    Renderiza todos los mensajes guardados.
    """

    for message in st.session_state.messages:
        role = message["role"]

        avatar = (
            "🔷"
            if role == "assistant"
            else "👤"
        )

        with st.chat_message(
            role,
            avatar=avatar,
        ):
            st.markdown(
                message["content"]
            )

            if role == "assistant":
                show_sources(
                    message.get(
                        "sources",
                        [],
                    )
                )

                show_feedback_controls(
                    message.get(
                        "interaction_id"
                    )
                )


def process_question(
    agent: SolAIAgent,
    question: str,
) -> None:
    """
    Procesa una pregunta y guarda el resultado.
    """

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user",
        avatar="👤",
    ):
        st.markdown(
            question
        )

    history = build_conversation_history()
    interaction_id = str(
        uuid.uuid4()
    )

    with st.chat_message(
        "assistant",
        avatar="🔷",
    ):
        with st.spinner(
            "Sol AI está consultando los documentos..."
        ):
            start_time = time.perf_counter()

            try:
                response: RAGResponse = agent.answer(
                    question=question,
                    conversation_history=history[:-1],
                )

                response_time = (
                    time.perf_counter()
                    - start_time
                )

                st.markdown(
                    response.answer
                )

                show_sources(
                    response.sources
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "interaction_id": interaction_id,
                        "confidence_status": (
                            response.confidence_status
                        ),
                        "response_time_seconds": (
                            response_time
                        ),
                    }
                )

                log_interaction(
                    interaction_id=interaction_id,
                    question=question,
                    response=response,
                    response_time_seconds=response_time,
                )

                show_feedback_controls(
                    interaction_id
                )

            except Exception as error:
                st.error(
                    "No pude procesar la consulta en este momento. "
                    "Comprueba la conexión con Gemini e inténtalo nuevamente."
                )

                with st.expander(
                    "Ver detalle técnico",
                    expanded=False,
                ):
                    st.code(
                        str(error)
                    )


# =========================================================
# APLICACIÓN PRINCIPAL
# =========================================================

def main() -> None:
    """
    Ejecuta la aplicación.
    """

    show_sidebar()
    show_header()

    try:
        with st.spinner(
            "Iniciando Sol AI..."
        ):
            agent = load_agent()

    except Exception as error:
        st.error(
            "No fue posible iniciar Sol AI."
        )

        st.markdown(
            """
Revisa que:

- exista la carpeta `vectorstore/`;
- la base ChromaDB haya sido creada;
- `GEMINI_API_KEY` esté configurada;
- las dependencias estén instaladas.
            """
        )

        with st.expander(
            "Ver detalle técnico",
        ):
            st.code(
                str(error)
            )

        st.stop()

    if not st.session_state.messages:
        show_welcome_message()

    render_chat_history()

    question = st.chat_input(
        placeholder=(
            "Escribe una pregunta sobre NexoData..."
        ),
        max_chars=1000,
    )

    if question:
        process_question(
            agent=agent,
            question=question,
        )


if __name__ == "__main__":
    main()
