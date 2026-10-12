
import os
import re
import importlib.resources
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from rapidfuzz import fuzz, process
from symspellpy import SymSpell

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from database import (
    create_table,
    save_chat,
    get_chat_history,
    delete_chat,
    save_feedback,
    get_feedback,
    get_feedback_stats,
    get_total_questions,
    get_daily_usage,
    get_feedback_breakdown,
    get_recent_feedback,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI IT Helpdesk Assistant",
    page_icon="🛠️",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
RAG_MAX_L2_DISTANCE = float(
    os.getenv("RAG_MAX_L2_DISTANCE", "1.25")
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
create_table()


# ==================================================
# 2. TECHNICAL VOCABULARY
# ==================================================

# Add important technical terms from your own
# knowledge base here. Acronyms are especially useful.

IT_TERMS = [
    "ssh",
    "linux",
    "rhel",
    "server",
    "database",
    "sql",
    "mysql",
    "mssql",
    "network",
    "troubleshooting",
    "disk",
    "space",
    "memory",
    "cpu",
    "service",
    "firewall",
    "permission",
    "authentication",
    "password",
    "terminal",
    "command",
    "error",
    "errors",
    "logs",
    "application",
    "deployment",
    "python",
    "api",
    "http",
    "https",
    "dns",
    "address",
    "port",
    "process",
    "restart",
    "connection",
    "timeout",
    "storage",
    "backup",
    "restore",
    "check",
    "where",
    "what",
    "which",
    "how",
    "find",
    "space",
    "file",
    "files",
    "running",
    "failed",
    "failure",
    "status",
    "install",
    "configure",
    "update",
    "delete",
    "create",
    "memory",
    "usage",
    "system",
    "operating",
]


# ==================================================
# 3. SPELL CHECKER
# ==================================================

@st.cache_resource
def load_spell_checker():
    spell_checker = SymSpell(
        max_dictionary_edit_distance=2,
        prefix_length=7,
    )

    dictionary_path = (
        importlib.resources.files("symspellpy")
        / "frequency_dictionary_en_82_765.txt"
    )

    loaded = spell_checker.load_dictionary(
        str(dictionary_path),
        term_index=0,
        count_index=1,
    )

    if not loaded:
        raise RuntimeError(
            "Could not load the English spelling dictionary."
        )

    return spell_checker


# ==================================================
# 4. AUTOMATIC TYPO CORRECTION
# ==================================================

# Explicit acronym variants handle cases such as:
# shs -> ssh, sshh -> ssh, s s h -> ssh.
# Add more variants if your users commonly misspell
# other short technical abbreviations.

ACRONYM_ALIASES = {
    "ssh": [
        "shs",
        "sshh",
        "s h",
        "s s h",
        "shh",
    ],
    "sql": [
        "s q l",
        "sqll",
    ],
    "api": [
        "a p i",
        "aip",
    ],
    "dns": [
        "d n s",
    ],
    "cpu": [
        "c p u",
    ],
    "rhel": [
        "r h e l",
    ],
}


def normalize_acronyms(question):
    """Normalize known acronym variants without changing
    the user's stored original question."""
    corrected = question.lower().strip()

    for correct_term, aliases in ACRONYM_ALIASES.items():
        for alias in aliases:
            pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )
            corrected = re.sub(
                pattern,
                correct_term,
                corrected,
                flags=re.IGNORECASE,
            )

    # Also recognize spaced-out known acronyms.
    for term in IT_TERMS:
        if len(term) <= 4 and term.isalpha():
            spaced_pattern = (
                r"(?<!\w)"
                + r"\s*".join(map(re.escape, term))
                + r"(?!\w)"
            )

            corrected = re.sub(
                spaced_pattern,
                term,
                corrected,
                flags=re.IGNORECASE,
            )

    return corrected


def correct_question(question):
    """Correct common spelling errors and known IT terms."""
    original = question.strip()

    if not original:
        return original

    corrected = normalize_acronyms(original)

    # SymSpell handles common English spelling mistakes.
    try:
        spell_checker = load_spell_checker()

        suggestions = spell_checker.lookup_compound(
            corrected,
            max_edit_distance=2,
            ignore_non_words=True,
        )

        if suggestions:
            corrected = suggestions[0].term

    except Exception as error:
        print(f"Spell correction warning: {error}")

    # Restore likely misspellings of known technical words.
    # Only single-word matches are used here.
    tokens = re.findall(r"\b[\w.-]+\b|\W+", corrected)
    output = []

    for token in tokens:
        if not re.fullmatch(r"[a-zA-Z]+", token):
            output.append(token)
            continue

        word = token.lower()

        if len(word) < 3:
            output.append(token)
            continue

        # Keep already-known vocabulary intact.
        if word in IT_TERMS:
            output.append(word)
            continue

        match = process.extractOne(
            word,
            IT_TERMS,
            scorer=fuzz.ratio,
        )

        if match:
            matched_term, score, _ = match

            # Higher threshold reduces accidental changes.
            if score >= 82:
                output.append(matched_term)
                continue

        output.append(word)

    return "".join(output)


# ==================================================
# 5. DOCUMENT LOADING
# ==================================================

def load_documents(uploaded_files=None):
    documents = []

    # Local text files
    for path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        try:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            if content.strip():
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": path.name,
                            "page": 1,
                        },
                    )
                )
        except OSError as error:
            st.warning(f"Cannot read {path.name}: {error}")

    # Uploaded PDFs
    for uploaded_file in uploaded_files or []:
        try:
            pdf = PdfReader(
                BytesIO(uploaded_file.getvalue())
            )

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):
                content = page.extract_text() or ""

                if content.strip():
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": uploaded_file.name,
                                "page": page_number,
                            },
                        )
                    )

        except Exception as error:
            st.warning(
                f"Cannot read {uploaded_file.name}: {error}"
            )

    return documents


# ==================================================
# 6. EMBEDDINGS AND FAISS
# ==================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource
def build_vectorstore(file_signature, _documents):
    if not _documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(_documents)

    if not chunks:
        return None

    return FAISS.from_documents(
        chunks,
        load_embeddings(),
    )


# ==================================================
# 7. GROQ MODEL
# ==================================================

@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key,
    )


# ==================================================
# 8. RAG ANSWER GENERATION
# ==================================================

def answer_question(question, vectorstore, llm):
    corrected_question = correct_question(question)

    if vectorstore is None:
        return (
            "No knowledge-base documents are available. "
            "Upload a PDF or add TXT files to knowledge_base.",
            [],
            corrected_question,
            [],
        )

    # Try corrected question first.
    try:
        results = vectorstore.similarity_search_with_score(
            corrected_question,
            k=4,
        )
    except Exception as error:
        return (
            f"Document retrieval failed: {error}",
            [],
            corrected_question,
            [],
        )

    # If correction produced no usable matches, try original.
    relevant = [
        (doc, score)
        for doc, score in results
        if score <= RAG_MAX_L2_DISTANCE
    ]

    if not relevant and corrected_question != question:
        try:
            original_results = (
                vectorstore.similarity_search_with_score(
                    question,
                    k=4,
                )
            )

            original_relevant = [
                (doc, score)
                for doc, score in original_results
                if score <= RAG_MAX_L2_DISTANCE
            ]

            if original_relevant:
                results = original_results
                relevant = original_relevant
                corrected_question = question

        except Exception:
            pass

    if not relevant:
        return (
            "I could not find sufficiently relevant information "
            "in the available knowledge base. Try rephrasing "
            "your question or adding a relevant document.",
            [],
            corrected_question,
            results,
        )

    context_parts = []
    sources = []

    for doc, score in relevant:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{doc.page_content}"
        )

        item = {
            "source": source,
            "page": page,
            "distance": round(float(score), 4),
        }

        if item not in sources:
            sources.append(item)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an IT Helpdesk Assistant.

Answer using the supplied context.
Do not invent information absent from the context.
If the answer is unavailable, say so clearly.
Explain troubleshooting steps in simple language.
Explain commands before recommending them.
Warn about commands that may delete or modify data.
Never request passwords, API keys, or other secrets.
Never claim you have accessed a real server.

Retrieved context:
{context}
            """,
        ),
        (
            "human",
            "Question: {question}",
        ),
    ])

    try:
        response = (prompt | llm).invoke({
            "context": "\n\n---\n\n".join(context_parts),
            "question": corrected_question,
        })

        answer = response.content

        if not isinstance(answer, str):
            answer = str(answer)

        return (
            answer,
            sources,
            corrected_question,
            results,
        )

    except Exception as error:
        return (
            f"Could not generate an answer. Check your Groq "
            f"API configuration and connection. Error: {error}",
            sources,
            corrected_question,
            results,
        )


# ==================================================
# 9. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}


# ==================================================
# 10. SIDEBAR
# ==================================================

st.sidebar.title("🛠️ AI IT Helpdesk")

page = st.sidebar.radio(
    "Navigation",
    [
        "Chatbot",
        "Analytics Dashboard",
        "Chat History",
    ],
)

st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Upload troubleshooting PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

show_diagnostics = st.sidebar.checkbox(
    "Show retrieval diagnostics",
    value=False,
)

st.sidebar.caption(
    "Local TXT files are loaded from knowledge_base."
)

st.sidebar.caption(
    "Uploaded PDFs are session-only in this implementation."
)


# ==================================================
# 11. INITIALIZE APP RESOURCES
# ==================================================

documents = load_documents(uploaded_files)

file_signature = tuple(
    (
        doc.metadata.get("source", ""),
        doc.metadata.get("page", 1),
        hash(doc.page_content),
    )
    for doc in documents
)

vectorstore = build_vectorstore(
    file_signature,
    documents,
)

llm = None

if GROQ_API_KEY:
    try:
        llm = load_llm(GROQ_API_KEY)
    except Exception as error:
        st.sidebar.error(f"Groq initialization failed: {error}")


# ==================================================
# 12. CHATBOT PAGE
# ==================================================

def show_chatbot():
    st.title("🛠️ AI IT Helpdesk Assistant")

    st.write(
        "Ask IT troubleshooting questions. The assistant "
        "attempts to correct common typos before searching "
        "the knowledge base."
    )

    if not documents:
        st.info(
            "Add TXT documents to knowledge_base or upload a PDF."
        )

    if llm is None:
        st.error(
            "GROQ_API_KEY is missing or the model could not "
            "be initialized. Check your local .env file."
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            for source in message.get("sources", []):
                st.caption(
                    f"Source: {source['source']} | "
                    f"Page: {source['page']}"
                )

    question = st.chat_input(
        "Ask an IT troubleshooting question..."
    )

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("assistant"):
        with st.spinner("Correcting typos and searching..."):
            answer, sources, corrected, raw_results = (
                answer_question(
                    question,
                    vectorstore,
                    llm,
                )
            )

        st.markdown(answer)

        if corrected.strip().casefold() != question.strip().casefold():
            st.caption(
                f"Search query after correction: {corrected}"
            )

        if sources:
            st.markdown("**Sources**")

            for source in sources:
                st.caption(
                    f"{source['source']} — Page {source['page']}"
                )

        if show_diagnostics:
            st.write({
                "Original question": question,
                "Corrected question": corrected,
                "Retrieved chunks": len(raw_results),
                "Accepted sources": len(sources),
                "Distance threshold": RAG_MAX_L2_DISTANCE,
            })

    source_text = "\n".join(
        f"{s['source']} (Page {s['page']})"
        for s in sources
    )

    chat_id = None

    try:
        chat_id = save_chat(
            question,
            answer,
            source_text,
        )
    except Exception as error:
        st.error(f"Could not save chat: {error}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    if chat_id is not None:
        st.divider()
        st.subheader("Was this answer helpful?")

        rating = st.radio(
            "Rate the answer",
            ["Helpful", "Not Helpful"],
            horizontal=True,
            key=f"rating_{chat_id}",
        )

        comment = st.text_area(
            "Optional feedback",
            key=f"comment_{chat_id}",
        )

        if st.button(
            "Submit Feedback",
            key=f"submit_{chat_id}",
        ):
            try:
                save_feedback(
                    chat_id,
                    "helpful" if rating == "Helpful"
                    else "not_helpful",
                    comment,
                )
                st.success("Feedback saved.")
            except Exception as error:
                st.error(f"Feedback could not be saved: {error}")


# ==================================================
# 13. ANALYTICS DASHBOARD
# ==================================================

def show_analytics():
    st.title("📊 Analytics Dashboard")

    total_questions = get_total_questions()
    total_ratings, helpful, not_helpful = get_feedback_stats()

    helpful_rate = (
        round(helpful * 100 / total_ratings, 1)
        if total_ratings
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Questions", total_questions)
    c2.metric("Total Ratings", total_ratings)
    c3.metric("Helpful", helpful)
    c4.metric("Helpful Rate", f"{helpful_rate}%")

    st.divider()

    st.subheader("Daily Question Usage")

    daily_df = pd.DataFrame(get_daily_usage(days=14))

    if not daily_df.empty:
        st.line_chart(
            daily_df,
            x="Date",
            y="Questions",
        )

    st.subheader("Feedback Breakdown")

    feedback_df = pd.DataFrame(get_feedback_breakdown())

    if not feedback_df.empty:
        st.bar_chart(
            feedback_df,
            x="Rating",
            y="Count",
        )

    st.subheader("Recent Feedback")

    feedback_rows = get_recent_feedback(limit=10)

    if feedback_rows:
        st.dataframe(
            pd.DataFrame(
                feedback_rows,
                columns=[
                    "Chat ID",
                    "Question",
                    "Rating",
                    "Comment",
                    "Created At",
                ],
            ),
            use_container_width=True,
        )
    else:
        st.info("No feedback submitted yet.")

    st.subheader("Recent Questions")

    history = get_chat_history(limit=10)

    if history:
        st.dataframe(
            pd.DataFrame(
                history,
                columns=[
                    "ID",
                    "Question",
                    "Answer",
                    "Sources",
                    "Created At",
                ],
            )[["ID", "Question", "Created At"]],
            use_container_width=True,
        )
    else:
        st.info("No saved questions yet.")


# ==================================================
# 14. CHAT HISTORY PAGE
# ==================================================

def show_chat_history():
    st.title("🗂️ Chat History")

    history = get_chat_history(limit=100)

    if not history:
        st.info("No saved chats.")
        return

    for chat in history:
        chat_id, question, answer, sources, created_at = chat

        with st.expander(
            f"#{chat_id} — {question[:100]}"
        ):
            st.caption(f"Created: {created_at}")

            st.markdown("**Question**")
            st.write(question)

            st.markdown("**Answer**")
            st.write(answer)

            if sources:
                st.markdown("**Sources**")
                st.write(sources)

            try:
                feedback = get_feedback(chat_id)

                if feedback:
                    st.write(f"Rating: {feedback[0]}")

                    if feedback[1]:
                        st.write(f"Comment: {feedback[1]}")
            except Exception:
                pass

            confirm = st.checkbox(
                "Confirm deletion",
                key=f"confirm_{chat_id}",
            )

            if st.button(
                "Delete Chat",
                key=f"delete_{chat_id}",
                disabled=not confirm,
            ):
                try:
                    delete_chat(chat_id)
                    st.success("Chat deleted.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not delete chat: {error}")


# ==================================================
# 15. NAVIGATION
# ==================================================

if page == "Chatbot":
    show_chatbot()

elif page == "Analytics Dashboard":
    show_analytics()

elif page == "Chat History":
    show_chat_history()
