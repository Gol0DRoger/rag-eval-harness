from pathlib import Path
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")



# ---------- session state ----------
# uploader_key: changing it resets the file uploader to empty
# ingested_files: {(name, size): (kind, message)} so each file is ingested once
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("ingested_files", {})
st.session_state.setdefault("clear_status", None)


# ---------- helpers ----------
def _check(r: requests.Response) -> None:
    """Raise a readable error if the backend returned a non-2xx response."""
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


def wait_for_run(run_id: str, timeout_s: float = 300.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    while True:
        r = requests.get(f"{BACKEND_URL}/api/runs/{run_id}", timeout=10)
        _check(r)
        run = r.json()
        if run["status"] == "Completed":
            return run
        if run["status"] == "Failed":
            raise RuntimeError(run.get("error") or "Ingestion failed")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for ingestion (status: {run['status']})")
        time.sleep(poll_interval_s)


def ingest_pdf(file) -> tuple[str, str]:
    """Save the PDF, start the background ingest, wait for it. Returns (kind, message)."""
    try:
        path = save_uploaded_pdf(file)
        with st.spinner("Ingesting PDF (chunking + embedding)..."):
            r = requests.post(
                f"{BACKEND_URL}/api/ingest",
                json={"pdf_path": str(path.resolve()), "source_id": path.name},
                timeout=30,
            )
            _check(r)
            run = wait_for_run(r.json()["run_id"])
        return "success", f"Ingested {run['output']['ingested']} chunks from {path.name}"
    except Exception as e:
        return "error", f"Ingestion failed: {e}"


def clear_context() -> None:
    """Runs as a button callback, before the script reruns."""
    try:
        r = requests.post(f"{BACKEND_URL}/api/clear", timeout=30)
        _check(r)
        st.session_state.uploader_key += 1
        st.session_state.ingested_files = {}
        st.session_state.messages = []
        st.session_state.clear_status = (
            "success",
            "Context cleared! You can now upload fresh PDFs.",
        )
    except Exception as e:
        st.session_state.clear_status = ("error", f"Could not clear context: {e}")


# ---------- upload section ----------
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    if file_id not in st.session_state.ingested_files:
        st.session_state.ingested_files[file_id] = ingest_pdf(uploaded)
    kind, message = st.session_state.ingested_files[file_id]
    if kind == "success":
        st.success(message)
    else:
        st.error(message)

# ---------- clear button ----------
st.button("Clear Context & Reset", on_click=clear_context)

status = st.session_state.clear_status
if status:
    kind, message = status
    if kind == "success":
        st.success(message)
    else:
        st.error(message)
    st.session_state.clear_status = None

st.divider()
st.title("Ask a question about your PDFs")

# ---------- question section ----------
with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=10, step=1)
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Generating answer..."):
            r = requests.post(
                f"{BACKEND_URL}/api/query",
                json={"question": question.strip(), "top_k": int(top_k)},
                timeout=180,
            )
            _check(r)
            output = r.json()
    except Exception as e:
        st.error(f"Query failed: {e}")
    else:
        st.subheader("Answer:")
        st.write(output.get("answer") or "(No answer)")
        sources = output.get("sources", [])
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")