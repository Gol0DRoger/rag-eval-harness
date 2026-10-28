from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

# Load the local BGE model (produces 384-dimensional vectors)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
EMBED_DIM = 384

splitter = SentenceSplitter(chunk_size=250, chunk_overlap=50)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # Generate vectors locally on your CPU
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()