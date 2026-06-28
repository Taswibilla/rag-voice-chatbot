import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from chunking import fixed_chunking, overlap_chunking, semantic_chunking

load_dotenv()

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text


def build_index(docs: dict, strategy: str = "overlap"):
    """
    docs = {"filename.pdf": "full text content..."}
    strategy = "fixed" | "overlap" | "semantic"
    Returns: (faiss_index, all_chunks, chunk_metadata)
    """
    all_chunks = []
    chunk_metadata = []  
    for doc_name, text in docs.items():
        print(f"Chunking {doc_name} with '{strategy}' strategy...")
        
        if strategy == "fixed":
            chunks = fixed_chunking(text)
        elif strategy == "overlap":
            chunks = overlap_chunking(text)
        elif strategy == "semantic":
            chunks = semantic_chunking(text, embed_model)
        else:
            chunks = overlap_chunking(text)  

        for chunk in chunks:
            if chunk.strip():
                all_chunks.append(chunk)
                chunk_metadata.append(doc_name)

    print(f"Total chunks created: {len(all_chunks)}")

    
    print("Creating embeddings...")
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")
    return index, all_chunks, chunk_metadata



def retrieve(query: str, index, chunks: list, k: int = 4) -> list:
    """Find the most relevant chunks for a query."""
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results



def generate_answer(query: str, context_chunks: list) -> str:
    """Use GROQ LLM to answer based on retrieved context."""
    context = "\n\n---\n\n".join(context_chunks)
    
    system_prompt = """You are a helpful document assistant. 
Answer questions based ONLY on the provided context. 
If the answer is not in the context, say "I couldn't find that information in the uploaded documents."
Keep answers clear and concise."""

    user_prompt = f"""Context from documents:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()