import numpy as np

def fixed_chunking(text: str, chunk_size: int = 50) -> list:
    """
    Splits text into equal-sized word chunks.
    Fast but loses context at boundaries.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def overlap_chunking(text: str, chunk_size: int = 50, overlap: int = 10) -> list:
    """
    Chunks share 'overlap' words with the next chunk.
    Prevents information loss at boundaries. RECOMMENDED.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


def semantic_chunking(text: str, model, threshold: float = 0.75) -> list:
    """
    Splits at points where meaning changes significantly.
    Best retrieval quality but slower at index time.
    """
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 5]
    
    if not sentences:
        return [text]
    
    embeddings = model.encode(sentences, show_progress_bar=False)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        vec_a = embeddings[i - 1]
        vec_b = embeddings[i]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            similarity = 0
        else:
            similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        
        if similarity >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i]]
    
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
    
    return chunks