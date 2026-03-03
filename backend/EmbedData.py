import ollama
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import uuid

class EmbeddingPipeline:    
    def __init__(self, embedding_model: str, chunk_size: int = 512, chunk_overlap: int = 50):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str, source: str) -> List[Dict]:
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': f"{doc_id}_chunk_{chunk_num}",
                    'doc_id': doc_id,
                    'source': source,
                    'created_at': datetime.utcnow().isoformat()
                })
                chunk_num += 1
            
            start += self.chunk_size - self.chunk_overlap
        
        print(f"Created {len(chunks)} chunks from {source}")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                embedding = response.get("embedding")
                if embedding:
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * 768)
        
        return np.array(embeddings, dtype=np.float32)
    
    def process_document(self, text: str, source: str) -> Tuple[List[Dict], np.ndarray]:
        doc_id = str(uuid.uuid4())
        chunks = self.chunk_text(text, doc_id, source)
        
        if not chunks:
            return [], np.array([], dtype=np.float32)
        
        texts_to_embed = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts_to_embed)
        
        return chunks, embeddings
