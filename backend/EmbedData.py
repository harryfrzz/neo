import ollama
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import Future, as_completed
import uuid

from shared_executor import executor


class EmbeddingPipeline:
    def __init__(self, embedding_model: str, chunk_size: int = 512, chunk_overlap: int = 50):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_segments(
        self,
        segments: List[Dict],
        source: str,
        category: str = "video_transcript",
        max_chars: int = 500,
    ) -> List[Dict]:
        chunks = []
        doc_id = str(uuid.uuid4())
        chunk_num = 0

        current_texts: List[str] = []
        current_start: float = 0.0
        current_end: float = 0.0
        current_len: int = 0

        def _flush():
            nonlocal chunk_num
            text = " ".join(current_texts).strip()
            if text:
                chunks.append({
                    "text": text,
                    "chunk_id": f"{doc_id}_chunk_{chunk_num}",
                    "doc_id": doc_id,
                    "source": source,
                    "category": category,
                    "start_time": current_start,
                    "end_time": current_end,
                    "created_at": datetime.utcnow().isoformat(),
                })
                chunk_num += 1

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue
            seg_start = float(seg.get("start", 0))
            seg_end   = float(seg.get("end", seg_start))

            if current_len + len(seg_text) > max_chars and current_texts:
                _flush()
                current_texts = []
                current_start = seg_start
                current_end   = seg_end
                current_len   = 0

            if not current_texts:
                current_start = seg_start
            current_texts.append(seg_text)
            current_end = seg_end
            current_len += len(seg_text)

        if current_texts:
            _flush()

        print(f"Created {len(chunks)} timestamped chunks from {source} (category={category})")
        return chunks

    def chunk_text(self, text: str, doc_id: str, source: str, category: str = "document") -> List[Dict]:
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
                    'category': category,
                    'created_at': datetime.utcnow().isoformat()
                })
                chunk_num += 1

            start += self.chunk_size - self.chunk_overlap

        print(f"Created {len(chunks)} chunks from {source} (category={category})")
        return chunks

    def embed_async(self, text: str) -> Future:
        model = self.embedding_model

        def _call():
            try:
                response = ollama.embeddings(model=model, prompt=text)
                return response.get("embedding") or [0.0] * 768
            except Exception as e:
                print(f"Error generating embedding: {e}")
                return [0.0] * 768

        return executor.submit(_call)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        # Submit all futures first (non-blocking)
        futures = [self.embed_async(text) for text in texts]

        # Collect in original order
        embeddings = []
        for fut in futures:
            embeddings.append(fut.result())

        return np.array(embeddings, dtype=np.float32)

    def process_document_async(self, text: str, source: str, category: str = "document") -> Tuple[List[Dict], List[Future]]:
        doc_id = str(uuid.uuid4())
        chunks = self.chunk_text(text, doc_id, source, category=category)

        if not chunks:
            return [], []

        futures = [self.embed_async(chunk['text']) for chunk in chunks]
        return chunks, futures

    @staticmethod
    def collect_embeddings(futures: List[Future]) -> np.ndarray:
        if not futures:
            return np.array([], dtype=np.float32)
        return np.array([f.result() for f in futures], dtype=np.float32)

    def process_document(self, text: str, source: str, category: str = "document") -> Tuple[List[Dict], np.ndarray]:
        chunks, futures = self.process_document_async(text, source, category=category)
        embeddings = self.collect_embeddings(futures)
        return chunks, embeddings
