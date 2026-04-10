from __future__ import annotations
import math
import re
from typing import Callable, List, Optional

# --- CHUNKING STRATEGIES ---

class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        # Tránh lặp vô hạn nếu overlap >= chunk_size
        if step <= 0:
            step = self.chunk_size 

        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.
    Dùng Regex để nhận diện dấu câu kết thúc.
    """
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        
        # Tách câu dựa trên . ! ? theo sau là khoảng trắng hoặc xuống dòng
        sentences = re.split(r'(?<=[.!?])(?:\s+|\n)', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Chia văn bản đệ quy theo thứ tự ưu tiên của dấu phân tách.
    """
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text.strip()] if current_text.strip() else []

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        
        # Tách văn bản
        if separator == "":
            splits = list(current_text)
        else:
            splits = current_text.split(separator)

        final_chunks = []
        current_chunk = ""

        for part in splits:
            # Nếu bản thân part quá lớn, gọi đệ quy sâu hơn
            if len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""
                final_chunks.extend(self._split(part, next_separators))
            # Gom các part nhỏ lại cho đến khi gần chạm ngưỡng chunk_size
            elif len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                sep = separator if current_chunk else ""
                current_chunk += sep + part
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                current_chunk = part

        if current_chunk:
            final_chunks.append(current_chunk.strip())
            
        return [c for c in final_chunks if c]


# --- MATH & SIMILARITY ---

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Tính Cosine Similarity giữa 2 vector.
    """
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)


# --- COMPARATOR ---

class ChunkingStrategyComparator:
    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        
        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            # Tính độ dài trung bình của các chunk
            avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
            
            comparison[name] = {
                # Sửa lỗi: lấy độ dài của danh sách 'chunks', không phải 'chunk_size'
                # Đổi tên key thành 'count' để khớp với test case
                "count": len(chunks), 
                "avg_length": round(avg_len, 2),
                "chunks": chunks
            }
        return comparison

