from typing import Callable
from .store import EmbeddingStore

class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        """
        Khởi tạo agent với một Vector Store và một hàm gọi LLM.
        """
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Thực hiện quy trình RAG để trả lời câu hỏi.
        """
        # 1. Retrieve: Tìm kiếm top-k đoạn văn bản liên quan nhất từ Vector Store
        results = self.store.search(question, top_k=top_k)
        
        # Trích xuất nội dung văn bản từ kết quả tìm kiếm (giả sử kết quả trả về list các dict có key 'text')
        # Tùy thuộc vào cấu trúc EmbeddingStore của bạn, có thể là r['text'] hoặc r.text
        context_chunks = [r['content'] for r in results]
        context_text = "\n---\n".join(context_chunks)

        # 2. Build Prompt: Tạo prompt bao gồm ngữ cảnh để "nhồi" vào LLM
        prompt = f"""Bạn là một chuyên gia tư vấn IELTS. Hãy sử dụng các đoạn thông tin dưới đây để trả lời câu hỏi của người dùng. 
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không biết, đừng tự bịa ra câu trả lời.

NGỮ CẢNH:
{context_text}

CÂU HỎI: {question}

TRẢ LỜI:"""
        # 3. Call LLM: Gửi prompt đã có context cho mô hình ngôn ngữ
        answer = self.llm_fn(prompt)
        
        return answer