# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thùy Linh
**Nhóm:** Nhóm 9
**Ngày:** 10/4/2026

### Danh sách thành viên nhóm

| STT | Họ tên | MSSV |
|-----|--------|------|
| 1 | Nguyễn Triệu Gia Khánh | 2A202600225 |
| 2 | Nguyễn Thùy Linh | 2A202600216 |
| 3 | Nguyễn Hoàng Khải Minh | 2A202600159 |
| 4 | Nguyễn Thị Diệu Linh | 2A202600209 |
| 5 | Nguyễn Hoàng Duy | 2A202600158 |

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
High cosine similarity nghĩa là góc giữa hai vector biểu diễn văn bản rất nhỏ, cho thấy chúng có hướng đi gần như trùng nhau trong không gian vector. Điều này đồng nghĩa với việc hai văn bản có sự tương đồng rất lớn về mặt ngữ nghĩa hoặc nội dung.

**Ví dụ HIGH similarity:**
- Sentence A:    "Tôi muốn mua một chiếc iPhone 15 Pro Max màu xanh"
- Sentence B: "Cho tôi xem các mẫu iPhone 15 Pro Max màu xanh"
- Tại sao tương đồng: Cả hai câu đều nói về việc tìm mua iPhone 15 Pro Max màu xanh, chỉ khác nhau về cách diễn đạt (một bên là "mua", một bên là "xem các mẫu").

**Ví dụ LOW similarity:**
- Sentence A: "Tôi muốn mua một chiếc iPhone 15 Pro Max màu xanh"
- Sentence B: "Hôm nay trời đẹp quá, tôi muốn đi dạo"
- Tại sao khác: Câu A nói về việc mua điện thoại, câu B nói về thời tiết và đi dạo, hoàn toàn không liên quan đến nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Cosine similarity được ưu tiên vì nó tập trung vào hướng của vector thay vì độ dài, giúp loại bỏ ảnh hưởng của việc độ dài văn bản khác nhau (một đoạn văn dài và một câu ngắn cùng chủ đề vẫn được coi là tương đồng). Trong khi đó, Euclidean distance rất nhạy cảm với độ dài, dễ dẫn đến kết quả sai lệch nếu hai văn bản có độ dài chênh lệch lớn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* [(10000 - 50) / (500 - 50)] = [22.11] → 23 chunks
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
Khi overlap tăng lên 100, số lượng chunk sẽ tăng lên (khoảng 25-26 chunks) vì "bước nhảy" giữa các đoạn bị thu hẹp lại. Người ta muốn overlap nhiều hơn để đảm bảo các thông tin quan trọng nằm ở biên đoạn văn không bị cắt đôi một cách vô nghĩa, giúp duy trì ngữ cảnh liên tục cho mô hình Embedding khi xử lý.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn


**Domain:** IELTS knowledge base (Reading, Listening, Writing, Speaking)

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*
Nhóm chọn IELTS vì dữ liệu có cấu trúc rõ (band descriptors, tips theo kỹ năng, dạng câu hỏi theo section) và rất phù hợp để kiểm thử retrieval theo ngữ nghĩa. Đây cũng là domain gần với nhu cầu học tập thực tế, nên dễ đánh giá chất lượng câu trả lời của agent. Ngoài ra, domain này có nhiều metadata tự nhiên như skill, task type, band level để áp dụng filtered search.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | IELTS Speaking part 2 Band Descriptors | Tài liệu nội bộ | 8,200 | skill=speaking task=task2; level=all; source=official |
| 2 | IELTS Speaking Part 2 Strategies | IDP Blog + ghi chú lớp | 6,100 | skill=speaking; part=2; level=intermediate |
| 3 | Common Collocations for IELTS Speaking | Corpus-based notes | 9,050 | skill=speaking; topic=lexical_resource; band=5-7 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| skill | string | writing / speaking | Cho phép lọc đúng kỹ năng trước khi semantic search |
| question_type | string | tfng / multiple_choice | Thu hẹp phạm vi theo dạng bài người học đang hỏi |
| band_level | string | 6.0-7.0 | Trả nội dung đúng độ khó hoặc mục tiêu điểm |
| source | string | official / internal_notes | Ưu tiên tài liệu tin cậy khi nhiều chunk cạnh tranh |

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

 Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| 01_ielts_kb.md | FixedSizeChunker (fixed_size) | 7 | 178.57 | |
| 01_ielts_kb.md | SentenceChunker (by_sentences) | 6 |206.33 | |
| 01_ielts_kb.md | RecursiveChunker (recursive) | 9 | 137.33 | |
| 02_ielts_kb.md | FixedSizeChunker (fixed_size) | 12 | 195.58 | |
| 02_ielts_kb.md | SentenceChunker (by_sentences) | 7 | 332.14 | |
| 02_ielts_kb.md | RecursiveChunker (recursive) | 17 | 136.59 |  |
| 03_ielts_kb.md | FixedSizeChunker (fixed_size) | 31 | 193.97 | |
| 03_ielts_kb.md | SentenceChunker (by_sentences) | 20 | 298.50 | |
| 03_ielts_kb.md | RecursiveChunker (recursive) | 52 | 114.02 | |

### Strategy Của Tôi

**Loại:** SentenceChunker

**Mô tả cách hoạt động:**
Chiến lược này thực hiện chia nhỏ văn bản dựa trên các dấu hiệu kết thúc câu phổ biến như dấu chấm (. ), dấu chấm hỏi (? ), dấu chấm than (! ) kết hợp với khoảng trắng hoặc ký tự xuống dòng. Sau khi nhận diện được danh sách các câu đơn lẻ thông qua biểu thức chính quy (Regex), thuật toán sẽ tiến hành gom chúng lại thành từng nhóm (chunk) với số lượng câu tối đa được quy định bởi tham số max_sentences_per_chunk. Cách tiếp cận này giúp đảm bảo mỗi đoạn văn bản được trích xuất luôn kết thúc trọn vẹn ở một ý, tránh tình trạng bị ngắt quãng giữa chừng gây khó hiểu cho mô hình ngôn ngữ.

**Tại sao tôi chọn strategy này cho domain nhóm?**
Trong tài liệu IELTS, các tips, từ vựng hoặc mô tả tiêu chí band điểm thường được trình bày dưới dạng các câu đơn hoặc danh sách gạch đầu dòng ngắn gọn và súc tích. Việc sử dụng SentenceChunker giúp khai thác đặc điểm này để mỗi chunk trả về là một tập hợp các lời khuyên hoàn chỉnh hoặc một ví dụ sử dụng từ vựng nguyên vẹn, từ đó giúp AI Agent đưa ra các phản hồi mạch lạc và đúng trọng tâm yêu cầu của người học.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 03_ielts_kb.md | Best baseline (Recursive) | 52 | 114.02 | High (Tối ưu cho cấu trúc Markdown/List) |
| 03_ielts_kb.md | Của tôi (SentenceChunker) | 20 | 298.50 | Medium-High (Tốt cho ngữ nghĩa câu đơn) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy (tóm tắt) | Điểm nhóm (/10) | Điểm mạnh | Điểm yếu |
|-------------|-------------------|-----------------|-----------|----------|
| Nguyễn Triệu Gia Khánh | Semantic Chunker | 10 | Các chunk giàu ngữ nghĩa hơn giúp cải thiện độ chính xác của bước retrieval, dẫn đến phản hồi mạch lạc và liên quan hơn từ LLM. | Tính toán embedding và độ tương đồng có thể tốn kém hơn fixed-size chunking. |
| Nguyễn Thùy Linh | SentenceChunker| 10 | Giữ được ngữ nghĩa tự nhiên của câu, ít bị cắt ngang ý giữa câu | Có thể thiếu thông tin quan trọng nếu câu standalone không đủ nghĩa |
| Nguyễn Hoàng Khải Minh | RecursiveChunker | 10 | Cố gắng tôn trọng cấu trúc logic (đoạn văn, câu) của văn bản nhiều nhất có thể trong khi vẫn đảm bảo kích thước chunk phù hợp. | Việc triển khai có thể phức tạp hơn một chút, chi phí tính toán có thể tăng lên do quá trình kiểm tra và chia đệ quy. |
| Nguyễn Thị Diệu Linh | FixedSizeChunker | 10 | Dễ triển khai và quản lý, kích thước chunk đồng nhất giúp đơn giản hóa việc xử lý hàng loạt (batch processing). | Rất dễ phá vỡ cấu trúc ngữ nghĩa tự nhiên của văn bản. Một câu, một ý tưởng quan trọng có thể bị chia cắt làm đôi, nằm ở hai chunk khác nhau, làm giảm chất lượng ngữ cảnh được cung cấp cho LLM. |
| Nguyễn Hoàng Duy | HeadingChunker | 10 | Giữ được context lớn theo từng mục (section-level) | Chunk có thể quá dài, có thể chứa nhiều thông tin không liên quan, giảm precision |

**Kết luận strategy tốt nhất cho domain này:**  
Nhóm thống nhất **`RecursiveChunker`** làm hướng chính cho IELTS (heading/bullet), đồng thời mỗi thành viên có nhánh so sánh riêng để học chéo. Sau benchmark và demo, nhóm **đồng thuận 10/10** cho từng thành viên về đóng góp strategy và phối hợp nhóm.


**Strategy nào tốt nhất cho domain này? Tại sao?**
Đối với domain IELTS, RecursiveChunker là tốt nhất vì tài liệu thường có cấu trúc phân cấp rõ ràng (Heading -> Paragraph -> List). Việc ưu tiên tách theo Header giúp "đóng gói" toàn bộ ngữ cảnh của một chủ đề (ví dụ: "Tips cho Part 2") vào cùng một không gian vector,

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
Sử dụng Regex (?<=[.!?])\s+ (lookbehind) để tách câu mà không làm mất dấu câu. Xử lý các edge case như viết tắt (Mr., Dr.) bằng cách kết hợp thêm điều kiện độ dài hoặc danh sách từ loại trừ để tránh ngắt câu sai.

**`RecursiveChunker.chunk` / `_split`** — approach:
`chunk()` gọi `_split()` với thứ tự separator từ thô đến mịn. Nếu đoạn vượt `chunk_size`, hàm tiếp tục đệ quy với separator tiếp theo; nếu đã hết separator thì cắt cứng theo độ dài. Cách này đảm bảo chunk cuối cùng luôn nằm trong giới hạn kích thước.

### EmbeddingStore

**`add_documents` + `search`** — approach:
`add_documents` tạo record gồm `id`, `content`, `metadata`, `embedding`; metadata mặc định luôn có `doc_id` để truy xuất/xóa theo tài liệu gốc. Hệ thống ưu tiên ChromaDB nếu khả dụng, nếu không sẽ fallback sang in-memory list. `search` tính embedding cho query một lần, chấm điểm bằng dot product (in-memory) hoặc dùng `distances` của Chroma rồi đổi dấu để giữ quy ước “score cao hơn = gần hơn”.

**`search_with_filter` + `delete_document`** — approach:
`search_with_filter` lọc theo metadata trước khi xếp hạng similarity, giúp tăng precision cho câu hỏi theo category/topic. `delete_document` xóa theo `doc_id` cả ở Chroma và in-memory, trả về boolean để xác nhận có dữ liệu bị xóa thật hay không.

### KnowledgeBaseAgent

**`answer`** — approach:
Agent retrieve `top_k` chunks, đóng gói thành context dạng `[1] ... [2] ...`, rồi tạo prompt theo khung: instruction -> context -> question -> answer. Thiết kế này giúp câu trả lời bám dữ liệu retrieve và dễ kiểm tra nguồn nội dung.

### Test Results

```
# Paste output of: pytest tests/ -v
```
py -m pytest tests/test_solution.py -v
============================= test session starts =============================
collected 42 items
...
============================= 42 passed in 0.21s ==============================
```

**Số tests pass:** **42 / 42**
---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "How to get band 8.0?"| "Ways to achieve IELTS 8.0" | high | 0.89 | Yes |
| 2 | "Apple is a fruit" | "I love my iPhone" | low | 0.35| Yes |
| 3 | "Practice speaking daily" | "Don't forget to talk every day" | high | 0.82 | Yes |
| 4 | "The exam is hard" | "The test is difficult" | high | 0.82 | Yes |
| 5 | "Bank of the river" | "Financial bank" | low | 0.75 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Cặp số 5 (Từ đồng âm khác nghĩa). Dù cùng từ "bank" nhưng embedding vẫn phân biệt được ngữ cảnh (sông vs tài chính), chứng tỏ model hiểu ngữ nghĩa thay vì chỉ khớp từ khóa (keyword matching).

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

 # | Query | Gold Answer |
|---|-------|-------------|
| 1 | In IELTS Speaking Part 2, how should I open my answer in the first 10-15 seconds so I sound clear and on-topic before adding details? | Start with a direct one-sentence answer to the prompt, then extend with reason/example instead of giving background first. |
| 2 | My ideas are too general in Speaking. What exact structure can I use to move from a broad claim to a specific personal example without losing coherence? | Use a 3-step structure: general statement -> narrow reason -> concrete personal example (time/place/result). |
| 3 | If I don't know much about a topic, what is the safest high-control response pattern that avoids silence but still sounds natural and balanced? | Use an "it depends" frame with two short contrasting cases, then close by choosing one side. |
| 4 | During Speaking, when I run out of ideas mid-answer, what language moves can I use to keep fluency while buying thinking time and still add value? | Use filler bridges plus extension templates (reason, example, comparison) to maintain flow instead of stopping abruptly. |
| 5 | For a band-5 to band-6 improvement path, which habit hurts score most in spontaneous speaking and what should I do immediately to replace it? | Avoid switching to L1; stay in English and paraphrase with simpler words when vocabulary gaps appear. |
### Kết Quả Của Tôi


| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Opening Part 2 | """Start with a direct answer... focus on the prompt""" | 0.88 | Yes | Use a direct opening sentence... |
| 2 | General to Specific | """Structure: Point -> Reason -> Example""" | 0.85 | Yes | Apply the 3-step PREP framework... |
| 3 | Lack of topic info | """Strategy: It depends/Contrast two sides""" | 0.79 | Yes | Use 'it depends' to talk about 2 sides... |
| 4 | Run out of ideas | """Fillers: 'That's an interesting question'...""" | 0.82 | Yes | Use thinking phrases and fillers... |
| 5 | Band 5 to 6 habit | """Avoid L1, use paraphrasing""" | 0.76 | Yes | Stop translating from mother tongue... |
**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Học được cách dùng Semantic Chunker (tách dựa trên độ biến thiên của cosine similarity giữa các câu liên tiếp). Đây là kỹ thuật rất hay để đảm bảo tính nhất quán chủ đề trong một chunk.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Một số nhóm sử dụng Hybrid Search (kết hợp BM25 và Vector Search). Điều này giúp cải thiện kết quả khi người dùng tìm kiếm các thuật ngữ chuyên môn chính xác trong IELTS.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Nếu làm lại, mình sẽ chú trọng hơn vào việc Data Cleaning, đặc biệt là loại bỏ các ký tự thừa từ file PDF/Markdown và gán Metadata chi tiết hơn cho từng dạng bài (Map, Diagram, ...) để filter chính xác hơn.
---


