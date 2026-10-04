# Kết quả lab Day 17: Memory Systems for AI Agent

## Phần đã hoàn thiện

- Baseline giữ lịch sử theo thread, không ghi hồ sơ người dùng và quên khi chuyển thread.
- Advanced giữ lịch sử gần nhất, lưu facts vào `User.md` và nén lịch sử cũ khi vượt ngưỡng.
- Cấu hình model chính và judge; factory hỗ trợ OpenAI, custom, Gemini, Anthropic, Ollama và OpenRouter. OpenRouter dùng giao thức OpenAI-compatible.
- Benchmark offline chạy trên hai bộ dữ liệu gốc, hỏi recall ở thread mới sau từng conversation.
- Test kiểm tra read/write/edit, compact, recall chéo phiên, giảm prompt load, correction, nhiễu và cô lập người dùng.

## Cách chạy

Chạy từ thư mục gốc repo với Python >= 3.11 và pytest:

```powershell
python src/benchmark.py
python -m pytest src/test_agents.py -v --basetemp=state/pytest-check
```

`--basetemp` đặt dữ liệu test trong workspace để tránh lỗi quyền truy cập thư mục tạm của Windows. Thư mục này chỉ dành cho pytest.

Benchmark luôn dùng `force_offline=True`, không cần API key. Mỗi suite dùng trạng thái tạm mới trong `state/`, tránh hồ sơ từ lần chạy trước làm tăng điểm recall. Bộ dữ liệu và `expected_contains` không được thay đổi; agent không đọc đáp án benchmark.

Ngưỡng mặc định: 1.200 token ước lượng; giữ 4 message gần nhất. Có thể cấu hình bằng `COMPACT_THRESHOLD_TOKENS` và `COMPACT_KEEP_MESSAGES`.

## Kết quả đo

Chạy ngày 04/10/2026 bằng Python 3.14.6, pytest 8.4.2: **7 test passed**.

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1098 | 15084 | 0.0% | 0.0% | 0 | 0 |
| Advanced | 1812 | 21790 | 100.0% | 100.0% | 257 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 269 | 22398 | 0.0% | 0.0% | 0 | 0 |
| Advanced | 365 | 14196 | 100.0% | 100.0% | 218 | 3 |

Các chỉ số bao gồm lượt hội thoại và lượt recall. `Agent tokens only` đếm token đầu ra ước lượng. `Prompt tokens processed` cộng system prompt và ngữ cảnh đưa vào mỗi lượt. Estimator dùng số ký tự chia 4, làm tròn lên; đây không phải usage hay chi phí API thực.

Recall là trung bình tỷ lệ chuỗi kỳ vọng xuất hiện trong câu trả lời, không phân biệt hoa/thường. `Response quality` là proxy dựa trên recall và độ dài câu trả lời; không phải đánh giá độc lập bởi người hay LLM judge. Trong lần chạy này câu trả lời đủ ngắn nên quality bằng recall.

## Phân tích

Advanced nhớ được qua thread mới vì đọc hồ sơ từ đĩa. Test còn tạo lại đối tượng agent để kiểm tra tính bền vững. Baseline chỉ có lịch sử của thread hiện tại nên không biết facts đã cung cấp ở thread cũ.

Ở benchmark thường, Advanced xử lý 21.790 prompt tokens so với 15.084 của Baseline, tăng khoảng 44,5%. Lịch sử chưa đủ dài để compact kích hoạt, trong khi Advanced vẫn mang thêm hồ sơ vào prompt và trả lời recall dài hơn.

Ở stress test, Advanced compact 3 lần và xử lý 14.196 prompt tokens so với 22.398 của Baseline, giảm khoảng 36,6%. Lịch sử cũ được thay bằng summary có giới hạn độ dài, còn các message gần nhất được giữ nguyên. Lợi ích nằm ở lượng ngữ cảnh phải xử lý qua nhiều lượt; token đầu ra của Advanced vẫn cao hơn Baseline.

Hồ sơ tăng 257 bytes ở bộ thường và 218 bytes ở bộ stress. Mỗi field được cập nhật thay vì nối thêm mọi lần nhắc lại, nên facts lặp lại không làm hồ sơ tăng vô hạn. Tuy nhiên, thêm nhiều field vẫn làm tăng dữ liệu lưu và prompt cost.

Correction thay thế giá trị cũ trong cùng field. Bộ trích xuất bỏ câu hỏi, yêu cầu nhắc lại, câu giả định và một số nhiễu; ví dụ nơi đi họp hoặc câu đùa đổi nghề không được dùng làm hồ sơ hiện tại. Test kiểm tra cả correction nằm sau từ “nhưng”.

## Giới hạn

Offline dùng regex tiếng Việt và câu trả lời xác định, phù hợp kiểm chứng cơ chế memory của lab. Recall 100% chỉ áp dụng cho hai bộ dữ liệu mẫu đã chạy, không chứng minh khả năng hiểu mọi cách diễn đạt hoặc chất lượng tư vấn kỹ thuật.

Summary dùng facts trích xuất và đoạn văn rút gọn, có thể mất chi tiết tạm thời khi nén nhiều lần. Heuristic nơi ở ưu tiên tên riêng viết hoa và bỏ địa điểm như quán cà phê; cách diễn đạt khác có thể bị bỏ sót. Các facts ổn định được giữ riêng trong `User.md` để giảm tác động này.

Factory và đường gọi live đã có nhưng chưa chạy với API thật. LangGraph, tool middleware và LLM judge là phần mở rộng; kết quả trên không sử dụng chúng. Không có memory decay hoặc confidence score định lượng.
