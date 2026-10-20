# Lab 03 — Nhận dạng lệnh tiếng Việt bằng Gaussian HMM

**Bản chạy minh họa từ đầu đến cuối bằng âm thanh tổng hợp.** Chưa phải đánh giá trên bản ghi âm người thật. Bộ `dataset_demo` có 40 WAV được tổng hợp độc lập bằng eSpeak NG: 5 lệnh × (6 train + 2 test). Train/test dùng cùng bộ tổng hợp và voice, nên accuracy cao không chứng minh tổng quát hóa sang người nói thật.

## Chạy nhanh trong VS Code

Mở **cả thư mục dự án** này. Dùng Python 3.12 và terminal tại thư mục chứa `Lab03.ipynb`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python lab3.py --data dataset_demo --output results_demo
.venv\Scripts\python -m unittest -v
```

Mở `Lab03.ipynb`, chọn kernel `.venv`, chọn **Run All**. Notebook đã lưu sẵn kết quả và hình. Không cần tạo lại dữ liệu để chạy vì WAV nhỏ đã được đóng gói. Các file kết quả ở `results_demo/` được cập nhật khi chạy lại; thời gian đo có thể khác theo máy.

## Nội dung

| File/thư mục | Vai trò |
|---|---|
| `Lab03.ipynb` | Notebook trình bày A–H, thí nghiệm, hình và câu trả lời lý thuyết |
| `lab3.py` | Pipeline trích MFCC, huấn luyện, đánh giá, báo cáo số liệu |
| `dataset_demo/` | 30 train + 10 test WAV mono 16 kHz PCM16, dữ liệu tổng hợp |
| `dataset_demo/PROVENANCE.json` | Nguồn, voice, tốc độ, cao độ và SHA-256 từng WAV |
| `results_demo/` | CSV, confusion matrices, Viterbi plots, model NPZ và phân tích |
| `Bao_cao_Lab03.pdf` | Báo cáo 6 trang về lần chạy minh họa |
| `lexicon.txt`, `corpus.txt` | Minh họa tự xây, chưa phải bản chuẩn của giảng viên |
| `test_lab3.py` | Kiểm thử DP với vét cạn, HMM, đặc trưng, DTW, bigram, dữ liệu |
| `predict.py` | Nạp model NPZ và nhận dạng một WAV |
| `generate_demo.py` | Tái tạo bộ dữ liệu tổng hợp ngoại tuyến bằng eSpeak NG |
| `record_commands.py` | Công cụ thu âm tương tác khi người dùng chủ động chạy |
| `make_report.py` | Tạo lại báo cáo PDF 6 trang từ thư mục kết quả |

Tạo lại báo cáo sau khi chạy pipeline: `python make_report.py --results results_demo --output Bao_cao_Lab03.pdf`. Windows dùng Arial; Linux cần font DejaVu Sans để hiển thị tiếng Việt. Khi đổi sang dữ liệu thật, cũng phải rà soát và sửa các đoạn mô tả dữ liệu/thí nghiệm trong báo cáo, không chỉ đổi các con số.

`requirements-tested.txt` ghi phiên bản các gói chính đã dùng trong lần kiểm thử bàn giao. `verification.json` ghi phương thức chạy notebook và kiểm tra model đã lưu.

Ví dụ dùng model đã lưu:

```powershell
.venv\Scripts\python predict.py dataset_demo/test/mo/mo_01.wav
```

Tạo một bộ demo mới (không ghi đè WAV đã có):

```powershell
.venv\Scripts\python generate_demo.py --output dataset_demo_new
```

## Cấu hình và chống rò rỉ dữ liệu

- Pre-emphasis 0.97; Hamming 25 ms, hop 10 ms, FFT 512, 26 mel filters.
- 13 MFCC + delta + delta-delta = 39 chiều; CMVN độc lập từng utterance. Đây là phép xử lý tại inference, không dùng thống kê test cho train.
- HMM chính cố định 5 state, diagonal covariance, self-loop/next-state, state đầu cố định; Baum–Welch 80 vòng tối đa. Khởi tạo trung bình/phương sai bằng các đoạn thời gian của train.
- Variance floor 0.01 sau M-step để ổn định; mọi lần giảm objective được ghi, không coi `monitor.converged` là bảo đảm nghiệm tốt.
- Chỉ dùng 4 train/lớp để fit và 2 train/lớp để validation trong thí nghiệm nội bộ. Model chính fit lại trên 6 train/lớp; cấu hình chính ấn định trước test. So sánh test E1/E2 là mô tả, không chọn tham số theo test.
- Phát hiện WAV trùng theo mẫu âm thanh đã giải mã; không chỉ so tên file.
- DTW13 dùng toàn bộ train templates, cùng split; thêm DTW39 để đối chiếu cùng số chiều. Thời gian score không tính trích đặc trưng và JIT compilation.
- Lexicon là inventory minh họa tự đặt. Tone tags là metadata ký hiệu, không tuyên bố là phone inventory chuẩn.
- Bigram dùng **số chuyển tiếp đi ra từ context** ở mẫu số; cộng add-0.5; không mô hình EOS. Context trong re-ranking là giả định, không dùng để báo accuracy LM.
- E3 full covariance được bỏ theo điều kiện “nếu đủ data”: 39D full cần 780 tham số covariance/state, diag chỉ 39. Tập nhỏ chưa thích hợp cho so sánh đáng tin cậy.

## Thay bằng dữ liệu người thật

```text
dataset/
  train/{mo,dong,len,xuong,dung}/*.wav  # ít nhất 6 file mỗi lớp
  test/{mo,dong,len,xuong,dung}/*.wav   # ít nhất 2 file mỗi lớp
```

Mỗi lần nói là bản ghi riêng. Không chia một file thành train và test, không dùng bản tăng cường của cùng một bản gốc ở hai split. Nếu đánh giá người nói mới, tách toàn bộ người nói giữa train/test; split mặc định của công cụ ghi âm dưới đây chỉ là speaker-dependent.

```powershell
.venv\Scripts\python -m pip install sounddevice
.venv\Scripts\python record_commands.py --speaker sv01 --root dataset
.venv\Scripts\python lab3.py --data dataset --output results_real
```

Công cụ chỉ thu sau mỗi lần bạn nhấn Enter. Hãy nói đúng một lệnh, chừa khoảng 0.3 s im lặng đầu/cuối. Nghe lại bản ghi trước khi dùng. Khi chuyển sang dữ liệu thật, sửa `DATA` và `OUTPUT` trong notebook; báo cáo PDF demo có sẵn **không tự biến thành báo cáo dữ liệu thật**.

## Những phần cần xác nhận trước khi nộp

1. Giảng viên có chấp nhận dữ liệu tổng hợp hay bắt buộc bản ghi người thật? Đề mô tả utterance/người; bản demo chưa đáp ứng dữ liệu người thật.
2. Thay `lexicon.txt` bằng phone inventory do giảng viên cung cấp nếu có; corpus hiện tại cũng là tự xây.
3. Chưa có mã Lab 2 nên DTW được cài lại; nếu lớp yêu cầu đúng baseline Lab 2, thay hoặc đối chiếu quy ước khoảng cách.
4. Thêm tên, MSSV, lớp vào notebook/báo cáo theo mẫu nộp của bạn. Không có thông tin cá nhân được tự điền.

## Đưa lên Git

Dự án được đăng tại [Tiemanh/CSE457_Lab03](https://github.com/Tiemanh/CSE457_Lab03). Sau khi clone và sửa bài, chạy tại thư mục dự án:

```powershell
git add .
git commit -m "Update Lab 3 experiments"
git push origin main
```

`.gitignore` bỏ qua môi trường, cache và dữ liệu thu âm cá nhân; bộ demo nhỏ và model NPZ ở `results_demo/models` được giữ để tái lập. Không đưa `.venv` lên Git.

## Nguồn

- Đề `Lab 3.pdf` do người dùng cung cấp, CSE457, Trường Đại học Thủy Lợi.
- [hmmlearn 0.3.3 tutorial](https://hmmlearn.readthedocs.io/en/0.3.3/tutorial.html): HMM trái-sang-phải và tham số cố định.
- [librosa MFCC](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html) và [delta](https://librosa.org/doc/0.11.0/generated/librosa.feature.delta.html).
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng), [espeakng-loader](https://github.com/thewh1teagle/espeakng-loader): tổng hợp ngoại tuyến; không đóng gói mã nhị phân thư viện trong repo này.

Các kết quả demo chỉ dùng học tập và kiểm chứng phần mềm. Mã nguồn thư viện ngoài được cài bằng pip, không sao chép vào dự án.
