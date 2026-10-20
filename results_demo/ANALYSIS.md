# Kết quả thực nghiệm Lab 3

**Dữ liệu:** synthetic_espeak_ng
**Không phải đánh giá trên giọng người thật** nếu dùng dataset_demo. Tất cả số liệu dưới đây được tính từ lần chạy hiện tại.

## E1–E2: số trạng thái và đặc trưng

| States | Chiều | Train | Validation | Test | Chạm 80 vòng |
|---|---|---|---|---|---|
| 3 | 39 | 100% | 100% | 100% | 0 |
| 5 | 39 | 100% | 100% | 100% | 0 |
| 7 | 39 | 100% | 100% | 100% | 0 |
| 5 | 13 | 100% | 100% | 100% | 0 |

Cấu hình chính 5-state/39D được ấn định trước. Validation lấy 2 file cuối mỗi lớp từ train; test chỉ dùng báo cáo. Với 10 test utterance, một lỗi tương ứng 10 điểm phần trăm; không đủ để kết luận một cấu hình tối ưu phổ quát.
Nếu train cao hơn validation/test: có dấu hiệu overfit hoặc lệch phân bố. Nếu cả train lẫn validation thấp: xem lại đặc trưng, khởi tạo và khả năng underfit. Chạm giới hạn vòng lặp không đồng nghĩa hội tụ; xem final_gain và objective_decreases trong convergence.json.

## E3: covariance

Skipped: only 6 utterances/class; 780 covariance parameters/state at 39D vs 39 diagonal.

## E4: DTW và HMM

| Phương pháp | Accuracy | Fit (s) | Score/trial (ms) |
|---|---|---|---|
| HMM39 | 100% | 0.040 | 0.673 |
| DTW13 | 100% | 0.000 | 1.960 |
| DTW39 | 100% | 0.000 | 2.207 |

Thời gian score đã loại trích đặc trưng và JIT warm-up cho cả hai phương pháp. DTW chỉ lưu template nên thời gian ước lượng tham số bằng 0; đây không phải tổng latency end-to-end. DTW13 là baseline MFCC; DTW39 giúp kiểm soát khác biệt số chiều.
HMM nén mỗi lớp thành phân bố theo trạng thái; DTW đối sánh mọi template nên chi phí tăng theo số mẫu. Xem learning_curve.csv cho 1, 3, 6 mẫu/lớp.

Ví dụ quan sát: noise20dB, dung_01.wav; nhãn dung, HMM → xuong, DTW → dung. Khác biệt cho thấy mô hình phân bố và đối sánh template phản ứng khác nhau với chuỗi này. Tác động của duration, delta và nhiễu là giả thuyết cần thêm ablation, không suy ra nguyên nhân duy nhất từ một mẫu.

## E5: robustness

Nhiễu trắng 20 dB: HMM 20%; DTW 40%. Nhiễu được thêm vào waveform trước endpoint/MFCC. Cùng nhiễu và seed cho hai phương pháp. Đây là một loại mismatch, không đại diện người nói mới hoặc phòng thu khác.

## E6: LM và lexicon

lm_reranking.csv chọn utterance có acoustic margin nhỏ nhất; thử các context giả định và λ = 0, 0.1, 0.3, 1.0. Không có context thật nên không báo LM accuracy. Score/frame + λ log P là re-ranking có trọng số, không phải MAP nguyên bản nếu λ tùy chọn. lm_controlled_example.csv là điểm âm học giả lập riêng để phân tích cơ chế; nếu nhãn thật là len thì prior ủng hộ len giúp, nếu nhãn thật là xuong thì cùng prior có thể làm sai. Không tính ví dụ giả lập vào accuracy âm thanh.
Đếm mẫu số bằng số bigram đi ra từ context; không dùng tổng unigram gồm cả từ cuối câu. Vì không mô hình EOS, tổng xác suất trên đúng 5 lệnh bằng 1 cả với context chưa gặp.

## Viterbi, Forward và reject

forward_viterbi.csv đối chiếu DP viết riêng với thư viện trên hai utterance. Forward cộng mọi đường; Viterbi lấy đường tốt nhất. Duration ≈ số frame × 10 ms; state không có nhãn phone. Hai heatmap dùng chung tham số; cường độ và nhịp biến đổi khác nhau phản ánh các tín hiệu được tổng hợp khác nhau, không phải bằng chứng trực tiếp về khác biệt phone.
Ngưỡng reject đề xuất -49.657 log-likelihood/frame từ inner-validation. Có 0 test từ đã biết bị reject. Chưa có unknown test nên chưa đo FAR. Sau fit lại full train, cần hiệu chỉnh ngưỡng bằng tập calibration riêng vì thang score có thể đổi.

## Kết luận

Pipeline đã chạy đủ từ WAV đến đặc trưng, HMM, Viterbi, DTW và LM minh họa. Để kết luận về nhận dạng tiếng Việt trên người thật cần thu âm độc lập và chạy lại. Lexicon/corpus cần thay bằng bản giảng viên nếu được yêu cầu.