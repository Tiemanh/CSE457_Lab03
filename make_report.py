"""Build a six-page report from an existing experiment directory (reportlab)."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def build(result_dir, destination):
    base=Path(result_dir)
    summary=json.loads((base/'summary.json').read_text(encoding='utf-8'))
    synthetic=summary['synthetic']
    fonts=[(Path('C:/Windows/Fonts/arial.ttf'),Path('C:/Windows/Fonts/arialbd.ttf')),
           (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))]
    pair=next((p for p in fonts if all(x.exists() for x in p)),None)
    if pair is None: raise FileNotFoundError('Install Arial (Windows) or DejaVu Sans (Linux) for Vietnamese report fonts.')
    pdfmetrics.registerFont(TTFont('Body',str(pair[0])))
    pdfmetrics.registerFont(TTFont('BodyBold',str(pair[1])))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='VN',fontName='Body',fontSize=9.3,leading=13,spaceAfter=7,textColor=colors.HexColor('#213145')))
    styles.add(ParagraphStyle(name='VNHead',fontName='BodyBold',fontSize=17,leading=22,spaceAfter=12,textColor=colors.HexColor('#143c64')))
    styles.add(ParagraphStyle(name='VNSub',fontName='BodyBold',fontSize=11,leading=15,spaceBefore=8,spaceAfter=6,textColor=colors.HexColor('#146c87')))
    styles.add(ParagraphStyle(name='SmallVN',fontName='Body',fontSize=7.5,leading=10,spaceAfter=5))
    story=[]
    def p(s): story.append(Paragraph(s,styles['VN']))
    def h(s): story.append(Paragraph(s,styles['VNHead']))
    def sub(s): story.append(Paragraph(s,styles['VNSub']))
    def page(): story.append(PageBreak())
    def table(rows,widths=None):
        wrapped=[[Paragraph(escape(str(v)),styles['SmallVN']) for v in r] for r in rows]
        t=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeef4')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),
                              ('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#5294ad')),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7fa')])]))
        story.append(t); story.append(Spacer(1,8))
    def image(name,width=490):
        path=base/'figures'/name
        from reportlab.lib.utils import ImageReader
        w,h0=ImageReader(str(path)).getSize()
        story.append(Image(str(path),width=width,height=width*h0/w))
    def pair_images(left,right,width=240):
        from reportlab.lib.utils import ImageReader
        objects=[]
        for name in [left,right]:
            path=base/'figures'/name; w,h0=ImageReader(str(path)).getSize()
            objects.append(Image(str(path),width=width,height=width*h0/w))
        story.append(Table([objects],colWidths=[250,250]))
    def read(name):
        with (base/name).open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))

    h('LAB 03 | NHẬN DẠNG TIẾNG NÓI VỚI HMM')
    p('CSE457 · Xử lý âm thanh và tiếng nói · Báo cáo thực hành')
    p('<b>Phạm vi:</b> Nhận dạng 5 lệnh tiếng Việt bằng whole-word Gaussian HMM; so sánh DTW, phân tích Viterbi và mở rộng language model.')
    if synthetic:
        p('<b>TRẠNG THÁI DỮ LIỆU: ÂM THANH TỔNG HỢP.</b> Đây là báo cáo chạy thật của phần mềm trên WAV tạo bằng eSpeak NG, không phải thí nghiệm với bản ghi âm người thật. Kết quả không được diễn giải thành chất lượng ASR trên người nói thực tế.')
    else:
        p('<b>Dữ liệu do người dùng cung cấp.</b> Nguồn và cách thu cần được mô tả thêm theo bộ dữ liệu thực tế.')
    sub('1. Thiết kế và dữ liệu')
    table([['Thuộc tính','Thiết lập'],['Lệnh','mở, đóng, lên, xuống, dừng'],
           ['Train / test',f'{summary["train_files"]} / {summary["test_files"]} utterance'],
           ['Audio','Mono 16 kHz, PCM 16-bit; chèn 0.25 s silence hai đầu cho demo'],
           ['Nguồn demo','eSpeak NG voice vi; thay tốc độ và cao độ mỗi lần tổng hợp'],
           ['Tách dữ liệu','6 train + 2 test/lớp; không trùng waveform'],
           ['Validation nội bộ','4 train để fit + 2 train/lớp để validation'],
           ['Cấu hình chính','5-state / 39D / diagonal, ấn định trước khi xem test']], [125,375])
    p('Train và test demo dùng chung engine và voice. Dù từng WAV được tổng hợp riêng và khác tham số, chúng không độc lập theo người nói. Không coi điều chỉnh pitch là một người nói mới. Hash waveform kiểm tra trùng dữ liệu, nhưng không loại bỏ sự phụ thuộc do cùng engine.')
    sub('2. Tổ chức tái lập')
    p('Lab03.ipynb trình bày các bước; lab3.py chạy pipeline; results_demo chứa bảng CSV, hình, model NPZ, summary và lịch sử EM. PROVENANCE.json lưu nguồn và tham số từng WAV. Lexicon/corpus là minh họa tự xây, chưa phải bản giảng viên cung cấp.')
    p('E1/E2 và đường học được báo cáo mô tả, không dùng test chọn cấu hình. E3 full covariance bỏ theo điều kiện “nếu đủ data” vì tập rất nhỏ. Chưa có mã Lab 2 nên DTW được cài lại và công bố quy ước.')
    page()

    h('ĐẶC TRƯNG VÀ MÔ HÌNH')
    sub('3. MFCC động và CMVN')
    p('Pipeline: mono/resample → trim 30 dB → pre-emphasis 0.97 → Hamming 25 ms, hop 10 ms, FFT 512 → 26 mel filters → 13 MFCC. Ghép MFCC, delta và delta-delta thành T × 39; delta dùng hồi quy cửa sổ tối đa 9 frame. CMVN độc lập từng utterance nên không dùng thống kê test để fit train.')
    pair_images('features_mo.png','features_dong.png',220)
    p('Hình 1. MFCC, delta và delta-delta của hai lệnh tổng hợp. Các panel có thang màu riêng. Delta thể hiện vùng thay đổi nhanh, bổ sung thông tin lân cận mà emission HMM độc lập theo state chưa trực tiếp mô hình hóa.')
    sub('4. Cấu trúc HMM và cập nhật EM')
    p('λ = (A, B, π). A chỉ cho self-loop và next-state, π chỉ bắt đầu state 0. B là Gaussian diagonal. Khởi tạo self-loop 0.7, next-state 0.3; state cuối hấp thụ. Means/covariances khởi tạo theo các đoạn thời gian của train. Nối các utterance khi fit nhưng truyền lengths để giữ ranh giới file.')
    p('<b>γ(t,i)</b> là trách nhiệm của state i đối với frame t; <b>ξ(t,i,j)</b> là trách nhiệm của chuyển i→j. Baum–Welch dùng kỳ vọng Forward–Backward để cập nhật transition, mean và variance. π giữ cố định; sau fit kiểm tra cấm chuyển ngược/skip và tổng hàng bằng 1.')
    p('Diagonal có 39 tham số phương sai/state; full có 39×40/2 = 780. Với ít dữ liệu, diagonal ổn định hơn nhưng bỏ tương quan giữa các chiều. Floor phương sai 0.01 được áp sau M-step; ghi lại mọi lần giảm objective, không che warning hội tụ.')
    page()

    h('LIKELIHOOD VÀ VITERBI')
    sub('5. Hai bài toán khác nhau')
    p('Forward: α(t,j) = b(j,x_t) Σ_i α(t−1,i) A(i,j). Viterbi thay tổng bằng max và lưu backpointer. Forward trả log P(X); Viterbi trả log P(X,Q*) và đường state tốt nhất. Triển khai dùng logsumexp để tránh underflow, cho phép kết thúc ở mọi state như hmmlearn.')
    fv=read('forward_viterbi.csv')
    table([['Lệnh','Forward','Viterbi','Chênh lệch']]+[[r['label'],f'{float(r["forward"]):.3f}',f'{float(r["viterbi"]):.3f}',f'{float(r["gap"]):.5f}'] for r in fv],[100,135,135,130])
    pair_images('viterbi_mo.png','viterbi_dong.png')
    p('Hình 2. State path không giảm. Số frame ở một state × 10 ms là duration xấp xỉ. Self-loop cho phép kéo dài khi nói chậm; duration ngầm là hình học. State không mặc định là phoneme. DTW path nối frame với frame template, khác Viterbi nối frame với state model.')
    sub('6. Bảng likelihood của ba utterance')
    rows=read('likelihood_examples.csv')
    table([['File','mo','dong','len','xuong','dung','Chọn']]+[[r['file']]+[f'{float(r[w]):.0f}' for w in ['mo','dong','len','xuong','dung']]+[r['prediction']] for r in rows], [88,68,68,68,68,68,72])
    p('Mỗi hàng chọn likelihood lớn nhất. Chia mọi score trong cùng hàng cho T không đổi argmax. So sánh score/frame giữa nhiều utterance có độ dài khác nhau chỉ là chuẩn hóa thực dụng, chưa tạo xác suất posterior đã hiệu chỉnh.')
    p('DP viết riêng được đối chiếu với thư viện trên hai utterance. Kiểm thử nhỏ còn vét cạn tất cả state paths để xác minh phép tổng và phép max bằng kết quả độc lập.')
    page()

    h('THÍ NGHIỆM VÀ SO SÁNH DTW')
    sub('7. E1–E3: số state, số chiều và covariance')
    table([['State/chiều','Train','Validation','Test','Giảm objective']]+[[f'{r["states"]}/{r["dimensions"]}',f'{r["train_accuracy"]:.0%}',f'{r["validation_accuracy"]:.0%}',f'{r["test_accuracy"]:.0%}',str(r['objective_decreases'])] for r in summary['experiments']],[100,90,110,90,110])
    p('Các cấu hình đều đạt điểm cao trên bộ demo dễ và cùng nguồn voice; chưa đủ để kết luận 7 state tốt hơn 3 hoặc delta giúp một lớp cụ thể. Số lần giảm objective cho thấy cần xem lịch sử thay vì chỉ đọc monitor.converged. EM có thể gặp nghiệm xấu do khởi tạo, ít mẫu, quá nhiều tham số hoặc variance sụp. Chưa thấy bằng chứng underfit/overfit từ accuracy sạch ở bộ nhỏ này.')
    sub('8. E4: DTW và HMM')
    table([['Mô hình','Accuracy','Fit (s)','Score/trial (ms)']]+[[r['method'],f'{r["accuracy"]:.0%}',f'{r["train_seconds"]:.3f}',f'{r["test_ms_per_trial"]:.3f}'] for r in summary['comparison']],[125,100,120,155])
    pair_images('cm_5state_39dim.png','cm_dtw13.png',220)
    p('Hình 3. Confusion matrix sạch, 10 test utterance. DTW13 dùng MFCC tĩnh; DTW39 kiểm soát số chiều. Thời gian loại trích đặc trưng và JIT warm-up ở cả hai phương pháp. DTW không ước lượng tham số nên fit = 0, không có nghĩa chuẩn bị template không tốn thời gian.')
    learn=read('learning_curve.csv')
    p('Đường học (mẫu/lớp: HMM, DTW): '+ '; '.join(f'{r["samples_per_class"]}: {float(r["hmm_accuracy"]):.0%}, {float(r["dtw_accuracy"]):.0%}' for r in learn)+'. Các điểm này chưa xác định lượng dữ liệu tối thiểu cho giọng người thật.')
    page()

    h('ROBUSTNESS, LEXICON VÀ LANGUAGE MODEL')
    sub('9. E5: nhiễu trắng 20 dB')
    p(f'HMM: {summary["noise"]["hmm_accuracy"]:.0%}; DTW13: {summary["noise"]["dtw_accuracy"]:.0%}. Nhiễu được thêm vào waveform trước endpoint detection/MFCC. Công suất tham chiếu lấy trên vùng speech. Cùng seed và waveform cho hai phương pháp.')
    pair_images('cm_noise.png','cm_noise_dtw.png',190)
    cases=summary['disagreement_cases']
    if cases:
        c=cases[0]
        p(f'Ví dụ: {c["file"]} ({c["condition"]}), nhãn {c["truth"]}; HMM → {c["hmm"]}, DTW → {c["dtw"]}. Đây là ca DTW tốt hơn HMM trong lần chạy này. Nhiễu có thể làm giữ lại silence và làm lệch delta/CMVN; cần ablation endpoint cố định để phân lập nguyên nhân.')
    else:
        p('Không quan sát ca một mô hình đúng còn mô hình kia sai; không tạo ví dụ giả để khẳng định ưu thế.')
    sub('10. Lexicon và E6: re-ranking')
    p('Whole-word HMM không cần lexicon để nhận dạng. Khi dùng phone HMM, lexicon xác định chuỗi model cần nối để biểu diễn một từ. File minh họa dùng ký hiệu tự đặt và tone tags; phải thay bằng inventory của lớp nếu yêu cầu.')
    p('Corpus tự xây; unigram đếm token, bigram dùng add-0.5. Mẫu số là số chuyển tiếp đi ra từ context + 0.5V, không phải số unigram có cả từ cuối câu. Tổng xác suất trên 5 lệnh bằng 1; không mô hình EOS.')
    lm=read('lm_reranking.csv'); first=lm[0]
    p(f'Chọn {first["file"]}, điều kiện {first["condition"]}, theo acoustic margin nhỏ nhất. Thử λ = 0, 0.1, 0.3, 1.0 và context giả định. Không có previous-word ground truth nên không báo LM accuracy. Điểm là acoustic/frame + λ log P, không phải MAP nguyên bản.')
    toy=read('lm_controlled_example.csv')
    p('Ví dụ điểm âm học giả lập gần nhau (tách khỏi kết quả WAV), previous = mo: '+ '; '.join(f'λ={r["weight"]} → {r["prediction"]}' for r in toy)+'. Nếu từ đúng là len, LM có thể sửa lỗi; nếu từ đúng là xuong, cùng prior có thể làm sai. Trọng số quá lớn lấn át acoustic evidence.')
    page()

    h('PHÂN TÍCH SCORE VÀ KẾT LUẬN')
    image('score_distribution.png',360)
    p('Hình 4. Phân bố score/frame của đúng lớp và sai lớp trên test sạch. Đây là phân tích sau đánh giá, không dùng để đặt ngưỡng.')
    sub('11. Từ ngoài từ vựng và reject')
    p(f'Argmax luôn chọn một nhãn trong 5 lệnh dù đầu vào là từ khác. Ngưỡng đề xuất: {summary["reject"]["threshold"]:.3f} log-likelihood/frame, lấy percentile 5% score đúng lớp trên inner-validation; {summary["reject"]["clean_known_rejected"]} test known bị từ chối. Chưa có unknown test nên không báo false-accept rate. Sau refit full train, thang score có thể đổi; cần calibration riêng và có thể kết hợp margin top-1/top-2.')
    sub('12. Kết luận và giới hạn')
    p('Đã triển khai và chạy từ WAV → 39D → 5 whole-word HMM → likelihood/Viterbi → so sánh DTW, nhiễu, lexicon và LM. Dữ liệu tổng hợp cho phép kiểm chứng quy trình nhưng không thay bộ ghi âm người thật. Kết quả sạch quá dễ và giảm mạnh khi nhiễu là lý do chưa suy rộng sang ứng dụng thực tế.')
    p('Để nộp theo mô tả utterance/người: thu ít nhất 8 bản/lệnh, tách 6 train + 2 test, nghe và kiểm tra nhãn; dùng phone inventory/corpus của giảng viên. Nếu đánh giá speaker-independent, tách theo người nói. Điền tên/MSSV/lớp theo mẫu nộp; báo cáo này không tự gán thông tin cá nhân.')
    sub('13. Tái lập và nguồn')
    p('Cài requirements.txt; mở Lab03.ipynb và Run All hoặc chạy lab3.py. Kiểm thử: python -m unittest -v. Model dạng NPZ không dùng pickle. README hướng dẫn Git; dữ liệu thu âm riêng và môi trường .venv được bỏ qua bởi .gitignore.')
    p('Nguồn: đề Lab 3.pdf (CSE457, Đại học Thủy Lợi); hmmlearn 0.3.3 tutorial; librosa 0.11.0 MFCC/delta; eSpeak NG và espeakng-loader. Đường dẫn nguồn đầy đủ ở README. Kết quả số, phiên bản thư viện và giới hạn được lưu trong summary.json.')

    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont('Body',8); canvas.setFillColor(colors.HexColor('#607286'))
        label='LAB 03 | SYNTHETIC DEMO - KHÔNG PHẢI GHI ÂM NGƯỜI THẬT' if synthetic else 'LAB 03 | USER DATA'
        canvas.drawString(42,25,label); canvas.drawRightString(A4[0]-42,25,str(doc.page)); canvas.restoreState()
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42,
                          title='Lab 03 - HMM speech recognition - synthetic demonstration',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created',destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='results_demo')
    p.add_argument('--output',default='Bao_cao_Lab03.pdf')
    args=p.parse_args(); build(args.results,args.output)
