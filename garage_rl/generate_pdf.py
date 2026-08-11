import os
from fpdf import FPDF

class PDFReport5Team(FPDF):
    def header(self):
        self.set_font("Malgun", "B", 9.5)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "garage_rl Challenge Level 3 RL 에이전트 최종 성과 리포트 (5조)", border=0, align="R")
        self.ln(7)
        self.set_draw_color(180, 180, 180)
        self.line(10, 15, 200, 15)

    def footer(self):
        self.set_y(-12)
        self.set_font("Malgun", "", 8.5)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, "Challenge Level 3 Report | 5조 | Page 1 / 1", align="C")

def create_5team_report():
    pdf = PDFReport5Team()
    
    font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    font_bold_path = "C:\\Windows\\Fonts\\malgunbd.ttf"
    pdf.add_font("Malgun", "", font_path)
    pdf.add_font("Malgun", "B", font_bold_path)
    
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(10, 16, 10)
    pdf.add_page()
    
    # Title
    pdf.set_font("Malgun", "B", 16)
    pdf.set_text_color(20, 40, 75)
    pdf.cell(0, 9, "Challenge Level 3 강화학습 에이전트 결과 보고서", align="L", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Malgun", "", 9.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "작성일: 2026.08.10 | 팀명: 5조 | 제출 대상: Level 3 (인내도 30틱) | 모델: model/level3/ppo.zip", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    
    # 1. 성과 요약
    pdf.set_font("Malgun", "B", 11.5)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "1. 최종 성과 및 Baseline 대비 비교 (Pass 달성)", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Malgun", "B", 9)
    pdf.set_fill_color(225, 238, 250)
    pdf.set_draw_color(170, 195, 220)
    
    col_w = [47, 47, 47, 49]
    pdf.cell(col_w[0], 6.5, "평가 지표", border=1, fill=True, align="C")
    pdf.cell(col_w[1], 6.5, "Rule-based Baseline", border=1, fill=True, align="C")
    pdf.cell(col_w[2], 6.5, "PPO 최종 모델", border=1, fill=True, align="C")
    pdf.cell(col_w[3], 6.5, "개선 성과 및 판정", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Malgun", "", 9)
    pdf.set_text_color(20, 20, 20)
    data = [
        ("Score (낮을수록 우수)", "1490.8 점", "1080.1 점", "+27.5% 개선 (Pass ≤1300 달성)"),
        ("Total Steps / Serviced", "710.8 틱 / 84.4 대", "638.6 틱 / 91.2 대", "72.2틱 단축 / +6.8대 정비"),
        ("Removed (이탈 차량)", "15.60 대", "8.83 대", "43.4% 이탈 감소 (8.8대 달성)"),
    ]
    for row in data:
        pdf.cell(col_w[0], 6, row[0], border=1, align="C")
        pdf.cell(col_w[1], 6, row[1], border=1, align="C")
        pdf.cell(col_w[2], 6, row[2], border=1, align="C")
        pdf.cell(col_w[3], 6, row[3], border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 2. 관측 및 보상 설계
    pdf.set_font("Malgun", "B", 11.5)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "2. 핵심 관측 및 보상 파이프라인 설계", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Malgun", "B", 9.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, "[관측 설계] O2 확장 관측 (30차원)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Malgun", "", 9)
    pdf.multi_cell(0, 5.2, 
        "- 구성: 슬롯 점유(3d) + 정비소 가동(3d) + 진행률(3d) + 대기차량 3대 x 7개 속성(21d)\n"
        "- 적합도 플래그: a_fit = 1.0 - damage (A 저손상) | b_fit = 1.0 if year>=20 else 0 (B 올드카) | c_fit = 1.0 if size<=4 else 0 (C 소형차)\n"
        "- 특징: 정비소별 혜택 힌트를 관측에 명시하여 PPO 신경망의 조건 추론 부담을 크게 제거함."
    )
    pdf.ln(3)
    
    pdf.set_font("Malgun", "B", 9.5)
    pdf.cell(0, 6, "[보상 & 하이퍼파라미터 설계] R1 균형 보상 & ent_coef=0.0", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Malgun", "", 9)
    pdf.multi_cell(0, 5.2, 
        "- 보상 수식: reward = -0.1 (시간비용) + 1.0 (배치) - 2.0 (무효배치) + 5.0*finished - 20.0*expired + 10.0*done\n"
        "- 주요 파라미터: total_timesteps=500k | lr=0.0005 | ent_coef=0.0 | n_steps=2048 | batch_size=64 | gamma=0.99"
    )
    pdf.ln(5)
    
    # 3. 실험 과정 및 튜닝 결과
    pdf.set_font("Malgun", "B", 11.5)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "3. 하이퍼파라미터 및 관측/보상 튜닝 비교 (100k steps 탐색)", new_x="LMARGIN", new_y="NEXT")
    
    col_w3 = [40, 25, 45, 80]
    pdf.set_font("Malgun", "B", 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(col_w3[0], 6, "실험 구분", border=1, fill=True, align="C")
    pdf.cell(col_w3[1], 6, "차원 / 설정", border=1, fill=True, align="C")
    pdf.cell(col_w3[2], 6, "100k Score (이탈 수)", border=1, fill=True, align="C")
    pdf.cell(col_w3[3], 6, "분석 및 성과 비고", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Malgun", "", 8.5)
    exp_data = [
        ("관측 O1 (기본 관측)", "21d", "1400.1 점 (12.55대)", "기본 속성만 제공하여 조건 인식 속도 부족"),
        ("관측 O2 (적합도 포함)", "30d", "1326.9 점 (12.06대)", "최우수: 적합도 플래그 직접 제공으로 탐색 가속"),
        ("관측 O3 (압축 관측)", "18d", "1452.5 점 (14.83대)", "주요 변수 누락으로 이탈 수 증가"),
        ("lr=0.0005, ent=0.01", "30d / R1", "1295.3 점 (11.12대)", "탐색 노이즈로 30틱 이탈 완벽 차단 실패"),
        ("lr=0.0005, ent=0.0 (최종)", "30d / R1", "1181.1 점 (9.91대)", "최우수: 탐색 노이즈 제거하여 확정적 이탈 차단"),
    ]
    for row in exp_data:
        pdf.cell(col_w3[0], 5.5, row[0], border=1, align="C")
        pdf.cell(col_w3[1], 5.5, row[1], border=1, align="C")
        pdf.cell(col_w3[2], 5.5, row[2], border=1, align="C")
        pdf.cell(col_w3[3], 5.5, row[3], border=1, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 4. 종합 결론
    pdf.set_font("Malgun", "B", 11.5)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "4. 핵심 결과 분석 및 결론", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Malgun", "", 9)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 5.2, 
        "1) O2 관측 피처 엔지니어링: A(damage), B(year), C(size) 조건별 정비시간 차이를 관측 플래그(a/b/c_fit)로 제공해 조건 추론 부담을 크게 줄임.\n"
        "2) ent_coef=0.0 결정론적 정책: 인내도가 30틱으로 매우 짧아 무작위 탐색 노이즈가 곧바로 이탈(벌점 50)로 연결됨. "
        "ent_coef=0.0으로 확정적 배치를 밀어붙여 이탈 수를 15.60대에서 8.83대로 대폭 감소시킴.\n"
        "3) 최종 성과: 500k 학습 모델은 Score 1080.1 (Pass 판정)을 달성하여 5조의 목표를 성공적으로 이뤄냄."
    )
    
    output_pdf = "C:\\hong\\project-2\\garage_rl\\Challenge_Level3_Report.pdf"
    pdf.output(output_pdf)
    print(f"5-Team 1-Page PDF Report generated successfully: {output_pdf}")

if __name__ == "__main__":
    create_5team_report()
