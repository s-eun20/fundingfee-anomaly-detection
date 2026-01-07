import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

# =========================================
# 0. 데이터 로드 및 초기 설정
# =========================================
FILE_PATH = "problem_data_final.xlsx"
FUNDING_WINDOW_SEC = 30
FUNDING_WINDOW_P2 = timedelta(minutes=10) # 패턴 2 분석용 10분 윈도우

try:
    df_trade = pd.read_excel(FILE_PATH, sheet_name="Trade")
    df_funding = pd.read_excel(FILE_PATH, sheet_name="Funding")
except FileNotFoundError:
    print(f"오류: {FILE_PATH} 파일을 찾을 수 없습니다.")
    exit()

df_trade['ts_dt'] = pd.to_datetime(df_trade['ts'])
df_funding['ts_dt'] = pd.to_datetime(df_funding['ts'])

# 펀딩 시각 목록 (P1, P2 공통 사용)
funding_ts_unique = df_funding['ts_dt'].unique()
df_funding_unique = pd.DataFrame({'funding_ts': funding_ts_unique}).sort_values('funding_ts')

# 계정 목록 (최종 점수 계산의 기준)
all_accounts = pd.DataFrame({'account_id': df_trade['account_id'].unique()})

print("분석을 시작합니다...")

# -----------------------------------------
# A. 패턴 1 계산을 위한 OPEN/CLOSE 매칭
# -----------------------------------------
open_trades = df_trade[df_trade['openclose'] == "OPEN"].copy()
close_trades = df_trade[df_trade['openclose'] == "CLOSE"].copy()
open_trades = open_trades.rename(columns={'ts_dt': 'ts_open'})
close_trades = close_trades.rename(columns={'ts_dt': 'ts_close'})
open_trades = open_trades.sort_values(['account_id', 'symbol', 'ts_open'])
close_trades = close_trades.sort_values(['account_id', 'symbol', 'ts_close'])

merged_list = []
for (acc, sym), g_open in open_trades.groupby(['account_id', 'symbol']):
    g_close = close_trades[(close_trades['account_id'] == acc) & (close_trades['symbol'] == sym)].sort_values('ts_close')
    n = min(len(g_open), len(g_close))
    if n > 0:
        merged_list.append(pd.DataFrame({
            'account_id': acc,
            'ts_open': g_open['ts_open'].iloc[:n].values,
            'ts_close': g_close['ts_close'].iloc[:n].values
        }))

merged_df = pd.concat(merged_list, ignore_index=True)
if merged_df.empty:
    print("오류: FIFO 매칭 거래가 없습니다.")
    exit()

# =========================================
# 1. 패턴 1: '직전/직후' 거래 비중 (Hunter Ratio)
# =========================================
print("1. 패턴 1 (헌팅 비중) 계산 중...")

# 1-1. 헌팅 거래 건수 계산
temp_df = pd.merge_asof(
    merged_df.sort_values('ts_open'),
    df_funding_unique,
    left_on='ts_open',
    right_on='funding_ts',
    direction='forward'
)
straddle_trades = temp_df[temp_df['ts_close'] > temp_df['funding_ts']].copy()
straddle_trades['delta_open'] = (straddle_trades['funding_ts'] - straddle_trades['ts_open']).dt.total_seconds()
straddle_trades['delta_close'] = (straddle_trades['ts_close'] - straddle_trades['funding_ts']).dt.total_seconds()
hunter_trades = straddle_trades[(straddle_trades['delta_open'] <= FUNDING_WINDOW_SEC) & (straddle_trades['delta_close'] <= FUNDING_WINDOW_SEC)]

hunter_counts = hunter_trades.groupby('account_id').size().rename('P1_hunter_trades')
total_counts = merged_df.groupby('account_id').size().rename('P1_total_trades')

df_p1 = pd.concat([total_counts, hunter_counts], axis=1).fillna(0).reset_index()
df_p1['P1_score'] = (df_p1['P1_hunter_trades'] / df_p1['P1_total_trades']) * 100 # % 단위
df_p1 = df_p1.rename(columns={'P1_score': 'Hunter_Ratio'}).drop(columns=['P1_hunter_trades', 'P1_total_trades'])


# =========================================
# 2. 패턴 2: 펀딩 집중도 (Funding Focus %)
# =========================================
print("2. 패턴 2 (펀딩 집중도) 계산 중...")

# 2-1. 모든 거래에 대해 가장 가까운 펀딩 시간을 찾음 (±10분 윈도우)
df_trade_p2 = df_trade[['account_id', 'ts_dt', 'amount']].sort_values('ts_dt')
df_p2_temp = pd.merge_asof(
    df_trade_p2,
    df_funding_unique,
    left_on='ts_dt',
    right_on='funding_ts',
    direction='nearest'
)
df_p2_temp['delta'] = (df_p2_temp['ts_dt'] - df_p2_temp['funding_ts']).abs()

# 2-2. 윈도우 내 거래액 (A_window) 합계
df_window_trades = df_p2_temp[df_p2_temp['delta'] <= FUNDING_WINDOW_P2]
A_window = df_window_trades.groupby('account_id')['amount'].sum().rename('P2_A_window').reset_index()

# 2-3. 총 거래액 (A_total) 합계 (P3 계산에서 재사용)
A_total = df_trade.groupby('account_id')['amount'].sum().rename('P3_A_total').reset_index()

# 2-4. 비율 계산
df_p2 = pd.merge(A_window, A_total, on='account_id', how='right').fillna(0) # 헌팅 외 거래도 포함
df_p2['P2_score'] = np.where(df_p2['P3_A_total'] > 0, (df_p2['P2_A_window'] / df_p2['P3_A_total']) * 100, 0)
df_p2 = df_p2.rename(columns={'P2_score': 'Funding_Focus_Pct'}).drop(columns=['P2_A_window', 'P3_A_total'])


# =========================================
# 3. 패턴 3: 거래액 대비 펀딩 수익 (Funding Ratio)
# =========================================
print("3. 패턴 3 (펀딩 수익률) 계산 중...")

# 3-1. 총 거래액 (A_total) (P2에서 계산된 P3_A_total 재사용)
df_p3 = A_total.copy()

# 3-2. 순 펀딩 수익 (Net Funding Fee) 합계
net_funding_fee = df_funding.groupby('account_id')['funding_fee'].sum().rename('P3_net_funding_fee').reset_index()

# 3-3. 비율 계산
df_p3 = pd.merge(df_p3, net_funding_fee, on='account_id', how='left').fillna(0)
df_p3['P3_score'] = np.where(df_p3['P3_A_total'] > 0, df_p3['P3_net_funding_fee'] / df_p3['P3_A_total'], 0)
df_p3 = df_p3.rename(columns={'P3_score': 'Funding_Ratio'}).drop(columns=['P3_net_funding_fee', 'P3_A_total'])


# =========================================
# 4. 종합 의심 점수 산출
# =========================================
print("4. 종합 의심 점수 산출 중...")

# 4-1. 모든 패턴 결과를 병합 (account_id 기준)
df_score = all_accounts.copy()
df_score = pd.merge(df_score, df_p1, on='account_id', how='left').fillna(0)
df_score = pd.merge(df_score, df_p2, on='account_id', how='left').fillna(0)
df_score = pd.merge(df_score, df_p3, on='account_id', how='left').fillna(0)

# 4-2. 정규화 (0~1 스케일링) - 모든 지표는 높을수록 의심스러움
scaler = MinMaxScaler()
scores_to_normalize = ['Hunter_Ratio', 'Funding_Focus_Pct', 'Funding_Ratio']

# Scaling을 위해 DataFrame 복사 후 작업
df_scaled = df_score[scores_to_normalize].copy()

# 0으로 나뉘는 것을 방지하기 위해 최소값이 0이 아닌 경우에만 MinMaxScaler 적용
for col in scores_to_normalize:
    if df_scaled[col].min() != df_scaled[col].max():
        df_score[f'Score_{col}'] = scaler.fit_transform(df_scaled[[col]])
    else:
        df_score[f'Score_{col}'] = 0 # 모든 값이 동일하면 점수는 0

# 4-3. 최종 의심 점수 계산 (가중치 적용)
WEIGHT_P1 = 0.5
WEIGHT_P2 = 0.3
WEIGHT_P3 = 0.2

df_score['Suspicion_Score'] = (df_score['Score_Hunter_Ratio'] * WEIGHT_P1) + \
                              (df_score['Score_Funding_Focus_Pct'] * WEIGHT_P2) + \
                              (df_score['Score_Funding_Ratio'] * WEIGHT_P3)

# 4-4. 상위 10명 추출 및 결과 정리
df_top10 = df_score.sort_values('Suspicion_Score', ascending=False).head(10).copy()

# 최종 테이블 구성을 위한 세부 점수 계산
df_top10['P1_Score'] = (df_top10['Score_Hunter_Ratio'] * WEIGHT_P1).round(3)
df_top10['P2_Score'] = (df_top10['Score_Funding_Focus_Pct'] * WEIGHT_P2).round(3)
df_top10['P3_Score'] = (df_top10['Score_Funding_Ratio'] * WEIGHT_P3).round(3)

df_top10['Suspicion_Score'] = df_top10['Suspicion_Score'].round(3)

# 출력 테이블 정리
final_columns = ['account_id', 'Suspicion_Score', 'P1_Score', 'P2_Score', 'P3_Score']
df_output = df_top10[final_columns].reset_index(drop=True)
df_output.index = df_output.index + 1
df_output.index.name = '순위'

print("\n\n=============== 종합 의심 점수 Top 10 계정 ===============")
print(f"가중치: P1 (헌팅 비중) 50% | P2 (펀딩 집중도) 30% | P3 (펀딩 수익률) 20%")
print("----------------------------------------------------------")

# 최종 결과 테이블 출력
print(df_output.to_markdown(floatfmt=".3f"))
print("----------------------------------------------------------")

print("\n[세부 점수 해석]")
print("P1_Score: 헌팅 전문성 (최대 0.5)")
print("P2_Score: 자금 집중도 (최대 0.3)")
print("P3_Score: 통계적 수익 이상치 (최대 0.2)")