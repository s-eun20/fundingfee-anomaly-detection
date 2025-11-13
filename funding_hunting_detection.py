import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('MacOSX') # Mac용 GUI 백엔드 설정
import matplotlib.pyplot as plt
from datetime import timedelta

# =========================================
# 0. Mac 한글 폰트 설정
# =========================================
try:
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("AppleGothic 폰트가 없어 NanumGothic으로 대체합니다.")
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")

print("분석을 시작합니다...")

# =========================================
# 1~3. 데이터 로드 및 매칭 (기존과 동일)
# =========================================
FILE_PATH = "problem_data_final.xlsx"

try:
    df_trade = pd.read_excel(FILE_PATH, sheet_name="Trade")
    df_funding = pd.read_excel(FILE_PATH, sheet_name="Funding")
except FileNotFoundError:
    print(f"오류: {FILE_PATH} 파일을 찾을 수 없습니다.")
    exit()

df_trade['ts_dt'] = pd.to_datetime(df_trade['ts'])
df_funding['ts_dt'] = pd.to_datetime(df_funding['ts'])

funding_sorted = df_funding[['ts_dt']].sort_values('ts_dt').rename(columns={'ts_dt': 'funding_ts'})
FUNDING_WINDOW_SEC = 30

open_trades  = df_trade[df_trade['openclose'] == "OPEN"].copy()
close_trades = df_trade[df_trade['openclose'] == "CLOSE"].copy()
open_trades  = open_trades.rename(columns={'ts_dt': 'ts_open'})
close_trades = close_trades.rename(columns={'ts_dt': 'ts_close'})
open_trades  = open_trades.sort_values(['account_id', 'symbol', 'ts_open'])
close_trades = close_trades.sort_values(['account_id', 'symbol', 'ts_close'])

merged_list = []
for (acc, sym), g_open in open_trades.groupby(['account_id', 'symbol']):
    g_close = close_trades[
        (close_trades['account_id'] == acc) &
        (close_trades['symbol'] == sym)
    ].sort_values('ts_close')
    n = min(len(g_open), len(g_close))
    if n == 0:
        continue
    temp = pd.DataFrame({
        'account_id': acc,
        'symbol': sym,
        'ts_open': g_open['ts_open'].iloc[:n].values,
        'ts_close': g_close['ts_close'].iloc[:n].values
    })
    merged_list.append(temp)

if not merged_list:
    print("오류: OPEN/CLOSE가 매칭된 거래가 0건입니다.")
    exit()

# A. 총 매칭 거래 (분모)
merged_df = pd.concat(merged_list, ignore_index=True)

# =========================================
# 4. 펀딩 시점 '통과' 거래(hunter_trades) 찾기 (기존과 동일)
# =========================================
temp_df = pd.merge_asof(
    merged_df.sort_values('ts_open'),
    funding_sorted,
    left_on='ts_open',
    right_on='funding_ts',
    direction='forward'
)
straddle_trades = temp_df[temp_df['ts_close'] > temp_df['funding_ts']].copy()
straddle_trades['delta_open'] = (straddle_trades['funding_ts'] - straddle_trades['ts_open']).dt.total_seconds()
straddle_trades['delta_close'] = (straddle_trades['ts_close'] - straddle_trades['funding_ts']).dt.total_seconds()

is_just_before = straddle_trades['delta_open'] <= FUNDING_WINDOW_SEC
is_just_after = straddle_trades['delta_close'] <= FUNDING_WINDOW_SEC

# B. 펀딩 헌팅 거래 (분자)
hunter_trades = straddle_trades[is_just_before & is_just_after]

print(f"총 매칭 거래(A): {len(merged_df)}건")
print(f"펀딩 헌팅 거래(B): {len(hunter_trades)}건")

# =========================================
# 5. '헌팅 전문' 계정 분석 (수정됨)
# =========================================
# A. 계정별 총 매칭 거래 횟수
total_counts = merged_df.groupby('account_id').size().rename('total_trades')

# B. 계정별 펀딩 헌팅 거래 횟수
hunter_counts = hunter_trades.groupby('account_id').size().rename('hunter_trades')

# A와 B를 합치고, 헌팅 0건인 계정은 0으로 채움
df_analysis = pd.concat([total_counts, hunter_counts], axis=1).fillna(0)

# C. 헌팅 거래 비율 (%)
df_analysis['hunter_ratio'] = (df_analysis['hunter_trades'] / df_analysis['total_trades']) * 100
df_analysis['log_total_trades'] = np.log10(df_analysis['total_trades'] + 1e-9) # 0 방지

# --- [수정] 95% 분위수를 Cutoff로 사용 ---
# 95%의 계정이 이 값 '이하'에 속함
cutoff = df_analysis['hunter_ratio'].quantile(0.95)

# 만약 Cutoff가 0이면 (95% 이상이 0%라는 뜻), 상위 1%(Q0.99)로 기준 강화
if cutoff == 0:
    print("95% 분위수(Q0.95)가 0%입니다. 기준을 상위 1%(Q0.99)로 강화합니다.")
    cutoff = df_analysis['hunter_ratio'].quantile(0.99)

print(f"\n[통계적 Cutoff 설정]")
if cutoff == 0:
    print("경고: 99% 계정의 헌팅 비중이 0%입니다. '우연히 1번' 겹친 계정도 '의심'으로 탐지됩니다.")
    # 0%보다 큰 최소값을 임시 Cutoff로 사용할 수 있으나, 여기서는 0% 초과를 모두 잡음
    df_normal = df_analysis[df_analysis['hunter_ratio'] == 0]
    df_hunter = df_analysis[df_analysis['hunter_ratio'] > 0]
else:
    print(f"정상/의심 기준선 (상위 5% Cutoff): {cutoff:.2f}%")
    df_normal = df_analysis[df_analysis['hunter_ratio'] <= cutoff]
    df_hunter = df_analysis[df_analysis['hunter_ratio'] > cutoff]
# ----------------------------------------

print(f"\n[분석 결과]")
print(f"총 계정 수: {len(df_analysis)}")
print(f"  - 정상 계정 (Cutoff 이하): {len(df_normal)}")
print(f"  - 의심 계정 (Cutoff 초과): {len(df_hunter)}")

print("\n[상위 의심 계정 (전문 헌터)]")
print(df_hunter.sort_values('hunter_ratio', ascending=False).head(10))

# =========================================
# 6. 시각화 (Cutoff 기준 - 수정됨)
# =========================================
print("\n[시각화] '헌팅 전문 계정' 탐지 그래프를 생성합니다...")
plt.figure(figsize=(12, 7))

# 1. 정상 (회색) - Cutoff 이하
plt.scatter(
    df_normal['log_total_trades'],
    df_normal['hunter_ratio'],
    alpha=0.3,
    label=f'정상 계정 (하위 95%)',
    color='gray'
)

# 2. 의심 (빨간색) - Cutoff 초과 (상위 5%)
if not df_hunter.empty:
    plt.scatter(
        df_hunter['log_total_trades'],
        df_hunter['hunter_ratio'],
        alpha=0.8,
        label=f'의심 계정 (상위 5%)',
        color='red',
        s=40
    )
# ----------------------------------------

plt.axhline(100, color='red', linestyle=':', label='100% (전문 헌터)')
if cutoff > 0:
    plt.axhline(cutoff, color='orange', linestyle='--',
                label=f'정상/의심 기준선 ({cutoff:.2f}%)')

# X축 레이블
ticks = [0, 1, 2, 3] # log10(1), log10(10), log10(100), log10(1000)
labels = ['1', '10', '100', '1000']
plt.xticks(ticks, labels)

plt.title('총 거래 대비 펀딩 헌팅 거래 비율 분석')
plt.xlabel('계정별 총 매칭 거래 횟수')
plt.ylabel("'펀딩 시점 직전/직후' 거래 비중 (%)")
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.ylim(-5, 105)
plt.tight_layout()
plt.show()