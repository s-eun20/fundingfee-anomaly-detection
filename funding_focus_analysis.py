import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 한글 폰트 설정 (Mac) ---
def setup_korean_font():
    try:
        plt.rc('font', family='AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

setup_korean_font()

# --- 데이터 로드 ---
excel_file = 'problem_data_final.xlsx'
window_minutes = 10  # ±10분 기준

df_trade = pd.read_excel(excel_file, sheet_name='Trade')
df_spec = pd.read_excel(excel_file, sheet_name='Spec')

# --- 펀딩 주기 병합 ---
df_spec_simple = df_spec[['symbol', 'funding_interval']].drop_duplicates(subset=['symbol'])
df_trade = pd.merge(df_trade, df_spec_simple, on='symbol', how='left')
df_trade = df_trade.dropna(subset=['funding_interval'])
df_trade['funding_interval'] = df_trade['funding_interval'].astype(int)

# --- 시간 컬럼 생성 ---
df_trade['ts_dt'] = pd.to_datetime(df_trade['ts'])
df_trade['hour'] = df_trade['ts_dt'].dt.hour
df_trade['minute'] = df_trade['ts_dt'].dt.minute

# --- 펀딩 시점 ±10분 윈도우 설정 ---
df_trade['is_funding_hour_block'] = df_trade['hour'] % df_trade['funding_interval'] == 0
df_trade['is_pre_funding_hour_block'] = (df_trade['hour'] + 1) % df_trade['funding_interval'] == 0

df_trade['in_window_10'] = (df_trade['is_funding_hour_block'] & (df_trade['minute'] < window_minutes)) | \
                           (df_trade['is_pre_funding_hour_block'] & (df_trade['minute'] >= (60 - window_minutes)))

# --- 계정별 거래액 집계 ---
total_amount = df_trade.groupby('account_id')['amount'].sum().reset_index(name='Amount_total')
window_amount = df_trade[df_trade['in_window_10']].groupby('account_id')['amount'].sum().reset_index(name='Amount_window')

# --- 거래 비율(%) 계산 ---
df_agg = pd.merge(total_amount, window_amount, on='account_id', how='left').fillna(0)
df_agg['FundingFocusPercent'] = np.where(
    df_agg['Amount_total'] > 0,
    (df_agg['Amount_window'] / df_agg['Amount_total']) * 100,
    0
)

# --- 95% 분위수 계산 (정상 상한선) ---
cutoff = df_agg['FundingFocusPercent'].quantile(0.95)
print(f"📊 펀딩 구간 내 거래 비율 95% 상한선: {cutoff:.2f}%")

# --- 시각화 ---
plt.figure(figsize=(10,6))
plt.hist(df_agg['FundingFocusPercent'], bins=50, range=(0,100),
         color='skyblue', edgecolor='black', alpha=0.8, label='계정별 펀딩 구간 거래 비율')
plt.axvline(cutoff, color='red', linestyle='--', linewidth=2, label=f'상위 5% 컷 ({cutoff:.2f}%)')

plt.yscale('log')
plt.title('펀딩 구간 내 거래 비율 분포 (Funding Focus %)')
plt.xlabel('펀딩 시점 ±10분 내 거래 비율 (%)')
plt.ylabel('계정 수 (로그 스케일)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('funding_focus_percent_hist.png', dpi=150)
plt.show()
