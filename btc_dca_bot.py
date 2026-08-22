import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

# ==========================================
# 1. 설정값 (GitHub Secrets 연동)
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TARGET_ASSETS = {
    "QQQ": {"name": "QQQ (나스닥100)"},
    "BTC-USD": {"name": "비트코인 (BTC)"}
}

# ==========================================
# 2. FRED 거시경제 수집
# ==========================================
def get_fred_macro_data():
    macro_data = {'macro_score': 50, 'macro_status': "중립"}
    try:
        df_fed = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS", timeout=10).dropna()
        fed_rate = float(pd.to_numeric(df_fed.iloc[-1]['FEDFUNDS'], errors='coerce'))

        df_spread = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y", timeout=10).dropna()
        yield_spread = float(pd.to_numeric(df_spread.iloc[-1]['T10Y2Y'], errors='coerce'))

        df_real = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10", timeout=10).dropna()
        real_yield = float(pd.to_numeric(df_real.iloc[-1]['DFII10'], errors='coerce'))

        macro_score = 0
        if fed_rate <= 3.0: macro_score += 35
        elif fed_rate <= 4.5: macro_score += 20
        
        if yield_spread > 0.2: macro_score += 35
        elif yield_spread > 0: macro_score += 20
        
        if real_yield < 1.5: macro_score += 30
        elif real_yield < 2.0: macro_score += 15

        if macro_score >= 80: macro_status = "🟢 완화적 유동성"
        elif macro_score >= 50: macro_status = "🟡 중립/과도기"
        else: macro_status = "🔴 긴축 환경"

        return {'macro_score': macro_score, 'macro_status': macro_status}
    except Exception:
        return macro_data

# ==========================================
# 3. 보조 지표 및 정밀 추세 전환 연산
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_cmf(df, period=20):
    high_low_diff = (df['High'] - df['Low']).replace(0, np.nan)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_diff
    mfv = mfv.fillna(0) * df['Volume']
    vol_sum = df['Volume'].rolling(period).sum().replace(0, np.nan)
    return (mfv.rolling(period).sum() / vol_sum).fillna(0)

def precision_reversal_analysis(df):
    sub = df.iloc[-90:].copy()
    cur_price = sub['Close'].iloc[-1]

    # 1. 차트 구조 파괴 (CHoCH)
    sub['High_Pivot'] = (sub['High'].shift(1) > sub['High'].shift(2)) & (sub['High'].shift(1) > sub['High'])
    recent_pivots = sub[sub['High_Pivot']]['High'].iloc[-3:]
    last_swing_high = recent_pivots.max() if not recent_pivots.empty else sub['High'].iloc[-20:-5].max()
    choch_break = cur_price > last_swing_high

    # 2. 수급 폭발 (CMF & Volume)
    sub['CMF'] = calculate_cmf(sub, 20)
    cmf_inflow = sub['CMF'].iloc[-1] > 0.03
    
    sub['Is_Bull'] = sub['Close'] > sub['Open']
    bear_vol = sub[~sub['Is_Bull']]['Volume'].iloc[-20:]
    bear_vol_avg = bear_vol.mean() if not bear_vol.empty else 1.0
    recent_bull_vol = sub[sub['Is_Bull']]['Volume'].iloc[-3:].max() if not sub[sub['Is_Bull']]['Volume'].iloc[-3:].empty else 0
    vol_surge = (recent_bull_vol >= bear_vol_avg * 1.7) if bear_vol_avg > 0 else False
    vol_ratio = round(recent_bull_vol / bear_vol_avg, 2) if bear_vol_avg > 0 else 1.0

    # 3. 다중 다이버전스
    sub['RSI'] = calculate_rsi(sub['Close'], 14)
    ema12 = sub['Close'].ewm(span=12).mean()
    ema26 = sub['Close'].ewm(span=26).mean()
    sub['MACD_Hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()

    low_idx = [i for i in range(2, len(sub)-2) if sub['Low'].iloc[i] < sub['Low'].iloc[i-1] and sub['Low'].iloc[i] < sub['Low'].iloc[i-2]]
    rsi_div, macd_div = False, False
    if len(low_idx) >= 2:
        i1, i2 = low_idx[-2], low_idx[-1]
        if sub['Low'].iloc[i2] <= sub['Low'].iloc[i1]:
            if sub['RSI'].iloc[i2] > sub['RSI'].iloc[i1] + 2: rsi_div = True
            if sub['MACD_Hist'].iloc[i2] > sub['MACD_Hist'].iloc[i1]: macd_div = True

    # 4. 이평선 우상향 턴어라운드 (상승추세 시작 판단)
    ma5 = sub['Close'].rolling(5).mean()
    ma20 = sub['Close'].rolling(20).mean()
    ma20_slope = (ma20.iloc[-1] - ma20.iloc[-4]) / ma20.iloc[-4] * 100 if ma20.iloc[-4] != 0 else 0
    ma_rebound = (cur_price > ma5.iloc[-1]) and (ma5.iloc[-1] > ma20.iloc[-1]) and (ma20_slope > 0.0)

    return {
        'choch_break': choch_break,
        'cmf_inflow': cmf_inflow,
        'vol_surge': vol_surge,
        'vol_ratio': vol_ratio,
        'rsi_div': rsi_div,
        'macd_div': macd_div,
        'ma_rebound': ma_rebound,
        'ma20_slope': round(ma20_slope, 2)
    }

# ==========================================
# 4. 자산 사이클 판단 및 매수 제어
# ==========================================
def analyze_asset_cycle(symbol, asset_info, macro_score):
    try:
        df = yf.Ticker(symbol).history(period="6y", interval="1d", timeout=12)
        if df.empty or len(df) < 1000: return None
    except Exception: return None

    cur_price = float(df['Close'].iloc[-1])

    # 1. 전고점(ATH) 및 고점 대비 하락률(MDD) 계산
    ath_price = float(df['Close'].max())
    mdd_pct = round(((cur_price - ath_price) / ath_price) * 100, 2)

    # 2. 실제 주봉 200MA 계산
    df_weekly = df['Close'].resample('W-FRI').last().dropna()
    if len(df_weekly) < 200: return None
    w_ma200 = float(df_weekly.rolling(200).mean().iloc[-1])
    dist_w200_pct = round(((cur_price - w_ma200) / w_ma200) * 100, 2)

    # 3. 추세 전환 정밀 분석
    rev = precision_reversal_analysis(df)
    tech_score = 0
    if rev['choch_break']: tech_score += 30
    if rev['vol_surge']: tech_score += 25
    if rev['cmf_inflow']: tech_score += 15
    if rev['rsi_div'] or rev['macd_div']: tech_score += 20
    if rev['ma_rebound']: tech_score += 10
    final_score = int(tech_score * 0.8 + macro_score * 0.2)

    # -----------------------------------------------------------
    # [사이클 핵심 로직] 사용자 정의 기준 적용
    # -----------------------------------------------------------
    is_at_bottom_zone = dist_w200_pct <= 5.0 # 200주선 터치/하방 이탈 (+5% 이하)
    is_uptrend_started = rev['ma_rebound'] and rev['choch_break'] # 상승 추세 재개 여부

    if not is_at_bottom_zone:
        # 200주선 바닥에 오지 않은 구간
        cycle_state = "1단계: 전고점 후 하락 대기중 (관망)"
        action = "⚪ [매수 금지] 200주선 바닥 미도달 ➔ 관망 / 다음 하락장 대기"
    else:
        # 200주선 바닥권 도착
        if is_uptrend_started:
            cycle_state = "3단계: 200주선 탈출 및 상승 추세 시작 (매수 중단)"
            action = "🛑 [매수 종료 / 홀딩] 바닥 탈출 및 상승 추세 진입 ➔ 추가 매수 중단 후 관망"
        else:
            cycle_state = "2단계: 200주선 바닥 진입 (매수 집행 구간)"
            if final_score >= 70:
                action = f"⚡ [분할 매수 집행] 바닥권 점수 높음 ({final_score}점) ➔ 비상 준비금 적극 분할 매수"
            elif final_score >= 40:
                action = f"👀 [DCA/정찰 매수] 바닥권이나 반격 약함 ({final_score}점) ➔ DCA 방식을 통한 소액 분할 매수"
            else:
                action = f"⏳ [대기] 200주선 타격했으나 추가 하락 위험 ({final_score}점) ➔ 관망 후 반격 신호 대기"

    return {
        'name': asset_info['name'],
        'cur_price': cur_price,
        'ath_price': ath_price,
        'mdd_pct': mdd_pct,
        'w_ma200': w_ma200,
        'dist_w200_pct': dist_w200_pct,
        'is_at_bottom_zone': is_at_bottom_zone,
        'cycle_state': cycle_state,
        'final_score': final_score,
        'rev': rev,
        'action': action
    }

# ==========================================
# 5. 실행 및 리포팅
# ==========================================
def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    macro = get_fred_macro_data()

    report = f"""🤖 [QQQ & BTC 200주선 사이클 매수 리포트]
📅 기준시각: {now_str}
🌐 거시 유동성: {macro['macro_status']} ({macro['macro_score']}/100)
================================="""

    for symbol, info in TARGET_ASSETS.items():
        res = analyze_asset_cycle(symbol, info, macro['macro_score'])
        if res:
            report += f"""

📌 [{res['name']}]
• 현재가: ${res['cur_price']:,.2f}
• 전고점: ${res['ath_price']:,.2f} (고점 대비: {res['mdd_pct']}%)
• 200주선: ${res['w_ma200']:,.2f} (격차: {res['dist_w200_pct']}%)

🔄 현재 사이클 상태: 
   {res['cycle_state']}

👉 최종 행동 지침: 
   {res['action']}
---------------------------------"""

    print(report)
    
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": report}, timeout=10)
        except Exception as e:
            print(f"텔레그램 발송 실패: {e}")

if __name__ == "__main__":
    main()
