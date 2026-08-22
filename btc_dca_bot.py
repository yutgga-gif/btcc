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
    "QQQ": {"name": "QQQ (나스닥100)", "ma200_days": 1000},       # 주 5일 거래 (200주 = 1000일)
    "BTC-USD": {"name": "비트코인 (BTC)", "ma200_days": 1400}     # 주 7일 거래 (200주 = 1400일)
}

# ==========================================
# 2. 거시경제(FRED API) 데이터 수집 엔진
# ==========================================
def get_fred_macro_data():
    macro_data = {'fed_rate': 5.25, 'yield_spread': 0.1, 'real_yield': 1.8, 'macro_score': 0, 'macro_status': "보통"}
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

        if macro_score >= 80: macro_status = "🟢 기술주/암호화폐 극호재 (유동성 확장기)"
        elif macro_score >= 50: macro_status = "🟡 중립 / 피벗 전환기"
        else: macro_status = "🔴 긴축 및 고금리 압박 구간"

        return {
            'fed_rate': fed_rate,
            'yield_spread': yield_spread,
            'real_yield': real_yield,
            'macro_score': macro_score,
            'macro_status': macro_status
        }
    except Exception as e:
        print(f"거시경제 데이터 수집 오류: {e}")
        return macro_data

# ==========================================
# 3. 기술적 추세 전환 분석 엔진
# ==========================================
def fetch_stock_data(symbol, period="10y"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", timeout=12)
        if not df.empty and len(df) > 500: return df
    except Exception: pass
    return pd.DataFrame()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(100)

def analyze_reversal_signals(df):
    if len(df) < 100: return {}
    sub = df.iloc[-60:].copy()

    # 1. 차트 구조 파괴 (Higher High)
    prev_lower_high = sub['High'].iloc[-20:-5].max()
    cur_price = sub['Close'].iloc[-1]
    structure_break = cur_price > prev_lower_high

    # 2. 거래량 수급 전환 (매수 폭발)
    sub['Is_Bull'] = sub['Close'] > sub['Open']
    bear_vol_avg = sub[~sub['Is_Bull']]['Volume'].iloc[-20:].mean()
    recent_bull_max_vol = sub[sub['Is_Bull']]['Volume'].iloc[-3:].max() if not sub[sub['Is_Bull']]['Volume'].iloc[-3:].empty else 0
    volume_reversal = (recent_bull_max_vol >= bear_vol_avg * 1.8) if bear_vol_avg > 0 else False
    vol_ratio = round(recent_bull_max_vol / bear_vol_avg, 2) if bear_vol_avg > 0 else 1.0

    # 3. 보조지표 다이버전스 (RSI / MACD)
    sub['RSI'] = calculate_rsi(sub['Close'], 14)
    sub['MACD'] = sub['Close'].ewm(span=12).mean() - sub['Close'].ewm(span=26).mean()
    sub['MACD_Hist'] = sub['MACD'] - sub['MACD'].ewm(span=9).mean()

    lows = [i for i in range(2, len(sub) - 2) if sub['Low'].iloc[i] < sub['Low'].iloc[i-1] and sub['Low'].iloc[i] < sub['Low'].iloc[i+1]]
    rsi_div, macd_div = False, False
    if len(lows) >= 2:
        idx1, idx2 = lows[-2], lows[-1]
        price_falling = sub['Low'].iloc[idx2] <= sub['Low'].iloc[idx1]
        rsi_div = price_falling and (sub['RSI'].iloc[idx2] > sub['RSI'].iloc[idx1])
        macd_div = price_falling and (sub['MACD_Hist'].iloc[idx2] > sub['MACD_Hist'].iloc[idx1])

    # 4. 이평선 우상향 전환 (5/20일선)
    ma5 = sub['Close'].rolling(5).mean()
    ma20 = sub['Close'].rolling(20).mean()
    ma_rebound = (cur_price > ma5.iloc[-1]) and (ma5.iloc[-1] > ma20.iloc[-1]) and (ma20.iloc[-1] > ma20.iloc[-2])

    return {
        'structure_break': structure_break,
        'volume_reversal': volume_reversal,
        'vol_ratio': vol_ratio,
        'rsi_div': rsi_div,
        'macd_div': macd_div,
        'ma_rebound': ma_rebound
    }

def analyze_asset(symbol, asset_info, macro_score):
    df = fetch_stock_data(symbol, period="10y")
    if df.empty: return None

    cur_price = float(df['Close'].iloc[-1])
    ma200_days = asset_info['ma200_days']
    w_ma200 = float(df['Close'].rolling(ma200_days).mean().iloc[-1])
    dist_w200_pct = round(((cur_price - w_ma200) / w_ma200) * 100, 2)

    reversal = analyze_reversal_signals(df)

    tech_score = 0
    if reversal.get('structure_break'): tech_score += 25
    if reversal.get('volume_reversal'): tech_score += 25
    if reversal.get('rsi_div') or reversal.get('macd_div'): tech_score += 15
    if reversal.get('ma_rebound'): tech_score += 15

    total_score = int(tech_score * 0.8 + macro_score * 0.2)

    # 매수 가이드 산출
    if dist_w200_pct <= 5:
        if total_score >= 75: action = "🚨🚨 [바닥 확증 + 추세전환] 비상 준비금 60~70% 적극 투입"
        elif total_score >= 50: action = "⚡ [1차 반격 신호] 비상 준비금 30~40% 분할 매수"
        elif total_score >= 25: action = "👀 [정찰대 진입] 비상 준비금 10% 제한 매수"
        else: action = "🛑 [매수 금지] 200주선 바닥권이나 추세 전환 신호 없음"
    else:
        if total_score >= 70: action = "🟢 [상승 추세 순항] 기본 DCA 1.5배 가속 집행"
        else: action = "⚪ 평시 정속 DCA 집행"

    return {
        'name': asset_info['name'],
        'cur_price': cur_price,
        'w_ma200': w_ma200,
        'dist_w200_pct': dist_w200_pct,
        'tech_score': tech_score,
        'total_score': total_score,
        'reversal': reversal,
        'action': action
    }

# ==========================================
# 4. 실행 및 텔레그램 발송
# ==========================================
def send_telegram(msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": msg}, timeout=10)
        except Exception as e:
            print(f"텔레그램 발송 실패: {e}")

def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    macro = get_fred_macro_data()

    report = f"""🤖 [QQQ & 비트코인 매크로 + 추세전환 리포트]
📅 기준시각: {now_str}

🌐 [거시경제 사이클 (Macro)]
• 기준금리: {macro['fed_rate']}% | 장단기금리차: {macro['yield_spread']}% | 실질금리: {macro['real_yield']}%
👉 매크로 평가: {macro['macro_status']} ({macro['macro_score']}/100점)
=================================
"""

    for symbol, info in TARGET_ASSETS.items():
        res = analyze_asset(symbol, info, macro['macro_score'])
        if res:
            rev = res['reversal']
            report += f"""
📌 [{res['name']}]
• 현재가: ${res['cur_price']:,.2f} (200주선 격차: {res['dist_w200_pct']}%)
• 추세전환 5대 지표 검증:
  - 구조 파괴(HH): {"✅" if rev['structure_break'] else "❌"} | 수급 폭발: {"✅ ("+str(rev['vol_ratio'])+"배)" if rev['volume_reversal'] else "❌"}
  - 다이버전스: {"✅" if (rev['rsi_div'] or rev['macd_div']) else "❌"} | 이평선 정렬: {"✅" if rev['ma_rebound'] else "❌"}
• 종합 평가 점수: {res['total_score']} / 100점 (기술점수: {res['tech_score']}/80)
👉 실행 지침: {res['action']}
---------------------------------"""

    print(report)
    send_telegram(report)

if __name__ == "__main__":
    main()
