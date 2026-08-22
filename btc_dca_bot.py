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
# 2. FRED 거시경제 데이터 수집
# ==========================================
def get_fred_macro_data():
    macro_data = {'fed_rate': 5.25, 'yield_spread': 0.1, 'real_yield': 1.8, 'macro_score': 50, 'macro_status': "중립"}
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

        if macro_score >= 80: macro_status = "🟢 완화적 유동성 환경"
        elif macro_score >= 50: macro_status = "🟡 과도기 / 중립"
        else: macro_status = "🔴 긴축 및 고금리 긴장"

        return {
            'fed_rate': fed_rate,
            'yield_spread': yield_spread,
            'real_yield': real_yield,
            'macro_score': macro_score,
            'macro_status': macro_status
        }
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
    cmf = mfv.rolling(period).sum() / vol_sum
    return cmf.fillna(0)

def precision_reversal_analysis(df):
    sub = df.iloc[-90:].copy()
    cur_price = sub['Close'].iloc[-1]

    # 1. 차트 구조 파괴 (CHoCH) - 미래 데이터 참조(shift(-1)) 오류 수정
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

    # 4. 이평선 턴어라운드 (5/20일선)
    ma5 = sub['Close'].rolling(5).mean()
    ma20 = sub['Close'].rolling(20).mean()
    ma20_slope = (ma20.iloc[-1] - ma20.iloc[-4]) / ma20.iloc[-4] * 100 if ma20.iloc[-4] != 0 else 0
    ma_rebound = (cur_price > ma5.iloc[-1]) and (ma5.iloc[-1] > ma20.iloc[-1]) and (ma20_slope > -0.1)

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
# 4. 자산 분석 및 주봉(200주선) 연산
# ==========================================
def fetch_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6y", interval="1d", timeout=12) # 200주 연산을 위해 최소 5년 이상 수집
        if not df.empty and len(df) > 1000: return df
    except Exception: pass
    return pd.DataFrame()

def analyze_asset_strict(symbol, asset_info, macro_score):
    df = fetch_stock_data(symbol)
    if df.empty: return None

    cur_price = float(df['Close'].iloc[-1])

    # -----------------------------------------------------------
    # [수정] 일봉 기반 200주선 오차 제거 -> 실제 주봉(Weekly) 200MA 계산
    # -----------------------------------------------------------
    df_weekly = df['Close'].resample('W-FRI').last().dropna()
    if len(df_weekly) < 200: return None
    
    w_ma200 = float(df_weekly.rolling(200).mean().iloc[-1])
    dist_w200_pct = round(((cur_price - w_ma200) / w_ma200) * 100, 2)

    # -----------------------------------------------------------
    # [GATE] 200주선 근접(+5% 이하) 또는 하방 이탈(-) 판별
    # -----------------------------------------------------------
    is_at_bottom_zone = dist_w200_pct <= 5.0 

    rev = precision_reversal_analysis(df)

    tech_score = 0
    if rev['choch_break']: tech_score += 30
    if rev['vol_surge']: tech_score += 25
    if rev['cmf_inflow']: tech_score += 15
    if rev['rsi_div'] or rev['macd_div']: tech_score += 20
    if rev['ma_rebound']: tech_score += 10

    final_score = int(tech_score * 0.8 + macro_score * 0.2)

    # -----------------------------------------------------------
    # 의사결정 코멘트 제어
    # -----------------------------------------------------------
    if is_at_bottom_zone:
        if dist_w200_pct < 0:
            zone_desc = f"🔴 [200주선 하방 이탈] 깊은 바닥 구간 (격차: {dist_w200_pct}%)"
        else:
            zone_desc = f"🟡 [200주선 터치/근접] 주요 바닥 매수 타겟권 (격차: {dist_w200_pct}%)"

        if final_score >= 80:
            action = "🚨🚨 [S급 반전 확증] 200주선 바닥 + 강력 추세전환 시그널! 비상 준비금 60~70% 적극 투입"
        elif final_score >= 60:
            action = "⚡ [A급 1차 반격] 200주선 바닥권 + 수급/구조 파괴 확인. 비상 준비금 30~40% 분할 매수"
        elif final_score >= 40:
            action = "👀 [B급 정찰대] 200주선 바닥권 접근 중이나 추세전환 신호 부족. 비상 준비금 10%만 정찰 투입"
        else:
            action = "🛑 [매수 금지 / 하락세 지속] 200주선 바닥권이나 상승 반전 신호가 전혀 없음 ➔ 전액 현금 대기"
    else:
        zone_desc = f"🟢 [200주선 상회 레벨] 평시 추세 구간 (격차: +{dist_w200_pct}%)"
        action = "⚪ [비상 준비금 집행 대상 아님] 200주선 바닥권 미도달 ➔ 관망 또는 정속 DCA 유지"

    return {
        'name': asset_info['name'],
        'cur_price': cur_price,
        'w_ma200': w_ma200,
        'dist_w200_pct': dist_w200_pct,
        'is_at_bottom_zone': is_at_bottom_zone,
        'zone_desc': zone_desc,
        'tech_score': tech_score,
        'final_score': final_score,
        'rev': rev,
        'action': action
    }

# ==========================================
# 5. 실행 및 텔레그램 리포팅
# ==========================================
def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    macro = get_fred_macro_data()

    report = f"""🤖 [QQQ & BTC 200주선 바닥 타겟팅 추세전환 리포트]
📅 기준시각: {now_str}
🌐 매크로 환경: {macro['macro_status']} (점수: {macro['macro_score']}/100)
================================="""

    for symbol, info in TARGET_ASSETS.items():
        res = analyze_asset_strict(symbol, info, macro['macro_score'])
        if res:
            rev = res['rev']
            report += f"""

📌 [{res['name']}]
• 현재가: ${res['cur_price']:,.2f} | 200주선: ${res['w_ma200']:,.2f}
📍 바닥권 위치 평가: {res['zone_desc']}"""

            if res['is_at_bottom_zone']:
                report += f"""
🔬 추세 전환(Reversal) 정밀 지표:
  - 구조 파괴 (CHoCH): {"✅ 성공" if rev['choch_break'] else "❌ 미달"}
  - 수급 폭발 (거래량): {"✅ (" + str(rev['vol_ratio']) + "배)" if rev['vol_surge'] else "❌ 미달"}
  - 스마트머니 (CMF): {"✅ 유입" if rev['cmf_inflow'] else "❌ 음수"}
  - 다중 다이버전스: {"✅ 포착" if (rev['rsi_div'] or rev['macd_div']) else "❌ 없음"}
  - 이평선 우상향 정렬: {"✅ 성공" if rev['ma_rebound'] else "❌ 미달"}
🏆 상승 전환 종합 점수: {res['final_score']} / 100점"""
            
            report += f"""
👉 최종 자금 집행 가이드: 
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
