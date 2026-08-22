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
    "QQQ": {"name": "QQQ (나스닥100)", "ma200_days": 1000},
    "BTC-USD": {"name": "비트코인 (BTC)", "ma200_days": 1400}
}

# ==========================================
# 2. 정밀 거시경제(Macro) 수집 및 모멘텀 산출
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
        # 기준금리 가중치
        if fed_rate <= 2.5: macro_score += 35
        elif fed_rate <= 4.0: macro_score += 25
        elif fed_rate <= 5.0: macro_score += 10
        
        # 장단기 금리차 (역전 해제 후 0.2% 이상 양수 전환 시 유동성 폭발)
        if yield_spread > 0.3: macro_score += 35
        elif yield_spread > 0.0: macro_score += 20
        elif yield_spread > -0.5: macro_score += 10
        
        # 실질금리 (기술주/비트코인 밸류에이션 한계선 1.5%)
        if real_yield < 1.2: macro_score += 30
        elif real_yield < 1.8: macro_score += 20
        elif real_yield < 2.2: macro_score += 10

        if macro_score >= 80: macro_status = "🟢 강한 유동성 확장기 (초우호적)"
        elif macro_score >= 50: macro_status = "🟡 피벗/완화 과도기 (중립)"
        else: macro_status = "🔴 유동성 수축/고금리 압박 (주의)"

        return {
            'fed_rate': fed_rate,
            'yield_spread': yield_spread,
            'real_yield': real_yield,
            'macro_score': macro_score,
            'macro_status': macro_status
        }
    except Exception as e:
        print(f"FRED API 데이터 수집 예외: {e}")
        return macro_data

# ==========================================
# 3. 보조 지표 연산 모듈
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_cmf(df, period=20):
    """Chaikin Money Flow (기관 수급 매집 지표)"""
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
    mfv = mfv.fillna(0) * df['Volume']
    cmf = mfv.rolling(period).sum() / df['Volume'].rolling(period).sum().replace(0, np.nan)
    return cmf.fillna(0)

# ==========================================
# 4. 정밀 추세 전환(Reversal) & 수급 파동 분석
# ==========================================
def precision_reversal_analysis(df):
    """
    5대 정밀 추세전환 시그널 연산
    """
    sub = df.iloc[-90:].copy() # 최근 90일 파동 추적
    cur_price = sub['Close'].iloc[-1]

    # --- [1] 차트 구조 파괴 (Market Structure Break: CHoCH / Higher High) ---
    # 최근 30일 간의 프랙탈 Swing High를 종가로 깨고 올라섰는가
    sub['High_Pivot'] = (sub['High'] > sub['High'].shift(1)) & (sub['High'] > sub['High'].shift(-1))
    recent_pivots = sub[sub['High_Pivot']]['High'].iloc[-4:-1]
    last_swing_high = recent_pivots.max() if not recent_pivots.empty else sub['High'].iloc[-20:-5].max()
    choch_break = cur_price > last_swing_high

    # --- [2] 스마트머니 수급 폭발 (CMF & Volume Reversal) ---
    sub['CMF'] = calculate_cmf(sub, 20)
    cmf_inflow = sub['CMF'].iloc[-1] > 0.05 # 기관 자금 순유입 영역
    
    sub['Is_Bull'] = sub['Close'] > sub['Open']
    bear_vol_avg = sub[~sub['Is_Bull']]['Volume'].iloc[-20:].mean()
    recent_bull_vol = sub[sub['Is_Bull']]['Volume'].iloc[-3:].max() if not sub[sub['Is_Bull']]['Volume'].iloc[-3:].empty else 0
    vol_surge = (recent_bull_vol >= bear_vol_avg * 1.7) if bear_vol_avg > 0 else False
    vol_ratio = round(recent_bull_vol / bear_vol_avg, 2) if bear_vol_avg > 0 else 1.0

    # --- [3] 다중 모멘텀 다이버전스 (Multi-Divergence) ---
    sub['RSI'] = calculate_rsi(sub['Close'], 14)
    ema12 = sub['Close'].ewm(span=12).mean()
    ema26 = sub['Close'].ewm(span=26).mean()
    sub['MACD_Hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()

    # 국소 저점 2개 검증
    low_idx = [i for i in range(2, len(sub)-2) if sub['Low'].iloc[i] < sub['Low'].iloc[i-1] and sub['Low'].iloc[i] < sub['Low'].iloc[i+1]]
    rsi_div, macd_div = False, False
    if len(low_idx) >= 2:
        i1, i2 = low_idx[-2], low_idx[-1]
        if sub['Low'].iloc[i2] <= sub['Low'].iloc[i1]: # 주가는 낮아지거나 더블바텀
            if sub['RSI'].iloc[i2] > sub['RSI'].iloc[i1] + 2: rsi_div = True
            if sub['MACD_Hist'].iloc[i2] > sub['MACD_Hist'].iloc[i1]: macd_div = True

    # --- [4] 이평선 정렬 및 기울기 턴어라운드 (Slope Turn) ---
    ma5 = sub['Close'].rolling(5).mean()
    ma20 = sub['Close'].rolling(20).mean()
    ma20_slope = (ma20.iloc[-1] - ma20.iloc[-4]) / ma20.iloc[-4] * 100 # 20일선 기울기
    
    ma_rebound = (cur_price > ma5.iloc[-1]) and (ma5.iloc[-1] > ma20.iloc[-1]) and (ma20_slope > -0.1)

    # --- [5] 변동성 응축 후 돌파 (Volatility Compression Squeeze) ---
    std20 = sub['Close'].rolling(20).std()
    upper_band = ma20 + (std20 * 2)
    lower_band = ma20 - (std20 * 2)
    bandwidth = (upper_band - lower_band) / ma20
    is_squeezed = bandwidth.iloc[-1] < bandwidth.iloc[-30:].quantile(0.25)
    squeeze_breakout = is_squeezed and (cur_price > upper_band.iloc[-2])

    return {
        'choch_break': choch_break,
        'cmf_inflow': cmf_inflow,
        'vol_surge': vol_surge,
        'vol_ratio': vol_ratio,
        'rsi_div': rsi_div,
        'macd_div': macd_div,
        'ma_rebound': ma_rebound,
        'ma20_slope': round(ma20_slope, 2),
        'squeeze_breakout': squeeze_breakout
    }

# ==========================================
# 5. 시장 국면(Phase) 정밀 체계화
# ==========================================
def determine_detailed_phase(df, w_ma200):
    cur_price = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    
    ma20_prev = df['Close'].rolling(20).mean().iloc[-5]
    ma50_prev = df['Close'].rolling(50).mean().iloc[-5]
    
    # 기울기 계산
    slope20 = (ma20 - ma20_prev) / ma20_prev
    slope50 = (ma50 - ma50_prev) / ma50_prev

    if cur_price > ma20 and ma20 > ma50 and ma50 > ma200 and slope20 > 0:
        return "🟢 [정배열 상승 국면] 파동 탄력 상승 진행 중"
    elif cur_price > w_ma200 and (cur_price < ma20 or cur_price < ma50):
        if slope50 >= 0:
            return "🟡 [대세 상승 속 건전한 조정] 장기 추세 살아있는 눌림목"
        else:
            return "🟠 [추세 균열 / 깊은 조정] 중기 이평선 꺾임 (주의 구간)"
    elif cur_price < ma200 and slope20 < 0:
        return "🟠 [하락 진입각 확정] 200일선 하회 및 단기 하락 파동 시작"
    else:
        return "🔴 [하락장 진행 중] 바닥 탐색 및 반전 시그널 대기"

# ==========================================
# 6. 데이터 연동 및 가중치 집행 산출
# ==========================================
def fetch_stock_data(symbol, period="10y"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", timeout=12)
        if not df.empty and len(df) > 500: return df
    except Exception: pass
    return pd.DataFrame()

def analyze_asset_precision(symbol, asset_info, macro_score):
    df = fetch_stock_data(symbol, period="10y")
    if df.empty: return None

    cur_price = float(df['Close'].iloc[-1])
    ma200_days = asset_info['ma200_days']
    w_ma200 = float(df['Close'].rolling(ma200_days).mean().iloc[-1])
    dist_w200_pct = round(((cur_price - w_ma200) / w_ma200) * 100, 2)

    phase = determine_detailed_phase(df, w_ma200)
    rev = precision_reversal_analysis(df)

    # 정밀 기술 점수 (100점 만점 설계)
    tech_score = 0
    if rev['choch_break']: tech_score += 25     # 구조 파괴 (핵심)
    if rev['vol_surge']: tech_score += 20       # 수급 거래량 폭발
    if rev['cmf_inflow']: tech_score += 15      # 스마트머니 유입
    if rev['rsi_div'] or rev['macd_div']: tech_score += 20 # 다중 다이버전스
    if rev['ma_rebound']: tech_score += 10      # 이평선 우상향
    if rev['squeeze_breakout']: tech_score += 10# 변동성 돌파

    # 최종 종합 점수 (기술적 추세전환 80% + 거시 매크로 20%)
    final_score = int(tech_score * 0.8 + macro_score * 0.2)

    # 비상 준비금 집행 가이드 세분화
    if "하락장" in phase or dist_w200_pct <= 5:
        if final_score >= 85:
            action = "🚨🚨 [S등급 완벽 반전] 비상 준비금의 70% 집행 (강력 바닥 확증)"
        elif final_score >= 65:
            action = "⚡ [A등급 반격 시그널] 비상 준비금의 40% 분할 매수"
        elif final_score >= 45:
            action = "👀 [B등급 정찰대] 비상 준비금의 15% 제한 입각"
        else:
            action = "🛑 [매수 엄금] 바닥권 접근했으나 추세 전환 동력 부족 (전액 대기)"
    elif "조정" in phase:
        if final_score >= 60:
            action = "🛒 [눌림목 분할 매수] 대세 우상향 유지 중, 평시 DCA 1.3배 가속"
        else:
            action = "👀 [조정 관망] 추가 하락 가능성 존재, 관찰 유지"
    elif "하락 진입각" in phase or "깊은 조정" in phase:
        action = "⚠️ [리스크 관리] 신규 진입 자제 및 현금 비중 방어"
    else:
        if final_score >= 70: action = "🟢 [상승 모멘텀 순항] 정속 DCA 1.5배 집행"
        else: action = "⚪ [평시 유동성] 정속 DCA 집행"

    return {
        'name': asset_info['name'],
        'cur_price': cur_price,
        'w_ma200': w_ma200,
        'dist_w200_pct': dist_w200_pct,
        'phase': phase,
        'tech_score': tech_score,
        'final_score': final_score,
        'rev': rev,
        'action': action
    }

# ==========================================
# 7. 실행 및 텔레그램 리포팅
# ==========================================
def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    macro = get_fred_macro_data()

    report = f"""🤖 [QQQ & BTC 초정밀 추세전환(Reversal) 리포트]
📅 기준시각: {now_str}

🌐 [거시경제 사이클 (Macro v3.0)]
• 미 기준금리: {macro['fed_rate']}%
• 장단기금리차(10Y-2Y): {macro['yield_spread']}%
• 10년물 실질금리: {macro['real_yield']}%
👉 매크로 스코어: {macro['macro_status']} ({macro['macro_score']}/100점)
================================="""

    for symbol, info in TARGET_ASSETS.items():
        res = analyze_asset_precision(symbol, info, macro['macro_score'])
        if res:
            rev = res['rev']
            report += f"""

📌 [{res['name']}]
• 현재가: ${res['cur_price']:,.2f} (200주선 격차: {res['dist_w200_pct']}%)
🧭 국면 평가: {res['phase']}

🔬 정밀 상승 전환(Reversal) 지표:
  - 구조 파괴 (CHoCH): {"✅ 성공" if rev['choch_break'] else "❌ 미달"}
  - 수급 폭발 (거래량): {"✅ (" + str(rev['vol_ratio']) + "배)" if rev['vol_surge'] else "❌ 미달"}
  - 스마트머니 유입 (CMF): {"✅ 양수 유입" if rev['cmf_inflow'] else "❌ 음수"}
  - 다중 다이버전스: {"✅ 포착 (RSI/MACD)" if (rev['rsi_div'] or rev['macd_div']) else "❌ 없음"}
  - 이평선 정렬/기울기: {"✅ 우상향 (" + str(rev['ma20_slope']) + "%)" if rev['ma_rebound'] else "❌ 미달"}
  - 변동성 축소후 돌파: {"✅ 포착" if rev['squeeze_breakout'] else "❌ 미달"}

🏆 최종 종합 점수: {res['final_score']} / 100점 (기술점수: {res['tech_score']}/100)
👉 자금 집행 지침: {res['action']}
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
