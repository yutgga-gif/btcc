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

# 비트코인 사이클 목표 고점 ($)
BTC_TARGET_PEAK_USD = 150000 

# ==========================================
# 2. 보조지표 계산 함수
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. 데이터 수집 함수
# ==========================================
def get_btc_data():
    """업비트에서 BTC/KRW 및 환율 데이터 조회"""
    try:
        url = "https://api.upbit.com/v1/ticker?markets=KRW-BTC,KRW-USDT"
        res = requests.get(url, timeout=10).json()
        btc_krw, usdt_krw = 0, 1350.0
        for item in res:
            if item['market'] == 'KRW-BTC':
                btc_krw = item['trade_price']
            elif item['market'] == 'KRW-USDT':
                usdt_krw = item['trade_price']
        btc_usd = btc_krw / usdt_krw if usdt_krw > 0 else 0
        return btc_krw, btc_usd, usdt_krw
    except Exception as e:
        print(f"BTC 데이터 조회 실패: {e}")
        return 0, 0, 1350.0

def get_qld_precision_data():
    """QLD 및 지표 수집"""
    try:
        qld_ticker = yf.Ticker("QLD")
        
        # 1) QLD 일봉 데이터
        df_qld = qld_ticker.history(period="1y", interval="1d")
        if df_qld.empty:
            return None
        
        df_qld['MA50'] = df_qld['Close'].rolling(50).mean()
        df_qld['MA200'] = df_qld['Close'].rolling(200).mean()
        df_qld['RSI'] = calculate_rsi(df_qld['Close'], 14)
        
        # 볼린저 밴드 (20일, 2σ)
        df_qld['BB_Mid'] = df_qld['Close'].rolling(20).mean()
        std20 = df_qld['Close'].rolling(20).std()
        df_qld['BB_Upper'] = df_qld['BB_Mid'] + (std20 * 2)
        df_qld['BB_Lower'] = df_qld['BB_Mid'] - (std20 * 2)
        
        # MACD
        ema12 = df_qld['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_qld['Close'].ewm(span=26, adjust=False).mean()
        df_qld['MACD'] = ema12 - ema26
        df_qld['MACD_Signal'] = df_qld['MACD'].ewm(span=9, adjust=False).mean()

        # 2) QLD 주봉 데이터
        df_qld_w = qld_ticker.history(period="2y", interval="1wk")
        df_qld_w['W_MA20'] = df_qld_w['Close'].rolling(20).mean()

        # 최신 데이터 추출
        cur_price = df_qld['Close'].iloc[-1]
        ath_price = df_qld['Close'].max()
        dd_pct = round(((cur_price - ath_price) / ath_price) * 100, 2)
        
        ma200 = df_qld['MA200'].iloc[-1]
        rsi = df_qld['RSI'].iloc[-1]
        bb_lower = df_qld['BB_Lower'].iloc[-1]
        bb_upper = df_qld['BB_Upper'].iloc[-1]
        macd = df_qld['MACD'].iloc[-1]
        macd_sig = df_qld['MACD_Signal'].iloc[-1]
        
        w_ma20 = df_qld_w['W_MA20'].iloc[-1] if not df_qld_w.empty else 0
        w_trend_ok = cur_price > w_ma20

        return {
            'price': cur_price,
            'ath': ath_price,
            'drawdown': dd_pct,
            'rsi': round(rsi, 1),
            'ma200': ma200,
            'bb_lower': bb_lower,
            'bb_upper': bb_upper,
            'macd_bull': macd > macd_sig,
            'above_ma200': cur_price > ma200,
            'w_trend_ok': w_trend_ok
        }
    except Exception as e:
        print(f"QLD 데이터 수집 실패: {e}")
        return None

# ==========================================
# 4. 평가 및 정밀 스코어링 로직
# ==========================================
def evaluate_btc(btc_usd, target_peak_usd=150000):
    if btc_usd <= 0:
        return {'upside_pct': 0, 'signal': "⚪ 관망", 'btc_ratio': "-", 'cash_ratio': "-"}
    
    upside_pct = round(((target_peak_usd - btc_usd) / btc_usd) * 100, 2)

    if upside_pct >= 150:
        signal, btc_ratio, cash_ratio = "🔥 적극 매수", "70% ~ 80%", "20% ~ 30%"
    elif upside_pct >= 80:
        signal, btc_ratio, cash_ratio = "🟢 매수", "50% ~ 65%", "35% ~ 50%"
    elif upside_pct >= 30:
        signal, btc_ratio, cash_ratio = "🟡 관망", "30% ~ 45%", "55% ~ 70%"
    elif upside_pct > 0:
        signal, btc_ratio, cash_ratio = "🚨 익절", "10% ~ 20%", "80% ~ 90%"
    else:
        signal, btc_ratio, cash_ratio = "⚠️ 손절 / 전량 익절", "0% ~ 10%", "90% ~ 100%"

    return {
        'upside_pct': upside_pct,
        'signal': signal,
        'btc_ratio': btc_ratio,
        'cash_ratio': cash_ratio
    }

def evaluate_qld_precision(data):
    if not data:
        return {'signal': "⚪ 관망", 'score': 0, 'qld_ratio': "-", 'cash_ratio': "-", 'dca_action': "데이터 없음"}

    score = 0
    dd = data['drawdown']
    rsi = data['rsi']
    price = data['price']

    # 1. 고점 대비 낙폭(MDD)
    if dd <= -30: score += 4
    elif dd <= -20: score += 3
    elif dd <= -10: score += 1

    # 2. RSI
    if rsi <= 30: score += 3
    elif rsi <= 40: score += 1
    elif rsi >= 65: score -= 2

    # 3. 볼린저 밴드
    if price <= data['bb_lower']: score += 2
    elif price >= data['bb_upper']: score -= 1

    # 4. 추세
    if data['above_ma200']: score += 1
    if data['macd_bull']: score += 1

    # 위험관리 (주봉 20선 이탈 + MDD -20% 이하)
    is_danger = (not data['w_trend_ok']) and (dd <= -20)

    # 8, 9, 10점 세부 분할 매수 실행 지침
    if score >= 10:
        dca_action = "🚨 10점 달성: [3차 매수] 매수 준비금의 30% 집행 (최대 대바닥 구간)"
    elif score == 9:
        dca_action = "🔥 9점 달성: [2차 매수] 매수 준비금의 20% 집행 (깊은 투매 구간)"
    elif score == 8:
        dca_action = "⚡ 8점 달성: [1차 매수] 매수 준비금의 10% 집행 (1차 바닥 진입)"
    elif score >= 4:
        dca_action = "🟢 정속 분할 매수 구간 (월별 기본 DCA 금액 집행)"
    elif score >= 1:
        dca_action = "🟡 관망 구간 (매수 보류 또는 최소 금액만 집행)"
    else:
        dca_action = "🚨 과열 구간 (신규 매수 중단 및 일부 분할 익절 고려)"

    if is_danger:
        signal = "⚠️ 손절 / 리스크 관리"
        qld_ratio, cash_ratio = "0% ~ 10%", "90% ~ 100%"
        dca_action = "⚠️ 주봉 20선 이탈 위험: 추가 매수 중단 및 현금 확보"
    elif score >= 8:
        signal = "🔥 적극 매수 (역사적 대바닥)"
        qld_ratio, cash_ratio = "40% ~ 50%", "50% ~ 60%"
    elif score >= 4:
        signal = "🟢 매수 (정속 분할 진입)"
        qld_ratio, cash_ratio = "25% ~ 35%", "65% ~ 75%"
    elif score >= 1:
        signal = "🟡 관망 (소량 DCA)"
        qld_ratio, cash_ratio = "10% ~ 20%", "80% ~ 90%"
    else:
        signal = "🚨 익절 / 비중 축소"
        qld_ratio, cash_ratio = "0% ~ 10%", "90% ~ 100%"

    return {
        'signal': signal,
        'score': score,
        'qld_ratio': qld_ratio,
        'cash_ratio': cash_ratio,
        'dca_action': dca_action
    }

# ==========================================
# 5. 텔레그램 전송 및 메인 실행
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("텔레그램 토큰 또는 Chat ID가 설정되지 않았습니다. 콘솔 메시지:")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("텔레그램 메시지 전송 성공!")
        else:
            print(f"텔레그램 전송 실패: {res.text}")
    except Exception as e:
        print(f"텔레그램 전송 오류: {e}")

def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 수집 및 평가
    btc_krw, btc_usd, usdt_krw = get_btc_data()
    qld = get_qld_precision_data()

    btc_eval = evaluate_btc(btc_usd, BTC_TARGET_PEAK_USD)
    qld_eval = evaluate_qld_precision(qld)

    macd_str = "🟢 상승 모멘텀" if qld and qld['macd_bull'] else "🔴 하락 모멘텀"
    w_trend_str = "🟢 상방 유지" if qld and qld['w_trend_ok'] else "🔴 20주선 이탈 (주의)"

    report = f"""🤖 [BTC & QLD 레버리지 정밀 DCA 리포트]
📅 기준시각: {now_str}

---------------------------------
🟠 비트코인 (BTC/USD)
• 현재 가격: ${btc_usd:,.2f} ({btc_krw:,.0f} 원)
• 목표 고점: ${BTC_TARGET_PEAK_USD:,.0f} (상방 여력: +{btc_eval['upside_pct']}%)
• [포트폴리오 가이드]
  - 투자 신호: {btc_eval['signal']}
  - 목표 BTC 비중: {btc_eval['btc_ratio']}
  - 목표 현금 비중: {btc_eval['cash_ratio']}

---------------------------------
⚡ QLD (ProShares Ultra QQQ - 2배 레버리지)
• 현재 가격: ${qld['price']:,.2f}
• 고점 대비 낙폭(MDD): {qld['drawdown']}%

📊 QLD 정밀 기술적 지표
• 일봉 RSI (14): {qld['rsi']} (30이하 과매도 / 65이상 과열)
• 주봉 20선 추세: {w_trend_str}
• MACD 모멘텀: {macd_str}
• 200일 이동평균선: ${qld['ma200']:,.2f}
• 볼린저 밴드 하단: ${qld['bb_lower']:,.2f}

🎯 [QLD 포트폴리오 가이드]
• 투자 신호: {qld_eval['signal']}
• 정밀 점수: {qld_eval['score']} / 10 점
• **오늘의 매수 실행 지침**:
  👉 `{qld_eval['dca_action']}`
• 목표 QLD 비중: {qld_eval['qld_ratio']}
• 목표 현금/안정자산 비중: {qld_eval['cash_ratio']}
"""

    print(report)
    send_telegram_message(report)

if __name__ == "__main__":
    main()
