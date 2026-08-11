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
# 2. 보조지표 및 수집 함수
# ==========================================
def calculate_rsi(series, period=14):
    """Wilder 방식 표준 RSI 계산 (0 나누기 예외 처리 포함)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)

def get_fed_rate():
    """미국 기준금리(Effective Federal Funds Rate) 수집"""
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        df = pd.read_csv(url)
        df['FEDFUNDS'] = pd.to_numeric(df['FEDFUNDS'], errors='coerce')
        df = df.dropna()
        return float(df.iloc[-1]['FEDFUNDS'])
    except Exception as e:
        print(f"기준금리 수집 실패 (기본값 5.25% 사용): {e}")
        return 5.25

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

def get_qqq_and_qld_data():
    """QQQ 정밀 파라미터(5년 기준 ATH) 및 QLD 가격 수집"""
    try:
        qqq_ticker = yf.Ticker("QQQ")
        qld_ticker = yf.Ticker("QLD")

        # ATH 및 장기 이평선 계산을 위해 5년치 수집
        df_qqq = qqq_ticker.history(period="5y", interval="1d")
        df_qld = qld_ticker.history(period="5d", interval="1d")

        if df_qqq.empty:
            return None

        # 1) QQQ 이동평균선 및 표준편차
        df_qqq['MA20'] = df_qqq['Close'].rolling(20).mean()
        df_qqq['STD20'] = df_qqq['Close'].rolling(20).std()
        df_qqq['MA200'] = df_qqq['Close'].rolling(200).mean()
        df_qqq['RSI'] = calculate_rsi(df_qqq['Close'], 14)

        # 2) 시그마 밴드 (1σ, 2σ)
        df_qqq['Sigma1_Upper'] = df_qqq['MA20'] + (df_qqq['STD20'] * 1)
        df_qqq['Sigma1_Lower'] = df_qqq['MA20'] - (df_qqq['STD20'] * 1)
        df_qqq['Sigma2_Upper'] = df_qqq['MA20'] + (df_qqq['STD20'] * 2)
        df_qqq['Sigma2_Lower'] = df_qqq['MA20'] - (df_qqq['STD20'] * 2)

        # 3) MACD
        ema12 = df_qqq['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_qqq['Close'].ewm(span=26, adjust=False).mean()
        df_qqq['MACD'] = ema12 - ema26
        df_qqq['MACD_Signal'] = df_qqq['MACD'].ewm(span=9, adjust=False).mean()

        # 4) QQQ 주봉 데이터
        df_qqq_w = qqq_ticker.history(period="2y", interval="1wk")
        df_qqq_w['W_MA20'] = df_qqq_w['Close'].rolling(20).mean()

        # 최신 값 추출
        cur_qqq = df_qqq['Close'].iloc[-1]
        ath_qqq = df_qqq['Close'].max()  # 5년 내 역대 최고가 (진짜 ATH)
        qqq_mdd = round(((cur_qqq - ath_qqq) / ath_qqq) * 100, 2)

        ma20 = df_qqq['MA20'].iloc[-1]
        std20 = df_qqq['STD20'].iloc[-1]
        z_score = round((cur_qqq - ma20) / std20, 2) if std20 > 0 else 0

        s1_upper = df_qqq['Sigma1_Upper'].iloc[-1]
        s1_lower = df_qqq['Sigma1_Lower'].iloc[-1]
        s2_upper = df_qqq['Sigma2_Upper'].iloc[-1]
        s2_lower = df_qqq['Sigma2_Lower'].iloc[-1]

        rsi = df_qqq['RSI'].iloc[-1]
        ma200 = df_qqq['MA200'].iloc[-1]
        macd = df_qqq['MACD'].iloc[-1]
        macd_sig = df_qqq['MACD_Signal'].iloc[-1]

        w_ma20 = df_qqq_w['W_MA20'].iloc[-1] if not df_qqq_w.empty else 0
        w_trend_ok = cur_qqq > w_ma20

        # QLD 최신 가격
        cur_qld = df_qld['Close'].iloc[-1] if not df_qld.empty else 0

        # 변동성 예상 범위 (1σ 기준)
        daily_vol = (std20 / ma20) if ma20 > 0 else 0
        weekly_vol = daily_vol * np.sqrt(5)
        monthly_vol = daily_vol * np.sqrt(21)

        w_1s_lower = cur_qqq * (1 - weekly_vol)
        w_1s_upper = cur_qqq * (1 + weekly_vol)
        m_1s_lower = cur_qqq * (1 - monthly_vol)
        m_1s_upper = cur_qqq * (1 + monthly_vol)

        return {
            'cur_qqq': cur_qqq,
            'cur_qld': cur_qld,
            'qqq_mdd': qqq_mdd,
            'rsi': round(rsi, 1),
            'z_score': z_score,
            's1_upper': s1_upper,
            's1_lower': s1_lower,
            's2_upper': s2_upper,
            's2_lower': s2_lower,
            'w_1s_lower': w_1s_lower,
            'w_1s_upper': w_1s_upper,
            'm_1s_lower': m_1s_lower,
            'm_1s_upper': m_1s_upper,
            'ma200': ma200,
            'macd_bull': macd > macd_sig,
            'above_ma200': cur_qqq > ma200,
            'w_trend_ok': w_trend_ok
        }
    except Exception as e:
        print(f"QQQ/QLD 데이터 수집 실패: {e}")
        return None

# ==========================================
# 3. 100점 만점 가중치 평가 시스템
# ==========================================
def evaluate_100point_system(data, fed_rate):
    if not data:
        return {'score': 0, 'signal': "⚪ 데이터 없음", 'dca_action': "데이터 수집 실패", 'qld_ratio': "-", 'cash_ratio': "-"}

    score = 0
    mdd = data['qqq_mdd']
    rsi = data['rsi']
    cur_qqq = data['cur_qqq']

    # 1. 고점 대비 낙폭 (MDD) - 최대 35점
    if mdd <= -30: score += 35
    elif mdd <= -25: score += 30
    elif mdd <= -20: score += 24
    elif mdd <= -15: score += 16
    elif mdd <= -10: score += 8

    # 2. 시그마 이탈도 - 최대 25점
    if cur_qqq <= data['s2_lower']: score += 25
    elif cur_qqq <= data['s1_lower']: score += 15
    elif cur_qqq >= data['s2_upper']: score -= 10

    # 3. RSI 과매도 - 최대 20점
    if rsi <= 30: score += 20
    elif rsi <= 35: score += 15
    elif rsi <= 40: score += 10
    elif rsi <= 45: score += 5
    elif rsi >= 65: score -= 10

    # 4. 기술적 모멘텀 및 이평선 - 최대 10점
    if not data['above_ma200']: score += 5
    if data['macd_bull']: score += 5

    # 5. 미국 기준금리 환경 - 최대 10점
    if fed_rate <= 3.5: score += 10
    elif fed_rate <= 4.5: score += 8
    elif fed_rate <= 5.5: score += 5
    else: score += 2

    # 점수 범위 제한 (0 ~ 100점)
    score = max(0, min(100, score))

    # 위험관리 조건 (주봉 20선 이탈 & MDD -20% 이하)
    is_danger = (not data['w_trend_ok']) and (mdd <= -20)
    is_overbought = (rsi >= 65) or (cur_qqq >= data['s2_upper'])

    # 매수 실행 지침
    if is_danger:
        signal = "⚠️ 손절 / 리스크 관리"
        dca_action = "⚠️ 주봉 20선 이탈 폭락: 추가 매수 중단 및 현금 확보"
        qld_ratio, cash_ratio = "0% ~ 10%", "90% ~ 100%"
    elif score >= 100:
        signal = "🚨 100점 만점! (역대급 대바닥)"
        dca_action = "🚨 100점 달성: [3차 매수] 매수 준비금의 30% 집행!"
        qld_ratio, cash_ratio = "50%", "50%"
    elif score >= 90:
        signal = "🔥 90점대 (깊은 투매 구간)"
        dca_action = "🔥 90점 이상: [2차 매수] 매수 준비금의 20% 집행!"
        qld_ratio, cash_ratio = "40% ~ 45%", "55% ~ 60%"
    elif score >= 80:
        signal = "⚡ 80점대 (1차 바닥 진입)"
        dca_action = "⚡ 80점 이상: [1차 매수] 매수 준비금의 10% 집행!"
        qld_ratio, cash_ratio = "30% ~ 35%", "65% ~ 70%"
    elif score >= 50:
        signal = "🟢 적극 DCA 구간"
        dca_action = "🟢 적극 분할 매수 (기본 DCA 금액 + α 집행)"
        qld_ratio, cash_ratio = "20% ~ 30%", "70% ~ 80%"
    elif score >= 20:
        signal = "🟡 정속 DCA 구간"
        dca_action = "🟡 정속 매수 구간 (월별 기본 DCA 정량 매수)"
        qld_ratio, cash_ratio = "15% ~ 20%", "80% ~ 85%"
    else:
        if is_overbought:
            signal = "🚨 시장 과열 구간"
            dca_action = "🚨 과열 구간: 신규 매수 중단 및 일부 분할 익절 고려"
            qld_ratio, cash_ratio = "0% ~ 10%", "90% ~ 100%"
        else:
            signal = "⚪ 평상시 / 중립 구간"
            dca_action = "⚪ 관망 및 기본 DCA 유지 (추가 매수 없음)"
            qld_ratio, cash_ratio = "10% ~ 20%", "80% ~ 90%"

    return {
        'score': score,
        'signal': signal,
        'dca_action': dca_action,
        'qld_ratio': qld_ratio,
        'cash_ratio': cash_ratio
    }

def evaluate_btc(btc_usd, target_peak_usd=150000):
    if btc_usd <= 0:
        return {'upside_pct': 0, 'signal': "⚪ 데이터 없음", 'btc_ratio': "-", 'cash_ratio': "-"}
    
    upside_pct = round(((target_peak_usd - btc_usd) / btc_usd) * 100, 2)

    if upside_pct >= 150:
        signal, btc_ratio, cash_ratio = "🔥 적극 매수", "70% ~ 80%", "20% ~ 30%"
    elif upside_pct >= 80:
        signal, btc_ratio, cash_ratio = "🟢 매수", "50% ~ 65%", "35% ~ 50%"
    elif upside_pct >= 30:
        signal, btc_ratio, cash_ratio = "🟡 관망", "30% ~ 45%", "55% ~ 70%"
    else:
        signal, btc_ratio, cash_ratio = "🚨 익절 / 관망", "10% ~ 20%", "80% ~ 90%"

    return {
        'upside_pct': upside_pct,
        'signal': signal,
        'btc_ratio': btc_ratio,
        'cash_ratio': cash_ratio
    }

# ==========================================
# 4. 텔레그램 전송 및 메인 실행
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("텔레그램 토큰 미설정. 콘솔 출력:\n", message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("텔레그램 전송 성공!")
        else:
            print(f"텔레그램 전송 실패: {res.text}")
    except Exception as e:
        print(f"텔레그램 전송 오류: {e}")

def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 데이터 수집
    fed_rate = get_fed_rate()
    btc_krw, btc_usd, usdt_krw = get_btc_data()
    qqq_qld = get_qqq_and_qld_data()

    btc_eval = evaluate_btc(btc_usd, BTC_TARGET_PEAK_USD)

    if qqq_qld:
        q_eval = evaluate_100point_system(qqq_qld, fed_rate)
        macd_str = "🟢 상승 모멘텀" if qqq_qld['macd_bull'] else "🔴 하락 모멘텀"
        w_trend_str = "🟢 상방 유지" if qqq_qld['w_trend_ok'] else "🔴 20주선 이탈 (주의)"

        report = f"""🤖 [BTC & QLD 레버리지 정밀 분석 리포트]
📅 기준시각: {now_str}
🏛️ 미국 기준금리: {fed_rate}%

---------------------------------
🟠 비트코인 (BTC/USD)
• 현재 가격: ${btc_usd:,.2f} ({btc_krw:,.0f} 원)
• 목표 고점 대비 상방 여력: +{btc_eval['upside_pct']}%
• 투자 신호: {btc_eval['signal']}
• 목표 비중: BTC {btc_eval['btc_ratio']} / 현금 {btc_eval['cash_ratio']}

---------------------------------
📊 QQQ 분석 기반 QLD 매수 전략
• QQQ 현재가: ${qqq_qld['cur_qqq']:,.2f} (5년 ATH 대비 MDD: {qqq_qld['qqq_mdd']}%)
• QLD 현재가: ${qqq_qld['cur_qld']:,.2f} (2배 레버리지)

📈 QQQ 표준편차(시그마) & 변동성 범위
• 현재 위치: Z-Score {qqq_qld['z_score']}σ
• 일봉 1σ 범위: ${qqq_qld['s1_lower']:,.1f} ~ ${qqq_qld['s1_upper']:,.1f}
• 일봉 2σ 범위: ${qqq_qld['s2_lower']:,.1f} ~ ${qqq_qld['s2_upper']:,.1f}
• 주단위 예상 1σ: ${qqq_qld['w_1s_lower']:,.1f} ~ ${qqq_qld['w_1s_upper']:,.1f}
• 월단위 예상 1σ: ${qqq_qld['m_1s_lower']:,.1f} ~ ${qqq_qld['m_1s_upper']:,.1f}

🔍 주요 기술적 지표 (QQQ 기준)
• RSI (14): {qqq_qld['rsi']}
• 주봉 20선 추세: {w_trend_str}
• MACD 모멘텀: {macd_str}

🎯 [QLD 포트폴리오 가이드 - 100점 만점 System]
• 정밀 평가 점수: {q_eval['score']} / 100 점
• 투자 신호: {q_eval['signal']}
• 오늘의 매수 실행 지침:
  👉 {q_eval['dca_action']}
• 목표 비중: QLD {q_eval['qld_ratio']} / 현금 {q_eval['cash_ratio']}
"""
    else:
        report = f"⚠️ QQQ/QLD 데이터를 불러오지 못했습니다. ({now_str})"

    print(report)
    send_telegram_message(report)

if __name__ == "__main__":
    main()
