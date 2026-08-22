import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================
# 1. 설정값 (GitHub Secrets 연동)
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TARGET_ASSETS = {
    "QQQ": {"name": "QQQ (나스닥 100 본주)", "period": "15y"},
    "BTC-USD": {"name": "비트코인 (BTC)", "period": "10y"}
}

# ==========================================
# 2. 로그 선형회귀 채널 & 3단계 분할 매수 연산 Engine
# ==========================================
def analyze_log_channel_system(symbol, asset_name, period):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", timeout=12)
        if df.empty or len(df) < 500: return None
    except Exception as e:
        print(f"[{symbol}] 데이터 수집 실패: {e}")
        return None

    df = df.dropna()
    df['Log_Close'] = np.log(df['Close'])
    df['Time_Index'] = np.arange(len(df))

    # 선형 회귀 연산 (기울기 및 절편)
    x = df['Time_Index'].values
    y = df['Log_Close'].values
    slope, intercept = np.polyfit(x, y, 1)

    # 잔차(Residuals) 기반 표준편차 연산
    log_fitted = slope * x + intercept
    residuals = y - log_fitted
    std_dev = np.std(residuals)

    # 채널 가격 $ 변환 (np.exp)
    upper_band = np.exp(log_fitted + (2.0 * std_dev))  # +2σ (과열 상단)
    center_band = np.exp(log_fitted)                   # 0σ  (적정 중앙)
    lower_band = np.exp(log_fitted - (2.0 * std_dev))  # -2σ (바닥 하단)

    cur_price = float(df['Close'].iloc[-1])
    cur_upper = float(upper_band[-1])
    cur_center = float(center_band[-1])
    cur_lower = float(lower_band[-1])

    # 전고점 대비 하락률 (MDD)
    ath_price = float(df['Close'].max())
    mdd_pct = round(((cur_price - ath_price) / ath_price) * 100, 2)

    # 채널 내 상대적 위치 (0% = 하단선, 50% = 중앙선, 100% = 상단선)
    min_log = log_fitted[-1] - (2.0 * std_dev)
    max_log = log_fitted[-1] + (2.0 * std_dev)
    cur_log = np.log(cur_price)
    
    channel_pos = round(((cur_log - min_log) / (max_log - min_log)) * 100, 1)

    # -----------------------------------------------------------
    # 3단계 분할 매수 및 익절 가이드 세부 판정
    # -----------------------------------------------------------
    if channel_pos <= 0.0:
        cycle_state = "🚨 [3차 필살기 구간] 채널 최하단선 이탈 (-2σ 이하)"
        action = "⚡ [3차 매수 집행] 역대급 바닥! QQQ/BTC 전용 비상금 남은 30% 전액 집행 (누적 100%)"
    elif channel_pos <= 5.0:
        cycle_state = "🚨 [2차 본동대 구간] 채널 Pos 5% 이하 (진성 바닥)"
        action = "⚡ [2차 매수 집행] 진성 바닥 밀착! QQQ/BTC 전용 비상금 40% 추가 집행 (누적 70%)"
    elif channel_pos <= 15.0:
        cycle_state = "🚨 [1차 정찰대 구간] 채널 Pos 15% 이하 진입"
        action = "⚡ [1차 매수 집행] 하단선 근처 접근! 모아둔 QQQ/BTC 전용 비상금의 30% 1차 집행"
    elif channel_pos >= 85.0:
        cycle_state = "🔥 [극단적 과열 구간] 로그 채널 상단선 도달 (+2σ)"
        action = "🛑 [전량 분할 익절] 역사적 고점 도달! 보유 물량 30%/30%/40% 나누어 현금화"
    elif cur_price > cur_center:
        cycle_state = "🟢 [상방 추세 진행중] 채널 중앙선 상회"
        action = "⚪ [관망 / 현금 적립] 중앙선 위 우상향 중. 매수 금지 및 월 100만 원 현금 축적"
    else:
        cycle_state = "🟡 [조정 진행중] 채널 중앙선 하회"
        action = "⏳ [관망 / 현금 적립] 조정 진행 중. 하단선(Pos 15% 이하) 진입 전까지 현금 축적"

    return {
        'name': asset_name,
        'cur_price': cur_price,
        'ath_price': ath_price,
        'mdd_pct': mdd_pct,
        'cur_upper': cur_upper,
        'cur_center': cur_center,
        'cur_lower': cur_lower,
        'channel_pos': channel_pos,
        'cycle_state': cycle_state,
        'action': action
    }

# ==========================================
# 3. 메인 실행 및 텔레그램 발송
# ==========================================
def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""📐 [QQQ & BTC 로그 채널 분할매수 리포트]
📅 검증 시각: {now_str}
================================="""

    for symbol, info in TARGET_ASSETS.items():
        res = analyze_log_channel_system(symbol, info['name'], info['period'])
        if res:
            report += f"""

📌 [{res['name']}]
• 현재가: ${res['cur_price']:,.2f} (전고점 대비 {res['mdd_pct']}%)
• 채널 상단 (+2σ) : ${res['cur_upper']:,.2f}
• 채널 중앙 ( 0σ) : ${res['cur_center']:,.2f}
• 채널 하단 (-2σ) : ${res['cur_lower']:,.2f}

📊 로그 채널 내 위치: {res['channel_pos']}%
  [하단 0% ───────────────── 100% 상단]

🔄 사이클 상태:
   {res['cycle_state']}

👉 자금 집행 가이드:
   {res['action']}
---------------------------------"""

    print(report)

    # 텔레그램 알림 전송
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": report},
                timeout=10
            )
            print("텔레그램 발송 성공!")
        except Exception as e:
            print(f"텔레그램 발송 실패: {e}")

if __name__ == "__main__":
    main()
