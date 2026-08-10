import os
import requests
import json
from datetime import datetime

# ==========================================
# 1. 설정값 (Environment Variables)
# ==========================================
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "your_access_key")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "your_secret_key")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_bot_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_chat_id")

# 이번 사이클 예상 고점 (기본값: $150,000)
# 원화 기준으로 환산하여 상방 여력을 계산하거나, USD 원/달러 환율을 곱해 계산합니다.
TARGET_PEAK_USD = 150000 

# ==========================================
# 2. 업비트 API & Market Data Fetcher
# ==========================================
def get_btc_price_and_usd_rate():
    """
    업비트에서 BTC/KRW 가격과 원/달러(USDT/KRW) 환율을 조회합니다.
    """
    try:
        # BTC/KRW 가격 조회
        ticker_url = "https://api.upbit.com/v1/ticker?markets=KRW-BTC,KRW-USDT"
        res = requests.get(ticker_url).json()
        
        btc_krw = 0
        usdt_krw = 1350.0 # 기본 환율 fallback
        
        for item in res:
            if item['market'] == 'KRW-BTC':
                btc_krw = item['trade_price']
            elif item['market'] == 'KRW-USDT':
                usdt_krw = item['trade_price']
                
        btc_usd = btc_krw / usdt_krw if usdt_krw > 0 else 0
        return btc_krw, btc_usd, usdt_krw
    except Exception as e:
        print(f"가격 데이터 조회 실패: {e}")
        return 0, 0, 1350.0

# ==========================================
# 3. 사이클 내 투자 비율 계산 로직
# ==========================================
def calculate_cycle_allocation(current_btc_usd, target_peak_usd=150000):
    """
    현재 USD 가격 대비 예상 고점까지의 상방 여력(Upside Potential)을 산출하고,
    사이클 내 포트폴리오 목표 비중 및 매수 강도를 제시합니다.
    """
    if current_btc_usd <= 0:
        return {
            'upside_pct': 0,
            'mode': "⚪ 데이터 없음",
            'btc_ratio': "-",
            'cash_ratio': "-",
            'dca_pct': "-"
        }

    # 상방 여력 (%) = (목표 고점 - 현재가) / 현재가 * 100
    upside_pct = round(((target_peak_usd - current_btc_usd) / current_btc_usd) * 100, 2)

    # 상방 여력에 따른 구간별 모드 설정
    if upside_pct >= 150:
        mode = "🔥 [1단계: 대바닥 매집] 적극 매수 구간"
        target_btc_ratio = "70% ~ 80%"
        target_cash_ratio = "20% ~ 30%"
        monthly_dca_pct = "현금 예산의 20% ~ 25% 집행"
    elif upside_pct >= 80:
        mode = "🟢 [2단계: 정속 상승] 정속 분할 매수"
        target_btc_ratio = "50% ~ 65%"
        target_cash_ratio = "35% ~ 50%"
        monthly_dca_pct = "현금 예산의 10% ~ 15% 집행"
    elif upside_pct >= 30:
        mode = "🟡 [3단계: 상승 후반기] 보수적 관망 / 소량 매수"
        target_btc_ratio = "30% ~ 45%"
        target_cash_ratio = "55% ~ 70%"
        monthly_dca_pct = "현금 예산의 5% 미만 집행 (홀딩 중심)"
    else:
        mode = "🚨 [4단계: 고점 과열] 신규 매수 중단 및 분할 익절"
        target_btc_ratio = "10% ~ 20%"
        target_cash_ratio = "80% ~ 90%"
        monthly_dca_pct = "0% (분할 익절 단계)"

    return {
        'target_peak_usd': target_peak_usd,
        'upside_pct': upside_pct,
        'mode': mode,
        'btc_ratio': target_btc_ratio,
        'cash_ratio': target_cash_ratio,
        'dca_pct': monthly_dca_pct
    }

# ==========================================
# 4. 텔레그램 메시지 전송
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("텔레그램 토큰이 설정되지 않았습니다.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            print("텔레그램 메시지 전송 성공!")
        else:
            print(f"전송 실패: {res.text}")
    except Exception as e:
        print(f"텔레그램 전송 중 오류 발생: {e}")

# ==========================================
# 5. 메인 실행 함수
# ==========================================
def main():
    btc_krw, btc_usd, usdt_krw = get_btc_price_and_usd_rate()
    target_peak_krw = TARGET_PEAK_USD * usdt_krw
    
    cycle_info = calculate_cycle_allocation(btc_usd, TARGET_PEAK_USD)
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 텔레그램 전송용 마크다운 메시지 리포트
    report = f"""
🤖 *[비트코인 DCA & 사이클 포트폴리오 리포트]*
📅 기준시각: `{now_str}`

💰 *현재 시장 가격*
• BTC/KRW: `{btc_krw:,.0f} 원`
• BTC/USD (추정): `${btc_usd:,.2f}`
• 적용 환율: `{usdt_krw:,.1f} 원/$`

🎯 *다음 사이클 목표 고점 (Target Peak)*
• USD 기준: `${TARGET_PEAK_USD:,.0f}`
• KRW 환산: `{target_peak_krw:,.0f} 원`
• 남은 상방 여력(Upside): *+{cycle_info['upside_pct']}%*

📊 *포트폴리오 비중 가이드*
• 현재 진단: {cycle_info['mode']}
• **목표 BTC 비중**: `{cycle_info['btc_ratio']}`
• **목표 현금 비중**: `{cycle_info['cash_ratio']}`
• **금회 매수 집행 강도**: `{cycle_info['dca_pct']}`
    """

    print(report)
    send_telegram_message(report)

if __name__ == "__main__":
    main()
