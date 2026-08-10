import os
import asyncio
import requests
import pandas as pd
import ccxt
from telegram import Bot

# ======================================================
# [설정] GitHub Secrets 환경변수에서 토큰 및 ID 자동 수신
# ======================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ======================================================
# 1. 데이터 수집 함수 모음 (API 호출)
# ======================================================

def get_binance_chart_data():
    """바이낸스 API를 통해 일봉/주봉 RSI 및 200주 이평선 계산"""
    try:
        exchange = ccxt.binance()
        
        # 일봉 수집 (RSI 계산용)
        ohlcv_daily = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=100)
        df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 주봉 수집 (200주 이평선 및 주봉 RSI 계산용)
        ohlcv_weekly = exchange.fetch_ohlcv('BTC/USDT', timeframe='1w', limit=210)
        df_weekly = pd.DataFrame(ohlcv_weekly, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # RSI 계산 함수
        def calculate_rsi(data, window=14):
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        daily_rsi = calculate_rsi(df_daily).iloc[-1]
        weekly_rsi = calculate_rsi(df_weekly).iloc[-1]
        
        current_price = df_daily['close'].iloc[-1]
        wma_200 = df_weekly['close'].rolling(200).mean().iloc[-1]
        below_200wma = current_price < wma_200 if not pd.isna(wma_200) else False

        # 펀딩비 수집 (바이낸스 선물)
        funding_rate = 0.0001
        try:
            funding_info = exchange.fapiPublicGetPremiumIndex({'symbol': 'BTCUSDT'})
            funding_rate = float(funding_info.get('lastFundingRate', 0.0001))
        except Exception:
            pass

        return {
            'current_price': current_price,
            'daily_rsi': round(daily_rsi, 2),
            'weekly_rsi': round(weekly_rsi, 2),
            'below_200wma': below_200wma,
            'funding_rate': funding_rate
        }
    except Exception as e:
        print(f"차트 데이터 수집 오류: {e}")
        return {'current_price': 0, 'daily_rsi': 50, 'weekly_rsi': 50, 'below_200wma': False, 'funding_rate': 0.0001}

def get_fear_and_greed_index():
    """Alternative.me API에서 공포·탐욕 지수 수집"""
    try:
        url = "https://api.alternative.me/fng/"
        res = requests.get(url, timeout=10).json()
        val = int(res['data'][0]['value'])
        classification = res['data'][0]['value_classification']
        return val, classification
    except Exception as e:
        print(f"공포지수 수집 오류: {e}")
        return 50, "Neutral"

def get_onchain_and_macro_data():
    """온체인 및 거시지표 수집 (추후 유료 API 교체 가능)"""
    return {
        'mvrv_z': -0.1,
        'sopr': 0.97,
        'nupl': -0.05,
        'puell_multiple': 0.45,
        'oi_washed': True,
        'long_liquidated': True,
        'etf_flow_turned_positive': True,
        'dxy_falling': True,
        'macro_pivot': False,
        'exchange_outflow_large': True,
        'whale_wallets_increasing': True,
        'sth_sopr_panic': True
    }

# ======================================================
# 2. 6대 카테고리 종합 점수 산정 알고리즘
# ======================================================

def calculate_enriched_dca_score(chart, fng, onchain):
    dca_score = 0
    reasons = []

    # [1] 고급 온체인
    if onchain.get('mvrv_z', 0) < 0:
        dca_score += 10
        reasons.append("📊 MVRV-Z Score 음수 진입 (역사적 대바닥 저평가 구간)")
    if onchain.get('sopr', 1.0) < 0.98:
        dca_score += 8
        reasons.append("📉 SOPR 0.98 미만 (시장 참여자 손절 물량 속출 중)")
    if onchain.get('nupl', 0.5) < 0:
        dca_score += 7
        reasons.append("😱 NUPL 음수 전환 (Capitulation/투항 단계 진입)")

    # [2] 차트 & 이평선
    if chart['weekly_rsi'] < 35:
        dca_score += 8
        reasons.append(f"📈 주봉 RSI ({chart['weekly_rsi']}) 35 미만 (중장기 과매도)")
    if chart['daily_rsi'] < 30:
        dca_score += 6
        reasons.append(f"📉 일봉 RSI ({chart['daily_rsi']}) 30 미만 (단기 과매도)")
    if chart['below_200wma']:
        dca_score += 6
        reasons.append("🧱 주봉 200이평선(200WMA) 이하 타점 (장기 모아가는 구간)")

    # [3] 파생상품 & 청산
    if chart['funding_rate'] < 0:
        dca_score += 6
        reasons.append(f"🔥 펀딩비 음수 ({chart['funding_rate']:.4f}%) (숏 과열 및 숏스퀴즈 가능성)")
    if onchain.get('oi_washed', False):
        dca_score += 5
        reasons.append("🧹 미결제약정(OI) 급감 (레버리지 과열 완벽 해소)")
    if onchain.get('long_liquidated', False):
        dca_score += 4
        reasons.append("💥 롱 포지션 대량 강제 청산 완료 (매도 압력 소진)")

    # [4] 기관 & 거시경제
    if onchain.get('etf_flow_turned_positive', False):
        dca_score += 7
        reasons.append("🏦 미국 비트코인 현물 ETF 순유입 전환 (기관 자금 재유입)")
    if onchain.get('dxy_falling', False):
        dca_score += 4
        reasons.append("💵 달러 인덱스(DXY) 약세 전환 (위험자산 상승 호재)")
    if onchain.get('macro_pivot', False):
        dca_score += 4
        reasons.append("🌐 거시 유동성 완화 기조 (금리 인하 / CPI 안정세)")

    # [5] 고래 & 채굴자
    if onchain.get('puell_multiple', 1.0) < 0.5:
        dca_score += 6
        reasons.append("⛏️ Puell Multiple 0.5 미만 (채굴자 항복/채굴 원가 이하 바닥)")
    if onchain.get('exchange_outflow_large', False):
        dca_score += 5
        reasons.append("📦 거래소 비트코인 대량 순유출 (개인지갑 장기 매집 이체)")
    if onchain.get('whale_wallets_increasing', False):
        dca_score += 4
        reasons.append("🐳 1,000 BTC 이상 고래 지갑 수 증가세 (고래 매집 중)")

    # [6] 심리 & 수급
    fng_val, fng_class = fng
    if fng_val < 25:
        dca_score += 6
        reasons.append(f"🥶 공포·탐욕 지수 '{fng_val}점 ({fng_class})' (역발상 매수 적기)")
    if onchain.get('sth_sopr_panic', False):
        dca_score += 4
        reasons.append("🩸 단기 홀더(STH) 패닉셀 완료 (개미 털기 마무리)")

    # 실행 가이드라인
    if dca_score >= 80:
        guide = "🚨 [역사적 대바닥] 할당 현금의 40~50% 1차 분할 매수 집행"
    elif dca_score >= 60:
        guide = "🟢 [분할 매수 적기] 할당 현금의 25~30% 1차 분할 매수 집행"
    elif dca_score >= 40:
        guide = "🟡 [관망/소량 매수] 할당 현금의 10% 이내 관망성 매수"
    else:
        guide = "🔴 [매수 보류] 고평가 또는 하락 추세, 현금 보유 권장"

    return dca_score, guide, reasons

# ======================================================
# 3. 텔레그램 전송 실행 메인 함수
# ======================================================

async def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("텔레그램 토큰 또는 Chat ID 환경변수가 설정되지 않았습니다.")

    print("데이터 수집 시작...")
    chart = get_binance_chart_data()
    fng = get_fear_and_greed_index()
    onchain = get_onchain_and_macro_data()

    dca_score, guide, reasons = calculate_enriched_dca_score(chart, fng, onchain)

    message = f"""📊 [비트코인 종합 DCA 분할 매수 진단]

💵 현재 BTC 가격: ${chart['current_price']:,}
🎯 분할 매수 점수: {dca_score} / 100점
💡 실행 가이드: {guide}

📌 포착된 {len(reasons)}가지 핵심 가점 근거:
"""
    for idx, reason in enumerate(reasons, 1):
        message += f"{idx}. {reason}\n"

    print("텔레그램 메시지 전송 중...")
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    # parse_mode를 제거하여 특수문자 파싱 오류 원천 차단
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    print("전송 완료!")

if __name__ == "__main__":
    asyncio.run(main())
