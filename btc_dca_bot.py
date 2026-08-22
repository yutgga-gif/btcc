import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

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
# 2. [축 2] HMM 기반 시장 국면(Regime) 연산 (버그 수정)
# ==========================================
def analyze_hmm_regime(df_daily):
    try:
        sub = df_daily.copy()
        sub['Return'] = np.log(sub['Close'] / sub['Close'].shift(1))
        sub['Volatility'] = sub['Return'].rolling(window=20).std()
        sub = sub.dropna()

        if len(sub) < 500: return False, 0.0, "데이터 부족"

        X = sub[['Return', 'Volatility']].values
        
        # HMM 모델 학습 (3가지 국면)
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
        model.fit(X)

        # 전체 시계열 기반 상태 확률 계산 (수정: 전체 X 입력 후 마지막 행 취득)
        all_probs = model.predict_proba(X)
        last_prob = all_probs[-1]

        # 변동성 대비 수익률이 가장 낮은 State(투매/공포 국면) 인덱스 찾기
        means = model.means_
        # means[:, 0] = 평균 수익률, means[:, 1] = 평균 변동성
        # 공포 국면: 수익률은 작고 변동성은 큰 상태
        panic_score = means[:, 0] - means[:, 1] 
        fear_state_idx = np.argmin(panic_score)

        fear_prob = round(last_prob[fear_state_idx] * 100, 1)

        # 공포/바닥 국면 확률이 40% 이상일 때 통과
        is_fear_zone = fear_prob >= 40.0
        return is_fear_zone, fear_prob, f"공포/바닥 확률: {fear_prob}%"
    except Exception as e:
        return False, 0.0, "HMM 연산 우회"

# ==========================================
# 3. [축 1 & 3] 주봉 기반 3축 앙상블 분석 엔진
# ==========================================
def analyze_ensemble_system(symbol, asset_info):
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period="6y", interval="1d", timeout=12)
        if df_daily.empty or len(df_daily) < 1000: return None
    except Exception: return None

    cur_price = float(df_daily['Close'].iloc[-1])

    # 주봉 데이터 변환
    weekly = pd.DataFrame()
    weekly['Close'] = df_daily['Close'].resample('W-FRI').last()
    weekly['High'] = df_daily['High'].resample('W-FRI').max()
    weekly['Low'] = df_daily['Low'].resample('W-FRI').min()
    weekly['Volume'] = df_daily['Volume'].resample('W-FRI').sum()
    weekly = weekly.dropna()

    ath_price = float(weekly['Close'].max())
    mdd_pct = round(((cur_price - ath_price) / ath_price) * 100, 2)

    # 200주 이동평균선
    weekly['MA200'] = weekly['Close'].rolling(200).mean()
    w_ma200 = float(weekly['MA200'].iloc[-1])
    dist_w200_pct = round(((cur_price - w_ma200) / w_ma200) * 100, 2)

    # -----------------------------------------------------------
    # [축 1] 가격 위치 필터: 200주선 터치/이탈 (+5% 이하)
    # -----------------------------------------------------------
    axis1_price_passed = dist_w200_pct <= 5.0

    # -----------------------------------------------------------
    # [축 2] HMM 통계 국면 필터
    # -----------------------------------------------------------
    axis2_hmm_passed, fear_prob, hmm_msg = analyze_hmm_regime(df_daily)

    # -----------------------------------------------------------
    # [축 3] 수급/구조 필터 (주봉 수급 유입)
    # -----------------------------------------------------------
    mfv = ((weekly['Close'] - weekly['Low']) - (weekly['High'] - weekly['Close'])) / (weekly['High'] - weekly['Low']).replace(0, np.nan)
    cmf_w = (mfv.fillna(0) * weekly['Volume']).rolling(10).sum() / weekly['Volume'].rolling(10).sum().replace(0, np.nan)
    cmf_passed = cmf_w.iloc[-1] > 0.0

    vol_avg_10w = weekly['Volume'].iloc[-11:-1].mean()
    vol_passed = weekly['Volume'].iloc[-1] >= vol_avg_10w * 1.3
    
    recent_swing_high = weekly['High'].iloc[-8:-2].max()
    choch_passed = cur_price > recent_swing_high

    axis3_flow_passed = (cmf_passed or vol_passed) and choch_passed

    # -----------------------------------------------------------
    # [상승 추세 전환 판단 (수정)]
    # 200주선 근처(+15% 이내)에서 5주선 > 20주선 골든크로스 발생 시
    # -----------------------------------------------------------
    w_ma5 = weekly['Close'].rolling(5).mean()
    w_ma20 = weekly['Close'].rolling(20).mean()
    is_near_bottom_rebound = dist_w200_pct <= 15.0 # 바닥권 탈출 과정인지 체크
    uptrend_rebound_started = is_near_bottom_rebound and (w_ma5.iloc[-1] > w_ma20.iloc[-1]) and choch_passed

    # -----------------------------------------------------------
    # 사이클 및 자금 집행 의사결정
    # -----------------------------------------------------------
    if uptrend_rebound_started:
        cycle_state = "3단계: 바닥 탈출 및 상승 추세 시작 (매수 중단)"
        action = "🛑 [매수 종료 / 홀딩] 바닥에서 반격 성공 ➔ 추가 매수 중단 후 수익 극대화"
    elif axis1_price_passed and axis2_hmm_passed and axis3_flow_passed:
        cycle_state = "2단계: 3축 ALL-PASS 진성 바닥 (매수 집행 구간)"
        action = f"⚡ [비상금 분할 매수 집행] 모든 노이즈 제거 승인 ({hmm_msg}) ➔ 1차 진입"
    elif axis1_price_passed:
        cycle_state = "2단계 대기: 200주선 접근 중 (수급/모델 미충족)"
        action = f"⏳ [관망] 200주선 바닥권 진입했으나 반격 수급 부재 (HMM: {'OK' if axis2_hmm_passed else 'NO'}, 수급: {'OK' if axis3_flow_passed else 'NO'})"
    else:
        cycle_state = "1단계: 평시 구간 (신고점 후 하락장 대기)"
        action = "⚪ [관망] 200주선 바닥 미도달 ➔ 비상 준비금 집행 금지"

    return {
        'name': asset_info['name'],
        'cur_price': cur_price,
        'ath_price': ath_price,
        'mdd_pct': mdd_pct,
        'w_ma200': w_ma200,
        'dist_w200_pct': dist_w200_pct,
        'axis1': axis1_price_passed,
        'axis2': axis2_hmm_passed,
        'axis3': axis3_flow_passed,
        'hmm_msg': hmm_msg,
        'cycle_state': cycle_state,
        'action': action
    }

# ==========================================
# 4. 실행 및 알림
# ==========================================
def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""🤖 [3축 앙상블 노이즈 제어 매수 리포트]
📅 검증 시각: {now_str}
================================="""

    for symbol, info in TARGET_ASSETS.items():
        res = analyze_ensemble_system(symbol, info)
        if res:
            report += f"""

📌 [{res['name']}]
• 현재가: ${res['cur_price']:,.2f} | 200주선: ${res['w_ma200']:,.2f}
• 전고점 대비: {res['mdd_pct']}% | 200주선 격차: {res['dist_w200_pct']}%

📊 3축 독립 검증 현황:
  [축 1] 가격 위치 (200주선 ≤ +5%): {"✅ PASS" if res['axis1'] else "❌ FAIL"}
  [축 2] 통계 모델 ({res['hmm_msg']}): {"✅ PASS" if res['axis2'] else "❌ FAIL"}
  [축 3] 주봉 수급/구조 (스마트머니): {"✅ PASS" if res['axis3'] else "❌ FAIL"}

🔄 사이클 상태: 
   {res['cycle_state']}

👉 최종 자금 집행 명령: 
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
