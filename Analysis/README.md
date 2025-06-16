# 고독사 감지 시스템 (Solitary Death Detection System)

고독사 노인의 가구에 설치된 4개의 LED 센서를 통해 활동을 추적하고, 머신러닝을 활용하여 고독사를 조기에 감지하는 시스템입니다.

## 프로젝트 개요

### 목표
- 기존 시스템(LED 12시간 연속 켜짐 → 고독사 판단)의 감지 시간 단축 및 정확도 향상
- 비지도 학습 머신러닝 방법 3개(Isolation Forest, One-Class SVM, DBSCAN) 활용
- 현실적인 더미 데이터를 통한 알고리즘 개발 및 검증

### 시스템 구성
- **LED 센서 4개**: 01(안방), 02(거실), 03(주방), 04(화장실)
- **데이터 수집**: 10초마다 센서 상태 수집
- **분석 단위**: 10분 타임슬롯 (하루 144개 데이터 포인트)
- **시뮬레이션 기간**: 30일

## 프로젝트 구조

```
SolitaryDeath-Analysis/
├── config.py                  # 프로젝트 설정 파일
├── generate_dummy_data.py      # 통합 실행 스크립트
├── normal_dummy.py             # 정상 데이터 생성기
├── abnormal_dummy.py           # 비정상 데이터 생성기  
├── visualize_dummy_data.py     # 데이터 시각화 도구
├── requirements.txt            # 패키지 의존성
├── README.md                   # 프로젝트 문서
├── data/
│   ├── raw/                    # 원본 더미 데이터
│   └── processed/              # 전처리된 데이터
├── models/                     # 머신러닝 모델 저장소
├── charts/                     # 시각화 결과 (영어)
└── reports/                    # 분석 보고서 (한글)
```

## 데이터 구성

### 정상 데이터 (300명, 30일)
다양한 생활 패턴을 확률적으로 반영:
- 외출/재실 패턴 (30% 외출 확률)
- 외출 시간 (1-12시간, 평균 4시간)
- 기상 시간 분포 (6-9시, 7-8시 집중)
- 아침식사 여부 (80% 확률)
- 낮시간 조명 사용 (60% 확률)
- 화장실 방문 빈도 (시간당 0.8회 평균)
- 수면 시간 분포 (21-24시, 22-23시 집중)

### 비정상 데이터 (60명, 30일)
3가지 고독사 패턴:
1. **점진적 악화** (40%): 1-2주에 걸쳐 서서히 활동 감소
2. **급격한 악화** (40%): 1-3일에 걸쳐 빠른 활동 감소  
3. **갑작스런 중단** (20%): 즉시 활동 완전 중단

## 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 더미 데이터 생성 및 시각화
```bash
# 통합 실행 (권장)
python generate_dummy_data.py

# 또는 개별 실행
python normal_dummy.py      # 정상 데이터만
python abnormal_dummy.py    # 비정상 데이터만  
python visualize_dummy_data.py  # 시각화만
```

## 생성되는 시각화

### LED 활동 분석
- **Toggle Frequency Heatmap**: 사용자별 LED 토글 횟수 히트맵
- **State Duration Heatmap**: LED 상태 유지 시간 히트맵
- **Toggle Comparison Charts**: 정상/비정상 그룹 간 비교 박스플롯
- **Daily Activity Patterns**: 일일 활동 패턴 시각화

### 악화 패턴 분석
- **Deterioration Analysis**: 비정상 사용자의 3가지 악화 패턴 시각화
- **Summary Statistics**: 전체 통계 요약

## 데이터 특성

### 정상 패턴 특징
- 규칙적인 생활 리듬 (기상, 식사, 수면)
- 공간별 적절한 LED 사용 빈도
- 외출 시 대부분 LED 꺼짐
- 개인별 변동성 반영

### 비정상 패턴 특징
- **점진적 악화**: 주방/화장실 사용 점진적 감소
- **급격한 악화**: 안방 위주 생활, 타 공간 사용 급감
- **갑작스런 중단**: 즉시 모든 활동 중단
- **사망 후**: 특정 LED 상태 고정 유지

## 다음 단계

1. **데이터 전처리**: 특성 엔지니어링, 정규화
2. **모델 개발**: Isolation Forest, One-Class SVM, DBSCAN 구현
3. **성능 평가**: 감지 시간, 정확도, 재현율 분석
4. **모델 최적화**: 하이퍼파라미터 튜닝, 앙상블 기법

## 설정 변경

`config.py`에서 다음 설정들을 조정할 수 있습니다:
- 사용자 수 (정상/비정상)
- 시뮬레이션 기간
- 생활 패턴 확률 분포
- 악화 패턴 특성

## 기술 스택

- **Python 3.8+**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn (예정)
- **Others**: tqdm, datetime

## 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다. 