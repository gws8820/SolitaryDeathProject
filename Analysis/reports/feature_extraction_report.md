# 고독사 감지 LED 센서 데이터 특성 추출 과정 보고서

## 1. 개요

본 보고서는 고독사 감지 LED 센서 시스템의 특성 추출 과정, 방법론, 그리고 최종 결과를 상세히 기록합니다.

### 1.1 프로젝트 배경
- **목적**: 기존 12시간 LED 켜짐 기반 고독사 감지 시스템의 정확도 및 반응속도 개선
- **데이터**: 4개 방(안방, 거실, 주방, 화장실)의 LED 센서 데이터
- **시간 단위**: 10분 타임슬롯 (하루 144개)
- **머신러닝 접근**: 비지도 학습 (Isolation Forest, One-Class SVM)

### 1.2 데이터셋 구성
- **Train Dataset**: 9,000개 정상 데이터 행
- **Normal Test Dataset**: 600개 정상 데이터 행
- **Abnormal Test Datasets**: 3개 유형 (Rapid, Immediate, Gradual)

## 2. 특성 설계 및 정의

### 2.1 특성 카테고리
총 13개 특성을 5개 카테고리로 분류하여 설계했습니다.

#### 2.1.1 비활성 시간 특성 (4개)
**정의**: 모든 LED가 이전 상태를 유지하는 타임슬롯 수

- `day_inactive_total`: 주간(06:00-18:00) 전체 비활성 시간
- `night_inactive_total`: 야간(18:00-06:00) 전체 비활성 시간  
- `day_inactive_room`: 주간 방별 비활성 시간
- `night_inactive_room`: 야간 방별 비활성 시간

**계산 로직**:
```python
# 전체 비활성: 모든 방이 동시에 이전 상태 유지
for i in range(1, len(time_filtered)):
    all_inactive = True
    for room in ['01', '02', '03', '04']:
        if time_filtered.iloc[i][room] != time_filtered.iloc[i-1][room]:
            all_inactive = False
            break
    if all_inactive:
        total_inactive += 1

# 방별 비활성: 각 방이 이전 상태 유지
for i in range(1, len(room_data)):
    if room_data[i] == room_data[i-1]:
        inactive_count += 1
```

#### 2.1.2 토글 횟수 특성 (4개)
**정의**: LED 상태 변화 횟수

- `day_toggle_total`: 주간 전체 토글 횟수
- `night_toggle_total`: 야간 전체 토글 횟수
- `day_toggle_room`: 주간 방별 토글 횟수
- `night_toggle_room`: 야간 방별 토글 횟수

**계산 로직**:
```python
# 상태 변화 카운트
toggles = sum(1 for i in range(1, len(room_data)) 
              if room_data[i] != room_data[i-1])
```

#### 2.1.3 ON 시간 비율 특성 (2개)
**정의**: 전체 ON 시간 대비 각 방의 ON 시간 비율

- `day_on_ratio_room`: 주간 방별 ON 시간 비율
- `night_on_ratio_room`: 야간 방별 ON 시간 비율

**계산 로직**:
```python
# 각 방의 ON 시간 비율
for room in ['01', '02', '03', '04']:
    on_time = time_filtered[room].sum()
    ratio = on_time / total_on_time if total_on_time > 0 else 0
```

#### 2.1.4 주방 사용률 특성 (2개)
**정의**: 전체 토글 중 주방 토글이 차지하는 비율

- `day_kitchen_usage_rate`: 주간 주방 사용률
- `night_kitchen_usage_rate`: 야간 주방 사용률

**계산 로직**:
```python
# 주방(03번 방) 사용률
usage_ratio = kitchen_toggles / total_toggles if total_toggles > 0 else 0
```

#### 2.1.5 심야 화장실 사용 특성 (1개)
**정의**: 심야 시간대(22:00-04:00) 화장실 사용 횟수

- `night_bathroom_usage`: 심야 화장실 사용 횟수

### 2.2 시간대 정의
- **주간**: 06:00-18:00 (12시간, 72 타임슬롯)
- **야간**: 18:00-06:00 (12시간, 72 타임슬롯) - 날짜 경계 처리
- **심야**: 22:00-04:00 (6시간, 36 타임슬롯)

### 2.3 야간 시간대 처리
야간 데이터는 날짜 경계를 넘나들기 때문에 특별한 처리가 필요했습니다.

```python
if start_hour > end_hour:  # 야간의 경우 (18:00-06:00)
    # 당일 저녁 데이터 (18:00-23:59)
    current_evening = user_data[
        (user_data['Date'] == date) & 
        (user_data['Hour'] >= start_hour)
    ]
    
    # 익일 새벽 데이터 (00:00-05:59)
    next_date = pd.to_datetime(date) + pd.Timedelta(days=1)
    next_morning = user_data[
        (user_data['Date'] == next_date.date()) & 
        (user_data['Hour'] < end_hour)
    ]
    
    # 야간 데이터 = 당일 저녁 + 익일 새벽
    time_filtered = pd.concat([current_evening, next_morning])
```

## 3. 특성 추출 구현

### 3.1 클래스 구조
```python
class FeatureExtractor:
    def __init__(self, raw_data_path, processed_data_path)
    def load_data(self, filename)
    def calculate_inactive_periods(self, df, start_hour, end_hour)
    def calculate_toggle_counts(self, df, start_hour, end_hour)
    def calculate_on_ratios(self, df, start_hour, end_hour)
    def calculate_kitchen_usage_rate(self, df, start_hour, end_hour)
    def calculate_bathroom_usage(self, df)
    def extract_all_features(self, filename)
```

### 3.2 데이터 전처리
1. **타임스탬프 변환**: `pd.to_datetime()`로 datetime 객체 생성
2. **시간 추출**: 날짜, 시간, 분 컬럼 생성
3. **타임슬롯 계산**: `Hour * 6 + Minute // 10`

### 3.3 특성별 계산 과정

#### 3.3.1 비활성 시간 계산 개선 과정
**초기 문제**: 연속된 동일 상태의 개수에서 1을 빼는 방식으로 계산하여 이론적 최대값을 초과

**개선 과정**:
1. **1차 수정**: 각 방별 비활성 시간을 단순 합산 (결과: 최대 288 초과)
2. **2차 수정**: 모든 방이 동시에 비활성 상태인 시간만 계산 (결과: 최대 72 이내)

**최종 로직**: 모든 LED가 이전 상태를 유지하는 타임슬롯만 카운트

## 4. 특성 추출 결과

### 4.1 최종 산출물
- **총 특성 테이블**: 65개 CSV 파일 (5개 데이터셋 × 13개 특성)
- **데이터 행 수**: Train(9,000행), Normal(600행), Abnormal(각 300행)

### 4.2 특성별 통계 요약

#### 4.2.1 비활성 시간 특성
- **주간 비활성 총 시간**: 평균 18.47 (Train) vs 18.78 (Normal)
- **야간 비활성 총 시간**: 평균 29.88 (Train) vs 28.60 (Normal)
- **방별 비활성**: 화장실 > 주방 > 거실 > 안방 순서

#### 4.2.2 토글 횟수 특성  
- **주간 토글 총 횟수**: 평균 75.98 (Train) vs 75.27 (Normal)
- **야간 토글 총 횟수**: 평균 47.64 (Train) vs 46.49 (Normal)
- **활동 패턴**: 주간 > 야간 (정상적 생활 패턴)

#### 4.2.3 ON 시간 비율 특성
- **주간 방별 비율**: 안방(41%) > 거실(38%) > 주방(18%) > 화장실(3%)
- **야간 방별 비율**: 안방(77%) > 거실(14%) > 주방(6%) > 화장실(3%)

#### 4.2.4 사용 패턴 특성
- **주방 사용률**: 주간 20.6% vs 야간 8.2%
- **심야 화장실 사용**: 평균 1.37회

### 4.3 데이터 품질 검증 결과

#### 4.3.1 검증 항목
- **결측값**: 0개 발견
- **무한값**: 0개 발견  
- **범위 초과값**: 0개 발견
- **음수값**: 0개 발견

#### 4.3.2 논리적 일관성
- ✅ **주간 > 야간 활동량**: 정상적 생활 패턴 확인
- ✅ **방별 사용 패턴**: 안방 > 거실 > 주방 > 화장실 순서 합리적
- ✅ **비활성 시간 범위**: 모든 값이 이론적 최대값(72) 이내

### 4.4 Train vs Normal 분포 비교

#### 4.4.1 유사도 분석
- **전체 유사한 특성**: 19/31개 (61.3%)
- **높은 유사성**: ON 비율, 주방 사용률, 화장실 사용
- **약간 차이**: 야간 비활성 시간, 일부 토글 특성

#### 4.4.2 시간적 패턴
- **요일별 패턴**: 주중과 주말 유사, 최대 차이 2.6%
- **사용자별 변동**: Normal 데이터가 약간 높은 변동성 (정상 범위)

## 5. 문제 해결 과정

### 5.1 주요 문제 및 해결

#### 5.1.1 비활성 시간 계산 오류
**문제**: 초기 로직이 이론적 최대값 초과
**원인**: 연속 상태 카운트 방식의 논리적 오류
**해결**: 시간슬롯별 상태 비교 방식으로 변경

#### 5.1.2 야간 시간대 처리
**문제**: 날짜 경계를 넘나드는 야간 데이터 처리
**해결**: 당일 저녁 + 익일 새벽 데이터 결합 로직 구현

#### 5.1.3 데이터 검증 기준
**문제**: 부정확한 최대값 검증 기준
**해결**: 특성별 이론적 최대값 재정의 및 적용

### 5.2 개발 효율성
- **모듈화**: 각 특성별 계산 함수 분리
- **재사용성**: 시간대별 필터링 로직 공통화
- **확장성**: 새로운 특성 추가 용이한 구조

## 6. 검증 및 품질 보증

### 6.1 자동화된 품질 검증
- **데이터 완성도**: 모든 파일 생성 확인
- **수치 범위**: 이론적 최대/최소값 검증
- **분포 일관성**: 통계적 검정을 통한 이상치 탐지

### 6.2 통계적 검증
- **T-test**: 평균 차이 검정
- **Kolmogorov-Smirnov test**: 분포 유사성 검정
- **Cohen's d**: 효과 크기 측정

## 7. 결론 및 다음 단계

### 7.1 달성 성과
- ✅ **13개 특성 완전 추출**: 모든 데이터셋에 대해 성공
- ✅ **데이터 품질 확보**: 품질 문제 0개
- ✅ **논리적 일관성**: 모든 특성이 합리적 범위 내
- ✅ **분포 안정성**: Train-Normal 간 적절한 유사성 (61.3%)

### 7.2 특성 추출의 우수성
1. **포괄적 커버리지**: 시간, 공간, 패턴의 다차원적 특성
2. **생활 패턴 반영**: 주간/야간, 방별 사용 패턴 고려
3. **고독사 특화**: 비활성 시간, 화장실 사용 등 고독사 관련 지표

### 7.3 다음 단계 준비 상태
**모델 훈련 진행 가능**: ✅ **권장함**

**근거**:
- 모든 특성이 올바르게 추출됨
- 데이터 품질 문제 완전 해결
- Train-Normal 분포 적절히 유사
- 논리적 일관성 확보

### 7.4 향후 개선 방향
1. **추가 특성 탐색**: 시간대별 연속성, 패턴 변화율 등
2. **특성 선택**: 중요도 기반 특성 선별
3. **정규화**: 스케일링 및 정규화 기법 적용
4. **시계열 특성**: 시간적 종속성을 고려한 특성 개발

---

**보고서 작성**: 2024년  
**특성 추출 완료**: 전체 5개 데이터셋, 총 65개 특성 테이블  
**검증 상태**: 품질 검증 완료, 모델 훈련 준비 완료 