# abnormal_detection 테이블 설명 보고서

생성일시: 2025-06-17

## 1. 테이블 개요

`abnormal_detection` 테이블은 실시간 고독사 감지 시스템의 핵심 결과 저장소입니다. 
머신러닝 모델(Isolation Forest, One-Class SVM)이 분석한 사용자별 이상치 탐지 결과를 체계적으로 저장하여 
실무진의 모니터링 및 의사결정을 지원합니다.

## 2. 테이블 구조

### 2.1 컬럼 정의

| 컬럼명 | 데이터 타입 | 설명 |
|--------|------------|------|
| `User` | VARCHAR(20) | 사용자 식별자 (Primary Key 구성요소) |
| `Date` | DATE | 분석 날짜 (Primary Key 구성요소) |
| `Type` | VARCHAR(10) | 모델 타입 - 'day' 또는 'night' (Primary Key 구성요소) |
| `OCSVM_prediction` | BOOLEAN | One-Class SVM 이상치 예측 결과 (TRUE: 이상, FALSE: 정상) |
| `OCSVM_score` | FLOAT | One-Class SVM 이상치 점수 (1-100점, 50점 이상 이상) |
| `Isforest_prediction` | BOOLEAN | Isolation Forest 이상치 예측 결과 (TRUE: 이상, FALSE: 정상) |
| `Isforest_score` | FLOAT | Isolation Forest 이상치 점수 (1-100점, 50점 이상 이상) |
| `Consensus_prediction` | BOOLEAN | 합의 이상치 판정 (두 모델 모두 이상으로 판단) |
| `Consensus_score` | FLOAT | 합의 점수 (두 모델 점수의 평균) |

### 2.2 인덱스 구조

```sql
PRIMARY KEY (User, Date, Type)
INDEX idx_user_date (User, Date)
INDEX idx_type (Type)
INDEX idx_consensus (Consensus_prediction)
```

## 3. 데이터 의미

### 3.1 모델 타입 (Type)
- **day**: 주간 모델 (06:00-18:00 활동 패턴 분석)
  - 15개 특성 기반 이상치 탐지
  - 주간 활동량, 방별 사용 패턴, 주방 사용률 등
  
- **night**: 야간 모델 (18:00-06:00 활동 패턴 분석)
  - 16개 특성 기반 이상치 탐지 (주간 특성 + 심야 화장실 사용)
  - 야간 활동 패턴, 수면 패턴 등

### 3.2 예측 결과 해석
- **TRUE (이상)**: 해당 사용자의 일일 활동 패턴이 정상 범위를 벗어남
- **FALSE (정상)**: 일반적인 활동 패턴 범위 내

### 3.3 점수 체계
- **OCSVM_score**: 
  - 1-49점: 정상 패턴
  - 50-100점: 이상 패턴 (점수가 높을수록 이상도 높음)
  - 범위: 1.0 ~ 100.0

- **Isforest_score**: 
  - 1-49점: 정상 패턴
  - 50-100점: 이상 패턴 (점수가 높을수록 이상도 높음)
  - 범위: 1.0 ~ 100.0

- **Consensus_score**:
  - 두 모델 점수의 평균
  - 50점 이상일 때 합의 이상치로 판정

## 4. 활용 방안

### 4.1 모니터링 우선순위
1. **최우선**: `Consensus_prediction = TRUE` (두 모델 모두 이상 탐지)
2. **우선**: 연속 3일 이상 단일 모델 이상 탐지
3. **관찰**: 단발성 이상 탐지

### 4.2 쿼리 예시

#### 4.2.1 합의 이상치 조회
```sql
SELECT User, Date, Type, Consensus_score
FROM abnormal_detection 
WHERE Consensus_prediction = TRUE
ORDER BY Date DESC, Consensus_score ASC;
```

#### 4.2.2 사용자별 이상치 빈도
```sql
SELECT User, 
       COUNT(*) as total_records,
       SUM(CASE WHEN OCSVM_prediction = TRUE THEN 1 ELSE 0 END) as ocsvm_anomalies,
       SUM(CASE WHEN Isforest_prediction = TRUE THEN 1 ELSE 0 END) as isforest_anomalies,
       SUM(CASE WHEN Consensus_prediction = TRUE THEN 1 ELSE 0 END) as consensus_anomalies
FROM abnormal_detection 
GROUP BY User
ORDER BY consensus_anomalies DESC;
```

#### 4.2.3 최근 7일 이상치 동향
```sql
SELECT Date, Type,
       COUNT(*) as total_users,
       SUM(CASE WHEN OCSVM_prediction = TRUE THEN 1 ELSE 0 END) as ocsvm_anomalies,
       SUM(CASE WHEN Isforest_prediction = TRUE THEN 1 ELSE 0 END) as isforest_anomalies,
       SUM(CASE WHEN Consensus_prediction = TRUE THEN 1 ELSE 0 END) as consensus_anomalies
FROM abnormal_detection 
WHERE Date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
GROUP BY Date, Type
ORDER BY Date DESC, Type;
```

## 5. 운영 고려사항

### 5.1 데이터 관리
- **보존 기간**: 최소 1년 이상 권장
- **백업**: 일일 백업을 통한 데이터 보호
- **아카이빙**: 6개월 이상 데이터는 별도 아카이브 테이블로 이관

### 5.2 성능 최적화
- 날짜 범위 기반 파티셔닝 고려
- 자주 사용되는 쿼리에 대한 추가 인덱스 생성
- 정기적인 통계 정보 업데이트

### 5.3 알림 체계 연동
이 테이블을 기반으로 자동 알림 시스템 구축 가능:
```sql
-- 긴급 알림: 합의 이상치
-- 주의 알림: 연속 3일 이상치
-- 정보 알림: 단발성 이상치
```

## 6. 확장 가능성

### 6.1 추가 컬럼 고려사항
- `severity_level`: 이상치 심각도 (1-5 단계)
- `confirmed_status`: 실무진 확인 상태
- `action_taken`: 취해진 조치 내용
- `false_positive`: 거짓 양성 표시

### 6.2 연관 테이블
- `user_profile`: 사용자 기본 정보
- `action_log`: 조치 이력
- `notification_log`: 알림 발송 이력

## 7. 결론

`abnormal_detection` 테이블은 고독사 감지 시스템의 핵심 데이터 저장소로서:
- 체계적인 이상치 정보 관리
- 실시간 모니터링 지원
- 데이터 기반 의사결정 지원
- 시스템 성능 분석 기반 제공

이를 통해 보다 효과적이고 신뢰성 높은 고독사 예방 시스템 운영이 가능합니다. 