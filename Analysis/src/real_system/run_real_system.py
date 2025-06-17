#!/usr/bin/env python3
"""
실제 고독사 감지 시스템 운용 파이프라인
- 실제 데이터베이스에서 데이터 로드
- 모델 훈련 및 저장
- 테스트 데이터 평가
- 결과 보고서 생성
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# 환경변수 로드
load_dotenv('../../config.env')

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_loader import RealDataLoader
from real_model_trainer import RealModelTrainer
from real_evaluator import RealModelEvaluator

class RealSystemPipeline:
    """실제 고독사 감지 시스템 파이프라인"""
    
    def __init__(self):
        self.train_start_date = '2025-04-12'
        self.train_end_date = '2025-05-31'
        self.test_start_date = '2025-05-31'
        self.test_end_date = '2025-06-05'
        
        self.loader = None
        self.trainer = None
        self.evaluator = None
        
    def initialize_components(self):
        """시스템 컴포넌트 초기화"""
        print("=== 실제 고독사 감지 시스템 초기화 ===")
        
        # 데이터 로더 초기화 (환경변수 사용)
        self.loader = RealDataLoader()
        
        # 모델 트레이너 초기화
        self.trainer = RealModelTrainer()
        
        # 모델 평가기 초기화는 훈련 후에 수행
        
        print("시스템 컴포넌트 초기화 완료")
        
    def connect_database(self):
        """데이터베이스 연결"""
        print("\n=== 데이터베이스 연결 ===")
        
        if not self.loader.connect():
            print("데이터베이스 연결 실패")
            return False
        
        return True
        
    def load_training_data(self):
        """훈련 데이터 로드"""
        print(f"\n=== 훈련 데이터 로드 ({self.train_start_date} ~ {self.train_end_date}) ===")
        
        train_data = self.loader.load_all_tables(self.train_start_date, self.train_end_date)
        
        if not train_data:
            print("훈련 데이터 로드 실패")
            return None, None
        
        # 특성 병합
        day_train, night_train = self.loader.merge_features_for_training(train_data)
        
        # 데이터 품질 확인
        self.loader.check_data_quality(day_train, "Day 훈련")
        self.loader.check_data_quality(night_train, "Night 훈련")
        
        return day_train, night_train
        
    def train_models(self, day_train, night_train):
        """모델 훈련"""
        print(f"\n=== 모델 훈련 시작 ===")
        
        day_columns, night_columns = self.loader.get_feature_columns()
        
        # Day 모델 훈련
        print("\n" + "="*50)
        print("DAY 모델 훈련")
        print("="*50)
        
        X_day, day_meta = self.trainer.prepare_training_data(day_train, day_columns)
        day_stats, day_corr = self.trainer.analyze_feature_importance(X_day, day_columns, 'day')
        day_results = self.trainer.train_models(X_day, 'day')
        self.trainer.save_models('day')
        
        # Night 모델 훈련
        print("\n" + "="*50)
        print("NIGHT 모델 훈련")
        print("="*50)
        
        X_night, night_meta = self.trainer.prepare_training_data(night_train, night_columns)
        night_stats, night_corr = self.trainer.analyze_feature_importance(X_night, night_columns, 'night')
        night_results = self.trainer.train_models(X_night, 'night')
        self.trainer.save_models('night')
        
        print(f"\n모델 훈련 완료")
        print(f"Day 모델: {len(day_columns)}개 특성, {X_day.shape[0]}개 샘플")
        print(f"Night 모델: {len(night_columns)}개 특성, {X_night.shape[0]}개 샘플")
        
        return day_results, night_results
        
    def load_test_data(self):
        """테스트 데이터 로드"""
        print(f"\n=== 테스트 데이터 로드 ({self.test_start_date} ~ {self.test_end_date}) ===")
        
        test_data = self.loader.load_all_tables(self.test_start_date, self.test_end_date)
        
        if not test_data:
            print("테스트 데이터 로드 실패")
            return None, None
        
        # 특성 병합
        day_test, night_test = self.loader.merge_features_for_training(test_data)
        
        # 데이터 품질 확인
        self.loader.check_data_quality(day_test, "Day 테스트")
        self.loader.check_data_quality(night_test, "Night 테스트")
        
        return day_test, night_test
        
    def evaluate_models(self, day_test, night_test):
        """모델 평가"""
        print(f"\n=== 모델 평가 시작 ===")
        
        # 모델 평가기 초기화 (훈련된 모델 로드)
        self.evaluator = RealModelEvaluator()
        
        day_columns, night_columns = self.loader.get_feature_columns()
        
        # Day 모델 평가
        print("\n" + "="*50)
        print("DAY 모델 평가")
        print("="*50)
        
        day_predictions = self.evaluator.predict_anomalies(day_test, day_columns, 'day')
        day_analysis = None
        day_results = None
        if day_predictions:
            day_results = self.evaluator.create_detailed_results(day_test, day_predictions, 'day')
            day_analysis = self.evaluator.analyze_anomalies(day_results, 'day')
            self.evaluator.save_results(day_results, day_analysis, 'day')
        
        # Night 모델 평가
        print("\n" + "="*50)
        print("NIGHT 모델 평가")
        print("="*50)
        
        night_predictions = self.evaluator.predict_anomalies(night_test, night_columns, 'night')
        night_analysis = None
        night_results = None
        if night_predictions:
            night_results = self.evaluator.create_detailed_results(night_test, night_predictions, 'night')
            night_analysis = self.evaluator.analyze_anomalies(night_results, 'night')
            self.evaluator.save_results(night_results, night_analysis, 'night')
        
        return day_analysis, night_analysis, day_results, night_results
    
    def create_and_save_abnormal_detection(self, day_results, night_results):
        """abnormal_detection 테이블 생성 및 데이터 저장"""
        print(f"\n=== abnormal_detection 테이블 처리 ===")
        
        # 테이블 생성
        self.loader.create_abnormal_detection_table()
        
        # 데이터 저장
        if day_results is not None:
            self.loader.insert_abnormal_detection_data(day_results, 'day')
        
        if night_results is not None:
            self.loader.insert_abnormal_detection_data(night_results, 'night')
        
    def generate_final_report(self, day_analysis, night_analysis):
        """최종 통합 보고서 생성"""
        print(f"\n=== 최종 보고서 생성 ===")
        
        report = []
        report.append("# 실제 고독사 감지 시스템 운용 결과 보고서")
        report.append(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 시스템 개요
        report.append(f"\n## 시스템 개요")
        report.append(f"- 데이터베이스: {os.getenv('DB_HOST')} ({os.getenv('DB_NAME')})")
        report.append(f"- 훈련 기간: {self.train_start_date} ~ {self.train_end_date}")
        report.append(f"- 테스트 기간: {self.test_start_date} ~ {self.test_end_date}")
        report.append(f"- 사용 모델: Isolation Forest, One-Class SVM")
        report.append(f"- Day 특성: 15개, Night 특성: 16개")
        
        # Day 모델 결과 요약
        if day_analysis:
            report.append(f"\n## Day 모델 결과 요약")
            report.append(f"- 분석 레코드 수: {day_analysis['total_records']:,}")
            report.append(f"- 분석 대상 사용자: {day_analysis['unique_users']}명")
            
            for model_name, stats in day_analysis['model_stats'].items():
                report.append(f"- {model_name}: 이상치 {stats['anomaly_count']}건 ({stats['anomaly_ratio']:.4f})")
            
            if 'consensus_anomalies' in day_analysis:
                consensus = day_analysis['consensus_anomalies']
                report.append(f"- 합의 이상치: {consensus['count']}건 ({consensus['ratio']:.4f})")
        
        # Night 모델 결과 요약
        if night_analysis:
            report.append(f"\n## Night 모델 결과 요약")
            report.append(f"- 분석 레코드 수: {night_analysis['total_records']:,}")
            report.append(f"- 분석 대상 사용자: {night_analysis['unique_users']}명")
            
            for model_name, stats in night_analysis['model_stats'].items():
                report.append(f"- {model_name}: 이상치 {stats['anomaly_count']}건 ({stats['anomaly_ratio']:.4f})")
            
            if 'consensus_anomalies' in night_analysis:
                consensus = night_analysis['consensus_anomalies']
                report.append(f"- 합의 이상치: {consensus['count']}건 ({consensus['ratio']:.4f})")
        
        # 권장 사항
        report.append(f"\n## 권장사항")
        report.append(f"- 이상치로 탐지된 사용자에 대한 추가 모니터링 실시")
        report.append(f"- 합의 이상치의 경우 우선순위를 두어 확인")
        report.append(f"- 정기적인 모델 재훈련을 통한 성능 최적화")
        report.append(f"- 실제 고독사 발생 시 모델 성능 평가 및 개선")
        
        # 최종 보고서는 콘솔에만 출력
        print('\n'.join(report))
        
        return "콘솔 출력 완료"
        
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        
        try:
            # 1. 시스템 초기화
            self.initialize_components()
            
            # 2. 데이터베이스 연결
            if not self.connect_database():
                return False
            
            # 3. 훈련 데이터 로드
            day_train, night_train = self.load_training_data()
            if day_train is None or night_train is None:
                return False
            
            # 4. 모델 훈련
            day_results, night_results = self.train_models(day_train, night_train)
            
            # 5. 테스트 데이터 로드
            day_test, night_test = self.load_test_data()
            if day_test is None or night_test is None:
                return False
            
            # 6. 모델 평가
            day_analysis, night_analysis, day_results, night_results = self.evaluate_models(day_test, night_test)
            
            # 7. abnormal_detection 테이블 생성 및 데이터 저장
            self.create_and_save_abnormal_detection(day_results, night_results)
            
            # 8. 사용자별 종합 보고서 생성
            self.evaluator.generate_user_summary_report(day_results, night_results)
            
            # 9. 최종 보고서 생성
            report_path = self.generate_final_report(day_analysis, night_analysis)
            
            # 완료
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n" + "="*60)
            print("실제 고독사 감지 시스템 운용 완료")
            print("="*60)
            print(f"실행 시간: {duration:.2f}초")
            print(f"데이터베이스 저장 완료: abnormal_detection 테이블")
            
            return True
            
        except Exception as e:
            print(f"파이프라인 실행 중 오류 발생: {e}")
            return False
            
        finally:
            # 데이터베이스 연결 종료
            if self.loader:
                self.loader.disconnect()

def main():
    """메인 실행 함수"""
    print("실제 고독사 감지 시스템 운용 시작")
    print("데이터베이스: 172.30.1.19 (mcs_led)")
    print("="*60)
    
    # 파이프라인 실행
    pipeline = RealSystemPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n시스템 운용이 성공적으로 완료되었습니다.")
    else:
        print("\n시스템 운용 중 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 