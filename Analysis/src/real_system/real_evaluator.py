import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict, List
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# 환경변수 로드
load_dotenv('../../config.env')

from database_loader import RealDataLoader

class RealModelEvaluator:
    """실제 데이터로 모델을 평가하고 이상치를 탐지하는 클래스"""
    
    def __init__(self, model_dir: str = "../../real_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """저장된 모델들을 로드"""
        for model_type in ['day', 'night']:
            type_dir = os.path.join(self.model_dir, model_type)
            
            if not os.path.exists(type_dir):
                print(f"경고: {type_dir} 폴더가 존재하지 않습니다.")
                continue
            
            # 스케일러 로드
            scaler_path = os.path.join(type_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scalers[model_type] = joblib.load(scaler_path)
                print(f"{model_type} 스케일러 로드 완료")
            
            # 모델들 로드
            self.models[model_type] = {}
            
            for model_name in ['isolation_forest', 'one_class_svm']:
                model_path = os.path.join(type_dir, f'{model_name}.pkl')
                if os.path.exists(model_path):
                    self.models[model_type][model_name] = joblib.load(model_path)
                    print(f"{model_type} {model_name} 로드 완료")
                else:
                    print(f"경고: {model_path} 파일이 존재하지 않습니다.")
    
    def predict_anomalies(self, df: pd.DataFrame, feature_columns: List[str], model_type: str) -> Dict:
        """이상치 예측"""
        if model_type not in self.models or model_type not in self.scalers:
            print(f"오류: {model_type} 모델 또는 스케일러가 로드되지 않았습니다.")
            return {}
        
        # 특성 데이터 준비
        X = df[feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        
        # 데이터 정규화
        X_scaled = self.scalers[model_type].transform(X)
        
        results = {}
        
        for model_name, model in self.models[model_type].items():
            # 예측
            predictions = model.predict(X_scaled)
            scores = model.decision_function(X_scaled)
            
            # 결과 저장
            results[model_name] = {
                'predictions': predictions,
                'scores': scores,
                'anomaly_count': (predictions == -1).sum(),
                'anomaly_ratio': (predictions == -1).mean()
            }
            
            print(f"{model_name}: 이상치 {results[model_name]['anomaly_count']}개 "
                  f"({results[model_name]['anomaly_ratio']:.4f})")
        
        return results
    
    def normalize_score_to_human_readable(self, scores, model_name):
        """점수를 0-100점 척도로 변환 (50점 이상이 이상)"""
        scores = np.array(scores)
        
        if model_name == 'isolation_forest':
            # Isolation Forest: 원래 점수가 낮을수록 이상
            # 점수 범위를 0-100으로 변환하되, 낮은 점수(이상)가 높은 점수가 되도록
            min_score, max_score = scores.min(), scores.max()
            if max_score != min_score:
                # 정규화 후 반전 (낮은 점수가 높은 점수가 되도록)
                normalized = (scores - min_score) / (max_score - min_score)
                human_scores = (1 - normalized) * 99 + 1  # 1-100 범위
            else:
                human_scores = np.full_like(scores, 25.0)  # 모든 값이 같으면 정상으로 처리
                
        elif model_name == 'one_class_svm':
            # One-Class SVM: 원래 점수가 낮을수록 이상
            # 점수 범위를 0-100으로 변환하되, 낮은 점수(이상)가 높은 점수가 되도록
            min_score, max_score = scores.min(), scores.max()
            if max_score != min_score:
                # 정규화 후 반전
                normalized = (scores - min_score) / (max_score - min_score)
                human_scores = (1 - normalized) * 99 + 1  # 1-100 범위
            else:
                human_scores = np.full_like(scores, 25.0)  # 모든 값이 같으면 정상으로 처리
        
        return human_scores

    def create_detailed_results(self, df: pd.DataFrame, predictions: Dict, model_type: str) -> pd.DataFrame:
        """상세 결과 데이터프레임 생성"""
        result_df = df[['User', 'Date']].copy()
        
        for model_name, pred_data in predictions.items():
            result_df[f'{model_name}_prediction'] = pred_data['predictions']
            
            # 원본 점수를 사람이 읽기 쉬운 0-100점 척도로 변환
            human_scores = self.normalize_score_to_human_readable(pred_data['scores'], model_name)
            result_df[f'{model_name}_score'] = human_scores
            
            # 변환된 점수 기준으로 이상치 판정 (50점 이상)
            result_df[f'{model_name}_is_anomaly'] = (human_scores >= 50).astype(int)
        
        # 합의 점수 (두 모델 점수의 평균)
        model_names = list(predictions.keys())
        if len(model_names) >= 2:
            score_cols = [f'{model_name}_score' for model_name in model_names]
            result_df['anomaly_severity'] = result_df[score_cols].mean(axis=1)
            
            # 합의 이상치 (평균 점수가 50점 이상)
            result_df['consensus_anomaly'] = (result_df['anomaly_severity'] >= 50).astype(int)
        
        return result_df
    
    def analyze_anomalies(self, result_df: pd.DataFrame, model_type: str) -> Dict:
        """이상치 분석"""
        analysis = {}
        
        # 전체 통계
        total_records = len(result_df)
        analysis['total_records'] = total_records
        analysis['unique_users'] = result_df['User'].nunique()
        analysis['date_range'] = (result_df['Date'].min(), result_df['Date'].max())
        
        # 모델별 이상치 통계
        model_stats = {}
        for col in result_df.columns:
            if col.endswith('_is_anomaly') and not col.startswith('consensus'):
                model_name = col.replace('_is_anomaly', '')
                anomaly_count = result_df[col].sum()
                model_stats[model_name] = {
                    'anomaly_count': int(anomaly_count),
                    'anomaly_ratio': float(anomaly_count / total_records),
                    'normal_count': int(total_records - anomaly_count)
                }
        
        analysis['model_stats'] = model_stats
        
        # 합의 이상치
        if 'consensus_anomaly' in result_df.columns:
            consensus_count = result_df['consensus_anomaly'].sum()
            analysis['consensus_anomalies'] = {
                'count': int(consensus_count),
                'ratio': float(consensus_count / total_records)
            }
        
        # 사용자별 이상치
        user_anomalies = {}
        for user in result_df['User'].unique():
            user_data = result_df[result_df['User'] == user]
            user_anomalies[user] = {}
            
            for model_name in model_stats.keys():
                anomaly_col = f'{model_name}_is_anomaly'
                if anomaly_col in user_data.columns:
                    user_anomaly_count = user_data[anomaly_col].sum()
                    user_anomalies[user][model_name] = {
                        'count': int(user_anomaly_count),
                        'ratio': float(user_anomaly_count / len(user_data))
                    }
            
            if 'consensus_anomaly' in user_data.columns:
                consensus_count = user_data['consensus_anomaly'].sum()
                user_anomalies[user]['consensus'] = {
                    'count': int(consensus_count),
                    'ratio': float(consensus_count / len(user_data))
                }
        
        analysis['user_anomalies'] = user_anomalies
        
        # 날짜별 이상치
        date_anomalies = result_df.groupby('Date').agg({
            col: 'sum' for col in result_df.columns if col.endswith('_is_anomaly')
        }).to_dict()
        
        analysis['date_anomalies'] = date_anomalies
        
        return analysis
    
    def generate_anomaly_report(self, analysis: Dict, model_type: str) -> str:
        """이상치 분석 보고서 생성"""
        report = []
        report.append(f"# {model_type.upper()} 모델 이상치 탐지 결과 보고서")
        report.append(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 전체 개요
        report.append(f"\n## 전체 개요")
        report.append(f"- 총 레코드 수: {analysis['total_records']:,}")
        report.append(f"- 분석 대상 사용자: {analysis['unique_users']}명")
        report.append(f"- 분석 기간: {analysis['date_range'][0]} ~ {analysis['date_range'][1]}")
        
        # 모델별 결과
        report.append(f"\n## 모델별 이상치 탐지 결과")
        for model_name, stats in analysis['model_stats'].items():
            report.append(f"\n### {model_name.upper()}")
            report.append(f"- 이상치 탐지: {stats['anomaly_count']:,}건 ({stats['anomaly_ratio']:.4f})")
            report.append(f"- 정상 패턴: {stats['normal_count']:,}건")
        
        # 합의 결과
        if 'consensus_anomalies' in analysis:
            consensus = analysis['consensus_anomalies']
            report.append(f"\n### 합의 이상치 (두 모델 모두 탐지)")
            report.append(f"- 합의 이상치: {consensus['count']:,}건 ({consensus['ratio']:.4f})")
        
        # 사용자별 분석
        report.append(f"\n## 사용자별 이상치 분석")
        for user, user_stats in analysis['user_anomalies'].items():
            report.append(f"\n### 사용자 {user}")
            
            for model_name, model_stats in user_stats.items():
                if model_name != 'consensus':
                    report.append(f"- {model_name}: {model_stats['count']}건 ({model_stats['ratio']:.4f})")
            
            if 'consensus' in user_stats:
                consensus_stats = user_stats['consensus']
                report.append(f"- 합의 이상치: {consensus_stats['count']}건 ({consensus_stats['ratio']:.4f})")
        
        return '\n'.join(report)
    
    def save_results(self, result_df: pd.DataFrame, analysis: Dict, model_type: str):
        """결과는 데이터베이스에만 저장 - 파일 저장 제거"""
        print(f"{model_type} 모델 결과 분석 완료: {len(result_df)}건")
    
    def generate_user_summary_report(self, day_results: pd.DataFrame, night_results: pd.DataFrame):
        """사용자별 종합 이상치 탐지 결과 보고서 생성"""
        # Day와 Night 결과 합치기
        all_results = pd.concat([day_results, night_results], ignore_index=True)
        
        # 사용자별 집계
        user_summary = {}
        
        for user in all_results['User'].unique():
            user_data = all_results[all_results['User'] == user]
            
            isolation_forest_count = user_data['isolation_forest_is_anomaly'].sum()
            one_class_svm_count = user_data['one_class_svm_is_anomaly'].sum()
            consensus_count = user_data['consensus_anomaly'].sum()
            total_records = len(user_data)
            
            user_summary[user] = {
                'isolation_forest': {
                    'count': int(isolation_forest_count),
                    'ratio': float(isolation_forest_count / total_records)
                },
                'one_class_svm': {
                    'count': int(one_class_svm_count),
                    'ratio': float(one_class_svm_count / total_records)
                },
                'consensus': {
                    'count': int(consensus_count),
                    'ratio': float(consensus_count / total_records)
                },
                'total_records': int(total_records)
            }
        
        # 보고서 생성
        report = []
        report.append("# 사용자별 이상치 탐지 결과 종합 보고서")
        report.append(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n분석 대상: {len(user_summary)}명")
        report.append(f"전체 레코드: {len(all_results)}건")
        
        report.append(f"\n## 사용자별 상세 결과")
        
        for user in sorted(user_summary.keys()):
            stats = user_summary[user]
            report.append(f"\n### 사용자 {user}")
            report.append(f"- isolation_forest: {stats['isolation_forest']['count']}건 ({stats['isolation_forest']['ratio']:.4f})")
            report.append(f"- one_class_svm: {stats['one_class_svm']['count']}건 ({stats['one_class_svm']['ratio']:.4f})")
            report.append(f"- 합의 이상치: {stats['consensus']['count']}건 ({stats['consensus']['ratio']:.4f})")
            report.append(f"- 총 레코드: {stats['total_records']}건")
        
        # 전체 요약
        total_isolation = sum([stats['isolation_forest']['count'] for stats in user_summary.values()])
        total_svm = sum([stats['one_class_svm']['count'] for stats in user_summary.values()])
        total_consensus = sum([stats['consensus']['count'] for stats in user_summary.values()])
        total_records = len(all_results)
        
        report.append(f"\n## 전체 요약")
        report.append(f"- isolation_forest: {total_isolation}건 ({total_isolation/total_records:.4f})")
        report.append(f"- one_class_svm: {total_svm}건 ({total_svm/total_records:.4f})")
        report.append(f"- 합의 이상치: {total_consensus}건 ({total_consensus/total_records:.4f})")
        
        # 보고서 저장
        results_dir = os.path.join("../../reports")
        os.makedirs(results_dir, exist_ok=True)
        
        report_path = os.path.join(results_dir, "user_anomaly_summary_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"사용자별 종합 보고서 저장: {report_path}")
        return report_path

def main():
    """메인 실행 함수"""
    # 데이터베이스 연결 (환경변수 사용)
    loader = RealDataLoader()
    
    if not loader.connect():
        print("데이터베이스 연결 실패")
        return
    
    try:
        # 모델 평가기 초기화
        evaluator = RealModelEvaluator()
        
        # 테스트 데이터 로드 (2025-05-31 ~ 2025-06-05)
        print("=== 실제 데이터 기반 모델 평가 시작 ===")
        test_data = loader.load_all_tables('2025-05-31', '2025-06-05')
        
        if not test_data:
            print("테스트 데이터 로드 실패")
            return
        
        # 특성 병합
        day_test, night_test = loader.merge_features_for_training(test_data)
        day_columns, night_columns = loader.get_feature_columns()
        
        # Day 모델 평가
        print("\n" + "="*50)
        print("DAY 모델 평가")
        print("="*50)
        
        day_predictions = evaluator.predict_anomalies(day_test, day_columns, 'day')
        if day_predictions:
            day_results = evaluator.create_detailed_results(day_test, day_predictions, 'day')
            day_analysis = evaluator.analyze_anomalies(day_results, 'day')
            evaluator.save_results(day_results, day_analysis, 'day')
        
        # Night 모델 평가
        print("\n" + "="*50)
        print("NIGHT 모델 평가")
        print("="*50)
        
        night_predictions = evaluator.predict_anomalies(night_test, night_columns, 'night')
        if night_predictions:
            night_results = evaluator.create_detailed_results(night_test, night_predictions, 'night')
            night_analysis = evaluator.analyze_anomalies(night_results, 'night')
            evaluator.save_results(night_results, night_analysis, 'night')
        
        print("\n" + "="*50)
        print("모델 평가 완료")
        print("="*50)
        print(f"결과 저장 위치: {evaluator.model_dir}/evaluation_results")
        
    finally:
        loader.disconnect()

if __name__ == "__main__":
    main() 