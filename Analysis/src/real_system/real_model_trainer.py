import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
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

class RealModelTrainer:
    """실제 데이터로 모델을 훈련하는 클래스"""
    
    def __init__(self, save_dir: str = "../../real_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델 설정
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1
            )
        }
        
        self.scalers = {}
        self.trained_models = {}
        
    def prepare_training_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        """훈련 데이터 준비"""
        # 메타데이터 저장 (User, Date)
        metadata = df[['User', 'Date']].copy()
        
        # 특성 데이터 추출
        X = df[feature_columns].values
        
        # 결측값 처리
        X = np.nan_to_num(X, nan=0.0)
        
        print(f"훈련 데이터 크기: {X.shape}")
        print(f"특성 개수: {len(feature_columns)}")
        
        return X, metadata
    
    def train_models(self, X: np.ndarray, model_type: str) -> Dict:
        """모델 훈련"""
        print(f"\n=== {model_type.upper()} 모델 훈련 시작 ===")
        
        # 데이터 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        training_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} 훈련 중...")
            
            # 모델 훈련
            start_time = datetime.now()
            model.fit(X_scaled)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 훈련 데이터에 대한 예측
            train_predictions = model.predict(X_scaled)
            train_scores = model.decision_function(X_scaled)
            
            # 이상치 비율 계산
            anomaly_ratio = (train_predictions == -1).mean()
            
            results = {
                'model': model,
                'training_time': training_time,
                'anomaly_ratio': anomaly_ratio,
                'train_scores': train_scores,
                'train_predictions': train_predictions
            }
            
            training_results[model_name] = results
            
            print(f"  - 훈련 시간: {training_time:.2f}초")
            print(f"  - 이상치 비율: {anomaly_ratio:.4f}")
            print(f"  - 점수 범위: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
        
        # 스케일러 저장
        self.scalers[model_type] = scaler
        self.trained_models[model_type] = training_results
        
        return training_results
    
    def save_models(self, model_type: str):
        """모델 저장"""
        model_dir = os.path.join(self.save_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # 스케일러 저장
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scalers[model_type], scaler_path)
        print(f"스케일러 저장: {scaler_path}")
        
        # 각 모델 저장
        for model_name, results in self.trained_models[model_type].items():
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            joblib.dump(results['model'], model_path)
            print(f"모델 저장: {model_path}")
        
        # 훈련 결과 저장
        stats = {}
        for model_name, results in self.trained_models[model_type].items():
            stats[model_name] = {
                'training_time': results['training_time'],
                'anomaly_ratio': results['anomaly_ratio'],
                'score_stats': {
                    'mean': float(results['train_scores'].mean()),
                    'std': float(results['train_scores'].std()),
                    'min': float(results['train_scores'].min()),
                    'max': float(results['train_scores'].max())
                }
            }
        
        stats_path = os.path.join(model_dir, 'training_stats.pkl')
        joblib.dump(stats, stats_path)
        print(f"훈련 통계 저장: {stats_path}")
    
    def analyze_feature_importance(self, X: np.ndarray, feature_columns: List[str], model_type: str):
        """특성 중요도 분석"""
        print(f"\n=== {model_type.upper()} 특성 분석 ===")
        
        # 특성 통계
        feature_stats = pd.DataFrame({
            'feature': feature_columns,
            'mean': X.mean(axis=0),
            'std': X.std(axis=0),
            'min': X.min(axis=0),
            'max': X.max(axis=0)
        })
        
        print("특성 기본 통계:")
        print(feature_stats.round(4))
        
        # 특성간 상관관계 (상위 5개만)
        corr_matrix = pd.DataFrame(X, columns=feature_columns).corr()
        high_corr = []
        
        for i in range(len(feature_columns)):
            for j in range(i+1, len(feature_columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.7:  # 높은 상관관계
                    high_corr.append({
                        'feature1': feature_columns[i],
                        'feature2': feature_columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr:
            print(f"\n높은 상관관계 특성 쌍 (>0.7): {len(high_corr)}개")
            for pair in sorted(high_corr, key=lambda x: x['correlation'], reverse=True)[:5]:
                print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.4f}")
        else:
            print("\n높은 상관관계 특성 쌍 없음")
        
        return feature_stats, high_corr

def main():
    """메인 실행 함수"""
    # 데이터베이스 연결 및 데이터 로드 (환경변수 사용)
    loader = RealDataLoader()
    
    if not loader.connect():
        print("데이터베이스 연결 실패")
        return
    
    try:
        # 훈련 데이터 로드
        print("=== 실제 데이터 기반 모델 훈련 시작 ===")
        train_data = loader.load_all_tables('2025-04-12', '2025-05-31')
        
        if not train_data:
            print("훈련 데이터 로드 실패")
            return
        
        # 특성 병합
        day_train, night_train = loader.merge_features_for_training(train_data)
        day_columns, night_columns = loader.get_feature_columns()
        
        # 모델 트레이너 초기화
        trainer = RealModelTrainer()
        
        # Day 모델 훈련
        print("\n" + "="*50)
        print("DAY 모델 훈련")
        print("="*50)
        
        X_day, day_meta = trainer.prepare_training_data(day_train, day_columns)
        
        # 특성 분석
        day_stats, day_corr = trainer.analyze_feature_importance(X_day, day_columns, 'day')
        
        # Day 모델 훈련
        day_results = trainer.train_models(X_day, 'day')
        trainer.save_models('day')
        
        # Night 모델 훈련
        print("\n" + "="*50)
        print("NIGHT 모델 훈련")
        print("="*50)
        
        X_night, night_meta = trainer.prepare_training_data(night_train, night_columns)
        
        # 특성 분석
        night_stats, night_corr = trainer.analyze_feature_importance(X_night, night_columns, 'night')
        
        # Night 모델 훈련
        night_results = trainer.train_models(X_night, 'night')
        trainer.save_models('night')
        
        print("\n" + "="*50)
        print("모델 훈련 완료")
        print("="*50)
        print(f"Day 모델: {len(day_columns)}개 특성, {X_day.shape[0]}개 샘플")
        print(f"Night 모델: {len(night_columns)}개 특성, {X_night.shape[0]}개 샘플")
        print(f"저장 위치: {trainer.save_dir}")
        
    finally:
        loader.disconnect()

if __name__ == "__main__":
    main() 