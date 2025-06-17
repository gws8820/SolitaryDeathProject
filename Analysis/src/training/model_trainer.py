import pandas as pd
import numpy as np
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class ModelTrainer:
    """고독사 감지를 위한 비지도 학습 모델 훈련 클래스"""
    
    def __init__(self, processed_data_path: str = "dummy_data/processed", models_path: str = "dummy_models"):
        self.processed_data_path = Path(processed_data_path)
        self.models_path = Path(models_path)
        
        # 모델 저장 경로 생성
        self.day_models_path = self.models_path / "day"
        self.night_models_path = self.models_path / "night"
        
        self.day_models_path.mkdir(parents=True, exist_ok=True)
        self.night_models_path.mkdir(parents=True, exist_ok=True)
        
        # 특징 그룹 정의
        self.day_features = [
            'day_inactive_total', 'day_inactive_room', 'day_toggle_total',
            'day_toggle_room', 'day_on_ratio_room', 'day_kitchen_usage_rate'
        ]
        
        self.night_features = [
            'night_inactive_total', 'night_inactive_room', 'night_toggle_total',
            'night_toggle_room', 'night_on_ratio_room', 'night_kitchen_usage_rate',
            'night_bathroom_usage'
        ]
        
        # 모델 초기화
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            ),
            'one_class_svm': OneClassSVM(
                nu=0.1,
                kernel='rbf',
                gamma='scale'
            )
        }
        
        self.scalers = {}
        self.training_stats = {}
    
    def load_feature_dataset(self, dataset_name: str, feature_group: List[str]) -> pd.DataFrame:
        """특정 데이터셋의 특징 그룹을 로드하여 병합"""
        dataset_path = self.processed_data_path / dataset_name
        
        combined_data = None
        
        for feature in feature_group:
            feature_file = dataset_path / f"{feature}.csv"
            
            if not feature_file.exists():
                print(f"경고: {feature_file}가 존재하지 않습니다.")
                continue
            
            # 특징 데이터 로드
            feature_data = pd.read_csv(feature_file)
            
            if combined_data is None:
                # 첫 번째 특징: User, Date 컬럼 포함
                combined_data = feature_data.copy()
                # 특징 컬럼명 변경
                feature_cols = [col for col in feature_data.columns if col not in ['User', 'Date']]
                if len(feature_cols) == 1:
                    # 단일 값 특징
                    combined_data = combined_data.rename(columns={feature_cols[0]: feature})
                else:
                    # 다중 값 특징 (방별 데이터)
                    for i, col in enumerate(feature_cols):
                        combined_data = combined_data.rename(columns={col: f"{feature}_{col}"})
            else:
                # 추가 특징: User, Date로 병합
                feature_cols = [col for col in feature_data.columns if col not in ['User', 'Date']]
                
                if len(feature_cols) == 1:
                    # 단일 값 특징
                    merge_data = feature_data[['User', 'Date'] + feature_cols].copy()
                    merge_data = merge_data.rename(columns={feature_cols[0]: feature})
                else:
                    # 다중 값 특징 (방별 데이터)
                    merge_data = feature_data.copy()
                    for i, col in enumerate(feature_cols):
                        merge_data = merge_data.rename(columns={col: f"{feature}_{col}"})
                
                combined_data = pd.merge(combined_data, merge_data, on=['User', 'Date'], how='inner')
        
        print(f"{dataset_name} 데이터셋 로드 완료: {combined_data.shape}")
        print(f"특징 컬럼: {[col for col in combined_data.columns if col not in ['User', 'Date']]}")
        
        return combined_data
    
    def prepare_training_data(self, time_of_day: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """훈련 데이터 준비 및 전처리"""
        if time_of_day == 'day':
            features = self.day_features
        else:
            features = self.night_features
        
        # 훈련 데이터 로드
        train_data = self.load_feature_dataset('train_dataset', features)
        
        # 특징 컬럼만 추출
        feature_columns = [col for col in train_data.columns if col not in ['User', 'Date']]
        X_train = train_data[feature_columns].values
        
        # 데이터 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 스케일러 저장
        self.scalers[time_of_day] = scaler
        
        print(f"{time_of_day.upper()} 훈련 데이터 준비 완료")
        print(f"데이터 형태: {X_train_scaled.shape}")
        print(f"특징 수: {len(feature_columns)}")
        
        return X_train_scaled, train_data[feature_columns]
    
    def train_models(self, time_of_day: str):
        """특정 시간대(day/night)의 모든 모델 훈련"""
        print(f"\n=== {time_of_day.upper()} 모델 훈련 시작 ===")
        
        # 훈련 데이터 준비
        X_train, feature_df = self.prepare_training_data(time_of_day)
        
        trained_models = {}
        model_stats = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()} 훈련 중...")
            
            try:
                # Isolation Forest, One-Class SVM
                model.fit(X_train)
                
                # 이상치 예측
                predictions = model.predict(X_train)
                anomaly_ratio = np.sum(predictions == -1) / len(predictions)
                
                model_stats[model_name] = {
                    'anomaly_ratio': anomaly_ratio,
                    'n_anomalies': np.sum(predictions == -1),
                    'n_normal': np.sum(predictions == 1)
                }
                
                print(f"이상치 비율: {anomaly_ratio:.3f}")
                
                trained_models[model_name] = model
                print(f"{model_name.upper()} 훈련 완료")
                
            except Exception as e:
                print(f"{model_name.upper()} 훈련 중 오류 발생: {str(e)}")
                continue
        
        # 모델 및 스케일러 저장
        self.save_models(time_of_day, trained_models)
        
        # 통계 저장
        self.training_stats[time_of_day] = {
            'feature_count': X_train.shape[1],
            'sample_count': X_train.shape[0],
            'feature_names': feature_df.columns.tolist(),
            'model_stats': model_stats
        }
        
        print(f"\n{time_of_day.upper()} 모델 훈련 완료")
        return trained_models
    
    def save_models(self, time_of_day: str, models: Dict):
        """훈련된 모델들을 저장"""
        save_path = self.day_models_path if time_of_day == 'day' else self.night_models_path
        
        # 각 모델 저장
        for model_name, model in models.items():
            model_file = save_path / f"{model_name}.pkl"
            joblib.dump(model, model_file)
            print(f"{model_name} 모델 저장: {model_file}")
        
        # 스케일러 저장
        scaler_file = save_path / "scaler.pkl"
        joblib.dump(self.scalers[time_of_day], scaler_file)
        print(f"스케일러 저장: {scaler_file}")
    
    def train_all_models(self):
        """모든 시간대의 모든 모델 훈련"""
        print("=== 고독사 감지 모델 훈련 시작 ===")
        
        # Day 모델 훈련
        day_models = self.train_models('day')
        
        # Night 모델 훈련  
        night_models = self.train_models('night')
        
        # 훈련 통계를 각 폴더에 저장
        day_stats_file = self.day_models_path / "training_stats.pkl"
        night_stats_file = self.night_models_path / "training_stats.pkl"
        
        with open(day_stats_file, 'wb') as f:
            pickle.dump(self.training_stats['day'], f)
        with open(night_stats_file, 'wb') as f:
            pickle.dump(self.training_stats['night'], f)
        
        print(f"\n=== 모든 모델 훈련 완료 ===")
        print(f"Day 통계 파일 저장: {day_stats_file}")
        print(f"Night 통계 파일 저장: {night_stats_file}")
        
        return {
            'day': day_models,
            'night': night_models
        }
    
    def print_training_summary(self):
        """훈련 결과 요약 출력"""
        print("\n=== 훈련 결과 요약 ===")
        
        for time_of_day, stats in self.training_stats.items():
            print(f"\n{time_of_day.upper()} 모델:")
            print(f"  특징 수: {stats['feature_count']}")
            print(f"  샘플 수: {stats['sample_count']}")
            print(f"  특징명: {stats['feature_names']}")
            
            for model_name, model_stats in stats['model_stats'].items():
                print(f"\n  {model_name.upper()}:")
                for key, value in model_stats.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.3f}")
                    else:
                        print(f"    {key}: {value}")

def main():
    """메인 실행 함수"""
    print("고독사 감지 모델 훈련을 시작합니다...")
    
    # 트레이너 초기화
    trainer = ModelTrainer()
    
    # 모든 모델 훈련
    models = trainer.train_all_models()
    
    # 결과 요약
    trainer.print_training_summary()
    
    print("\n모델 훈련이 완료되었습니다!")

if __name__ == "__main__":
    main() 