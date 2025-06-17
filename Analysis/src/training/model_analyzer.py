import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score

warnings.filterwarnings('ignore')

class ModelAnalyzer:
    """훈련된 모델들의 차원 특성 분석 클래스"""
    
    def __init__(self, models_path: str = "dummy_models", charts_path: str = "charts"):
        self.models_path = Path(models_path)
        self.charts_path = Path(charts_path)
        self.charts_path.mkdir(exist_ok=True)
        
        # 모델 및 통계 로드
        self.load_training_stats()
        self.load_models()
        
    def load_training_stats(self):
        """훈련 통계 로드"""
        day_stats_file = self.models_path / "day" / "training_stats.pkl"
        night_stats_file = self.models_path / "night" / "training_stats.pkl"
        
        with open(day_stats_file, 'rb') as f:
            day_stats = pickle.load(f)
        with open(night_stats_file, 'rb') as f:
            night_stats = pickle.load(f)
            
        self.training_stats = {
            'day': day_stats,
            'night': night_stats
        }
        print("훈련 통계 로드 완료")
    
    def load_models(self):
        """모든 모델과 스케일러 로드"""
        self.models = {'day': {}, 'night': {}}
        self.scalers = {}
        
        for time_of_day in ['day', 'night']:
            model_path = self.models_path / time_of_day
            
            # 모델 로드
            for model_file in model_path.glob("*.pkl"):
                if model_file.name != "scaler.pkl":
                    model_name = model_file.stem
                    self.models[time_of_day][model_name] = joblib.load(model_file)
                    print(f"{time_of_day} {model_name} 모델 로드 완료")
            
            # 스케일러 로드
            scaler_file = model_path / "scaler.pkl"
            self.scalers[time_of_day] = joblib.load(scaler_file)
            print(f"{time_of_day} 스케일러 로드 완료")
    
    def analyze_feature_importance(self, time_of_day: str) -> Dict:
        """특징 중요도 분석"""
        stats = self.training_stats[time_of_day]
        feature_names = stats['feature_names']
        
        analysis = {
            'feature_count': stats['feature_count'],
            'feature_names': feature_names,
            'feature_categories': self.categorize_features(feature_names),
            'dimensionality_insights': self.get_dimensionality_insights(stats['feature_count'])
        }
        
        return analysis
    
    def categorize_features(self, feature_names: List[str]) -> Dict:
        """특징을 카테고리별로 분류"""
        categories = {
            'inactive_features': [],
            'toggle_features': [],
            'ratio_features': [],
            'usage_features': []
        }
        
        for feature in feature_names:
            if 'inactive' in feature:
                categories['inactive_features'].append(feature)
            elif 'toggle' in feature:
                categories['toggle_features'].append(feature)
            elif 'ratio' in feature:
                categories['ratio_features'].append(feature)
            elif 'usage' in feature:
                categories['usage_features'].append(feature)
        
        return categories
    
    def get_dimensionality_insights(self, feature_count: int) -> Dict:
        """차원 수에 따른 인사이트"""
        insights = {
            'dimension_level': '',
            'curse_of_dimensionality_risk': '',
            'recommended_techniques': [],
            'sample_to_feature_ratio': 9000 / feature_count if feature_count > 0 else 0
        }
        
        if feature_count <= 5:
            insights['dimension_level'] = 'Low'
            insights['curse_of_dimensionality_risk'] = 'Very Low'
            insights['recommended_techniques'] = ['Direct visualization', 'Simple clustering']
        elif feature_count <= 10:
            insights['dimension_level'] = 'Medium'
            insights['curse_of_dimensionality_risk'] = 'Low'
            insights['recommended_techniques'] = ['PCA', 'Feature selection', 'Standard ML algorithms']
        elif feature_count <= 20:
            insights['dimension_level'] = 'High'
            insights['curse_of_dimensionality_risk'] = 'Medium'
            insights['recommended_techniques'] = ['PCA', 'Feature selection', 'Regularization', 'Ensemble methods']
        else:
            insights['dimension_level'] = 'Very High'
            insights['curse_of_dimensionality_risk'] = 'High'
            insights['recommended_techniques'] = ['Aggressive feature selection', 'Dimensionality reduction', 'Deep learning']
        
        return insights
    
    def perform_pca_analysis(self, time_of_day: str) -> Dict:
        """PCA 차원 축소 분석"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.training.model_trainer import ModelTrainer
        
        # 데이터 재로드
        trainer = ModelTrainer()
        if time_of_day == 'day':
            features = trainer.day_features
        else:
            features = trainer.night_features
        
        train_data = trainer.load_feature_dataset('train_dataset', features)
        feature_columns = [col for col in train_data.columns if col not in ['User', 'Date']]
        X = train_data[feature_columns].values
        
        # 데이터 표준화
        X_scaled = self.scalers[time_of_day].transform(X)
        
        # PCA 수행
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # 분산 비율 계산
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 95% 분산을 설명하는 컴포넌트 수
        n_components_95 = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1
        n_components_90 = np.where(cumulative_variance_ratio >= 0.90)[0][0] + 1
        
        analysis = {
            'original_dimensions': X_scaled.shape[1],
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
            'n_components_90_percent': int(n_components_90),
            'n_components_95_percent': int(n_components_95),
            'dimension_reduction_potential': {
                '90_percent': f"{(1 - n_components_90/X_scaled.shape[1])*100:.1f}%",
                '95_percent': f"{(1 - n_components_95/X_scaled.shape[1])*100:.1f}%"
            }
        }
        
        return analysis
    
    def analyze_model_characteristics(self, time_of_day: str) -> Dict:
        """모델별 특성 분석"""
        stats = self.training_stats[time_of_day]['model_stats']
        models = self.models[time_of_day]
        
        analysis = {}
        
        for model_name, model_stats in stats.items():
            model_analysis = {
                'algorithm_type': self.get_algorithm_type(model_name),
                'dimensionality_handling': self.get_dimensionality_handling(model_name),
                'performance_stats': model_stats,
                'strengths': self.get_model_strengths(model_name),
                'weaknesses': self.get_model_weaknesses(model_name)
            }
            
            analysis[model_name] = model_analysis
        
        return analysis
    
    def get_algorithm_type(self, model_name: str) -> str:
        """알고리즘 유형 반환"""
        types = {
            'isolation_forest': 'Tree-based Ensemble',
            'one_class_svm': 'Kernel-based'
        }
        return types.get(model_name, 'Unknown')
    
    def get_dimensionality_handling(self, model_name: str) -> str:
        """차원 처리 능력 반환"""
        handling = {
            'isolation_forest': 'Good - Tree structure naturally handles high dimensions',
            'one_class_svm': 'Moderate - Kernel trick helps but can struggle with very high dimensions'
        }
        return handling.get(model_name, 'Unknown')
    
    def get_model_strengths(self, model_name: str) -> List[str]:
        """모델별 강점 반환"""
        strengths = {
            'isolation_forest': [
                'Fast training and prediction',
                'Good performance with high-dimensional data',
                'No assumption about data distribution',
                'Built-in anomaly scoring'
            ],
            'one_class_svm': [
                'Strong theoretical foundation',
                'Flexible kernel functions',
                'Good generalization',
                'Robust to outliers in training data'
            ]
        }
        return strengths.get(model_name, [])
    
    def get_model_weaknesses(self, model_name: str) -> List[str]:
        """모델별 약점 반환"""
        weaknesses = {
            'isolation_forest': [
                'Random sampling may miss patterns',
                'Performance depends on contamination parameter',
                'Less interpretable than simpler methods'
            ],
            'one_class_svm': [
                'Computationally expensive for large datasets',
                'Sensitive to parameter tuning',
                'Memory intensive'
            ]
        }
        return weaknesses.get(model_name, [])
    
    def generate_comprehensive_analysis(self) -> Dict:
        """종합 분석 생성"""
        analysis = {}
        
        for time_of_day in ['day', 'night']:
            time_analysis = {
                'feature_analysis': self.analyze_feature_importance(time_of_day),
                'pca_analysis': self.perform_pca_analysis(time_of_day),
                'model_analysis': self.analyze_model_characteristics(time_of_day)
            }
            analysis[time_of_day] = time_analysis
        
        return analysis
    
    def create_visualization(self, time_of_day: str):
        """PCA 시각화 생성"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.training.model_trainer import ModelTrainer
        
        # 데이터 재로드
        trainer = ModelTrainer()
        if time_of_day == 'day':
            features = trainer.day_features
        else:
            features = trainer.night_features
        
        train_data = trainer.load_feature_dataset('train_dataset', features)
        feature_columns = [col for col in train_data.columns if col not in ['User', 'Date']]
        X = train_data[feature_columns].values
        
        # 데이터 표준화
        X_scaled = self.scalers[time_of_day].transform(X)
        
        # PCA 수행
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 분산 설명 비율 플롯
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    np.cumsum(pca.explained_variance_ratio_), 'bo-')
        axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        axes[0].axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance Ratio')
        axes[0].set_title(f'{time_of_day.upper()} - PCA Explained Variance')
        axes[0].legend()
        axes[0].grid(True)
        
        # 처음 두 주성분 산점도
        axes[1].scatter(X_pca[:1000, 0], X_pca[:1000, 1], alpha=0.6)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1].set_title(f'{time_of_day.upper()} - First Two Principal Components')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.charts_path / f'{time_of_day}_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{time_of_day.upper()} PCA 시각화 저장 완료")

def main():
    """메인 실행 함수"""
    print("모델 차원 특성 분석을 시작합니다...")
    
    # 분석기 초기화
    analyzer = ModelAnalyzer()
    
    # 종합 분석 수행
    analysis = analyzer.generate_comprehensive_analysis()
    
    # 시각화 생성
    for time_of_day in ['day', 'night']:
        analyzer.create_visualization(time_of_day)
    
    print("모델 차원 특성 분석이 완료되었습니다!")
    
    return analysis

if __name__ == "__main__":
    main() 