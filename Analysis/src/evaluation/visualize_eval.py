import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.evaluation.comprehensive_evaluation import ComprehensiveEvaluator

# 스타일 설정
plt.style.use('default')
sns.set_palette("husl")

# 출력 폴더 생성
os.makedirs('charts/anomaly_detection', exist_ok=True)

class DetailedAnomalyVisualizer:
    def __init__(self):
        self.evaluator = ComprehensiveEvaluator()
        self.datasets = [
            'immediate_abnormal_test_dataset',
            'rapid_abnormal_test_dataset', 
            'gradual_abnormal_test_dataset'
        ]
        self.dataset_labels = {
            'immediate_abnormal_test_dataset': 'Immediate Abnormal',
            'rapid_abnormal_test_dataset': 'Rapid Abnormal',
            'gradual_abnormal_test_dataset': 'Gradual Abnormal'
        }
        self.methods = ['baseline', 'isolation_forest', 'one_class_svm', 'dbscan']
        self.method_labels = {
            'baseline': 'Traditional Method\n(24h LED inactivity)',
            'isolation_forest': 'Isolation Forest',
            'one_class_svm': 'One-Class SVM',
            'dbscan': 'DBSCAN'
        }
        
    def run_evaluation_for_charts(self):
        """차트 생성을 위한 평가 실행"""
        results = {}
        
        for dataset in self.datasets:
            print(f"Evaluating {dataset}...")
            # 기존 메서드 활용
            dataset_results = self.evaluator.evaluate_dataset(dataset)
            
            if dataset_results:
                # 차트용 형식으로 변환
                formatted_results = {}
                for method, data in dataset_results.items():
                    if method == 'traditional':
                        method_key = 'baseline'
                    else:
                        method_key = method
                    
                    formatted_results[method_key] = {
                        'detection_rate': data['detection_rate_72h'],
                        'avg_detection_time': data['average_detection_time'],
                        'detection_times': data['detection_times']
                    }
                
                results[dataset] = formatted_results
            
        return results
        
    def calculate_false_positive_rates(self):
        """정상 데이터에 대한 오탐율 계산"""
        normal_day, normal_night = self.evaluator.load_test_data('normal_test_dataset')
        
        total_samples = 200  # 100명 * 2일간 평가
        false_positives = {
            'isolation_forest': 0,
            'one_class_svm': 0,
            'dbscan': 0
        }
        
        # 사용자별, 날짜별로 예측 수행
        for user_id in range(1, 11):  # 10명 체크
            for day_idx in range(10):  # 10일간 체크
                date_str = f"2024-01-{day_idx + 1:02d}"
                
                # Day 데이터 체크
                user_day_data = normal_day[
                    (normal_day['User'] == user_id) & 
                    (normal_day['Date'] == date_str)
                ]
                
                if not user_day_data.empty:
                    day_predictions = self.evaluator.predict_anomaly(user_day_data, 'day')
                    for method, detected in day_predictions.items():
                        if detected:
                            false_positives[method] += 1
                
                # Night 데이터 체크
                user_night_data = normal_night[
                    (normal_night['User'] == user_id) & 
                    (normal_night['Date'] == date_str)
                ]
                
                if not user_night_data.empty:
                    night_predictions = self.evaluator.predict_anomaly(user_night_data, 'night')
                    for method, detected in night_predictions.items():
                        if detected:
                            false_positives[method] += 1
        
        # 백분율로 변환
        fp_rates = {}
        for method, count in false_positives.items():
            fp_rates[method] = count / total_samples * 100
            
        return fp_rates
        
    def create_detection_rate_by_dataset(self, results):
        """데이터셋별 감지율 비교 차트"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        datasets = list(results.keys())
        methods = ['baseline', 'isolation_forest', 'one_class_svm']  # DBSCAN 제외
        
        x = np.arange(len(datasets))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, method in enumerate(methods):
            rates = [results[dataset][method]['detection_rate'] for dataset in datasets]
            bars = ax.bar(x + i*width, rates, width, label=self.method_labels[method], 
                         color=colors[i], alpha=0.8)
            
            # 값 표시
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.annotate(f'{rate:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Dataset Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('72-Hour Detection Rate by Dataset Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([self.dataset_labels[d] for d in datasets])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig('charts/anomaly_detection/detection_rate_by_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_detection_time_distribution(self, results):
        """감지 시간 분포 상세 분석"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, method in enumerate(['baseline', 'isolation_forest', 'one_class_svm']):
            ax = axes[idx]
            
            all_times = []
            labels = []
            
            for dataset in self.datasets:
                times = results[dataset][method]['detection_times']
                if times:  # 감지된 경우만
                    all_times.extend(times)
                    labels.extend([self.dataset_labels[dataset]] * len(times))
            
            if all_times:
                # 박스플롯
                data_by_dataset = []
                dataset_names = []
                for dataset in self.datasets:
                    times = results[dataset][method]['detection_times']
                    if times:
                        data_by_dataset.append(times)
                        dataset_names.append(self.dataset_labels[dataset])
                
                if data_by_dataset:
                    bp = ax.boxplot(data_by_dataset, labels=dataset_names, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor(colors[idx])
                        patch.set_alpha(0.7)
            
            ax.set_title(f'{self.method_labels[method]}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Detection Time (hours)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # 네 번째 subplot 제거
        axes[3].remove()
        
        plt.suptitle('Detection Time Distribution by Method and Dataset', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('charts/anomaly_detection/detection_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_performance_summary_heatmap(self, results):
        """성능 요약 히트맵"""
        # 데이터 준비
        methods = ['baseline', 'isolation_forest', 'one_class_svm']
        datasets = self.datasets
        
        # 감지율 데이터
        detection_data = []
        for method in methods:
            row = []
            for dataset in datasets:
                rate = results[dataset][method]['detection_rate']
                row.append(rate)
            detection_data.append(row)
        
        # 평균 감지 시간 데이터
        time_data = []
        for method in methods:
            row = []
            for dataset in datasets:
                time = results[dataset][method]['avg_detection_time']
                row.append(time if time > 0 else 0)
            time_data.append(row)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 감지율 히트맵
        detection_df = pd.DataFrame(detection_data, 
                                   index=[self.method_labels[m].replace('\n', ' ') for m in methods],
                                   columns=[self.dataset_labels[d] for d in datasets])
        
        sns.heatmap(detection_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   ax=ax1, cbar_kws={'label': 'Detection Rate (%)'})
        ax1.set_title('Detection Rate by Method and Dataset (%)', fontweight='bold')
        
        # 평균 감지 시간 히트맵
        time_df = pd.DataFrame(time_data,
                              index=[self.method_labels[m].replace('\n', ' ') for m in methods],
                              columns=[self.dataset_labels[d] for d in datasets])
        
        sns.heatmap(time_df, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   ax=ax2, cbar_kws={'label': 'Average Detection Time (hours)'})
        ax2.set_title('Average Detection Time by Method and Dataset (hours)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('charts/anomaly_detection/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_false_positive_analysis(self):
        """오탐율 분석 차트"""
        fp_rates = self.calculate_false_positive_rates()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(fp_rates.keys())
        rates = list(fp_rates.values())
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax.bar(methods, rates, color=colors, alpha=0.8)
        
        # 값 표시
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.annotate(f'{rate:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Machine Learning Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('False Positive Rate Analysis on Normal Data', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(rates) * 1.2)
        
        # 방법명 정리
        ax.set_xticklabels(['Isolation\nForest', 'One-Class\nSVM', 'DBSCAN'])
        
        plt.tight_layout()
        plt.savefig('charts/anomaly_detection/false_positive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_comprehensive_comparison(self, results):
        """종합 비교 차트"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = ['baseline', 'isolation_forest', 'one_class_svm']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. 전체 평균 감지율
        avg_rates = []
        for method in methods:
            rates = [results[dataset][method]['detection_rate'] for dataset in self.datasets]
            avg_rates.append(np.mean(rates))
        
        bars1 = ax1.bar(methods, avg_rates, color=colors, alpha=0.8)
        for bar, rate in zip(bars1, avg_rates):
            height = bar.get_height()
            ax1.annotate(f'{rate:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Average Detection Rate Across All Datasets', fontweight='bold')
        ax1.set_ylabel('Detection Rate (%)')
        ax1.set_xticklabels(['Traditional\nMethod', 'Isolation\nForest', 'One-Class\nSVM'])
        ax1.grid(True, alpha=0.3)
        
        # 2. 전체 평균 감지 시간
        avg_times = []
        for method in methods:
            times = [results[dataset][method]['avg_detection_time'] for dataset in self.datasets]
            avg_times.append(np.mean([t for t in times if t > 0]))
        
        bars2 = ax2.bar(methods, avg_times, color=colors, alpha=0.8)
        for bar, time in zip(bars2, avg_times):
            height = bar.get_height()
            ax2.annotate(f'{time:.1f}h',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Average Detection Time Across All Datasets', fontweight='bold')
        ax2.set_ylabel('Detection Time (hours)')
        ax2.set_xticklabels(['Traditional\nMethod', 'Isolation\nForest', 'One-Class\nSVM'])
        ax2.grid(True, alpha=0.3)
        
        # 3. 데이터셋별 성능 비교 (레이더 차트)
        from math import pi
        
        categories = [self.dataset_labels[d] for d in self.datasets]
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax3 = plt.subplot(223, projection='polar')
        
        for i, method in enumerate(methods):
            values = [results[dataset][method]['detection_rate'] for dataset in self.datasets]
            values += values[:1]
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=self.method_labels[method].replace('\n', ' '), color=colors[i])
            ax3.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 100)
        ax3.set_title('Detection Rate by Dataset Type', fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. 감지 시간 효율성 비교
        datasets_short = ['Immediate', 'Rapid', 'Gradual']
        x = np.arange(len(datasets_short))
        width = 0.25
        
        for i, method in enumerate(methods):
            times = [results[dataset][method]['avg_detection_time'] for dataset in self.datasets]
            bars = ax4.bar(x + i*width, times, width, label=self.method_labels[method].replace('\n', ' '), 
                          color=colors[i], alpha=0.8)
            
            # 값 표시
            for bar, time in zip(bars, times):
                if time > 0:
                    height = bar.get_height()
                    ax4.annotate(f'{time:.1f}h',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        ax4.set_xlabel('Dataset Type')
        ax4.set_ylabel('Average Detection Time (hours)')
        ax4.set_title('Detection Time Efficiency by Dataset', fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(datasets_short)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('charts/anomaly_detection/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_charts(self):
        """모든 차트 생성"""
        print("Starting evaluation for chart generation...")
        results = self.run_evaluation_for_charts()
        
        print("Creating detection rate by dataset chart...")
        self.create_detection_rate_by_dataset(results)
        
        print("Creating detection time distribution chart...")
        self.create_detection_time_distribution(results)
        
        print("Creating performance heatmap...")
        self.create_performance_summary_heatmap(results)
        
        print("Creating false positive analysis...")
        self.create_false_positive_analysis()
        
        print("Creating comprehensive comparison...")
        self.create_comprehensive_comparison(results)
        
        print("All charts have been generated in charts/anomaly_detection/")
        
        return results

if __name__ == "__main__":
    visualizer = DetailedAnomalyVisualizer()
    results = visualizer.generate_all_charts() 