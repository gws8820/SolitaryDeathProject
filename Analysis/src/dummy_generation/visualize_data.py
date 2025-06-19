import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class DataVisualizer:
    def __init__(self):
        self.led_positions = {
            '01': 'Bedroom',
            '02': 'Living Room', 
            '03': 'Kitchen',
            '04': 'Bathroom'
        }
        
        # 그래프 설정
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 한글 폰트 설정 (matplotlib에서 한글 깨짐 방지)
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_dataset(self, filename):
        """데이터셋을 로드합니다."""
        filepath = f'../../dummy_data/raw/{filename}'
        return pd.read_csv(filepath)
    
    def calculate_hourly_toggle_and_ontime(self, df):
        """시간대별 토글 횟수와 ON 시간을 계산합니다."""
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['hour'] = df['Timestamp'].dt.hour
        df['date'] = df['Timestamp'].dt.date
        
        # 각 사용자별, 날짜별, 시간별, LED별 토글 횟수 계산
        toggle_data = []
        ontime_data = []
        
        for user_id in df['User'].unique():
            user_data = df[df['User'] == user_id]
            
            for date in user_data['date'].unique():
                date_data = user_data[user_data['date'] == date]
                
                for hour in range(24):
                    hour_data = date_data[date_data['hour'] == hour].sort_values('Timestamp')
                    
                    for led_col in ['01', '02', '03', '04']:
                        if len(hour_data) > 1:
                            # 토글 횟수 계산
                            led_series = hour_data[led_col]
                            led_series_shifted = led_series.shift(1)
                            toggles = len(led_series[led_series != led_series_shifted]) - 1
                            toggles = max(0, toggles)
                        else:
                            toggles = 0
                        
                        # ON 시간 계산 (10분 단위)
                        on_time = len(hour_data[hour_data[led_col] == 1]) * 10
                        
                        toggle_data.append({
                            'hour': hour,
                            'led_id': led_col,
                            'led_position': self.led_positions[led_col],
                            'toggle_count': toggles
                        })
                        
                        ontime_data.append({
                            'hour': hour,
                            'led_id': led_col,
                            'led_position': self.led_positions[led_col],
                            'on_time': on_time
                        })
        
        toggle_df = pd.DataFrame(toggle_data)
        ontime_df = pd.DataFrame(ontime_data)
        
        # 시간대별 평균 계산
        hourly_toggle = toggle_df.groupby(['hour', 'led_id', 'led_position']).agg({
            'toggle_count': 'mean'
        }).reset_index()
        
        hourly_ontime = ontime_df.groupby(['hour', 'led_id', 'led_position']).agg({
            'on_time': 'mean'
        }).reset_index()
        
        return hourly_toggle, hourly_ontime
    
    def calculate_overall_stats(self, df):
        """전체 통계를 계산합니다."""
        # 각 사용자별 총 토글 횟수와 ON 시간 계산
        user_stats = []
        
        for user_id in df['User'].unique():
            user_data = df[df['User'] == user_id].sort_values('Timestamp')
            
            total_toggles = 0
            total_ontime = 0
            
            for led_col in ['01', '02', '03', '04']:
                # 토글 횟수 계산
                if len(user_data) > 1:
                    led_series = user_data[led_col]
                    led_series_shifted = led_series.shift(1)
                    toggles = len(led_series[led_series != led_series_shifted]) - 1
                    toggles = max(0, toggles)
                else:
                    toggles = 0
                
                # ON 시간 계산
                on_time = len(user_data[user_data[led_col] == 1]) * 10
                
                total_toggles += toggles
                total_ontime += on_time
            
            user_stats.append({
                'user_id': user_id,
                'total_toggle_count': total_toggles,
                'total_on_time': total_ontime
            })
        
        stats_df = pd.DataFrame(user_stats)
        return {
            'avg_toggle_count': stats_df['total_toggle_count'].mean(),
            'avg_on_time': stats_df['total_on_time'].mean()
        }
    
    def create_hourly_heatmap(self, hourly_data, title, value_column, output_filename, dataset_folder):
        """시간대별 히트맵을 생성합니다."""
        # 피벗 테이블 생성 (시간 x LED 위치)
        pivot_data = hourly_data.pivot_table(
            index='led_position', 
            columns='hour', 
            values=value_column,
            fill_value=0
        )
        
        # 히트맵 생성
        plt.figure(figsize=(16, 6))
        sns.heatmap(pivot_data, 
                    annot=True, 
                    fmt='.1f',
                    cmap='YlOrRd', 
                    cbar_kws={'label': value_column.replace('_', ' ').title()})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('LED Position', fontsize=12)
        plt.tight_layout()
        
        # 저장
        output_path = f'../../charts/dummy_data/{dataset_folder}/{output_filename}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_hourly_line_plot(self, toggle_data, ontime_data, title, output_filename, dataset_folder):
        """시간대별 토글 횟수와 ON 시간 선 그래프를 생성합니다."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 토글 횟수 그래프
        for led_id in ['01', '02', '03', '04']:
            led_data = toggle_data[toggle_data['led_id'] == led_id]
            ax1.plot(led_data['hour'], led_data['toggle_count'], 
                    marker='o', linewidth=2, markersize=6,
                    label=self.led_positions[led_id])
        
        ax1.set_title(f'{title} - Toggle Count by Hour', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Average Toggle Count', fontsize=12)
        ax1.legend(title='LED Position', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xlim(0, 23)
        
        # ON 시간 그래프
        for led_id in ['01', '02', '03', '04']:
            led_data = ontime_data[ontime_data['led_id'] == led_id]
            ax2.plot(led_data['hour'], led_data['on_time'], 
                    marker='s', linewidth=2, markersize=6,
                    label=self.led_positions[led_id])
        
        ax2.set_title(f'{title} - ON Time by Hour', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Average ON Time (minutes)', fontsize=12)
        ax2.legend(title='LED Position', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_xlim(0, 23)
        
        plt.tight_layout()
        
        # 저장
        output_path = f'../../charts/dummy_data/{dataset_folder}/{output_filename}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_comparison_heatmaps(self, all_data, value_column, title_prefix):
        """모든 데이터셋을 비교하는 히트맵을 생성합니다."""
        # Normal Train 제외하고 비교
        comparison_data = {k: v for k, v in all_data.items() if k != 'Normal Train'}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        # 색상 범위 통일을 위해 전체 데이터의 min/max 계산
        all_values = []
        for data in comparison_data.values():
            pivot_data = data.pivot_table(
                index='led_position', 
                columns='hour', 
                values=value_column,
                fill_value=0
            )
            all_values.extend(pivot_data.values.flatten())
        
        vmin, vmax = min(all_values), max(all_values)
        
        for i, (dataset_name, data) in enumerate(comparison_data.items()):
            if i < len(axes):
                pivot_data = data.pivot_table(
                    index='led_position', 
                    columns='hour', 
                    values=value_column,
                    fill_value=0
                )
                
                sns.heatmap(pivot_data, 
                           annot=False, 
                           cmap='YlOrRd',
                           vmin=vmin,
                           vmax=vmax,
                           ax=axes[i],
                           cbar=i==0,
                           cbar_kws={'label': value_column.replace('_', ' ').title()})
                
                axes[i].set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Hour of Day', fontsize=10)
                axes[i].set_ylabel('LED Position', fontsize=10)
        
        plt.suptitle(f'{title_prefix} - Dataset Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # comparison 폴더에 저장
        os.makedirs('../../charts/dummy_data/comparison', exist_ok=True)
        output_path = f'../../charts/dummy_data/comparison/{value_column}_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_statistics_comparison(self, all_stats):
        """모든 데이터셋의 통계를 비교하는 표를 생성합니다."""
        # Normal Train 제외
        comparison_stats = {k: v for k, v in all_stats.items() if k != 'Normal Train'}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 데이터 준비
        datasets = list(comparison_stats.keys())
        toggle_counts = [comparison_stats[ds]['avg_toggle_count'] for ds in datasets]
        on_times = [comparison_stats[ds]['avg_on_time'] for ds in datasets]
        
        # 색상 설정 (정상: 파란색, 비정상: 빨간색 계열)
        colors = ['#A23B72', '#F18F01', '#C73E1D', '#592E83']
        
        # 토글 횟수 비교
        bars1 = ax1.bar(range(len(datasets)), toggle_counts, color=colors)
        ax1.set_title('Average Toggle Count Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Dataset', fontsize=12)
        ax1.set_ylabel('Average Toggle Count', fontsize=12)
        ax1.set_xticks(range(len(datasets)))
        ax1.set_xticklabels([ds.replace(' ', '\n') for ds in datasets], fontsize=10)
        
        # 값 표시
        for bar, value in zip(bars1, toggle_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # ON 시간 비교
        bars2 = ax2.bar(range(len(datasets)), on_times, color=colors)
        ax2.set_title('Average ON Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dataset', fontsize=12)
        ax2.set_ylabel('Average ON Time (minutes)', fontsize=12)
        ax2.set_xticks(range(len(datasets)))
        ax2.set_xticklabels([ds.replace(' ', '\n') for ds in datasets], fontsize=10)
        
        # 값 표시
        for bar, value in zip(bars2, on_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # comparison 폴더에 저장
        os.makedirs('../../charts/dummy_data/comparison', exist_ok=True)
        output_path = '../../charts/dummy_data/comparison/statistics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_pattern_analysis(self, all_toggle_data):
        """정상 vs 비정상 패턴 분석 차트를 생성합니다."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Normal Test와 각 비정상 데이터 비교
        normal_data = all_toggle_data['Normal Test']
        normal_pivot = normal_data.groupby('hour')['toggle_count'].mean()
        
        abnormal_types = ['Immediate Abnormal', 'Rapid Abnormal', 'Gradual Abnormal']
        colors = ['#F18F01', '#C73E1D', '#592E83']
        
        for i, (abnormal_type, color) in enumerate(zip(abnormal_types, colors)):
            row = i // 2
            col = i % 2
            
            abnormal_data = all_toggle_data[abnormal_type]
            abnormal_pivot = abnormal_data.groupby('hour')['toggle_count'].mean()
            
            axes[row, col].plot(normal_pivot.index, normal_pivot.values, 
                              marker='o', linewidth=3, markersize=8, 
                              color='#2E86AB', label='Normal Pattern')
            axes[row, col].plot(abnormal_pivot.index, abnormal_pivot.values, 
                              marker='s', linewidth=3, markersize=8, 
                              color=color, label=f'{abnormal_type}')
            
            axes[row, col].set_title(f'Normal vs {abnormal_type}', 
                                   fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel('Hour', fontsize=10)
            axes[row, col].set_ylabel('Average Toggle Count', fontsize=10)
            axes[row, col].legend(fontsize=10)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlim(0, 23)
        
        # 전체 비교 (4번째 subplot)
        axes[1, 1].plot(normal_pivot.index, normal_pivot.values, 
                       marker='o', linewidth=3, markersize=8, 
                       color='#2E86AB', label='Normal Pattern')
        
        for abnormal_type, color in zip(abnormal_types, colors):
            abnormal_data = all_toggle_data[abnormal_type]
            abnormal_pivot = abnormal_data.groupby('hour')['toggle_count'].mean()
            axes[1, 1].plot(abnormal_pivot.index, abnormal_pivot.values, 
                           marker='s', linewidth=2, markersize=6, 
                           color=color, label=abnormal_type, alpha=0.8)
        
        axes[1, 1].set_title('Overall Pattern Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Hour', fontsize=10)
        axes[1, 1].set_ylabel('Average Toggle Count', fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 23)
        
        plt.tight_layout()
        
        # comparison 폴더에 저장
        os.makedirs('../../charts/dummy_data/comparison', exist_ok=True)
        output_path = '../../charts/dummy_data/comparison/pattern_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_room_usage_comparison(self, all_toggle_data):
        """방별 사용량 비교 차트를 생성합니다."""
        # Normal Train 제외
        comparison_data = {k: v for k, v in all_toggle_data.items() if k != 'Normal Train'}
        
        # 각 데이터셋별 방별 평균 토글 횟수 계산
        room_stats = {}
        rooms = ['Bedroom', 'Living Room', 'Kitchen', 'Bathroom']
        
        for dataset_name, data in comparison_data.items():
            room_stats[dataset_name] = {}
            for room in rooms:
                room_data = data[data['led_position'] == room]
                room_stats[dataset_name][room] = room_data['toggle_count'].mean()
        
        # 차트 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        dataset_names = list(comparison_data.keys())
        colors = ['#A23B72', '#F18F01', '#C73E1D', '#592E83']
        
        for i, room in enumerate(rooms):
            ax = axes[i]
            
            # 각 데이터셋별 해당 방의 사용량
            room_values = [room_stats[dataset][room] for dataset in dataset_names]
            
            # 막대 그래프 생성
            bars = ax.bar(range(len(dataset_names)), room_values, color=colors, alpha=0.8)
            
            # 차트 설정
            ax.set_title(f'{room} - Average Toggle Count', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dataset', fontsize=10)
            ax.set_ylabel('Average Toggle Count', fontsize=10)
            ax.set_xticks(range(len(dataset_names)))
            ax.set_xticklabels([name.replace(' ', '\n') for name in dataset_names], fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 값 표시
            for j, (bar, value) in enumerate(zip(bars, room_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(room_values) * 0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle('Room Usage Comparison Across Datasets', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # comparison 폴더에 저장
        os.makedirs('../../charts/dummy_data/comparison', exist_ok=True)
        output_path = '../../charts/dummy_data/comparison/room_usage_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def analyze_dataset(self, dataset_name, filename, dataset_folder):
        """데이터셋을 분석하고 시각화합니다."""
        print(f"Analyzing {dataset_name}...")
        
        # 데이터 로드
        df = self.load_dataset(filename)
        
        # 시간대별 토글 횟수와 ON 시간 계산
        hourly_toggle, hourly_ontime = self.calculate_hourly_toggle_and_ontime(df)
        
        # 전체 통계 계산
        stats = self.calculate_overall_stats(df)
        stats['total_records'] = len(df)
        stats['unique_users'] = df['User'].nunique()
        
        # 최고 활동 시간 계산
        peak_hour = hourly_toggle.groupby('hour')['toggle_count'].sum().idxmax()
        stats['max_activity_hour'] = peak_hour
        
        # 히트맵 생성
        toggle_heatmap = self.create_hourly_heatmap(
            hourly_toggle, 
            f'{dataset_name} - Toggle Count by Hour',
            'toggle_count',
            'toggle_heatmap.png',
            dataset_folder
        )
        
        ontime_heatmap = self.create_hourly_heatmap(
            hourly_ontime,
            f'{dataset_name} - ON Time by Hour (minutes)',
            'on_time',
            'ontime_heatmap.png',
            dataset_folder
        )
        
        # 시간대별 선 그래프
        activity_plot = self.create_hourly_line_plot(
            hourly_toggle,
            hourly_ontime,
            dataset_name,
            'hourly_activity.png',
            dataset_folder
        )
        
        print(f"  - Total records: {stats['total_records']:,}")
        print(f"  - Unique users: {stats['unique_users']}")
        print(f"  - Average Toggle Count: {stats['avg_toggle_count']:.1f}")
        print(f"  - Average ON Time: {stats['avg_on_time']:.1f} minutes")
        print(f"  - Peak activity hour: {stats['max_activity_hour']}:00")
        print(f"  - Charts saved to: charts/dummy_data/{dataset_folder}/\n")
        
        return stats, hourly_toggle, hourly_ontime

def main():
    # charts 폴더 생성
    os.makedirs('../../charts/dummy_data', exist_ok=True)
    
    visualizer = DataVisualizer()
    
    # 분석할 데이터셋 목록 (이름, 파일명, 폴더명)
    datasets = {
        'Normal Train': ('train_dataset.csv', 'normal_train'),
        'Normal Test': ('normal_test_dataset.csv', 'normal_test'),
        'Immediate Abnormal': ('immediate_abnormal_test_dataset.csv', 'immediate_abnormal'),
        'Rapid Abnormal': ('rapid_abnormal_test_dataset.csv', 'rapid_abnormal'),
        'Gradual Abnormal': ('gradual_abnormal_test_dataset.csv', 'gradual_abnormal')
    }
    
    all_stats = {}
    all_toggle_data = {}
    all_ontime_data = {}
    
    for dataset_name, (filename, folder_name) in datasets.items():
        try:
            # 각 데이터셋별 폴더 생성
            os.makedirs(f'../../charts/dummy_data/{folder_name}', exist_ok=True)
            
            stats, toggle_data, ontime_data = visualizer.analyze_dataset(dataset_name, filename, folder_name)
            all_stats[dataset_name] = stats
            all_toggle_data[dataset_name] = toggle_data
            all_ontime_data[dataset_name] = ontime_data
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
    
    # Comparison analysis charts
    print("=" * 60)
    print("Creating comparison analysis charts...")
    print("=" * 60)
    
    try:
        # Toggle count comparison heatmap
        toggle_comparison = visualizer.create_comparison_heatmaps(
            all_toggle_data, 'toggle_count', 'Toggle Count')
        print(f"✓ Toggle count comparison heatmap: {toggle_comparison}")
        
        # ON time comparison heatmap
        ontime_comparison = visualizer.create_comparison_heatmaps(
            all_ontime_data, 'on_time', 'ON Time')
        print(f"✓ ON time comparison heatmap: {ontime_comparison}")
        
        # Statistics comparison chart
        stats_comparison = visualizer.create_statistics_comparison(all_stats)
        print(f"✓ Statistics comparison chart: {stats_comparison}")
        
        # Pattern analysis chart
        pattern_analysis = visualizer.create_pattern_analysis(all_toggle_data)
        print(f"✓ Pattern analysis chart: {pattern_analysis}")
        
        # Room usage comparison chart
        room_comparison = visualizer.create_room_usage_comparison(all_toggle_data)
        print(f"✓ Room usage comparison chart: {room_comparison}")
        
    except Exception as e:
        print(f"Error creating comparison charts: {e}")
    
    # Summary output
    print("\n" + "=" * 60)
    print("Dataset Comparison Analysis")
    print("=" * 60)
    
    # Calculate changes compared to Normal Test
    normal_toggle = all_stats['Normal Test']['avg_toggle_count']
    normal_ontime = all_stats['Normal Test']['avg_on_time']
    
    for dataset_name, stats in all_stats.items():
        if dataset_name not in ['Normal Train', 'Normal Test']:
            toggle_change = ((stats['avg_toggle_count'] - normal_toggle) / normal_toggle) * 100
            ontime_change = ((stats['avg_on_time'] - normal_ontime) / normal_ontime) * 100
            
            print(f"\n{dataset_name}:")
            print(f"  Toggle count change: {toggle_change:+.1f}% ({stats['avg_toggle_count']:.1f} vs {normal_toggle:.1f})")
            print(f"  ON time change: {ontime_change:+.1f}% ({stats['avg_on_time']:.1f} vs {normal_ontime:.1f})")
    
    print(f"\n✓ All visualizations completed! Check charts/dummy_data/ folder.")

if __name__ == "__main__":
    main() 