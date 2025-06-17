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
        
        return stats

def main():
    # charts 폴더 생성
    os.makedirs('../../charts/dummy_data', exist_ok=True)
    
    visualizer = DataVisualizer()
    
    # 분석할 데이터셋 목록 (이름, 파일명, 폴더명)
    datasets = {
        'Train Dataset': ('train_dataset.csv', 'normal_train'),
        'Normal Test': ('normal_test_dataset.csv', 'normal_test'),
        'Immediate Abnormal': ('immediate_abnormal_test_dataset.csv', 'immediate_abnormal'),
        'Rapid Abnormal': ('rapid_abnormal_test_dataset.csv', 'rapid_abnormal'),
        'Gradual Abnormal': ('gradual_abnormal_test_dataset.csv', 'gradual_abnormal')
    }
    
    all_stats = {}
    
    for dataset_name, (filename, folder_name) in datasets.items():
        try:
            # 각 데이터셋별 폴더 생성
            os.makedirs(f'../../charts/dummy_data/{folder_name}', exist_ok=True)
            
            stats = visualizer.analyze_dataset(dataset_name, filename, folder_name)
            all_stats[dataset_name] = stats
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
    
    # 전체 요약 출력
    print("=" * 60)
    print("SUMMARY OF ALL DATASETS")
    print("=" * 60)
    
    for dataset_name, stats in all_stats.items():
        print(f"{dataset_name}:")
        print(f"  Users: {stats['unique_users']}, Records: {stats['total_records']:,}")
        print(f"  Avg Toggle Count: {stats['avg_toggle_count']:.1f}, Avg ON Time: {stats['avg_on_time']:.1f}min")
        print()

if __name__ == "__main__":
    main() 