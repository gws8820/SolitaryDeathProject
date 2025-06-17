import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from database_loader import RealDataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AnomalyHeatmapVisualizer:
    """사용자별 이상치 정도를 히트맵으로 시각화하는 클래스"""
    
    def __init__(self):
        self.loader = RealDataLoader()
        self.charts_path = Path("../../charts/detection_real")
        self.charts_path.mkdir(parents=True, exist_ok=True)
    
    def load_anomaly_data(self):
        """데이터베이스에서 이상치 감지 결과를 로드"""
        if not self.loader.connect():
            print("데이터베이스 연결 실패")
            return None
        
        query = """
        SELECT User, Date, Type, 
               OCSVM_prediction, OCSVM_score,
               Isforest_prediction, Isforest_score,
               Consensus_prediction, Consensus_score
        FROM abnormal_detection
        ORDER BY User, Date, Type
        """
        
        try:
            df = pd.read_sql(query, self.loader.connection)
            self.loader.disconnect()
            return df
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            self.loader.disconnect()
            return None
    
    def prepare_heatmap_data(self, df):
        """히트맵을 위한 데이터 준비"""
        # 사용자별 날짜별 이상치 점수 집계
        user_date_scores = {}
        
        for _, row in df.iterrows():
            user = row['User']
            date = row['Date']
            type_name = row['Type']
            
            key = f"{user}_{date}"
            
            if key not in user_date_scores:
                user_date_scores[key] = {
                    'user': user,
                    'date': date,
                    'ocsvm_max': 0,
                    'isforest_max': 0,
                    'consensus_max': 0,
                    'day_ocsvm': 0,
                    'day_isforest': 0,
                    'night_ocsvm': 0,
                    'night_isforest': 0
                }
            
            # 최대 점수 업데이트
            user_date_scores[key]['ocsvm_max'] = max(user_date_scores[key]['ocsvm_max'], row['OCSVM_score'])
            user_date_scores[key]['isforest_max'] = max(user_date_scores[key]['isforest_max'], row['Isforest_score'])
            user_date_scores[key]['consensus_max'] = max(user_date_scores[key]['consensus_max'], row['Consensus_score'])
            
            # 시간대별 점수 저장
            if type_name == 'day':
                user_date_scores[key]['day_ocsvm'] = row['OCSVM_score']
                user_date_scores[key]['day_isforest'] = row['Isforest_score']
            else:
                user_date_scores[key]['night_ocsvm'] = row['OCSVM_score']
                user_date_scores[key]['night_isforest'] = row['Isforest_score']
        
        return pd.DataFrame(list(user_date_scores.values()))
    
    def calculate_user_risk_levels(self, df):
        """사용자별 위험도 레벨 계산"""
        user_risk = df.groupby('user').agg({
            'ocsvm_max': ['mean', 'max', 'std'],
            'isforest_max': ['mean', 'max', 'std'],
            'consensus_max': ['mean', 'max', 'std']
        }).round(2)
        
        # 컬럼명 정리
        user_risk.columns = ['_'.join(col).strip() for col in user_risk.columns.values]
        
        # 종합 위험도 점수 계산
        user_risk['total_risk_score'] = (
            user_risk['ocsvm_max_mean'] * 0.3 +
            user_risk['isforest_max_mean'] * 0.3 +
            user_risk['consensus_max_mean'] * 0.4
        ).round(2)
        
        # 위험도 등급 분류
        user_risk['risk_level'] = pd.cut(
            user_risk['total_risk_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return user_risk.reset_index()
    
    def create_user_date_heatmap(self, df):
        """사용자별 날짜별 이상치 점수 히트맵"""
        # 피벗 테이블 생성 (사용자 x 날짜)
        pivot_ocsvm = df.pivot(index='user', columns='date', values='ocsvm_max')
        pivot_isforest = df.pivot(index='user', columns='date', values='isforest_max')
        pivot_consensus = df.pivot(index='user', columns='date', values='consensus_max')
        
        fig, axes = plt.subplots(3, 1, figsize=(20, 15))
        
        # OCSVM 히트맵
        sns.heatmap(pivot_ocsvm, annot=False, cmap='Reds', 
                   vmin=0, vmax=100, ax=axes[0], cbar_kws={'label': 'OCSVM Score'})
        axes[0].set_title('User Daily OCSVM Anomaly Scores', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('User')
        
        # Isolation Forest 히트맵
        sns.heatmap(pivot_isforest, annot=False, cmap='Blues',
                   vmin=0, vmax=100, ax=axes[1], cbar_kws={'label': 'Isolation Forest Score'})
        axes[1].set_title('User Daily Isolation Forest Anomaly Scores', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('User')
        
        # Consensus 히트맵
        sns.heatmap(pivot_consensus, annot=False, cmap='Purples',
                   vmin=0, vmax=100, ax=axes[2], cbar_kws={'label': 'Consensus Score'})
        axes[2].set_title('User Daily Consensus Anomaly Scores', fontsize=16, fontweight='bold')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('User')
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'user_daily_anomaly_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("사용자별 일일 이상치 히트맵 생성 완료")
    
    def create_user_risk_summary(self, user_risk):
        """사용자 위험도 요약 히트맵"""
        # 위험도 관련 컬럼만 선택
        risk_cols = ['ocsvm_max_mean', 'isforest_max_mean', 'consensus_max_mean', 'total_risk_score']
        risk_data = user_risk.set_index('user')[risk_cols]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        
        # 위험도 점수 히트맵
        sns.heatmap(risk_data, annot=True, fmt='.1f', cmap='YlOrRd',
                   vmin=0, vmax=100, ax=axes[0], cbar_kws={'label': 'Risk Score'})
        axes[0].set_title('User Risk Score Summary', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Risk Metrics')
        axes[0].set_ylabel('User')
        
        # 위험도 등급별 사용자 수
        risk_counts = user_risk['risk_level'].value_counts()
        colors = ['green', 'yellow', 'orange', 'red']
        axes[1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('User Risk Level Distribution', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'user_risk_summary.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("사용자 위험도 요약 차트 생성 완료")
    
    def create_time_based_anomaly_pattern(self, df):
        """시간대별 이상치 패턴 분석"""
        # Day vs Night 비교
        day_night_comparison = []
        
        for _, row in df.iterrows():
            day_night_comparison.append({
                'user': row['user'],
                'date': row['date'],
                'day_avg': (row['day_ocsvm'] + row['day_isforest']) / 2,
                'night_avg': (row['night_ocsvm'] + row['night_isforest']) / 2,
                'day_max': max(row['day_ocsvm'], row['day_isforest']),
                'night_max': max(row['night_ocsvm'], row['night_isforest'])
            })
        
        comparison_df = pd.DataFrame(day_night_comparison)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Day vs Night 평균 점수 산점도
        axes[0,0].scatter(comparison_df['day_avg'], comparison_df['night_avg'], alpha=0.6)
        axes[0,0].plot([0, 100], [0, 100], 'r--', label='Equal Score Line')
        axes[0,0].set_xlabel('Day Average Score')
        axes[0,0].set_ylabel('Night Average Score')
        axes[0,0].set_title('Day vs Night Average Anomaly Scores')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Day vs Night 최대 점수 산점도
        axes[0,1].scatter(comparison_df['day_max'], comparison_df['night_max'], alpha=0.6, color='orange')
        axes[0,1].plot([0, 100], [0, 100], 'r--', label='Equal Score Line')
        axes[0,1].set_xlabel('Day Max Score')
        axes[0,1].set_ylabel('Night Max Score')
        axes[0,1].set_title('Day vs Night Maximum Anomaly Scores')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 사용자별 Day 점수 분포
        user_day_pivot = comparison_df.pivot(index='user', columns='date', values='day_avg')
        sns.heatmap(user_day_pivot, annot=False, cmap='Reds', ax=axes[1,0],
                   cbar_kws={'label': 'Day Score'})
        axes[1,0].set_title('User Daily Day Period Scores')
        
        # 사용자별 Night 점수 분포
        user_night_pivot = comparison_df.pivot(index='user', columns='date', values='night_avg')
        sns.heatmap(user_night_pivot, annot=False, cmap='Blues', ax=axes[1,1],
                   cbar_kws={'label': 'Night Score'})
        axes[1,1].set_title('User Daily Night Period Scores')
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'time_based_anomaly_pattern.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("시간대별 이상치 패턴 분석 차트 생성 완료")
    
    def create_high_risk_user_detail(self, df, user_risk, top_n=10):
        """고위험 사용자 상세 분석"""
        # 상위 N명의 고위험 사용자 선별
        high_risk_users = user_risk.nlargest(top_n, 'total_risk_score')['user'].tolist()
        high_risk_data = df[df['user'].isin(high_risk_users)]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 고위험 사용자별 점수 박스플롯
        melted_data = pd.melt(high_risk_data, 
                             id_vars=['user', 'date'],
                             value_vars=['ocsvm_max', 'isforest_max', 'consensus_max'],
                             var_name='model', value_name='score')
        
        sns.boxplot(data=melted_data, x='user', y='score', hue='model', ax=axes[0,0])
        axes[0,0].set_title(f'Top {top_n} High-Risk Users Score Distribution')
        axes[0,0].set_xlabel('User')
        axes[0,0].set_ylabel('Anomaly Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 고위험 사용자 시계열 패턴
        for i, user in enumerate(high_risk_users[:5]):  # 상위 5명만
            user_data = high_risk_data[high_risk_data['user'] == user]
            axes[0,1].plot(user_data['date'], user_data['consensus_max'], 
                          marker='o', label=f'User {user}', alpha=0.7)
        
        axes[0,1].set_title('High-Risk Users Consensus Score Timeline')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Consensus Score')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True)
        
        # 모델간 상관관계 (고위험 사용자)
        correlation_matrix = high_risk_data[['ocsvm_max', 'isforest_max', 'consensus_max']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Model Score Correlation (High-Risk Users)')
        
        # 위험도 분포 히스토그램
        axes[1,1].hist(user_risk['total_risk_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,1].axvline(user_risk['total_risk_score'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {user_risk["total_risk_score"].mean():.1f}')
        axes[1,1].set_xlabel('Total Risk Score')
        axes[1,1].set_ylabel('Number of Users')
        axes[1,1].set_title('User Risk Score Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'high_risk_user_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"상위 {top_n}명 고위험 사용자 분석 차트 생성 완료")
    
    def generate_summary_report(self, user_risk):
        """요약 보고서 생성"""
        report_path = self.charts_path / 'anomaly_integration_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== User Anomaly Detection Integration Summary ===\n\n")
            
            f.write(f"Total Users Analyzed: {len(user_risk)}\n")
            f.write(f"Analysis Period: {user_risk.index.min()} to {user_risk.index.max()}\n\n")
            
            f.write("Risk Level Distribution:\n")
            risk_counts = user_risk['risk_level'].value_counts()
            for level, count in risk_counts.items():
                percentage = (count / len(user_risk)) * 100
                f.write(f"  {level}: {count} users ({percentage:.1f}%)\n")
            
            f.write(f"\nTop 10 High-Risk Users:\n")
            top_users = user_risk.nlargest(10, 'total_risk_score')
            for _, user_data in top_users.iterrows():
                f.write(f"  User {user_data['user']}: {user_data['total_risk_score']:.1f} points "
                       f"({user_data['risk_level']})\n")
            
            f.write(f"\nOverall Statistics:\n")
            f.write(f"  Average Risk Score: {user_risk['total_risk_score'].mean():.2f}\n")
            f.write(f"  Median Risk Score: {user_risk['total_risk_score'].median():.2f}\n")
            f.write(f"  Max Risk Score: {user_risk['total_risk_score'].max():.2f}\n")
            f.write(f"  Min Risk Score: {user_risk['total_risk_score'].min():.2f}\n")
        
        print(f"요약 보고서 생성 완료: {report_path}")
    
    def run_visualization(self):
        """전체 시각화 프로세스 실행"""
        print("=== 사용자별 이상치 히트맵 시각화 시작 ===")
        
        # 데이터 로드
        df = self.load_anomaly_data()
        if df is None:
            return
        
        print(f"총 {len(df)}개 레코드 로드 완료")
        
        # 히트맵 데이터 준비
        heatmap_data = self.prepare_heatmap_data(df)
        print(f"히트맵 데이터 준비 완료: {len(heatmap_data)}개 사용자-날짜 조합")
        
        # 사용자 위험도 계산
        user_risk = self.calculate_user_risk_levels(heatmap_data)
        print(f"사용자 위험도 계산 완료: {len(user_risk)}명")
        
        # 각종 시각화 생성
        self.create_user_date_heatmap(heatmap_data)
        self.create_user_risk_summary(user_risk)
        self.create_time_based_anomaly_pattern(heatmap_data)
        self.create_high_risk_user_detail(heatmap_data, user_risk)
        self.generate_summary_report(user_risk)
        
        print("=== 모든 시각화 완료 ===")
        print(f"결과 저장 위치: {self.charts_path}")

def main():
    """메인 실행 함수"""
    visualizer = AnomalyHeatmapVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main() 