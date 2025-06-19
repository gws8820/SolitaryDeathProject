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

class SimpleAnomalyVisualizer:
    """Simple and intuitive anomaly visualization class"""
    
    def __init__(self):
        self.loader = RealDataLoader()
        self.charts_path = Path("../../charts/detection_real")
        self.charts_path.mkdir(parents=True, exist_ok=True)
    
    def load_anomaly_data(self):
        """Load anomaly detection results from database"""
        if not self.loader.connect():
            print("Database connection failed")
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
            print(f"Data loading error: {e}")
            self.loader.disconnect()
            return None
    
    def calculate_detection_stats(self, df):
        """Calculate simple detection statistics"""
        total_records = len(df)
        
        # Model-wise detection count (50+ score)
        ocsvm_detections = len(df[df['OCSVM_score'] >= 50])
        isforest_detections = len(df[df['Isforest_score'] >= 50])
        consensus_detections = len(df[df['Consensus_score'] >= 50])
        
        # User-wise detection statistics
        user_stats = []
        for user in df['User'].unique():
            user_data = df[df['User'] == user]
            user_consensus_detections = len(user_data[user_data['Consensus_score'] >= 50])
            user_stats.append({
                'user': user,
                'total_records': len(user_data),
                'detections': user_consensus_detections
            })
        
        user_df = pd.DataFrame(user_stats)
        users_with_detections = len(user_df[user_df['detections'] > 0])
        total_users = len(user_df)
        
        return {
            'total_records': total_records,
            'ocsvm_detections': ocsvm_detections,
            'isforest_detections': isforest_detections,
            'consensus_detections': consensus_detections,
            'total_users': total_users,
            'users_with_detections': users_with_detections,
            'user_df': user_df
        }
    
    def create_total_detection_chart(self, stats):
        """Total records vs detection count chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Data preparation
        labels = ['Normal', 'Detected']
        normal_count = stats['total_records'] - stats['consensus_detections']
        detection_count = stats['consensus_detections']
        sizes = [normal_count, detection_count]
        colors = ['lightblue', 'red']
        
        # Pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         colors=colors, startangle=90, textprops={'fontsize': 14})
        
        # Add numerical info in center
        ax.text(0, -0.3, f'Total: {stats["total_records"]} records\nDetected: {detection_count}\nNormal: {normal_count}', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('Total Records: Anomaly Detection Status', fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'total_detection_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Total detection status chart created")
    
    def create_user_detection_chart(self, stats):
        """Total users vs detected users chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Data preparation
        labels = ['Normal Users', 'Detected Users']
        normal_users = stats['total_users'] - stats['users_with_detections']
        detected_users = stats['users_with_detections']
        sizes = [normal_users, detected_users]
        colors = ['lightgreen', 'orange']
        
        # Pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         colors=colors, startangle=90, textprops={'fontsize': 14})
        
        # Add numerical info in center
        ax.text(0, -0.3, f'Total: {stats["total_users"]} users\nDetected: {detected_users}\nNormal: {normal_users}', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('Total Users: Anomaly Detection Status', fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'user_detection_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("User detection status chart created")
    
    def create_user_detection_count_chart(self, stats):
        """Detection count by user chart"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        user_df = stats['user_df']
        
        # Filter users with detections only
        detected_users = user_df[user_df['detections'] > 0].sort_values('detections', ascending=False)
        
        if len(detected_users) == 0:
            ax.text(0.5, 0.5, 'No detected users', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
        else:
            # Bar chart
            bars = ax.bar(range(len(detected_users)), detected_users['detections'], 
                         color='red', alpha=0.7)
            
            # User names on x-axis
            ax.set_xticks(range(len(detected_users)))
            ax.set_xticklabels(detected_users['user'], rotation=45, ha='right')
            
            # Numbers on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('User', fontsize=14)
            ax.set_ylabel('Detection Count', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
        
        ax.set_title(f'Detection Count by User (Total: {len(detected_users)} users)', 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'user_detection_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("User detection count chart created")
    
    def create_model_comparison_chart(self, stats):
        """Model performance comparison chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Data preparation
        models = ['OCSVM', 'Isolation Forest', 'Consensus']
        detections = [stats['ocsvm_detections'], stats['isforest_detections'], stats['consensus_detections']]
        colors = ['skyblue', 'lightcoral', 'gold']
        
        # Bar chart
        bars = ax.bar(models, detections, color=colors, alpha=0.8)
        
        # Numbers and percentages on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / stats['total_records']) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}\n({percentage:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Detection Count', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Show total records
        ax.text(0.02, 0.98, f'Total Records: {stats["total_records"]}', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Model comparison chart created")
    
    def create_detection_distribution_chart(self, stats):
        """Detection count distribution chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        user_df = stats['user_df']
        
        # Calculate user count by detection count
        detection_counts = user_df['detections'].value_counts().sort_index()
        
        # Bar chart
        bars = ax.bar(detection_counts.index, detection_counts.values, 
                     color='purple', alpha=0.7)
        
        # Numbers on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Detection Count', fontsize=14)
        ax.set_ylabel('Number of Users', fontsize=14)
        ax.set_title('User Distribution by Detection Count', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Statistics info
        total_users = len(user_df)
        no_detection_users = len(user_df[user_df['detections'] == 0])
        with_detection_users = total_users - no_detection_users
        
        stats_text = f'Total Users: {total_users}\nNo Detection: {no_detection_users}\nWith Detection: {with_detection_users}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', horizontalalignment='right')
        
        plt.tight_layout()
        plt.savefig(self.charts_path / 'detection_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Detection distribution chart created")
    
    def generate_simple_summary(self, stats):
        """Generate simple summary report"""
        report_path = self.charts_path / 'simple_summary.txt'
        
        user_df = stats['user_df']
        detected_users = user_df[user_df['detections'] > 0].sort_values('detections', ascending=False)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Solitary Death Detection System Summary ===\n\n")
            
            f.write("📊 Overall Status\n")
            f.write(f"- Total Records: {stats['total_records']}\n")
            f.write(f"- Detected Records: {stats['consensus_detections']} ({stats['consensus_detections']/stats['total_records']*100:.1f}%)\n")
            f.write(f"- Total Users: {stats['total_users']}\n")
            f.write(f"- Detected Users: {stats['users_with_detections']} ({stats['users_with_detections']/stats['total_users']*100:.1f}%)\n\n")
            
            f.write("🔍 Model Performance\n")
            f.write(f"- OCSVM: {stats['ocsvm_detections']} detections\n")
            f.write(f"- Isolation Forest: {stats['isforest_detections']} detections\n")
            f.write(f"- Consensus: {stats['consensus_detections']} detections\n\n")
            
            if len(detected_users) > 0:
                f.write("⚠️ Top Detected Users\n")
                for _, user in detected_users.head(10).iterrows():
                    f.write(f"- User {user['user']}: {user['detections']} detections\n")
            else:
                f.write("⚠️ No detected users\n")
        
        print(f"Simple summary report created: {report_path}")
    
    def run_visualization(self):
        """Run complete visualization process"""
        print("=== Simple Anomaly Visualization Started ===")
        
        # Load data
        df = self.load_anomaly_data()
        if df is None:
            return
        
        print(f"Total {len(df)} records loaded")
        
        # Calculate statistics
        stats = self.calculate_detection_stats(df)
        print(f"Statistics calculated")
        
        # Generate individual charts
        self.create_total_detection_chart(stats)
        self.create_user_detection_chart(stats)
        self.create_user_detection_count_chart(stats)
        self.create_model_comparison_chart(stats)
        self.create_detection_distribution_chart(stats)
        self.generate_simple_summary(stats)
        
        print("=== All Visualizations Completed ===")
        print(f"📊 Charts saved to: {self.charts_path}")
        
        # Key results summary
        print(f"\n=== Key Results ===")
        print(f"Total Records: {stats['total_records']}")
        print(f"Detected Records: {stats['consensus_detections']} ({stats['consensus_detections']/stats['total_records']*100:.1f}%)")
        print(f"Total Users: {stats['total_users']}")
        print(f"Detected Users: {stats['users_with_detections']} ({stats['users_with_detections']/stats['total_users']*100:.1f}%)")

def main():
    """Main execution function"""
    visualizer = SimpleAnomalyVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main() 