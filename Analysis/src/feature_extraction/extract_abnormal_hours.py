import pandas as pd
import os

class AbnormalHourExtractor:
    def __init__(self, raw_data_path="dummy_data/raw", output_path="dummy_data/abnormal_hour"):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        
        # Create output directory if not exists
        os.makedirs(output_path, exist_ok=True)
        
        # Abnormal datasets to process (정상 데이터는 제외)
        self.abnormal_datasets = [
            'rapid_abnormal_test_dataset.csv',
            'immediate_abnormal_test_dataset.csv', 
            'gradual_abnormal_test_dataset.csv'
        ]
    
    def extract_user_abnormal_hours(self, dataset_filename):
        """Extract abnormal_hour for each user from a dataset"""
        file_path = os.path.join(self.raw_data_path, dataset_filename)
        
        if not os.path.exists(file_path):
            print(f"파일이 존재하지 않습니다: {dataset_filename}")
            return None
        
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Check if abnormal_hour column exists
        if 'abnormal_hour' not in df.columns:
            print(f"abnormal_hour 컬럼이 없습니다: {dataset_filename}")
            return None
        
        print(f"처리 중: {dataset_filename}")
        print(f"  - 총 행 수: {len(df)}")
        print(f"  - 사용자 수: {df['User'].nunique()}")
        
        # Extract unique abnormal_hour for each user
        user_abnormal_hours = df.groupby('User')['abnormal_hour'].first().reset_index()
        
        # Rename columns for clarity
        user_abnormal_hours.columns = ['User', 'abnormal_hour']
        
        # Sort by User
        user_abnormal_hours = user_abnormal_hours.sort_values('User')
        
        print(f"  - 추출된 사용자별 abnormal_hour: {len(user_abnormal_hours)}개")
        
        # Display sample data
        print(f"  - 샘플 데이터:")
        print(user_abnormal_hours.head(5).to_string(index=False))
        
        # Check for consistency (each user should have same abnormal_hour throughout)
        inconsistent_users = []
        for user in df['User'].unique():
            user_data = df[df['User'] == user]
            unique_hours = user_data['abnormal_hour'].unique()
            if len(unique_hours) > 1:
                inconsistent_users.append({
                    'user': user,
                    'abnormal_hours': unique_hours.tolist()
                })
        
        if inconsistent_users:
            print(f"  일관성 없는 사용자들 ({len(inconsistent_users)}명):")
            for user_info in inconsistent_users[:3]:  # Show first 3
                print(f"    User {user_info['user']}: {user_info['abnormal_hours']}")
        else:
            print(f"  모든 사용자의 abnormal_hour가 일관성 있음")
        
        return user_abnormal_hours
    
    def save_abnormal_hours(self, user_abnormal_hours, dataset_filename):
        """Save user abnormal hours to CSV file"""
        if user_abnormal_hours is None:
            return
        
        # Create output filename (remove .csv extension and add it back)
        output_filename = dataset_filename.replace('.csv', '.csv')
        output_path = os.path.join(self.output_path, output_filename)
        
        # Save to CSV
        user_abnormal_hours.to_csv(output_path, index=False)
        print(f"  저장 완료: {output_path}")
    
    def analyze_abnormal_hour_patterns(self):
        """Analyze patterns in abnormal hours across datasets"""
        print(f"\n{'='*60}")
        print("비정상 시간 패턴 분석")
        print(f"{'='*60}")
        
        all_data = {}
        
        for dataset in self.abnormal_datasets:
            user_hours = self.extract_user_abnormal_hours(dataset)
            if user_hours is not None:
                dataset_name = dataset.replace('.csv', '')
                all_data[dataset_name] = user_hours
                
                # Analyze distribution
                hour_counts = user_hours['abnormal_hour'].value_counts().sort_index()
                print(f"\n{dataset_name} 비정상 시간 분포:")
                for hour, count in hour_counts.items():
                    percentage = count / len(user_hours) * 100
                    print(f"  {hour:2d}시: {count:3d}명 ({percentage:5.1f}%)")
        
        # Compare patterns across datasets
        if len(all_data) > 1:
            print(f"\n데이터셋별 비정상 시간 비교:")
            all_hours = set()
            for data in all_data.values():
                all_hours.update(data['abnormal_hour'].unique())
            
            print(f"{'시간':>4} | {'Rapid':>7} | {'Immediate':>9} | {'Gradual':>8}")
            print("-" * 40)
            
            for hour in sorted(all_hours):
                row = f"{hour:4d} |"
                for dataset_name in ['rapid_abnormal_test_dataset', 'immediate_abnormal_test_dataset', 'gradual_abnormal_test_dataset']:
                    if dataset_name in all_data:
                        count = (all_data[dataset_name]['abnormal_hour'] == hour).sum()
                        row += f" {count:7d} |"
                    else:
                        row += "      - |"
                print(row)
        
        return all_data
    
    def extract_all_abnormal_hours(self):
        """Extract abnormal hours from all abnormal datasets
        
        Note: 이 함수는 비정상 데이터에서만 사용됩니다.
        정상 데이터에는 abnormal_hour 개념이 없으므로 별도 처리가 필요합니다.
        """
        print("="*60)
        print("비정상 데이터셋 abnormal_hour 추출")
        print("="*60)
        
        for dataset in self.abnormal_datasets:
            print(f"\n{'='*60}")
            user_abnormal_hours = self.extract_user_abnormal_hours(dataset)
            self.save_abnormal_hours(user_abnormal_hours, dataset)
        
        # Analyze patterns
        all_data = self.analyze_abnormal_hour_patterns()
        
        print(f"\n{'='*60}")
        print("추출 완료!")
        print(f"{'='*60}")
        print(f"출력 폴더: {self.output_path}")
        print(f"생성된 파일 수: {len(self.abnormal_datasets)}")
        
        return all_data

def main():
    """Main function to extract abnormal hours"""
    extractor = AbnormalHourExtractor()
    results = extractor.extract_all_abnormal_hours()
    return results

if __name__ == "__main__":
    main() 