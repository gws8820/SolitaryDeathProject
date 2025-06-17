import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# 환경변수 로드
load_dotenv('../../.env')

class RealDataLoader:
    """실제 운용 시스템의 데이터베이스에서 데이터를 로드하는 클래스"""
    
    def __init__(self, host: str = None, user: str = None, password: str = None, database: str = None):
        self.host = host or os.getenv('DB_HOST')
        self.user = user or os.getenv('DB_USER')
        self.password = password or os.getenv('DB_PASSWORD')
        self.database = database or os.getenv('DB_NAME')
        self.connection = None
        
        # 테이블 이름 정의
        self.tables = [
            'day_inactive_room', 'day_inactive_total', 'day_kitchen_usage_rate', 
            'day_on_ratio_room', 'day_toggle_room', 'day_toggle_total',
            'night_bathroom_usage', 'night_inactive_room', 'night_inactive_total',
            'night_kitchen_usage_rate', 'night_on_ratio_room', 'night_toggle_room', 
            'night_toggle_total'
        ]
        
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print(f"데이터베이스 '{self.database}' 연결 성공")
                return True
        except Error as e:
            print(f"데이터베이스 연결 오류: {e}")
            return False
    
    def disconnect(self):
        """데이터베이스 연결 종료"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("데이터베이스 연결 종료")
    
    def load_table_data(self, table_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """특정 테이블에서 날짜 범위에 해당하는 데이터 로드"""
        if not self.connection or not self.connection.is_connected():
            print("데이터베이스 연결이 필요합니다.")
            return pd.DataFrame()
        
        try:
            query = f"""
            SELECT * FROM {table_name} 
            WHERE Date >= %s AND Date <= %s
            ORDER BY User, Date
            """
            
            df = pd.read_sql(query, self.connection, params=(start_date, end_date))
            print(f"{table_name}: {len(df)}개 레코드 로드")
            return df
            
        except Error as e:
            print(f"테이블 {table_name} 로드 오류: {e}")
            return pd.DataFrame()
    
    def load_all_tables(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """모든 테이블에서 데이터 로드"""
        all_data = {}
        
        for table in self.tables:
            data = self.load_table_data(table, start_date, end_date)
            if not data.empty:
                all_data[table] = data
            else:
                print(f"경고: {table} 테이블이 비어있습니다.")
        
        return all_data
    
    def merge_features_for_training(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """머신러닝 훈련을 위한 특성 병합"""
        
        # Day 특성 (15개)
        day_features = None
        
        # 1. day_inactive_total
        if 'day_inactive_total' in data_dict:
            df = data_dict['day_inactive_total'].copy()
            df = df.rename(columns={'timeslot_count': 'day_inactive_total'})
            day_features = df[['User', 'Date', 'day_inactive_total']]
        
        # 2. day_inactive_room (4개 특성)
        if 'day_inactive_room' in data_dict:
            df = data_dict['day_inactive_room'].copy()
            df = df.rename(columns={
                '01': 'day_inactive_room_01', '02': 'day_inactive_room_02',
                '03': 'day_inactive_room_03', '04': 'day_inactive_room_04'
            })
            if day_features is not None:
                day_features = pd.merge(day_features, df, on=['User', 'Date'], how='inner')
            else:
                day_features = df
        
        # 3. day_toggle_total
        if 'day_toggle_total' in data_dict:
            df = data_dict['day_toggle_total'].copy()
            df = df.rename(columns={'toggle_count': 'day_toggle_total'})
            day_features = pd.merge(day_features, df[['User', 'Date', 'day_toggle_total']], 
                                  on=['User', 'Date'], how='inner')
        
        # 4. day_toggle_room (4개 특성)
        if 'day_toggle_room' in data_dict:
            df = data_dict['day_toggle_room'].copy()
            df = df.rename(columns={
                '01': 'day_toggle_room_01', '02': 'day_toggle_room_02',
                '03': 'day_toggle_room_03', '04': 'day_toggle_room_04'
            })
            day_features = pd.merge(day_features, df, on=['User', 'Date'], how='inner')
        
        # 5. day_on_ratio_room (4개 특성)
        if 'day_on_ratio_room' in data_dict:
            df = data_dict['day_on_ratio_room'].copy()
            df = df.rename(columns={
                '01': 'day_on_ratio_room_01', '02': 'day_on_ratio_room_02',
                '03': 'day_on_ratio_room_03', '04': 'day_on_ratio_room_04'
            })
            day_features = pd.merge(day_features, df, on=['User', 'Date'], how='inner')
        
        # 6. day_kitchen_usage_rate
        if 'day_kitchen_usage_rate' in data_dict:
            df = data_dict['day_kitchen_usage_rate'].copy()
            df = df.rename(columns={'usage_ratio': 'day_kitchen_usage_rate'})
            day_features = pd.merge(day_features, df[['User', 'Date', 'day_kitchen_usage_rate']], 
                                  on=['User', 'Date'], how='inner')
        
        # Night 특성 (16개 = day 15개 + night_bathroom_usage 1개)
        night_features = day_features.copy()
        
        # 7. night_bathroom_usage 추가
        if 'night_bathroom_usage' in data_dict:
            df = data_dict['night_bathroom_usage'].copy()
            df = df.rename(columns={'bathroom_count': 'night_bathroom_usage'})
            night_features = pd.merge(night_features, df[['User', 'Date', 'night_bathroom_usage']], 
                                    on=['User', 'Date'], how='inner')
        
        print(f"Day 특성 병합 완료: {day_features.shape}")
        print(f"Night 특성 병합 완료: {night_features.shape}")
        
        return day_features, night_features
    
    def get_feature_columns(self) -> Tuple[List[str], List[str]]:
        """특성 컬럼명 반환"""
        day_columns = [
            'day_inactive_total',
            'day_inactive_room_01', 'day_inactive_room_02', 'day_inactive_room_03', 'day_inactive_room_04',
            'day_toggle_total',
            'day_toggle_room_01', 'day_toggle_room_02', 'day_toggle_room_03', 'day_toggle_room_04',
            'day_on_ratio_room_01', 'day_on_ratio_room_02', 'day_on_ratio_room_03', 'day_on_ratio_room_04',
            'day_kitchen_usage_rate'
        ]
        
        night_columns = day_columns + ['night_bathroom_usage']
        
        return day_columns, night_columns
    
    def check_data_quality(self, df: pd.DataFrame, feature_type: str) -> Dict:
        """데이터 품질 확인"""
        quality_report = {
            'total_records': len(df),
            'unique_users': df['User'].nunique(),
            'date_range': (df['Date'].min(), df['Date'].max()),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated(subset=['User', 'Date']).sum()
        }
        
        print(f"\n=== {feature_type} 데이터 품질 보고서 ===")
        print(f"총 레코드 수: {quality_report['total_records']}")
        print(f"고유 사용자 수: {quality_report['unique_users']}")
        print(f"날짜 범위: {quality_report['date_range'][0]} ~ {quality_report['date_range'][1]}")
        print(f"중복 레코드: {quality_report['duplicate_records']}")
        
        missing_count = sum(quality_report['missing_values'].values())
        if missing_count > 0:
            print(f"경고: 결측값 {missing_count}개 발견")
            for col, count in quality_report['missing_values'].items():
                if count > 0:
                    print(f"  - {col}: {count}개")
        else:
            print("결측값 없음")
        
        return quality_report
    
    def create_abnormal_detection_table(self):
        """abnormal_detection 테이블 생성"""
        if not self.connection or not self.connection.is_connected():
            print("데이터베이스 연결이 필요합니다.")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # 테이블 생성 SQL
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS abnormal_detection (
                User VARCHAR(20) NOT NULL,
                Date DATE NOT NULL,
                Type VARCHAR(10) NOT NULL,
                OCSVM_prediction BOOLEAN NOT NULL,
                OCSVM_score FLOAT NOT NULL,
                Isforest_prediction BOOLEAN NOT NULL,
                Isforest_score FLOAT NOT NULL,
                Consensus_prediction BOOLEAN NOT NULL,
                Consensus_score FLOAT NOT NULL,
                PRIMARY KEY (User, Date, Type),
                INDEX idx_user_date (User, Date),
                INDEX idx_type (Type),
                INDEX idx_consensus (Consensus_prediction)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            cursor.execute(create_table_sql)
            self.connection.commit()
            print("abnormal_detection 테이블 생성 완료")
            return True
            
        except Error as e:
            print(f"테이블 생성 오류: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def insert_abnormal_detection_data(self, detection_results: pd.DataFrame, model_type: str):
        """abnormal_detection 테이블에 데이터 삽입"""
        if not self.connection or not self.connection.is_connected():
            print("데이터베이스 연결이 필요합니다.")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # 삽입할 데이터의 날짜 범위 확인
            min_date = detection_results['Date'].min()
            max_date = detection_results['Date'].max()
            
            # 해당 날짜 범위와 타입의 기존 데이터 삭제
            delete_sql = """
            DELETE FROM abnormal_detection 
            WHERE Type = %s AND Date >= %s AND Date <= %s
            """
            cursor.execute(delete_sql, (model_type, min_date, max_date))
            deleted_count = cursor.rowcount
            print(f"{model_type} 타입의 기존 데이터 {deleted_count}건 삭제")
            
            # 새 데이터 삽입 (REPLACE INTO 사용으로 중복 방지)
            insert_sql = """
            REPLACE INTO abnormal_detection 
            (User, Date, Type, OCSVM_prediction, OCSVM_score, Isforest_prediction, Isforest_score, Consensus_prediction, Consensus_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for _, row in detection_results.iterrows():
                cursor.execute(insert_sql, (
                    row['User'],
                    row['Date'],
                    model_type,
                    bool(row['one_class_svm_is_anomaly']),
                    float(row['one_class_svm_score']),
                    bool(row['isolation_forest_is_anomaly']),
                    float(row['isolation_forest_score']),
                    bool(row['consensus_anomaly']),
                    float(row['anomaly_severity'])
                ))
            
            self.connection.commit()
            print(f"{model_type} 모델 결과 {len(detection_results)}건 저장 완료")
            return True
            
        except Error as e:
            print(f"데이터 삽입 오류: {e}")
            return False
        finally:
            if cursor:
                cursor.close()

def main():
    """메인 실행 함수 - 데이터 로딩 테스트"""
    # 데이터 로더 초기화 및 연결 (환경변수 사용)
    loader = RealDataLoader()
    
    if not loader.connect():
        print("데이터베이스 연결 실패")
        return
    
    try:
        # 훈련 데이터 로드 (2025-04-12 ~ 2025-05-31)
        print("\n=== 훈련 데이터 로드 ===")
        train_data = loader.load_all_tables('2025-04-12', '2025-05-31')
        
        if train_data:
            # 특성 병합
            day_train, night_train = loader.merge_features_for_training(train_data)
            
            # 데이터 품질 확인
            day_quality = loader.check_data_quality(day_train, "Day 훈련")
            night_quality = loader.check_data_quality(night_train, "Night 훈련")
            
            print(f"\nDay 특성: {day_train.shape[1] - 2}개 (User, Date 제외)")
            print(f"Night 특성: {night_train.shape[1] - 2}개 (User, Date 제외)")
        
        # 테스트 데이터 로드 (2025-05-31 ~ 2025-06-05)  
        print("\n=== 테스트 데이터 로드 ===")
        test_data = loader.load_all_tables('2025-05-31', '2025-06-05')
        
        if test_data:
            # 특성 병합
            day_test, night_test = loader.merge_features_for_training(test_data)
            
            # 데이터 품질 확인
            day_test_quality = loader.check_data_quality(day_test, "Day 테스트")
            night_test_quality = loader.check_data_quality(night_test, "Night 테스트")
        
    finally:
        loader.disconnect()

if __name__ == "__main__":
    main() 