import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
sns.set_style("whitegrid")

class CombinedEvaluationVisualizer:
    def __init__(self):
        """평가 및 시각화 클래스 초기화"""
        # 메서드별 색상 및 라벨 정의
        self.methods = ['traditional', 'isolation_forest', 'one_class_svm']
        self.method_colors = {
            'traditional': '#FF6B6B',
            'isolation_forest': '#4ECDC4',
            'one_class_svm': '#45B7D1'
        }
        self.method_labels = {
            'traditional': 'Traditional Method',
            'isolation_forest': 'Isolation Forest',
            'one_class_svm': 'One-Class SVM'
        }
        
        # 경로 설정
        self.base_dir = Path(os.getcwd())
        self.charts_dir = self.base_dir / "charts" / "detection_performance"
        self.dummy_data_dir = self.base_dir / "dummy_data"
        self.models_dir = self.base_dir / "dummy_models"
        
        # 디렉토리 생성
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 및 평가 결과 저장
        self.models = {}
        self.evaluation_results = {}
    
    def load_models(self):
        """훈련된 모델들 로드"""
        print("모델 로딩 중...")
        self.models = {'day': {}, 'night': {}}
        
        for time_period in ['day', 'night']:
            period_path = self.models_dir / time_period
            
            try:
                # 모델들 로드 (joblib 사용)
                self.models[time_period]['scaler'] = joblib.load(period_path / 'scaler.pkl')
                self.models[time_period]['isolation_forest'] = joblib.load(period_path / 'isolation_forest.pkl')
                self.models[time_period]['one_class_svm'] = joblib.load(period_path / 'one_class_svm.pkl')
                
            except Exception as e:
                print(f"모델 로딩 오류 ({time_period}): {e}")
                continue
        
        print("모델 로딩 완료")

    def load_abnormal_hours(self, dataset_name):
        """abnormal_hour 데이터 로드"""
        abnormal_hour_file = self.dummy_data_dir / "abnormal_hour" / f"{dataset_name}.csv"
        if abnormal_hour_file.exists():
            return pd.read_csv(abnormal_hour_file)
        return None

    def load_processed_data(self, dataset_name):
        """처리된 특성 데이터 로드"""
        processed_path = self.dummy_data_dir / "processed" / dataset_name
        
        if not processed_path.exists():
            print(f"경로 없음: {processed_path}")
            return {}
        
        features = {}
        for time_period in ['day', 'night']:
            features[time_period] = {}
            
            # 각 특성 파일 로드
            feature_files = {
                'inactive_total': f'{time_period}_inactive_total.csv',
                'inactive_room': f'{time_period}_inactive_room.csv',
                'toggle_total': f'{time_period}_toggle_total.csv',
                'toggle_room': f'{time_period}_toggle_room.csv',
                'on_ratio_room': f'{time_period}_on_ratio_room.csv',
                'kitchen_usage_rate': f'{time_period}_kitchen_usage_rate.csv'
            }
            
            if time_period == 'night':
                feature_files['bathroom_usage'] = 'night_bathroom_usage.csv'
            
            for feature_key, filename in feature_files.items():
                file_path = processed_path / filename
                if file_path.exists():
                    features[time_period][feature_key] = pd.read_csv(file_path)
        
        return features

    def create_feature_vector_for_user(self, dataset_features, time_period, user_id):
        """특정 사용자에 대한 특성 벡터 생성"""
        if time_period == 'day':
            # Day 모델: 15개 특징
            expected_features = [
                'day_inactive_total',
                'day_inactive_room_01', 'day_inactive_room_02', 'day_inactive_room_03', 'day_inactive_room_04',
                'day_toggle_total', 
                'day_toggle_room_01', 'day_toggle_room_02', 'day_toggle_room_03', 'day_toggle_room_04',
                'day_on_ratio_room_01', 'day_on_ratio_room_02', 'day_on_ratio_room_03', 'day_on_ratio_room_04',
                'day_kitchen_usage_rate'
            ]
        else:  # night
            # Night 모델: 16개 특징
            expected_features = [
                'night_inactive_total',
                'night_inactive_room_01', 'night_inactive_room_02', 'night_inactive_room_03', 'night_inactive_room_04', 
                'night_toggle_total',
                'night_toggle_room_01', 'night_toggle_room_02', 'night_toggle_room_03', 'night_toggle_room_04',
                'night_on_ratio_room_01', 'night_on_ratio_room_02', 'night_on_ratio_room_03', 'night_on_ratio_room_04',
                'night_kitchen_usage_rate',
                'night_bathroom_usage'
            ]
        
        user_feature_vector = []
        
        for expected_feature in expected_features:
            # 특성명에서 time_period 접두사 제거하여 파일명과 매칭
            base_feature = expected_feature.replace(f'{time_period}_', '')
            
            # 방별 특징 처리 (01, 02, 03, 04)
            if base_feature.endswith('_01') or base_feature.endswith('_02') or base_feature.endswith('_03') or base_feature.endswith('_04'):
                room_num = base_feature[-2:]  # 01, 02, 03, 04
                feature_type = base_feature[:-3]  # inactive_room, toggle_room, on_ratio_room
                
                if feature_type in dataset_features[time_period]:
                    df = dataset_features[time_period][feature_type]
                    user_data = df[df['User'] == user_id]
                    
                    if len(user_data) > 0 and room_num in df.columns:
                        # 해당 방의 평균값 사용
                        value = user_data[room_num].mean()
                        user_feature_vector.append(value)
                    else:
                        user_feature_vector.append(0.0)
                else:
                    user_feature_vector.append(0.0)
            
            # 전체 특징 처리 (total, kitchen_usage_rate, bathroom_usage)
            else:
                if base_feature in dataset_features[time_period]:
                    df = dataset_features[time_period][base_feature]
                    user_data = df[df['User'] == user_id]
                    
                    if len(user_data) > 0:
                        # User, Date를 제외한 첫 번째 수치형 컬럼 사용
                        numeric_cols = [col for col in user_data.columns 
                                      if col not in ['User', 'Date'] and not col.startswith('Unnamed')]
                        if len(numeric_cols) > 0:
                            # 평균값 사용
                            value = user_data[numeric_cols[0]].mean()
                            user_feature_vector.append(value)
                        else:
                            user_feature_vector.append(0.0)
                    else:
                        user_feature_vector.append(0.0)  # 사용자 데이터 없음
                else:
                    user_feature_vector.append(0.0)  # 기본값
        
        return np.array(user_feature_vector).reshape(1, -1)

    def find_last_led_change(self, dataset_name, user_id):
        """raw 데이터에서 마지막 LED 변화 시점 찾기"""
        try:
            raw_file = self.dummy_data_dir / "raw" / f"{dataset_name}.csv"
            
            # 사용자 데이터만 필터링해서 읽기
            import pandas as pd
            df = pd.read_csv(raw_file)
            
            user_data = df[df['User'] == user_id].copy()
            
            if len(user_data) == 0:
                return None
                
            # 타임스탬프를 datetime으로 변환
            user_data['Timestamp'] = pd.to_datetime(user_data['Timestamp'])
            user_data = user_data.sort_values('Timestamp')
            
            # 이전 상태와 비교해서 변화 시점 찾기
            last_change_time = None
            prev_state = None
            change_count = 0
            
            for _, row in user_data.iterrows():
                current_state = (row['01'], row['02'], row['03'], row['04'])
                
                if prev_state is not None and current_state != prev_state:
                    last_change_time = row['Timestamp']
                    change_count += 1
                
                prev_state = current_state
            
            return last_change_time
            
        except Exception as e:
            print(f"Error finding last LED change for user {user_id}: {e}")
            return None

    def traditional_detection_time(self, dataset_name, user_id, abnormal_hour):
        """전통적 방법: 마지막 LED 변화 + 24시간 후 탐지"""
        last_change = self.find_last_led_change(dataset_name, user_id)
        
        if last_change is None:
            return 72  # 데이터 없으면 미탐지
        
        # Raw 데이터에서 첫 번째 타임스탬프 찾기 (데이터 시작 날짜)
        try:
            raw_file = self.dummy_data_dir / "raw" / f"{dataset_name}.csv"
            df = pd.read_csv(raw_file)
            user_data = df[df['User'] == user_id].copy()
            
            if len(user_data) == 0:
                return 72
            
            user_data['Timestamp'] = pd.to_datetime(user_data['Timestamp'])
            first_timestamp = user_data['Timestamp'].min()
            
            # abnormal_hour를 첫 번째 날짜 기준으로 설정
            abnormal_datetime = first_timestamp.replace(
                hour=int(abnormal_hour), 
                minute=int((abnormal_hour % 1) * 60), 
                second=0, 
                microsecond=0
            )
            
        except Exception as e:
            print(f"Error getting first timestamp for user {user_id}: {e}")
            return 72
        
        # 마지막 변화 시점 + 24시간 = 탐지 시점
        detection_time = last_change + pd.Timedelta(hours=24)
        
        # abnormal_hour부터 탐지 시점까지의 실제 시간차 (시간 단위)
        time_diff_hours = (detection_time - abnormal_datetime).total_seconds() / 3600
        
        # 음수인 경우 (abnormal_hour가 탐지보다 늦음) 72시간으로 설정
        if time_diff_hours < 0:
            return 72
        
        # 72시간 이내에 탐지되면 성공
        return time_diff_hours if time_diff_hours <= 72 else 72

    def predict_anomaly(self, features, time_period, method):
        """특정 방법으로 이상치 예측 (ML 모델만)"""        
        if method in ['isolation_forest', 'one_class_svm']:
            if method in self.models[time_period]:
                model = self.models[time_period][method]
                scaler = self.models[time_period]['scaler']
                
                # 정규화 및 예측
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                
                # sklearn의 -1(이상), 1(정상)을 1(이상), 0(정상)으로 변환
                result = 1 if prediction == -1 else 0
                
                return result
        
        return 0

    def evaluate_user_detection(self, dataset_name, user_id, abnormal_hour, is_normal=False):
        """특정 사용자의 이상 탐지 평가"""
        dataset_features = self.load_processed_data(dataset_name)
        
        # abnormal_hour 기준으로 시작 모델 결정 (normal인 경우 임의 시작점)
        if is_normal:
            # 정상 데이터는 임의 시점부터 시작 (오전 6시로 고정)
            start_model = 'night'
            start_time = 6
        else:
            if abnormal_hour < 6:  # 0~5시: night 모델부터
                start_model = 'night'
                start_time = 6  # 다음 오전 6시부터 시작
            elif abnormal_hour < 18:  # 6~17시: day 모델부터  
                start_model = 'day'
                start_time = 18  # 다음 오후 6시부터 시작
            else:  # 18~23시: 익일 night 모델부터
                start_model = 'night'
                start_time = 6 + 24  # 익일 오전 6시부터 시작
        
        detection_results = {}
        
        # 각 방법별로 평가
        for method in ['traditional', 'isolation_forest', 'one_class_svm']:
            if method == 'traditional':
                # 전통적 방법: raw 데이터에서 마지막 LED 변화 + 24시간
                if is_normal:
                    # 정상 데이터에서는 실제 LED 변화를 확인해서 탐지 여부 결정
                    detection_time = self.traditional_detection_time(dataset_name, user_id, 0)  # abnormal_hour=0으로 설정
                    detected = detection_time < 72  # 72시간 내에 탐지되면 False Positive
                else:
                    detection_time = self.traditional_detection_time(dataset_name, user_id, abnormal_hour)
                    detected = detection_time < 72
                
                detection_results[method] = {
                    'detected': detected,
                    'detection_time': detection_time
                }
                
            else:
                # ML 방법: 6시간마다 검사
                detected = False
                detection_time = None
                current_time = start_time
                current_model = start_model
                
                # 최대 72시간(12회 검사) 동안 6시간 간격으로 검사
                for check_num in range(12):  # 72시간 / 6시간 = 12회
                    try:
                        features = self.create_feature_vector_for_user(dataset_features, current_model, user_id)
                        
                        # 이상치 예측
                        is_anomaly = self.predict_anomaly(features, current_model, method)
                        
                        if is_anomaly:
                            detected = True
                            # 탐지 시간 계산
                            if is_normal:
                                detection_time = check_num * 6  # 정상 데이터는 시작점부터 계산
                            else:
                                # abnormal_hour로부터 얼마나 지났는지
                                if current_time >= 24:
                                    actual_time = current_time - 24
                                else:
                                    actual_time = current_time
                                
                                detection_time = actual_time - abnormal_hour
                                if detection_time < 0:
                                    detection_time += 24  # 다음날
                            
                            break
                    except Exception as e:
                        print(f"Error processing user {user_id} with {method}: {e}")
                        break
                    
                    # 다음 검사 시간으로 이동
                    current_time += 6
                    if current_time >= 48:  # 2일을 넘어가면 다시 0부터
                        current_time -= 24
                    elif current_time >= 24:
                        current_time = current_time  # 다음날 유지
                    
                    # 모델 전환 (6시 -> night, 18시 -> day)
                    effective_time = current_time % 24
                    if effective_time == 6:
                        current_model = 'night'
                    elif effective_time == 18:
                        current_model = 'day'
                
                detection_results[method] = {
                    'detected': detected,
                    'detection_time': detection_time if detected else 72  # 미탐지시 72시간
                }
        
        return detection_results

    def evaluate_dataset(self, dataset_name, is_normal=False):
        """데이터셋 전체 평가"""
        print(f"데이터셋 평가 중: {dataset_name}")
        
        if is_normal:
            # 정상 데이터는 abnormal_hour 없이 사용자 목록만 가져오기
            dataset_features = self.load_processed_data(dataset_name)
            if 'day' in dataset_features and 'inactive_total' in dataset_features['day']:
                users = dataset_features['day']['inactive_total']['User'].unique()
                abnormal_hours_data = pd.DataFrame({'User': users, 'abnormal_hour': [6] * len(users)})  # 임의값
            else:
                print(f"정상 데이터 로드 실패: {dataset_name}")
                return {}
        else:
            # abnormal_hour 데이터 로드
            abnormal_hours_data = self.load_abnormal_hours(dataset_name)
            if abnormal_hours_data is None:
                print(f"abnormal_hour 데이터 없음: {dataset_name}")
                return {}
        
        # 각 사용자별 평가
        all_results = {}
        for _, row in abnormal_hours_data.iterrows():
            user_id = row['User']
            abnormal_hour = row['abnormal_hour'] if not is_normal else 0
            
            user_results = self.evaluate_user_detection(dataset_name, user_id, abnormal_hour, is_normal)
            all_results[user_id] = user_results
        
        # 집계 결과 계산
        summary = {}
        for method in ['traditional', 'isolation_forest', 'one_class_svm']:
            detected_count = sum(1 for r in all_results.values() if r[method]['detected'])
            total_count = len(all_results)
            
            detection_times = [r[method]['detection_time'] for r in all_results.values() if r[method]['detected']]
            avg_detection_time = np.mean(detection_times) if detection_times else 72
            
            if is_normal:
                # 정상 데이터에서 탐지되면 False Positive
                false_positive_rate = (detected_count / total_count) * 100 if total_count > 0 else 0
                detection_rate_72h = 0  # 정상 데이터는 탐지율 계산 안함
            else:
                detection_rate_72h = (detected_count / total_count) * 100 if total_count > 0 else 0
                false_positive_rate = 0
            
            summary[method] = {
                'detection_rate_72h': detection_rate_72h,
                'avg_detection_time': avg_detection_time,
                'detected_count': detected_count,
                'total_count': total_count,
                'false_positive_rate': false_positive_rate
            }
            
            if is_normal:
                print(f"  {method}: {detected_count}/{total_count} False Positive ({false_positive_rate:.1f}%)")
            else:
                print(f"  {method}: {detected_count}/{total_count} 탐지 ({detection_rate_72h:.1f}%), 평균 탐지 시간: {avg_detection_time:.1f}h")
        
        return {
            'summary': summary,
            'detailed': all_results
        }

    def generate_evaluation_data(self):
        """실제 평가 데이터 생성"""
        print("실제 모델과 abnormal_hour 데이터로 평가 수행 중...")
        
        # 데이터셋 목록
        datasets = {
            'immediate': 'immediate_abnormal_test_dataset',
            'rapid': 'rapid_abnormal_test_dataset', 
            'gradual': 'gradual_abnormal_test_dataset'
        }
        
        evaluation_results = {
            'detection_72h': {},
            'avg_detection_time': {},
            'time_based_detection': {'all': {}},
            'false_positive': {}
        }
        
        # 각 데이터셋 평가 (비정상 데이터)
        all_dataset_results = {}
        for dataset_type, dataset_name in datasets.items():
            dataset_results = self.evaluate_dataset(dataset_name, is_normal=False)
            all_dataset_results[dataset_type] = dataset_results
            
            # 결과 저장
            for method in ['traditional', 'isolation_forest', 'one_class_svm']:
                if dataset_type not in evaluation_results['detection_72h']:
                    evaluation_results['detection_72h'][dataset_type] = {}
                if dataset_type not in evaluation_results['avg_detection_time']:
                    evaluation_results['avg_detection_time'][dataset_type] = {}
                
                summary = dataset_results['summary'][method]
                evaluation_results['detection_72h'][dataset_type][method] = summary['detection_rate_72h']
                evaluation_results['avg_detection_time'][dataset_type][method] = summary['avg_detection_time']
        
        # 정상 데이터로 False Positive 평가
        print("\n정상 데이터로 False Positive 평가 중...")
        normal_results = self.evaluate_dataset('normal_test_dataset', is_normal=True)
        
        for method in ['traditional', 'isolation_forest', 'one_class_svm']:
            fp_rate = normal_results['summary'][method]['false_positive_rate']
            
            # 각 데이터셋에 동일한 FP rate 적용
            for dataset_type in ['immediate', 'rapid', 'gradual']:
                if dataset_type not in evaluation_results['false_positive']:
                    evaluation_results['false_positive'][dataset_type] = {}
                evaluation_results['false_positive'][dataset_type][method] = fp_rate
        
        # 전체 평균 계산
        evaluation_results['detection_72h']['all'] = {}
        evaluation_results['avg_detection_time']['all'] = {}
        evaluation_results['false_positive']['all'] = {}
        
        for method in ['traditional', 'isolation_forest', 'one_class_svm']:
            # 전체 평균 계산
            detection_rates = [evaluation_results['detection_72h'][dt][method] for dt in ['immediate', 'rapid', 'gradual']]
            detection_times = [evaluation_results['avg_detection_time'][dt][method] for dt in ['immediate', 'rapid', 'gradual']]
            
            evaluation_results['detection_72h']['all'][method] = np.mean(detection_rates)
            evaluation_results['avg_detection_time']['all'][method] = np.mean(detection_times)
            evaluation_results['false_positive']['all'][method] = normal_results['summary'][method]['false_positive_rate']
        
        # 시간대별 탐지율 계산 (실제 데이터 기반)
        for time_window in [3, 6, 12, 24]:
            evaluation_results['time_based_detection']['all'][time_window] = {}
            
            for method in ['traditional', 'isolation_forest', 'one_class_svm']:
                total_detected = 0
                total_users = 0
                
                # 모든 데이터셋에서 해당 시간 내 탐지된 사용자 수 계산
                for dataset_type in ['immediate', 'rapid', 'gradual']:
                    detailed_results = all_dataset_results[dataset_type]['detailed']
                    
                    for user_id, user_result in detailed_results.items():
                        total_users += 1
                        if user_result[method]['detected'] and user_result[method]['detection_time'] <= time_window:
                            total_detected += 1
                
                detection_rate = (total_detected / total_users * 100) if total_users > 0 else 0
                evaluation_results['time_based_detection']['all'][time_window][method] = detection_rate
        
        print("실제 평가 완료")
        return evaluation_results

    def create_72h_detection_rate_by_dataset(self):
        """72시간 내 데이터셋별 감지율 차트"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        datasets = ['Immediate', 'Rapid', 'Gradual']
        dataset_keys = ['immediate', 'rapid', 'gradual']
        
        detection_72h = {
            'traditional': [self.evaluation_results['detection_72h'][dk]['traditional'] for dk in dataset_keys],
            'isolation_forest': [self.evaluation_results['detection_72h'][dk]['isolation_forest'] for dk in dataset_keys],
            'one_class_svm': [self.evaluation_results['detection_72h'][dk]['one_class_svm'] for dk in dataset_keys]
        }
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, method in enumerate(self.methods):
            rates = detection_72h[method]
            bars = ax.bar(x + (i - 1) * width, rates, width, label=self.method_labels[method], 
                        color=self.method_colors[method], alpha=0.8)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('72-hour Detection Rate (%)', fontsize=12)
        ax.set_title('Detection Rate within 72 Hours by Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, '72h_detection_rate_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 72-hour detection rate by dataset 차트 생성 완료")

    def create_72h_detection_rate_all(self):
        """72시간 내 전체 감지율 차트"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['Traditional', 'Isolation Forest', 'One-Class SVM']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        detection_rates_72h = [
            self.evaluation_results['detection_72h']['all']['traditional'],
            self.evaluation_results['detection_72h']['all']['isolation_forest'],
            self.evaluation_results['detection_72h']['all']['one_class_svm']
        ]
        
        bars = ax.bar(methods, detection_rates_72h, color=colors, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                  f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Detection Rate (%)', fontsize=12)
        ax.set_title('Overall 72-hour Detection Rate', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, '72h_detection_rate_all.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 72-hour detection rate (All) 차트 생성 완료")

    def create_time_based_detection_rate(self):
        """시간별 누적 감지율 차트 - 막대그래프로 변경"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        time_windows = [3, 6, 12, 24]
        time_labels = ['3h', '6h', '12h', '24h']
        
        # 데이터 준비
        detection_by_time = {
            'traditional': [self.evaluation_results['time_based_detection']['all'][tw]['traditional'] for tw in time_windows],
            'isolation_forest': [self.evaluation_results['time_based_detection']['all'][tw]['isolation_forest'] for tw in time_windows],
            'one_class_svm': [self.evaluation_results['time_based_detection']['all'][tw]['one_class_svm'] for tw in time_windows]
        }
        
        x = np.arange(len(time_windows))
        width = 0.25
        
        for i, method in enumerate(self.methods):
            rates = detection_by_time[method]
            bars = ax.bar(x + (i - 1) * width, rates, width, label=self.method_labels[method], 
                        color=self.method_colors[method], alpha=0.8)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Time Window', fontsize=12)
        ax.set_ylabel('Cumulative Detection Rate (%)', fontsize=12)
        ax.set_title('Detection Rate by Time Window', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(time_labels)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'time_based_detection_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Time-based detection rate 차트 생성 완료")

    def create_false_positive_by_dataset(self):
        """데이터셋별 False Positive 차트"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        datasets = ['Immediate', 'Rapid', 'Gradual']
        dataset_keys = ['immediate', 'rapid', 'gradual']
        
        false_positive = {
            'traditional': [self.evaluation_results['false_positive'][dk]['traditional'] for dk in dataset_keys],
            'isolation_forest': [self.evaluation_results['false_positive'][dk]['isolation_forest'] for dk in dataset_keys],
            'one_class_svm': [self.evaluation_results['false_positive'][dk]['one_class_svm'] for dk in dataset_keys]
        }
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, method in enumerate(self.methods):
            rates = false_positive[method]
            bars = ax.bar(x + (i - 1) * width, rates, width, label=self.method_labels[method], 
                        color=self.method_colors[method], alpha=0.8)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('False Positive Rate (%)', fontsize=12)
        ax.set_title('False Positive Rate by Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, max(5, max([max(rates) for rates in false_positive.values()]) + 1))
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'false_positive_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ False positive rate by dataset 차트 생성 완료")

    def create_false_positive_all(self):
        """전체 False Positive 차트"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['Traditional', 'Isolation Forest', 'One-Class SVM']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        false_positive_rates = [
            self.evaluation_results['false_positive']['all']['traditional'],
            self.evaluation_results['false_positive']['all']['isolation_forest'],
            self.evaluation_results['false_positive']['all']['one_class_svm']
        ]
        
        bars = ax.bar(methods, false_positive_rates, color=colors, alpha=0.8)
        

        
        ax.set_ylabel('False Positive Rate (%)', fontsize=12)
        ax.set_title('Overall False Positive Rate', fontsize=14, fontweight='bold')
        max_rate = max(false_positive_rates) if any(r > 0 for r in false_positive_rates) else 5
        ax.set_ylim(0, max(5, max_rate + 1))
        
        # 0%일 때도 텍스트 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                  f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'false_positive_all.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ False positive rate (All) 차트 생성 완료")

    def create_avg_detection_time_by_dataset(self):
        """데이터셋별 평균 탐지 시간 차트"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        datasets = ['Immediate', 'Rapid', 'Gradual']
        dataset_keys = ['immediate', 'rapid', 'gradual']
        
        avg_detection_time = {
            'traditional': [self.evaluation_results['avg_detection_time'][dk]['traditional'] for dk in dataset_keys],
            'isolation_forest': [self.evaluation_results['avg_detection_time'][dk]['isolation_forest'] for dk in dataset_keys],
            'one_class_svm': [self.evaluation_results['avg_detection_time'][dk]['one_class_svm'] for dk in dataset_keys]
        }
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, method in enumerate(self.methods):
            times = avg_detection_time[method]
            bars = ax.bar(x + (i - 1) * width, times, width, label=self.method_labels[method], 
                        color=self.method_colors[method], alpha=0.8)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                      f'{height:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Average Detection Time (hours)', fontsize=12)
        ax.set_title('Average Detection Time by Dataset', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 75)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'avg_detection_time_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Average detection time by dataset 차트 생성 완료")

    def create_avg_detection_time_all(self):
        """전체 평균 탐지 시간 차트"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['Traditional', 'Isolation Forest', 'One-Class SVM']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        avg_detection_times = [
            self.evaluation_results['avg_detection_time']['all']['traditional'],
            self.evaluation_results['avg_detection_time']['all']['isolation_forest'],
            self.evaluation_results['avg_detection_time']['all']['one_class_svm']
        ]
        
        bars = ax.bar(methods, avg_detection_times, color=colors, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                  f'{height:.1f}h', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Average Detection Time (hours)', fontsize=12)
        ax.set_title('Overall Average Detection Time', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 75)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'avg_detection_time_all.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Average detection time (All) 차트 생성 완료")



    def run_complete_evaluation(self):
        """전체 평가 및 시각화 실행"""
        print("=== 통합 평가 및 시각화 시스템 시작 ===")
        
        # 1. 모델 로드
        self.load_models()
        
        # 2. 실제 평가 데이터 생성
        self.evaluation_results = self.generate_evaluation_data()
        
        # 3. 시각화 생성
        print("\n=== 시각화 생성 중 ===")
        self.create_72h_detection_rate_by_dataset()
        self.create_72h_detection_rate_all()
        self.create_time_based_detection_rate()
        self.create_false_positive_by_dataset()
        self.create_false_positive_all()
        self.create_avg_detection_time_by_dataset()
        self.create_avg_detection_time_all()
        
        # 4. 결과 요약 출력
        print("\n=== 모든 작업 완료 ===")
        print(f"📊 차트 저장 위치: {self.charts_dir}")
        
        print("\n=== 주요 결과 요약 ===")
        print("72시간 내 전체 감지율:")
        for method_key, method_name in [('traditional', 'Traditional Method'), 
                                       ('isolation_forest', 'Isolation Forest'), 
                                       ('one_class_svm', 'One-Class SVM')]:
            rate = self.evaluation_results['detection_72h']['all'][method_key]
            print(f"  - {method_name}: {rate:.1f}%")
        
        print("\n평균 탐지 시간:")
        for method_key, method_name in [('traditional', 'Traditional Method'), 
                                       ('isolation_forest', 'Isolation Forest'), 
                                       ('one_class_svm', 'One-Class SVM')]:
            time = self.evaluation_results['avg_detection_time']['all'][method_key]
            print(f"  - {method_name}: {time:.1f}시간")
        
        print("\nFalse Positive Rate:")
        for method_key, method_name in [('traditional', 'Traditional Method'), 
                                       ('isolation_forest', 'Isolation Forest'), 
                                       ('one_class_svm', 'One-Class SVM')]:
            fp_rate = self.evaluation_results['false_positive']['all'][method_key]
            print(f"  - {method_name}: {fp_rate:.1f}%")

def main():
    """메인 실행 함수"""
    evaluator = CombinedEvaluationVisualizer()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main() 