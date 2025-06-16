import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

class DummyDataGenerator:
    def __init__(self):
        self.led_positions = {
            '01': '안방',
            '02': '거실', 
            '03': '주방',
            '04': '화장실'
        }
        self.abnormal_hour = None  # 비정상 상태 시작 시간을 저장
        
    def generate_normal_daily_pattern(self, user_id, date):
        """정상적인 하루 패턴을 생성합니다."""
        data = []
        
        # 하루를 144개의 10분 타임슬롯으로 나눔 (24시간 * 6슬롯/시간)
        for time_slot in range(144):
            hour = time_slot // 6
            minute = (time_slot % 6) * 10
            timestamp = f"{date} {hour:02d}:{minute:02d}:00"
            
            # 기본적으로 모든 LED는 꺼진 상태
            led_states = {'01': 0, '02': 0, '03': 0, '04': 0}
            
            # 특성 기반 활동 패턴 생성
            led_states = self._apply_activity_patterns(led_states, hour, minute)
            
            # 와이드 포맷으로 데이터 생성 (정상 패턴)
            data.append({
                'User': user_id,
                'Timestamp': timestamp,
                '01': led_states['01'],
                '02': led_states['02'],
                '03': led_states['03'],
                '04': led_states['04'],
                'abnormal_hour': None  # 정상 데이터는 abnormal_hour가 None
            })
        
        return data
    
    def _apply_activity_patterns(self, led_states, hour, minute):
        """시간대별 활동 패턴을 적용합니다."""
        
        # 1. 기상시간 (6시~9시 중 랜덤)
        wake_time = random.choice([6, 7, 8, 9])
        
        # 2. 수면시간 (21시~24시 중 랜덤)
        sleep_time = random.choice([21, 22, 23, 24])
        if sleep_time == 24:
            sleep_time = 0
            
        # 수면 시간대
        if hour < wake_time or hour >= sleep_time:
            if random.random() < 0.6:  # 60% 확률로 안방에 있음
                led_states['01'] = 1
            # 화장실 방문 (수면 중 가끔)
            if random.random() < 0.02:  # 화장실 방문 확률
                led_states['04'] = 1
                if random.random() < 0.9:  # 90% 확률로 이전 방 ON
                    led_states['01'] = 1
            return led_states
        
        # 활동 시간대
        # 3. 외출 여부 (20% 확률로 감소)
        is_out = random.random() < 0.2
        if is_out:
            # 외출 시간 (1~8시간으로 감소)
            out_duration = random.randint(1, 8)
            out_start = random.randint(9, 16)  # 9시~16시 사이 외출 시작
            if out_start <= hour < out_start + out_duration:
                return led_states  # 모든 LED 꺼짐
        
        # 4. 아침식사 (60% 확률로 기상 후 2시간 내)
        if wake_time <= hour <= wake_time + 2 and random.random() < 0.6:
            led_states['03'] = 1  # 주방
            if random.random() < 0.4:  # 40% 확률로 이전 방 ON
                led_states['01'] = 1
        
        # 5. 낮에 거실 활동 (50% 확률)
        if 10 <= hour <= 18 and random.random() < 0.5:
            led_states['02'] = 1  # 거실
            # 안방/거실 동시에 켤 확률 20%
            if random.random() < 0.2:
                led_states['01'] = 1
        
        # 6. 화장실 방문 (하루 4~6회)
        if random.random() < 0.03:  # 시간당 약 0.18회
            led_states['04'] = 1
            if random.random() < 0.9:  # 90% 확률로 이전 방 ON
                if led_states['01'] == 1 or led_states['02'] == 1:
                    pass  # 이미 켜진 방 유지
                else:
                    led_states[random.choice(['01', '02'])] = 1
        
        # 7. 주방 방문 (식사 시간대)
        if hour in [12, 13, 18, 19] and random.random() < 0.3:
            led_states['03'] = 1
            if random.random() < 0.4:  # 40% 확률로 이전 방 ON
                led_states[random.choice(['01', '02'])] = 1
        
        # 기본 활동 (안방이나 거실에 있을 확률)
        if sum(led_states.values()) == 0:
            if random.random() < 0.4:  # 기본 활동 확률
                # 거실 선택 확률을 더 높임
                if random.random() < 0.7:
                    led_states['02'] = 1  # 거실
                else:
                    led_states['01'] = 1  # 안방
        
        return led_states
    
    def generate_abnormal_pattern(self, user_id, date_list, pattern_type):
        """비정상 패턴을 생성합니다."""
        data = []
        
        # 각 사용자마다 비정상 상태 시작 시간 미리 결정 (0시부터 23시까지)
        if pattern_type == 'immediate':
            death_day = 0
            self.abnormal_hour = random.randint(0, 23)
        elif pattern_type == 'rapid':
            death_day = 2  
            self.abnormal_hour = random.randint(0, 23)
        elif pattern_type == 'gradual':
            death_day = random.randint(4, 7)
            self.abnormal_hour = random.randint(0, 23)
        
        # 첫날부터 비정상 패턴 시작
        pattern_start_day = 0
        
        for day_idx, date in enumerate(date_list):
            if day_idx < pattern_start_day:
                # 정상 패턴 (비정상 데이터셋 내의 정상 기간)
                daily_data = self.generate_normal_daily_pattern(user_id, date)
                # 비정상 데이터셋이므로 모든 레코드에 미리 결정된 abnormal_hour 기록
                for record in daily_data:
                    record['abnormal_hour'] = self.abnormal_hour
                data.extend(daily_data)
            else:
                # 비정상 패턴 적용
                days_into_pattern = day_idx - pattern_start_day
                daily_data = self._apply_abnormal_pattern(
                    user_id, date, pattern_type, days_into_pattern, death_day
                )
                data.extend(daily_data)
        
        return data
    
    def _apply_abnormal_pattern(self, user_id, date, pattern_type, days_into_pattern, death_day):
        """비정상 패턴을 적용합니다."""
        data = []
        
        current_day = days_into_pattern  # 현재 날짜 계산
        
        if pattern_type == 'immediate':
            # 즉시 중단: 첫날부터 활동 중단
            if current_day == death_day:
                return self._generate_death_pattern(user_id, date)
            elif current_day > death_day:
                return self._generate_post_death_pattern(user_id, date)
                
        elif pattern_type == 'rapid':
            # 급격한 악화: 1~2일 내 악화
            if current_day < death_day:
                activity_level = 1.0 - (days_into_pattern * 0.4)  # 더 급격한 감소
                return self._generate_deteriorating_pattern(user_id, date, activity_level)
            elif current_day == death_day:
                return self._generate_death_pattern(user_id, date)
            else:
                return self._generate_post_death_pattern(user_id, date)
                
        elif pattern_type == 'gradual':
            # 점진적 악화: 4~7일 내 악화
            deterioration_days = death_day  # 악화 기간 계산
            if current_day < death_day:
                activity_level = 1.0 - (days_into_pattern / deterioration_days * 0.8)  # 점진적 감소
                return self._generate_deteriorating_pattern(user_id, date, activity_level)
            elif current_day == death_day:
                return self._generate_death_pattern(user_id, date)
            else:
                return self._generate_post_death_pattern(user_id, date)
        
        return data
    
    def _generate_deteriorating_pattern(self, user_id, date, activity_level):
        """활동량이 감소하는 패턴을 생성합니다."""
        data = []
        
        for time_slot in range(144):
            hour = time_slot // 6
            minute = (time_slot % 6) * 10
            timestamp = f"{date} {hour:02d}:{minute:02d}:00"
            
            led_states = {'01': 0, '02': 0, '03': 0, '04': 0}
            
            # 활동량 감소 적용 - 토글 횟수를 크게 줄임
            if random.random() < activity_level * 0.2:  # 토글 빈도 대폭 감소
                led_states = self._apply_activity_patterns(led_states, hour, minute)
                # 활동량에 따라 더 많은 LED 끄기
                for led_id in led_states:
                    if led_states[led_id] == 1 and random.random() > activity_level:
                        led_states[led_id] = 0
            else:
                # 대부분의 시간을 한 곳에서만 보냄 (토글 없이)
                if random.random() < 0.9:
                    led_states['01'] = 1  # 주로 안방에만 있음
            
            # 와이드 포맷으로 데이터 생성 (악화 패턴)
            data.append({
                'User': user_id,
                'Timestamp': timestamp,
                '01': led_states['01'],
                '02': led_states['02'],
                '03': led_states['03'],
                '04': led_states['04'],
                'abnormal_hour': self.abnormal_hour  # 미리 결정된 비정상 상태 시작 시간 기록
            })
        
        return data
    
    def _generate_death_pattern(self, user_id, date):
        """사망이 발생하는 날의 패턴을 생성합니다."""
        data = []
        
        for time_slot in range(144):
            hour = time_slot // 6
            minute = (time_slot % 6) * 10
            timestamp = f"{date} {hour:02d}:{minute:02d}:00"
            
            if hour < self.abnormal_hour:
                # 사망 전: 정상 또는 악화된 패턴
                led_states = {'01': 0, '02': 0, '03': 0, '04': 0}
                led_states = self._apply_activity_patterns(led_states, hour, minute)
            else:
                # 비정상 상태 이후: LED 상태 고정 (비정상 시점의 상태 유지)
                if hour == self.abnormal_hour and minute == 0:
                    # 비정상 시점 상태 설정
                    abnormal_location = random.choice(['01', '02', '03', '04'])
                    led_states = {'01': 0, '02': 0, '03': 0, '04': 0}
                    led_states[abnormal_location] = 1
                    self.abnormal_led_state = led_states
                else:
                    # 이전 상태 유지
                    led_states = self.abnormal_led_state
            
            # 와이드 포맷으로 데이터 생성 (비정상 패턴)
            data.append({
                'User': user_id,
                'Timestamp': timestamp,
                '01': led_states['01'],
                '02': led_states['02'],
                '03': led_states['03'],
                '04': led_states['04'],
                'abnormal_hour': self.abnormal_hour  # 비정상 상태 시작 시간 기록
            })
        
        return data
    
    def _generate_post_death_pattern(self, user_id, date):
        """비정상 상태 이후 LED 상태를 고정으로 생성합니다."""
        data = []
        
        for time_slot in range(144):
            hour = time_slot // 6
            minute = (time_slot % 6) * 10
            timestamp = f"{date} {hour:02d}:{minute:02d}:00"
            
            # 와이드 포맷으로 데이터 생성 (비정상 상태 이후 패턴)
            data.append({
                'User': user_id,
                'Timestamp': timestamp,
                '01': self.abnormal_led_state['01'],
                '02': self.abnormal_led_state['02'],
                '03': self.abnormal_led_state['03'],
                '04': self.abnormal_led_state['04'],
                'abnormal_hour': self.abnormal_hour  # 비정상 상태 시작 시간 기록
            })
        
        return data
    
    def generate_dataset(self, dataset_type, num_users, num_days):
        """데이터셋을 생성합니다."""
        all_data = []
        
        # 날짜 리스트 생성
        start_date = datetime(2024, 1, 1)
        date_list = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                    for i in range(num_days)]
        
        for user_id in range(1, num_users + 1):
            user_id_str = f"{user_id:04d}"  # 4자리 숫자로 변경 (10001, 10002, ...)
            
            if dataset_type == 'normal':
                for date in date_list:
                    daily_data = self.generate_normal_daily_pattern(user_id_str, date)
                    all_data.extend(daily_data)
            else:
                # 비정상 패턴
                pattern_data = self.generate_abnormal_pattern(
                    user_id_str, date_list, dataset_type
                )
                all_data.extend(pattern_data)
        
        return pd.DataFrame(all_data)

def main():
    generator = DummyDataGenerator()
    
    # 데이터 생성 설정
    datasets = {
        'train_dataset.csv': ('normal', 300, 30),
        'normal_test_dataset.csv': ('normal', 60, 10),
        'immediate_abnormal_test_dataset.csv': ('immediate', 60, 10),
        'rapid_abnormal_test_dataset.csv': ('rapid', 60, 10),
        'gradual_abnormal_test_dataset.csv': ('gradual', 60, 10)
    }
    
    # 데이터 폴더 생성 (절대경로로 수정)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    for filename, (dataset_type, num_users, num_days) in datasets.items():
        print(f"Generating {filename}...")
        
        df = generator.generate_dataset(dataset_type, num_users, num_days)
        
        # CSV 파일로 저장
        output_path = os.path.join(data_dir, filename)
        df.to_csv(output_path, index=False)
        
        print(f"Generated {filename}: {len(df)} records")
        print(f"Users: {num_users}, Days: {num_days}")
        print(f"Saved to: {output_path}\n")

if __name__ == "__main__":
    main() 