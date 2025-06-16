import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

class FeatureExtractor:
    def __init__(self, raw_data_path="data/raw", processed_data_path="data/processed"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
        # Create processed directory if not exists
        os.makedirs(processed_data_path, exist_ok=True)
    
    def load_data(self, filename):
        """Load and preprocess data from CSV file"""
        filepath = os.path.join(self.raw_data_path, filename)
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        
        # Create timeslot (10-minute intervals)
        df['Timeslot'] = df['Hour'] * 6 + df['Minute'] // 10
        
        return df
    
    def calculate_inactive_periods(self, df, start_hour, end_hour):
        """Calculate inactive periods for given time range"""
        results = []
        
        for user in df['User'].unique():
            user_data = df[df['User'] == user].copy()
            
            for date in user_data['Date'].unique():
                # Filter by time range
                if start_hour <= end_hour:
                    # Day time: same date
                    daily_data = user_data[user_data['Date'] == date].copy()
                    time_filtered = daily_data[
                        (daily_data['Hour'] >= start_hour) & 
                        (daily_data['Hour'] < end_hour)
                    ]
                else:  # Night time (crosses midnight)
                    # Night time: current evening + next morning
                    current_date = pd.to_datetime(date)
                    next_date = current_date + pd.Timedelta(days=1)
                    
                    # Current date evening (18:00-23:59)
                    current_evening = user_data[
                        (user_data['Date'] == date) & 
                        (user_data['Hour'] >= start_hour)
                    ].copy()
                    
                    # Next date morning (00:00-05:59)
                    next_morning = user_data[
                        (user_data['Date'] == next_date.date()) & 
                        (user_data['Hour'] < end_hour)
                    ].copy()
                    
                    # Combine evening and morning data
                    time_filtered = pd.concat([current_evening, next_morning], ignore_index=True)
                
                if len(time_filtered) == 0:
                    continue
                
                # Calculate inactive timeslots - when ALL rooms maintain previous state
                inactive_counts = {}
                
                # Calculate per-room inactive counts first
                for room in ['01', '02', '03', '04']:
                    room_data = time_filtered[room].values
                    if len(room_data) <= 1:
                        inactive_counts[room] = len(room_data) - 1 if len(room_data) > 0 else 0
                        continue
                    
                    # Count timeslots with no state change from previous timeslot
                    inactive_count = 0
                    for i in range(1, len(room_data)):
                        if room_data[i] == room_data[i-1]:
                            inactive_count += 1
                    
                    inactive_counts[room] = inactive_count
                
                # Calculate total inactive: when ALL rooms are inactive simultaneously
                if len(time_filtered) <= 1:
                    total_inactive = len(time_filtered) - 1 if len(time_filtered) > 0 else 0
                else:
                    total_inactive = 0
                    for i in range(1, len(time_filtered)):
                        # Check if ALL rooms maintained their previous state
                        all_inactive = True
                        for room in ['01', '02', '03', '04']:
                            if time_filtered.iloc[i][room] != time_filtered.iloc[i-1][room]:
                                all_inactive = False
                                break
                        if all_inactive:
                            total_inactive += 1
                
                results.append({
                    'User': user,
                    'Date': date,
                    'total_inactive': total_inactive,
                    **inactive_counts
                })
        
        return pd.DataFrame(results)
    
    def calculate_toggle_counts(self, df, start_hour, end_hour):
        """Calculate toggle counts for given time range"""
        results = []
        
        for user in df['User'].unique():
            user_data = df[df['User'] == user].copy()
            
            for date in user_data['Date'].unique():
                # Filter by time range
                if start_hour <= end_hour:
                    # Day time: same date
                    daily_data = user_data[user_data['Date'] == date].copy()
                    time_filtered = daily_data[
                        (daily_data['Hour'] >= start_hour) & 
                        (daily_data['Hour'] < end_hour)
                    ]
                else:  # Night time (crosses midnight)
                    # Night time: current evening + next morning
                    current_date = pd.to_datetime(date)
                    next_date = current_date + pd.Timedelta(days=1)
                    
                    # Current date evening (18:00-23:59)
                    current_evening = user_data[
                        (user_data['Date'] == date) & 
                        (user_data['Hour'] >= start_hour)
                    ].copy()
                    
                    # Next date morning (00:00-05:59)
                    next_morning = user_data[
                        (user_data['Date'] == next_date.date()) & 
                        (user_data['Hour'] < end_hour)
                    ].copy()
                    
                    # Combine evening and morning data
                    time_filtered = pd.concat([current_evening, next_morning], ignore_index=True)
                
                if len(time_filtered) == 0:
                    continue
                
                # Calculate toggle counts for each room
                toggle_counts = {}
                total_toggles = 0
                
                for room in ['01', '02', '03', '04']:
                    room_data = time_filtered[room].values
                    if len(room_data) <= 1:
                        toggle_counts[room] = 0
                        continue
                    
                    # Count state changes
                    toggles = sum(1 for i in range(1, len(room_data)) 
                                if room_data[i] != room_data[i-1])
                    
                    toggle_counts[room] = toggles
                    total_toggles += toggles
                
                results.append({
                    'User': user,
                    'Date': date,
                    'total_toggles': total_toggles,
                    **toggle_counts
                })
        
        return pd.DataFrame(results)
    
    def calculate_on_ratios(self, df, start_hour, end_hour):
        """Calculate ON time ratios for given time range"""
        results = []
        
        for user in df['User'].unique():
            user_data = df[df['User'] == user].copy()
            
            for date in user_data['Date'].unique():
                # Filter by time range
                if start_hour <= end_hour:
                    # Day time: same date
                    daily_data = user_data[user_data['Date'] == date].copy()
                    time_filtered = daily_data[
                        (daily_data['Hour'] >= start_hour) & 
                        (daily_data['Hour'] < end_hour)
                    ]
                else:  # Night time (crosses midnight)
                    # Night time: current evening + next morning
                    current_date = pd.to_datetime(date)
                    next_date = current_date + pd.Timedelta(days=1)
                    
                    # Current date evening (18:00-23:59)
                    current_evening = user_data[
                        (user_data['Date'] == date) & 
                        (user_data['Hour'] >= start_hour)
                    ].copy()
                    
                    # Next date morning (00:00-05:59)
                    next_morning = user_data[
                        (user_data['Date'] == next_date.date()) & 
                        (user_data['Hour'] < end_hour)
                    ].copy()
                    
                    # Combine evening and morning data
                    time_filtered = pd.concat([current_evening, next_morning], ignore_index=True)
                
                if len(time_filtered) == 0:
                    continue
                
                # Calculate ON time for each room
                on_times = {}
                total_on_time = 0
                
                for room in ['01', '02', '03', '04']:
                    on_time = time_filtered[room].sum()
                    on_times[room] = on_time
                    total_on_time += on_time
                
                # Calculate ratios
                ratios = {}
                for room in ['01', '02', '03', '04']:
                    if total_on_time > 0:
                        ratios[room] = on_times[room] / total_on_time
                    else:
                        ratios[room] = 0
                
                results.append({
                    'User': user,
                    'Date': date,
                    **ratios
                })
        
        return pd.DataFrame(results)
    
    def calculate_kitchen_usage_rate(self, df, start_hour, end_hour):
        """Calculate kitchen usage rate for given time range"""
        results = []
        
        for user in df['User'].unique():
            user_data = df[df['User'] == user].copy()
            
            for date in user_data['Date'].unique():
                # Filter by time range
                if start_hour <= end_hour:
                    # Day time: same date
                    daily_data = user_data[user_data['Date'] == date].copy()
                    time_filtered = daily_data[
                        (daily_data['Hour'] >= start_hour) & 
                        (daily_data['Hour'] < end_hour)
                    ]
                else:  # Night time (crosses midnight)
                    # Night time: current evening + next morning
                    current_date = pd.to_datetime(date)
                    next_date = current_date + pd.Timedelta(days=1)
                    
                    # Current date evening (18:00-23:59)
                    current_evening = user_data[
                        (user_data['Date'] == date) & 
                        (user_data['Hour'] >= start_hour)
                    ].copy()
                    
                    # Next date morning (00:00-05:59)
                    next_morning = user_data[
                        (user_data['Date'] == next_date.date()) & 
                        (user_data['Hour'] < end_hour)
                    ].copy()
                    
                    # Combine evening and morning data
                    time_filtered = pd.concat([current_evening, next_morning], ignore_index=True)
                
                if len(time_filtered) == 0:
                    continue
                
                # Calculate total toggles and kitchen toggles
                total_toggles = 0
                kitchen_toggles = 0
                
                for room in ['01', '02', '03', '04']:
                    room_data = time_filtered[room].values
                    if len(room_data) <= 1:
                        continue
                    
                    toggles = sum(1 for i in range(1, len(room_data)) 
                                if room_data[i] != room_data[i-1])
                    
                    total_toggles += toggles
                    if room == '03':  # Kitchen
                        kitchen_toggles = toggles
                
                # Calculate usage rate
                usage_ratio = kitchen_toggles / total_toggles if total_toggles > 0 else 0
                
                results.append({
                    'User': user,
                    'Date': date,
                    'usage_ratio': usage_ratio
                })
        
        return pd.DataFrame(results)
    
    def calculate_bathroom_usage(self, df):
        """Calculate bathroom usage during late night (22:00-04:00)"""
        results = []
        
        for user in df['User'].unique():
            user_data = df[df['User'] == user].copy()
            
            for date in user_data['Date'].unique():
                # Late night time: current evening + next morning
                current_date = pd.to_datetime(date)
                next_date = current_date + pd.Timedelta(days=1)
                
                # Current date evening (22:00-23:59)
                current_evening = user_data[
                    (user_data['Date'] == date) & 
                    (user_data['Hour'] >= 22)
                ].copy()
                
                # Next date morning (00:00-03:59)
                next_morning = user_data[
                    (user_data['Date'] == next_date.date()) & 
                    (user_data['Hour'] < 4)
                ].copy()
                
                # Combine evening and morning data
                late_night = pd.concat([current_evening, next_morning], ignore_index=True)
                
                if len(late_night) == 0:
                    continue
                
                # Calculate bathroom toggles
                bathroom_data = late_night['04'].values  # Room 04 is bathroom
                if len(bathroom_data) <= 1:
                    bathroom_count = 0
                else:
                    bathroom_count = sum(1 for i in range(1, len(bathroom_data)) 
                                       if bathroom_data[i] != bathroom_data[i-1])
                
                results.append({
                    'User': user,
                    'Date': date,
                    'bathroom_count': bathroom_count
                })
        
        return pd.DataFrame(results)
    
    def extract_all_features(self, filename):
        """Extract all 13 feature tables from the given dataset"""
        print(f"Processing {filename}...")
        
        # Load data
        df = self.load_data(filename)
        
        # 1. Day inactive total (06:00-18:00)
        day_inactive = self.calculate_inactive_periods(df, 6, 18)
        day_inactive_total = day_inactive[['User', 'Date', 'total_inactive']].rename(
            columns={'total_inactive': 'timeslot_count'})
        
        # 2. Night inactive total (18:00-06:00)
        night_inactive = self.calculate_inactive_periods(df, 18, 6)
        night_inactive_total = night_inactive[['User', 'Date', 'total_inactive']].rename(
            columns={'total_inactive': 'timeslot_count'})
        
        # 3. Day inactive room
        day_inactive_room = day_inactive[['User', 'Date', '01', '02', '03', '04']]
        
        # 4. Night inactive room
        night_inactive_room = night_inactive[['User', 'Date', '01', '02', '03', '04']]
        
        # 5. Day toggle room (06:00-18:00)
        day_toggle = self.calculate_toggle_counts(df, 6, 18)
        day_toggle_room = day_toggle[['User', 'Date', '01', '02', '03', '04']]
        
        # 6. Night toggle room (18:00-06:00)
        night_toggle = self.calculate_toggle_counts(df, 18, 6)
        night_toggle_room = night_toggle[['User', 'Date', '01', '02', '03', '04']]
        
        # 7. Day toggle total
        day_toggle_total = day_toggle[['User', 'Date', 'total_toggles']].rename(
            columns={'total_toggles': 'toggle_count'})
        
        # 8. Night toggle total
        night_toggle_total = night_toggle[['User', 'Date', 'total_toggles']].rename(
            columns={'total_toggles': 'toggle_count'})
        
        # 9. Day ON ratio room
        day_on_ratio_room = self.calculate_on_ratios(df, 6, 18)
        
        # 10. Night ON ratio room
        night_on_ratio_room = self.calculate_on_ratios(df, 18, 6)
        
        # 11. Day kitchen usage rate
        day_kitchen_usage_rate = self.calculate_kitchen_usage_rate(df, 6, 18)
        
        # 12. Night kitchen usage rate
        night_kitchen_usage_rate = self.calculate_kitchen_usage_rate(df, 18, 6)
        
        # 13. Night bathroom usage
        night_bathroom_usage = self.calculate_bathroom_usage(df)
        
        # Save all feature tables
        dataset_name = filename.replace('.csv', '')
        dataset_dir = os.path.join(self.processed_data_path, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        feature_tables = {
            'day_inactive_total': day_inactive_total,
            'night_inactive_total': night_inactive_total,
            'day_inactive_room': day_inactive_room,
            'night_inactive_room': night_inactive_room,
            'day_toggle_room': day_toggle_room,
            'night_toggle_room': night_toggle_room,
            'day_toggle_total': day_toggle_total,
            'night_toggle_total': night_toggle_total,
            'day_on_ratio_room': day_on_ratio_room,
            'night_on_ratio_room': night_on_ratio_room,
            'day_kitchen_usage_rate': day_kitchen_usage_rate,
            'night_kitchen_usage_rate': night_kitchen_usage_rate,
            'night_bathroom_usage': night_bathroom_usage
        }
        
        for table_name, table_data in feature_tables.items():
            output_path = os.path.join(dataset_dir, f"{table_name}.csv")
            table_data.to_csv(output_path, index=False)
            print(f"Saved {table_name}.csv ({len(table_data)} rows)")
        
        print(f"Completed processing {filename}")
        return feature_tables

def main():
    """Main function to extract features from all datasets"""
    extractor = FeatureExtractor()
    
    # Get all CSV files in raw data directory
    csv_files = glob.glob(os.path.join(extractor.raw_data_path, "*.csv"))
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        try:
            extractor.extract_all_features(filename)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print("Feature extraction completed for all datasets!")

if __name__ == "__main__":
    main() 