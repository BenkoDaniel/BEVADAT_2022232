import pandas as pd


class NJCleaner:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def order_by_scheduled_time(self):
        sorted = self.data.sort_values(by=['scheduled_time'], ascending=True)
        return sorted

    def drop_columns_and_nan(self) -> pd.DataFrame:
        self.data = self.data.drop(['from', 'to'], axis=1)
        self.data = self.data.dropna()
        return self.data

    def convert_date_to_day(self) -> pd.DataFrame:
        self.data['day'] = pd.to_datetime(self.data['date']).dt.day_name()
        self.data = self.data.drop('date', axis=1)
        return self.data

    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame:
        def part_of_the_day(dt):
            x = int(dt.split(' ')[1].split(':')[0])
            if 0 <= x <= 3:
                return 'late_night'
            if 4 <= int(x) <= 7:
                return 'early_morning'
            if 8 <= int(x) <= 11:
                return 'morning'
            if 12 <= int(x) <= 15:
                return 'afternoon'
            if 16 <= int(x) <= 19:
                return 'evening'
            else:
                return 'night'

        new_df = self.data.copy()
        new_df['part_of_the_day'] = new_df['scheduled_time'].apply(part_of_the_day)
        new_df.drop('scheduled_time', axis=1)
        return new_df

    def convert_delay(self) -> pd.DataFrame:
        self.data['delay'] = (self.data['delay_minutes'] >= 5).astype(int)
        return self.data

    def drop_unnecessary_columns(self) -> pd.DataFrame:
        return self.data.drop(['train_id', 'actual_time', 'delay_minutes'], axis=1)

    def save_first_60k(self, path):
        new_df = self.data[:60000]
        new_df.to_csv(path)

    def prep_df(self, path='data/NJ.csv'):
        self.data = self.order_by_scheduled_time()
        self.data = self.drop_columns_and_nan()
        self.data = self.convert_date_to_day()
        self.data = self.convert_scheduled_time_to_part_of_the_day()
        self.data = self.convert_delay()
        self.data = self.drop_unnecessary_columns()
        self.save_first_60k(path)





