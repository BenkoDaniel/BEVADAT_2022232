import pandas as pd
class NJCleaner:

    def __int__(self, path: str):
        self.data = pd.read_csv(path)


    def order_by_scheduled_time(self):
        return self.data.sort_values(by=['scheduled_time'], ascending=True)

    def drop_columns_and_nan(self):
        labels = ['from', 'to']
        self.data = self.data.drop(labels=labels)
        return self.data.dropna()


    def convert_date_to_day(self):
        self.data['day'] = self.data['date'].strftime()
        self.data.drop('date')
        return self.data

    def convert_scheduled_time_to_part_of_the_day(self):
        helper = []
        for x in pd.to_datetime(self.data['scheduled_time']).dt.hour():
            if(x < 4):
                helper.append('late night')
            if(x < 8):
                helper.append('early_morning')
            if(x <12):
                helper.append('morning')
            if(x < 16):
                helper.append('afternoon')
            if(x < 20):
                helper.append('evening')
            else:
                helper.append('night')

        self.data['part_of_the_day'] = helper
        return self.data

    def convert_delay(self):
        self.data['delay'] = 0 if (self.data['delay_minutes'] < 5) else 1
        return self.data

    def drop_unnecessary_columns(self):
        labels = ['train_id', 'scheduled_time', 'actual_time', 'delay_minutes']
        return self.data.drop(labels=labels)

    def save_first_60k(self, save_path):
        new_df = self.data[:60000]
        new_df.to_csv(save_path)

    def prep_df(self, save_path='data/NJ.csv'):
        self.data = self.order_by_scheduled_time()
        self.data = self.drop_columns_and_nan()
        self.data = self.convert_date_to_day()
        self.data = self.convert_scheduled_time_to_part_of_the_day()
        self.data = self.convert_delay()
        self.data = self.drop_unnecessary_columns()
        self.save_first_60k(save_path)





