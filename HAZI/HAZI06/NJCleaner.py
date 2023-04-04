import pandas as pd
class NJCleaner:

    def __int__(self, path: str):
        self.data = pd.read_csv(path)


    def order_by_scheduled_time(self):
        new_df = self.data.copy()
        return new_df.sort_values(by=['scheduled_time'], ascending=True)

    def drop_columns_and_nan(self):
        new_df = self.data.copy()
        labels = ['from', 'to']
        new_df = new_df.drop(labels=labels)
        return new_df.dropna()


    def convert_date_to_day(self):
        new_df = self.data.copy()
        new_df['day'] = new_df['date']._convert()
        new_df.drop('date')

    def convert_scheduled_time_to_part_of_the_day(self):
        new_df = self.data.copy()
        helper = []
        for x in new_df['scheduled_time']:
            if(x > 4 & x < 8):
                helper.append('early morning')
            if(x > 8 & x < 12):
                helper.append('morning')
            if(x > 12 & x < 16):
                helper.append('afternoon')
            if(x > 16 & x < 20):
                helper.append('evening')
            if(x > 20 & x < 24):
                helper.append('night')
            else:
                helper.append('late night')

        new_df['part_of_the_day'] = helper
        return new_df

    #def convert_delay(self):



