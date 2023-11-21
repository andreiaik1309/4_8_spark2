from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window
import calendar
import pandas as pd


def main(year: int, month: int):
    spark = SparkSession.builder \
        .appName("test 4.8") \
        .master("local[*]") \
        .enableHiveSupport() \
        .getOrCreate()
    
    # Task 1
    data = [(1, 1562007679), (1, 1562007710), (1, 1562007720),
            (1, 1562007750), (2, 1564682430), (2, 1564682450), (2, 1564682480)]
    columns = ["id", "timestamp"]
    df = spark.createDataFrame(data, columns)
    #  Преобразовываем timestamp в формат даты
    df = df.withColumn("date", F.to_date(F.from_unixtime("timestamp").cast("timestamp")))
    # Определение оконной спецификации для разбиения на дни 
    window_spec = Window.partitionBy("id", "date").orderBy("timestamp")
    # Рассчитываем разницу между первым и последним действием для каждого пользователя и каждого дня
    df1 = df.withColumn("session_length", F.last("timestamp").over(window_spec) - F.first("timestamp").over(window_spec))

    # Task 2
    # Пример данных
    data_demand = [(1, '01', 10), (1, '02', 11), (2, '01', 12), (2, '02', 9), (3, '01', 7), (3, '02', 8)]
    data_stock = [(1, '01', 1000), (1, '02', 400), (2, '01', 390), (2, '02', 350), (3, '01', 500), (3, '02', 450)]
    columns_demand = ["product", "location", "demand"]
    columns_stock = ["product", "location", "stock"]
    df_demand = spark.createDataFrame(data_demand, columns_demand)
    df_stock = spark.createDataFrame(data_stock, columns_stock)
    
    # сформируем календарь с дискретностью 1 техническая неделя
    days_in_month = calendar.monthrange(year, month)[1]
    ind = pd.date_range(start=f'2023-{month}-1', end=f'2023-{month}-{days_in_month}') 
    pd_days = pd.DataFrame(index=ind).reset_index().rename(columns={'index': 'date'})
    pd_days['dow'] = pd_days['date'].dt.dayofweek
    pd_days['days'] = pd_days['date'].dt.day
    pd_days['start_week'] = 'no'
    pd_days.loc[(pd_days['dow'] == 0) | (pd_days['days'] == 1), 'start_week'] = 'yes'
    pd_days['end_week'] = 'no'
    pd_days.loc[(pd_days['dow'] == 6) | (pd_days['days'] == days_in_month ), 'end_week'] = 'yes'
    pd_week_calendar = pd_days.loc[pd_days['start_week'] == 'yes', ['days']]
    pd_week_calendar = pd_week_calendar.rename(columns={'days': 'day_start_week'}).reset_index(drop=True)
    pd_week_end = pd_days.loc[pd_days['end_week'] == 'yes', ['days']]
    pd_week_end = pd_week_end.rename(columns={'days': 'day_end_week'}).reset_index(drop=True)
    pd_week_calendar = pd_week_calendar.merge(pd_week_end, left_index=True, right_index=True)

    df_week_calendar = spark.createDataFrame(pd_week_calendar)

    # Объединяем датафреймы
    df_result = df_week_calendar.crossJoin(df_stock)
    df_result = df_result.join(df_demand, ['product', 'location'])

    # Считаем показатели
    partition=Window.partitionBy('product', 'location').orderBy('day_start_week')
    df_result  = df_result \
        .withColumn('sales_from_start_month', F.col('day_end_week')*F.col('demand'))\
        .withColumn('sales_before_week', F.lag('sales_from_start_month', 1).over(partition)) \
        .fillna(0, subset=['sales_before_week']) \
        .withColumn('sales_for_week', F.col('sales_from_start_month') - F.col('sales_before_week')) \
        .withColumn('stock_end_week', F.col('stock') - F.col('sales_from_start_month'))  
 
    # Показываем результат Task 1
    df1.show()
    df1.printSchema()

    # Показываем результат Task 2
    df_result.show(40)
    df_result.printSchema()

    print('################# END ##########################')


if __name__ == '__main__':
    year = 2023
    month = 6
    main(year, month)