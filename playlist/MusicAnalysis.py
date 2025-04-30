import pandas as pd

df = pd.read_csv('D:/GW_World/AI/playlist/tracks_features.csv')
print(df.head())
df_after_2000 = df[df['year'] > 2000]

year_counts = df_after_2000.groupby('year').size().reset_index(name='count')
print(year_counts)

total_count = year_counts['count'].sum()

print(f"Total records after 2000: {total_count}")