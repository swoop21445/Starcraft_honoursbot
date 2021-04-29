import pandas as pd



game_data_file = "game_data/model_vs_medium_random_training.csv"
lenght = 600
output = 'model_vs_medium_random_score.csv'

df = pd.read_csv(game_data_file)

df.columns = ['game_result', 'game_score']

winrates = []

for index,row in df.iterrows():
    if index < lenght:
        winrate_counter = 0
        win_total = 0
        for x in range(index, index + 14):
            print(x)
            if x < 0:
                pass
            else:
                winrate_counter += 1
                xrow = df.loc[x]
                if xrow['game_result'] == 1:
                    win_total += 1
        winrate = (win_total/winrate_counter) * 100
        winrates.append(winrate)

for index,row in df.iterrows():
    if row['game_result'] == 1:
        row['game_score'] = int(row['game_score']) + 5000

excess_rows = len(df.index) - lenght
drop_list = []
for x in range(0,excess_rows):
    x = x + lenght
    drop_list.append(x)


df.drop(labels=[*drop_list],
        axis = 0,
        inplace= True)
df['win_rate'] = winrates

    
print(df.head)
df.to_csv(output)