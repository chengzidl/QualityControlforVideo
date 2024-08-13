import pandas as pd

dfFeatures = pd.read_csv('Features_4K.csv')
dfQuality = pd.read_csv('Quality_4K.csv')
df = pd.merge(dfFeatures, dfQuality)

# match 90
def find_closest_row(group, target=90):
    return group.iloc[(group['vmafmean'] - target).abs().argmin()]

closest_rows = df.groupby(['file', 'chunk']).apply(find_closest_row).reset_index(drop=True)

# del
columns_to_remove = ['res', 'vmafmean', 'psnrmean', 'ssimmean'] # res

closest_rows = closest_rows.drop(columns=columns_to_remove)

# save
closest_rows.to_csv('dataset.csv', index=False)