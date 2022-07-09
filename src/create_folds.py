import enum
import pandas as pd
from sklearn import model_selection

import config

if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE)
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df['Transported'])):
        df.loc[val_idx, 'kfold'] = f

    print(df.head())

    df.to_csv("../input/train_folds.csv", index=False)
