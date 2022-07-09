import os
import argparse
import tarfile
from categorical import CategoricalFeatures
import joblib
import os
import pandas as pd
import numpy as np
import argparse

import config
import model_dispatcher
from metrics import ClassficationMetrics


def run(fold, model, enc_type):

    df = pd.read_csv(config.TRAINING_FOLDS_FILE)
    mp = {True: 1, False: 0}
    df['Transported'] = df['Transported'].map(mp)

    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)

    ytrain = train_df['Transported'].values
    yvalid = valid_df['Transported'].values

    train_df = train_df.drop(['Transported', 'kfold', 'PassengerId'], axis=1)
    valid_df = valid_df.drop(['Transported', 'kfold', 'PassengerId'], axis=1)

    cat_feats = [
        col for col in train_df.columns if train_df[col].dtype == object]

    num_feats = [
        col for col in train_df.columns if train_df[col].dtype != object]

# Handling Numerical Missing Values

    for c in num_feats:
        val = train_df.loc[:, c].mean()
        train_df.loc[:, c] = train_df.loc[:, c].fillna(val)
        valid_df.loc[:, c] = valid_df.loc[:, c].fillna(val)

    cv = CategoricalFeatures(train_df, cat_feats, enc_type, handle_na=True)


    train_df = cv.fit_transform()
    valid_df = cv.transform(valid_df)

    print(valid_df.head())

    model = model_dispatcher.models[model]
    model.fit(train_df, ytrain)



    ypred = model.predict(valid_df)

    # yproba = model.predict_proba(valid_df)

    metric = ClassficationMetrics()
    acc = metric.__call__("accuracy", yvalid, ypred)

    print("Fold: {}, Acc: {}".format(fold, acc))

    joblib.dump(cv.ohe, os.path.join(
        config.MODEL_OUTPUT, f"encoder_{fold}.pkl"))
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.pkl"))
    joblib.dump(train_df.columns, os.path.join(
        config.MODEL_OUTPUT, f"columns_{fold}.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )

    parser.add_argument(
        "--model",
        type=str
    )

    parser.add_argument(
        "--enc_type",
        type=str
    )

    args = parser.parse_args()

    run(
        fold=args.fold,
        model=args.model,
        enc_type=args.enc_type
    )
