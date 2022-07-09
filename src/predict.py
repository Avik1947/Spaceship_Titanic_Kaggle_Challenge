import numpy as np
import joblib
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble
import pandas as pd
import os
from categorical import CategoricalFeatures
import config
import model_dispatcher


def predict(model_type, model_path):
    df = pd.read_csv(config.TEST_FILE)
    test_idx = df["PassengerId"].values

    predictions = None

    for fold in range(5):
        df = pd.read_csv(config.TEST_FILE)
        encoders = joblib.load(os.path.join(model_path, f"encoder_{fold}.pkl"))
        # cols = joblib.load(os.path.join(model_path, f"columns_{fold}.pkl"))
        # for labelencoding
        # for c in encoders:
        #     lbl = encoders[c]
        #     df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
        #     df.loc[:, c] = lbl.transform(df[c].values.tolist())

        model = joblib.load(os.path.join(model_path, f"dt_{fold}.pkl"))

        # df = df[cols]
        df = df.drop(['PassengerId'], axis=1)

        cat_feats = [col for col in df.columns if df[col].dtype == object]

        num_feats = [
            col for col in df.columns if df[col].dtype != object]

        for c in num_feats:
            val = df.loc[:, c].mean()
            df.loc[:, c] = df.loc[:, c].fillna(val)

        for c in cat_feats:
            val = df.loc[:, c].mode()[0]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna(val)

        transformed = encoders.transform(
            df[cat_feats].values).toarray()
        ohe_df = pd.DataFrame(
            transformed, columns=encoders.get_feature_names())
        # concat with original data
        df = pd.concat(
            [df, ohe_df], axis=1).drop(cat_feats, axis=1)

        preds = model.predict_proba(df)[:, 1]

        if fold == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack(
        (test_idx, predictions)), columns=["PassengerId", "Transported"])

    sub['Transported'] = sub['Transported'].map(lambda x: 1 if x >= 0.5 else 0)
    print(sub.head())

    return sub


if __name__ == "__main__":
    submission = predict(model_type="gbc",
                         model_path=config.MODEL_OUTPUT)

    mp = {1: True, 0: False}

    submission['Transported'] = submission["Transported"].map(mp)
    print(submission.head())

    submission.to_csv("../input/submission.csv", index=False)
