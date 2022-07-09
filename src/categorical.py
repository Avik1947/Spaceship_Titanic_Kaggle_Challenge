from sklearn import preprocessing
import pandas as pd

class CategoricalFeatures:

    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names,
        encoding_type: label, binary, ohe,
        handle_na: True/False
        """

        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.lable_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                val = self.df.loc[:, c].mode()[0]
                # val = "-999999"
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna(val)

        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df.loc[:, c].values)
            self.lable_encoders[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin__{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder(drop = "first")
        ohe.fit(self.df[self.cat_feats].values)
        self.ohe = ohe
        transformed = ohe.transform(self.df[self.cat_feats].values).toarray()
        ohe_df = pd.DataFrame(
            transformed, columns=ohe.get_feature_names())
        #concat with original data
        self.output_df = pd.concat([self.output_df, ohe_df], axis=1).drop(self.cat_feats, axis=1)
        return self.output_df



    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                val = self.df.loc[:, c].mode()[0]
                # val = "-999999"
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna(val)

        if self.enc_type == "label":
            for c, lbl in self.lable_encoders.item():
                dataframe.loc[:, c] = lbl.transform(dataframe.loc[:, c].values)
            return dataframe

        elif self.enc_type == "ohe":
            transformed = self.ohe.transform(
                dataframe[self.cat_feats].values).toarray()
            ohe_df = pd.DataFrame(
            transformed, columns=self.ohe.get_feature_names())
            # concat with original data
            dataframe = pd.concat(
            [dataframe, ohe_df], axis=1).drop(self.cat_feats, axis=1)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        else:
            raise Exception("Encoding type not understood")
