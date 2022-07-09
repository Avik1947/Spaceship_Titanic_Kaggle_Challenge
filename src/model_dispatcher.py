from sklearn import tree
from sklearn import ensemble
import xgboost as xgb

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion='gini'
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion='entropy'
    ),
    "rf": ensemble.RandomForestClassifier(),
    "gbc": ensemble.GradientBoostingClassifier(),
    "xgb": xgb.XGBClassifier(njobs=- 1, objective="binary:logistic", n_estimators=200, max_depth=5, random_state=42)
}
