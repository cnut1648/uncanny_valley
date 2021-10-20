from argparse import ArgumentParser

import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

from tune.config import NEPTUNE_API, MODEL_CONFIG
from tune.data import get_data
# from utils import get_default_data_manager
from sklearn.model_selection import train_test_split
# from config import train_data_path, test_data_path, sample_submission_path, NEPTUNE_API, MODEL_CONFIG, BEST_MODEL_CONFIG
import numpy as np
from io import StringIO
import neptune
# from neptunecontrib.monitoring.sklearn import log_regressor_summary
import neptunecontrib.monitoring.optuna as opt_utils
from neptunecontrib.monitoring.xgboost import neptune_callback

#
# def run(
#         regressor,
#         params: dict,
#         alg: str,
#         tags=None,
#         preprocessors=None,
#         test_size=0.2,
#         random_state=42
# ):
#     """
#     :param regressor: sklearn regressor
#     :param params: dict params for regressor
#     :param alg: either 'xgboost' or 'sklearn'
#     :param tags: optional tags for neptune exps, by default module name
#     :param preprocessors: optional preprocessors
#     :param test_size: size for test datamodules
#     :param random_state: random seed for split
#     """
#     data_manager = get_default_data_manager() if preprocessors is None \
#         else DataManager(train_data_path, test_data_path, preprocessors)
#     train_features, test_features = data_manager.get_features()
#     X, y = train_features.drop("log_SalePrice", axis=1), train_features["log_SalePrice"]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                         test_size=test_size,
#                                                         random_state=random_state)
#
#     reg = regressor(**params)
#     model_name = type(reg).__name__
#     tags = tags if tags is not None else []
#     tags.append(model_name)
#
#     neptune.init(
#         project_qualified_name='jiashuxu/Kaggle-housing-price',
#         api_token=NEPTUNE_API
#     )
#
#     neptune.create_experiment(
#         params=params,
#         name=model_name,
#         tags=tags
#     )
#
#     if alg == "sklearn":
#         reg.fit(X_train, y_train)
#         log_regressor_summary(reg,
#                               X_train, X_test, y_train, y_test)
#
#     else:
#         reg.fit(X_train, y_train,
#                 eval_metric=['mae', 'rmse'],
#                 eval_set=[(X_test, y_test)],
#                 callbacks=[neptune_callback()]
#                 )
#
#     submission = pd.read_csv(sample_submission_path)
#     # refit all datamodules
#     reg.fit(X, y)
#     submission.iloc[:, 1] = np.floor(np.expm1(reg.predict(test_features)))
#
#     buffer = StringIO(submission.to_csv(index=False))
#     buffer.seek(0)
#     neptune.log_artifact(buffer, 'csv/vanilla submission.csv')
#
#     # postprocessing
#     q1 = submission['SalePrice'].quantile(0.0045)
#     q2 = submission['SalePrice'].quantile(0.99)
#     submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x * 0.77)
#     submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x * 1.1)
#
#     buffer = StringIO(submission.to_csv(index=False))
#     buffer.seek(0)
#     neptune.log_artifact(buffer, 'csv/processed submission.csv')
#
#     neptune.stop()
#

class Objective:
    def __init__(self, model, params: dict, alg:str,
                 X_train, X_test, y_train, y_test
                 ):
        self.model = model
        self.params = params
        self.alg = alg
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def __call__(self, trial):
        params = {}
        for name, value in self.params.items():
            if type(value) is list:
                params[name] = trial.suggest_categorical(name, value)
            elif type(value) is tuple:
                if all(type(i) is int for i in value):
                    params[name] = trial.suggest_int(name, *value, log=True)
                else:
                    params[name] = trial.suggest_float(name, *value,
                                                       log=True if value[1] >= 1.0 else False)
            # if scalar
            else:
                params[name] = value
        model = self.model(**params)
        if alg == "xgboost":
            model.fit(self.X_train, self.y_train,
                      eval_metric=['merror', 'mlogloss'],
                      eval_set=[(self.X_test, self.y_test)],
                      callbacks=[optuna.integration.XGBoostPruningCallback(trial, "validation_0-mlogloss")]
                      )

            # minimize error
            return model.evals_result_["validation_0"]["merror"][-1]
        else:
            model.fit(self.X_train, self.y_train)

        # y_pred = module.predict(self.X_test)
        # return
        # # return np.sqrt(mean_squared_error(self.y_test, y_pred))


def tune(
        classifer,
        params: dict,
        alg: str,
        tags=None,
        preprocessors=None,
        test_size=0.2,
        random_state=42
):
    """
    :param classifer: sklearn regressor
    :param params: dict params for regressor for tuning
    :param tags: optional tags for neptune exps, by default module name
    :param preprocessors: optional preprocessors
    :param test_size: size for test datamodules
    :param random_state: random seed for split
    """

    model_name = classifer.__name__
    tags = tags if tags is not None else []
    tags.append(model_name)

    neptune.init(
        project_qualified_name='jiashuxu/folklore',
        api_token=NEPTUNE_API
    )
    neptune.create_experiment(
        name=model_name,
        tags=tags
    )

    neptune_callback = opt_utils.NeptuneCallback(
        log_study=True,
        log_charts=True
    )

    study = optuna.create_study(direction="minimize")

    objective = Objective(
        classifer,
        params,
        alg,
        *get_data(filter_no=10, preprocess=[
            "standard_scaler",
            "pca"
        ])
    )
    study.optimize(
        objective,
        n_trials=50,
        callbacks=[neptune_callback]
    )

    opt_utils.log_study_info(study)

    print(f"best merror score: {study.best_value} with {study.best_params}")

    neptune.stop()

    # # run best params
    # # tag: run, alg
    # tags = ["run"] + [t for t in tags[:-1] if t != "tune"]
    # run(classifer, study.best_params, alg, tags,
    #     preprocessors, test_size, random_state)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("task", type=str, choices=["run", "tune"],
                        help="task to run")
    parser.add_argument("module", type=str, choices=MODEL_CONFIG.keys(),
                        help="desired module's name")

    args = parser.parse_args()

    alg = "xgboost" if args.model == "xgboost" else "sklearn"
    if args.task == "run":
        pass
        # module = BEST_MODEL_CONFIG[args.module]
        # run(module["module"], module["params"],
        #     alg, tags=[alg, "run"])
    else:
        model = MODEL_CONFIG[args.model]
        tune(model["module"], model["params"],
             alg, tags=[alg, "tune"])
