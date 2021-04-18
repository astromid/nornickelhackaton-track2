import random
from functools import partial

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm.auto import tqdm

SEED = 14300631
TRAIN_PKL = 'data/preprocessed/train_dedup.pkl'


def objective(trial, folds):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Poisson', 'No']),
        'depth': trial.suggest_int('depth', 3, 12),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'one_hot_max_size': trial.suggest_categorical('one_hot_max_size', [2, 10]),
        'fold_permutation_block': trial.suggest_int('fold_permutation_block', 1, 2),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced', 'SqrtBalanced']),
        'border_count': trial.suggest_categorical('border_count', [64, 128, 256]),
        'feature_border_type': trial.suggest_categorical('feature_border_type', ['Median', 'MinEntropy', 'GreedyLogSum']),
        'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 3, 5),
    }
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 1, 6)
    elif params['bootstrap_type'] == 'Poisson':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    if params['grow_policy'] in ['Depthwise', 'Lossguide']:
        params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 10)
    if params['grow_policy'] == 'Lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 15, 63)
        params['score_function'] = trial.suggest_categorical('score_function', ['L2', 'NewtonL2'])
    print(f'Start trial with parameters: {params}')
    trial_cv_metrics = {
        'F1': [],
        'AP': [],
        'ROC-AUC': [],
        'threshold': [],
        'iterations': [],
    }
    for fold_idx in tqdm(range(3), desc='Folds'):
        fold_train = pd.concat(folds[:fold_idx + 1])
        fold_val = folds[fold_idx + 1].copy()

        y_train = fold_train['sick']
        y_val = fold_val['sick']

        X_train = fold_train.drop(['date', 'cutoff_date', 'sick'], axis=1)
        X_val = fold_val.drop(['date', 'cutoff_date', 'sick'], axis=1)

        cat_features = X_train.select_dtypes('category').columns.tolist()
        # cat_features = [idx for idx, col in enumerate(X_train.columns) if col in cat_col_names]
        # 1st model - zeros classifier
        clf = CatBoostClassifier(
            iterations=3500,
            random_seed=SEED,
            task_type='GPU',
            use_best_model=True,
            od_pval=1e-6,
            learning_rate=params['learning_rate'],
            l2_leaf_reg=params['l2_leaf_reg'],
            bootstrap_type=params['bootstrap_type'],
            depth=params['depth'],
            grow_policy=params['grow_policy'],
            one_hot_max_size=params['one_hot_max_size'],
            fold_permutation_block=params['fold_permutation_block'],
            auto_class_weights=params['auto_class_weights'],
            border_count=params['border_count'],
            feature_border_type=params['feature_border_type'],
            max_ctr_complexity=params['max_ctr_complexity'],
            bagging_temperature=params.get('bagging_temperature', None),
            subsample=params.get('subsample', None),
            min_data_in_leaf=params.get('min_data_in_leaf', None),
            max_leaves=params.get('max_leaves', None),
            score_function=params.get('score_function', None),
        )
        clf.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            verbose_eval=250,
        )
        trial_cv_metrics['iterations'].append(clf.best_iteration_)
        y_probs = clf.predict_proba(X_val)[:, 1]
        # various metrics
        ap = average_precision_score(y_val, y_probs)
        auc = roc_auc_score(y_val, y_probs)

        trial_cv_metrics['AP'].append(ap)
        trial_cv_metrics['ROC-AUC'].append(auc)
        
        p, r, thresholds = precision_recall_curve(y_val, y_probs)
        # cut last 1
        p = p[:-1]
        r = r[:-1]
        # filter zerop precision thresholds
        thresholds = thresholds[p > 0]
        r = r[p > 0]
        p = p[p > 0]
        f1_scores = 2 * r * p / (r + p)
        th = thresholds[np.argmax(f1_scores)]
        f1 = f1_scores.max()
        
        trial_cv_metrics['F1'].append(f1)
        trial_cv_metrics['threshold'].append(th)
        tqdm.write(f'Fold {fold_idx}: F1 = {f1_scores.max():.4f} [th {th:.4f}], AP = {ap:.4f}, ROC-AUC = {auc:.4f}, trees: {clf.best_iteration_}')
    print('Trial CV metrics:')
    for metric, lst in trial_cv_metrics.items():
        print(f'{metric}: {lst} => {np.mean(lst)}')
    return np.mean(trial_cv_metrics['F1'])


if __name__ == '__main__':
    # set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    # load data
    train = pd.read_pickle(TRAIN_PKL)
    # make folds
    folds = [
        train[train['date'] <= '2016-08-01'].copy(),
        train[(train['date'] > '2016-08-01') & (train['date'] <= '2017-08-01')].copy(),
        train[(train['date'] > '2017-08-01') & (train['date'] <= '2018-08-01')].copy(),
        train[(train['date'] > '2018-08-01') & (train['date'] <= '2019-08-01')].copy(),
    ]
    study = optuna.create_study(direction='maximize')
    objective_func = partial(objective, folds=folds)
    study.optimize(objective_func, n_trials=500)
    print('Best trial:')
    print(f'  Value: {study.best_trial.value}')
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')
