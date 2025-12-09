import json
import numpy as np
import os
import optuna

from exp_utils import (build_binary_corpus, build_multi_class_corpus,
                       compute_evaluation_measures, compute_means_std_eval_measures)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


def get_model_and_params(trial, clf_name_: str):
    if clf_name_ == 'logistic_regression':
        return LogisticRegression(
            class_weight='balanced',
            max_iter=trial.suggest_int(name='max_iter', low=100, high=1000, step=200),
            C=trial.suggest_float('C', 1e-3, 10.0, log=True)
        )

    elif clf_name_ == 'knn':
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int('n_neighbors', 1, 30),
            weights=trial.suggest_categorical('weights', ['uniform', 'distance'])
        )

    elif clf_name_ == 'decision_tree':
        return DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=trial.suggest_int('max_depth', 2, 50)
        )

    elif clf_name_ == 'random_forest':
        return RandomForestClassifier(
            class_weight='balanced',
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 2, 50)
        )

    elif clf_name_ == 'extra_trees_classifier':
        return ExtraTreesClassifier(
            class_weight='balanced',
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 2, 50)
        )

    elif clf_name_ == 'xgboost':
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 2, 12),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0)
        )

    elif clf_name_ == 'lgbm':
        return LGBMClassifier(
            class_weight='balanced',
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', -1, 50),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
        )

    elif clf_name_ == 'svc':
        return SVC(
            class_weight='balanced',
            C=trial.suggest_float('C', 1e-3, 10.0, log=True),
            kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        )

    elif clf_name_ == 'cat_boost_classifier':
        return CatBoostClassifier(
            verbose=False,
            depth=trial.suggest_int('depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            iterations=trial.suggest_int('iterations', 50, 300)
        )

    elif clf_name_ == 'mlp_classifier':
        return MLPClassifier(
            hidden_layer_sizes=(
                trial.suggest_int('hidden_units', 50, 300),
            ),
            learning_rate_init=trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            max_iter=trial.suggest_int(name='max_iter', low=100, high=1000, step=200),
        )

    return None


def objective(trial):

    model_ = get_model_and_params(trial, clf_name)

    model_.fit(X_train, y_train)

    y_preds_ = model_.predict(X_val)

    eval_f1_score = f1_score(y_val, y_preds_, average='macro')

    return eval_f1_score


if __name__ == '__main__':

    corpus_file_path = '../../data/corpus/general/corpus_grammar_spell_errors.json'

    # problem_type = 'binary'
    problem_type = 'multi_class'

    max_features = None

    n_splits = 5

    results_model_dir = f'../../data/experiments/classification/results/{problem_type}'

    os.makedirs(results_model_dir, exist_ok=True)

    print('\nReading Corpus')

    with open(file=corpus_file_path, mode='r', encoding='utf-8') as file:
        original_corpus = json.load(file)

    print(f'\n\tTotal of Sentences: {len(original_corpus)}')

    print(f'\nRunning Experiment ML-based Models - {problem_type}\n')

    corpus = np.array(original_corpus)

    data_labels = np.zeros(len(original_corpus))

    classifiers = {
        'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=500),
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(class_weight='balanced'),
        'random_forest': RandomForestClassifier(class_weight='balanced'),
        'extra_trees_classifier': ExtraTreesClassifier(class_weight='balanced'),
        'xgboost': XGBClassifier(),
        'lgbm': LGBMClassifier(class_weight='balanced'),
        'svc': SVC(class_weight='balanced'),
        'cat_boost_classifier': CatBoostClassifier(verbose=False),
        'mlp_classifier': MLPClassifier()
    }

    for clf_name, clf_base in classifiers.items():

        print(f'\n  Classifier: {clf_name}')

        results_dict = {
            'all_accuracy': [],
            'all_macro_avg_p': [],
            'all_macro_avg_r': [],
            'all_macro_avg_f1': [],
            'all_weighted_avg_p': [],
            'all_weighted_avg_r': [],
            'all_weighted_avg_f1': []
        }

        all_y_test = []
        all_y_pred = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for k, (train_idx, test_idx) in enumerate(skf.split(corpus, data_labels)):

            train_idx, validation_idx, _, _ = train_test_split(
                train_idx, train_idx, test_size=0.1, shuffle=True, random_state=42)

            X_train = corpus[train_idx]
            X_val = corpus[validation_idx]
            X_test = corpus[test_idx]

            if problem_type == 'multi_class':
                X_train, y_train = build_multi_class_corpus(X_train)
                X_val, y_val = build_multi_class_corpus(X_val)
                X_test, y_test = build_multi_class_corpus(X_test)
            else:
                X_train, y_train = build_binary_corpus(X_train)
                X_val, y_val = build_binary_corpus(X_val)
                X_test, y_test = build_binary_corpus(X_test)

            vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=max_features)

            X_train = vectorizer.fit_transform(X_train).toarray()
            X_val = vectorizer.transform(X_val).toarray()
            X_test = vectorizer.transform(X_test).toarray()

            num_classes = len(set(y_train))

            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(y_train)
            y_val = label_encoder.transform(y_val)
            y_test = label_encoder.transform(y_test)

            # print(label_encoder.classes_)

            print(f'\n\tFolder {k + 1} - {len(X_train)} - {len(X_val)} - {len(X_test)}')

            study = optuna.create_study(direction='maximize')

            study.optimize(objective, n_trials=30, show_progress_bar=False)

            print(f'    Best params for {clf_name}: {study.best_params}')

            print(f'    Best F1-macro: {study.best_value:.4f}')

            best_trial = study.best_trial

            best_clf_base = get_model_and_params(best_trial, clf_name)

            best_classifier = clone(best_clf_base)

            best_classifier.fit(X_train, y_train)

            y_pred = best_classifier.predict(X_test)

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)

            compute_evaluation_measures(y_test, y_pred, results_dict)

        compute_means_std_eval_measures(clf_name, all_y_test, all_y_pred,
                                        results_dict, results_model_dir)

    print('\n\n***** Experiment Completed')
