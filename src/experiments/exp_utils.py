import numpy as np
import json
import os
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay


def build_binary_corpus(data):
    X_data = []
    y_data = []
    for example in data:
        X_data.append(example['frase_original'])
        y_data.append('CORRETA')
        X_data.append(example['frase_com_erros'])
        y_data.append('ERRO')
    return X_data, y_data


def build_multi_class_corpus(data):
    X_data = []
    y_data = []
    for example in data:
        X_data.append(example['frase_original'])
        y_data.append('CORRETA')
        for error in example['erros']:
            frase_with_error = example['frase_original'].replace(error['correto'], error['erro'])
            X_data.append(frase_with_error)
            y_data.append(error['tipo'].upper())
    return X_data, y_data


def tokenize_text(examples_, tokenizer_, max_len_):
    return tokenizer_(examples_['text'], padding='max_length', max_length=max_len_, truncation=True)


def compute_metrics_classification(eval_pred_):
    logits_, labels_ = eval_pred_
    predictions_ = np.argmax(logits_, axis=-1)
    labels_ = np.argmax(labels_, axis=-1)
    f1_macro_ = f1_score(labels_, predictions_, average='macro', zero_division=0)
    return {
        'f1_macro': f1_macro_
    }


def compute_evaluation_measures(y_true_: list, y_pred_: list, results_dict_: dict):
    report_dict = classification_report(y_true_, y_pred_, zero_division=0, output_dict=True)
    results_dict_['all_accuracy'].append(report_dict['accuracy'])
    results_dict_['all_macro_avg_p'].append(dict(report_dict['macro avg'])['precision'])
    results_dict_['all_macro_avg_r'].append(dict(report_dict['macro avg'])['recall'])
    results_dict_['all_macro_avg_f1'].append(dict(report_dict['macro avg'])['f1-score'])
    results_dict_['all_weighted_avg_p'].append(dict(report_dict['weighted avg'])['precision'])
    results_dict_['all_weighted_avg_r'].append(dict(report_dict['weighted avg'])['recall'])
    results_dict_['all_weighted_avg_f1'].append(dict(report_dict['weighted avg'])['f1-score'])


def compute_means_std_eval_measures(clf_name: str, all_y_test_: list, all_y_pred_: list, results_dict_: dict,
                                    results_dir_: str):

    new_results_dict = {}

    for measure_name, measure_values in results_dict_.items():
        mean_label = measure_name.replace('all_', 'mean_')
        std_label = measure_name.replace('all_', 'std_')
        new_results_dict[mean_label] = np.mean(measure_values)
        new_results_dict[std_label] = np.std(measure_values)

    results_dict_.update(new_results_dict)

    all_y_test_ = [int(y) for y in all_y_test_]
    all_y_pred_ = [int(y) for y in all_y_pred_]

    results_dict_['all_y_test'] = all_y_test_
    results_dict_['all_y_pred'] = all_y_pred_

    classification_report_file_name = f'{clf_name}_report.json'.lower()

    classification_report_file_path = os.path.join(results_dir_, classification_report_file_name)

    with open(file=classification_report_file_path, mode='w') as file_:
        json.dump(results_dict_, file_, indent=4)

    ConfusionMatrixDisplay.from_predictions(all_y_test_, all_y_pred_)

    confusion_matrix_name = f'{clf_name}_confusion_matrix.pdf'.lower()

    img_path = os.path.join(results_dir_, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)
