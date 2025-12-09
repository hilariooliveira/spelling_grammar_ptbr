import json
import torch
import numpy as np
import os
import torch.nn.functional as f

from exp_utils import (build_binary_corpus, build_multi_class_corpus, tokenize_text,
                       compute_metrics_classification, compute_evaluation_measures,
                       compute_means_std_eval_measures)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback


if __name__ == '__main__':

    corpus_file_path = '../data/corpus/general/corpus_grammar_spell_errors.json'

    # model_tuple = ('distilbert_base', 'adalbertojunior/distilbert-portuguese-cased')
    # model_tuple = ('bertimbau_base', 'neuralmind/bert-base-portuguese-cased')
    model_tuple = ('roberta_base', 'josu/roberta-pt-br')
    # model_tuple = ('bertimbau_large', 'neuralmind/bert-large-portuguese-cased')

    # problem_type = 'binary'
    problem_type = 'multi_class'

    num_epochs = 20

    early_stopping_patience = 3

    n_splits = 5

    batch_size = 8
    max_len = 512

    if 'albertina' in model_tuple[0]:
        batch_size = 4

    output_dir = f'../data/experiments/classification/model/{problem_type}'
    results_model_dir = f'../data/experiments/classification/results/{problem_type}'

    os.makedirs(results_model_dir, exist_ok=True)

    output_dir = os.path.join(output_dir, model_tuple[0])

    os.makedirs(output_dir, exist_ok=True)

    gradient_accumulation_steps = 1
    gradient_checkpointing = False
    fp16 = False
    optim = 'adamw_torch'

    print('\nReading Corpus')

    with open(file=corpus_file_path, mode='r', encoding='utf-8') as file:
        original_corpus = json.load(file)

    print(f'\n\tTotal of Sentences: {len(original_corpus)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\nDevice: {device}')

    print(f'\nRunning Experiment BERT-based Models - {problem_type} - {model_tuple[0]}\n')

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

    corpus = np.array(original_corpus)

    data_labels = np.zeros(len(original_corpus))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_tuple[1])

    for k, (train_idx, test_idx) in enumerate(skf.split(corpus, data_labels)):

        train_idx, validation_idx, _, _ = train_test_split(train_idx, train_idx, test_size=0.1,
                                                           shuffle=True, random_state=42)

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

        num_classes = len(set(y_train))

        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

        y_train = torch.tensor(y_train)
        y_val = torch.tensor(y_val)
        y_test = torch.tensor(y_test)

        y_train = f.one_hot(y_train.to(torch.int64), num_classes=num_classes)
        y_val = f.one_hot(y_val.to(torch.int64), num_classes=num_classes)
        y_test = f.one_hot(y_test.to(torch.int64), num_classes=num_classes)

        print(f'\n\tFolder {k + 1} - {len(X_train)} - {len(X_val)} - {len(X_test)}')

        train_dict = {'text': X_train, 'label': y_train}
        val_dict = {'text': X_val, 'label': y_val}
        test_dict = {'text': X_test, 'label': y_test}

        train_dataset = Dataset.from_dict(train_dict)
        val_dataset = Dataset.from_dict(val_dict)
        test_dataset = Dataset.from_dict(test_dict)

        encoded_train_dataset = train_dataset.map(lambda x: tokenize_text(x, tokenizer, max_len),
                                                  batched=True, batch_size=batch_size)

        encoded_val_dataset = val_dataset.map(lambda x: tokenize_text(x, tokenizer, max_len),
                                              batched=True, batch_size=batch_size)

        encoded_test_dataset = test_dataset.map(lambda x: tokenize_text(x, tokenizer, max_len),
                                                batched=True, batch_size=batch_size)

        output_dir_folder = os.path.join(output_dir, f'folder_{k}')

        model = AutoModelForSequenceClassification.from_pretrained(model_tuple[1], num_labels=num_classes)

        training_args = TrainingArguments(
            output_dir=output_dir_folder,
            logging_strategy='epoch',
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            optim=optim,
            weight_decay=0.01,
            eval_steps=100,
            logging_steps=100,
            learning_rate=5e-5,
            eval_strategy='epoch',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            save_total_limit=2,
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
            greater_is_better=True,
            report_to=['none']
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_val_dataset,
            compute_metrics=compute_metrics_classification,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )

        trainer.train()

        y_pred, _, _ = trainer.predict(encoded_test_dataset)

        y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        y_test = [int(y.item()) for y in y_test]
        y_pred = [int(y.item()) for y in y_pred]

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        compute_evaluation_measures(y_test, y_pred, results_dict)

    compute_means_std_eval_measures(model_tuple[0], all_y_test, all_y_pred, results_dict, results_model_dir)

    print('\n\n***** Experiment Completed')
