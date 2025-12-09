import json
import numpy as np
import os

from exp_utils import (build_binary_corpus, build_multi_class_corpus,
                       compute_evaluation_measures, compute_means_std_eval_measures)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rnn_exp_utils import (build_feed_foward, build_feed_foward_emb, build_cnn_model,
                           build_lstm, build_hybrid)
from keras.callbacks import ModelCheckpoint


if __name__ == '__main__':

    corpus_file_path = '../../data/corpus/general/corpus_grammar_spell_errors.json'

    model_name = 'cnn'
    # model_name = 'lstm'
    # model_name = 'hybrid'

    problem_type = 'binary'
    # problem_type = 'multi_class'

    num_epochs = 20
    batch_size = 4

    n_splits = 5

    vocab_size = 5_000
    emb_dim = 300

    checkpoint_dir = f'../data/experiments/classification/model/{problem_type}/{model_name}/'
    results_model_dir = f'../data/experiments/classification/results/{problem_type}'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_model_dir, exist_ok=True)

    print('\nReading Corpus')

    with open(file=corpus_file_path, mode='r', encoding='utf-8') as file:
        original_corpus = json.load(file)

    print(f'\n\tTotal of Sentences: {len(original_corpus)}')

    print(f'\nRunning Experiment RNN-based Models - {problem_type}\n')

    corpus = np.array(original_corpus)

    data_labels = np.zeros(len(original_corpus))

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

        num_classes = len(set(y_train))

        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)

        tokenizer = Tokenizer(oov_token='<OOV>')

        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_val = tokenizer.texts_to_sequences(X_val)
        X_test = tokenizer.texts_to_sequences(X_test)

        max_len = max([len(x) for x in X_train])

        X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
        X_val = pad_sequences(X_val, maxlen=max_len, padding='post')
        X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

        print(f'\n\tFolder {k + 1} - {len(X_train)} - {len(X_val)} - {len(X_test)}')

        if model_name == 'cnn':
            model = build_cnn_model(vocab_size, max_len, num_classes, emb_dim, num_filters=16,
                                    kernel_size=3)
        elif model_name == 'lstm':
            model = build_lstm(vocab_size, max_len, num_classes, emb_dim)
        else:
            model = build_hybrid(vocab_size, max_len, num_classes, emb_dim)

        checkpoint_file_path = os.path.join(checkpoint_dir, f'folder_{k}')

        os.makedirs(checkpoint_file_path, exist_ok=True)

        checkpoint_file_path = os.path.join(checkpoint_file_path, f'model_folder_{k}.weights.h5')

        model_checkpoint = ModelCheckpoint(filepath=checkpoint_file_path,
                                           save_weights_only=True, monitor='val_accuracy',
                                           mode='max', save_best_only=True)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                            validation_data=(X_val, y_val), callbacks=[model_checkpoint])

        model.load_weights(checkpoint_file_path)

        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)

        y_pred = [y for y in y_pred]

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        compute_evaluation_measures(y_test, y_pred, results_dict)

    compute_means_std_eval_measures(model_name, all_y_test, all_y_pred,
                                    results_dict, results_model_dir)

    print('\n\n***** Experiment Completed')
