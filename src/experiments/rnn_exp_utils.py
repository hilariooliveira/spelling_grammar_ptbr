from tensorflow.keras import layers, models


def build_cnn_model(vocab_size: int, max_len: int, num_classes: int, emb_dim: int,
                    num_filters: int, kernel_size: int) -> models.Sequential:
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=num_classes, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_lstm(vocab_size: int, max_len: int,
               num_classes: int, emb_dim: int) -> models.Sequential:
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.LSTM(units=128))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=num_classes, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_hybrid(vocab_size: int, max_len: int,
                 num_classes: int, emb_dim: int) -> models.Sequential:
    model = models.Sequential(name='CNN_with_BiLSTM')
    model.add(layers.Embedding(vocab_size, emb_dim, input_length=max_len))
    model.add(layers.Conv1D(16, 3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
