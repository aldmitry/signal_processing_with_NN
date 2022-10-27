def concatenate_signals(window , mean1, mean2, overlap ):
    """
    Функция принимает размер будущего сигнала, параметры усреднения стыка сигналов, перекрытие, и частоту дискретизации
    :param window:
    :param mean1:
    :param mean2:
    :param overlap:
    :param SAMPLE_RATE:
    :return: signal + label
    """
    duration = window * 2  # Секунды
    if overlap == 0:
        x1 = np.random.normal(0,1,duration)
        x1 = pd.Series(x1).rolling(window = mean1).mean().dropna()
        x1 = x1/(np.abs(x1).max())
        return x1[:window], 0
    else:
        x1 = np.random.normal(0,1,duration) # Первый сигнал
        x2 = np.random.normal(0,1,duration) # Второй сигнал

        # Усреднение, обрезание, нормировка, склейка
        x1 = pd.Series(x1).rolling(window = mean1).mean().dropna().values
        x2 = pd.Series(x2).rolling(window = mean2).mean().dropna().values
        x1 = x1/(np.abs(x1).max())
        x2 = x2/(np.abs(x2).max())
        len_1 = int(window * (1 - overlap))
        len_2 = window - len_1
        x1 = x1
        x2 = x2
        x = np.concatenate((x1[:len_1],x2[:len_2]))
        return x, 1

def make_signals(WINDOW, N_signals, overlap):
    mean1 = random.randint(2,6)
    mean2 = random.randint(2,6)
    while mean1 == mean2:
        mean2 = random.randint(2,5)
    signals = []
    y = []

    for i in range(int(N_signals/2)):
        signal, label = concatenate_signals(window = WINDOW, mean1 = mean1, mean2 = mean2, overlap = overlap)
        signals.append(signal)
        y.append(label)
        signal, label = concatenate_signals(window = WINDOW, mean1 = mean1, mean2 = mean2, overlap = 0)
        signals.append(signal)
        y.append(label)

    signals = [np.array(i) for i in signals]

    return signals, y

def make_spectr(signals,second_part = 0.5):
    ind = int(len(signals[0]) * (1 - second_part))
    signal_1 = [i[:ind] for i in signals]
    signal_2 = [i[ind:] for i in signals]
    signal_1_fft = [np.abs(rfft(i.flatten())) for i in signal_1]
    signal_1_fft = [i/i.max() for i in signal_1_fft]
    signal_2_fft = [np.abs(rfft(i.flatten())) for i in signal_2]
    signal_2_fft = [i/i.max() for i in signal_2_fft]
    signal_1_fft = np.array(signal_1_fft)
    signal_2_fft = np.array(signal_2_fft)
    print('Размеры спектров', signal_1_fft.shape, signal_2_fft.shape)
    return signal_1_fft, signal_2_fft

def compile_and_learn(model, X_train, y_train, X_test, y_test,  epochs, optimizer, loss, metrics, batch_size):

    # Компиляция модели
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
    model.summary()

    # Обучение модели
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=16,
                        validation_data=(X_test, y_test),
                        verbose = 1)

    f, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15,5))
    axes[0].plot(history.history['loss'], label='Ошибка на обучающем наборе')
    axes[0].plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Ошибка')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Точность на обучающем наборе')
    axes[1].plot(history.history['val_accuracy'], label='Точность на проверочном наборе')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Точность')
    axes[1].legend()

    print('Тестовые данные')
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(np.argmax(y_test, axis=1), y_pred))
    print('Тренировочные данные')
    y_pred = np.argmax(model.predict(X_train), axis=1)
    print(classification_report(np.argmax(y_train, axis=1), y_pred))
    return model

def prepare_data_1(signal_1_fft, signal_2_fft, y):
    X = np.concatenate([signal_1_fft, signal_2_fft], axis=1)
    y = to_categorical(y)
    print('signal_1_fft', signal_1_fft.shape)
    print('signal_2_fft', signal_2_fft.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.3)
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)
    return X_train, X_test, y_train, y_test

