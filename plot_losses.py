import matplotlib.pyplot as plt


def main():
    plt.rcParams.update({'font.size': 13})
    train_loss_lstm = [0.1101, 0.0757, 0.0690, 0.0659, 0.0637, 0.0618, 0.0605,
                       0.0594, 0.0585, 0.0578]
    val_loss_lstm = [0.0756, 0.0670, 0.0632, 0.0618, 0.0598, 0.0588, 0.0578,
                     0.0572, 0.0567, 0.0561]
    train_loss_cnn = [0.1077, 0.0831, 0.0780, 0.0753, 0.0735, 0.0722, 0.0712,
                      0.0705, 0.0699, 0.0693]
    val_loss_cnn = [0.0826, 0.0760, 0.0729, 0.0709, 0.0696, 0.0688, 0.0683,
                    0.0678, 0.0671, 0.0669]
    train_loss_transformer = [0.3652, 0.3454, 0.3394, 0.3355, 0.3324, 0.3295,
                              0.3270, 0.3244, 0.3218, 0.3190]
    val_loss_transformer = [0.3453, 0.3362, 0.3304, 0.3264, 0.3227, 0.3196,
                            0.3166, 0.3125, 0.3091, 0.3051]
    epochs = list(range(1, 11))

    plt.figure()
    plt.plot(epochs, train_loss_lstm, label='LSTM', marker='o')
    plt.plot(epochs, train_loss_cnn, label='CNN', marker='s')
    plt.plot(epochs, train_loss_transformer, label='Transformer', marker='^')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.pdf')
    plt.show()

    plt.figure()
    plt.plot(epochs, val_loss_lstm, label='LSTM', marker='o')
    plt.plot(epochs, val_loss_cnn, label='CNN', marker='s')
    plt.plot(epochs, val_loss_transformer, label='Transformer', marker='^')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('validation_loss.pdf')
    plt.show()


if __name__ == '__main__':
    main()
