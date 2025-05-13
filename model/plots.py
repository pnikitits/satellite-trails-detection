import matplotlib.pyplot as plt
import os


def plot_metrics(history, display=True, save_at=None):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.plot(epochs, history['real_loss'], label='Real Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    # plt.plot(epochs, history['real_acc'], label='Real Acc')
    plt.plot(epochs, history['real_acc_0s'], label='real_acc_0s')
    plt.plot(epochs, history['real_acc_1s'], label='real_acc_1s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1'], label='Train F1')
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.plot(epochs, history['real_f1'], label='Real F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()

    plt.tight_layout()
    
    if display:
        plt.show()
        
    if save_at != None:
        plt.savefig(os.path.join(save_at, 'history_plot.png'))
        plt.clf()