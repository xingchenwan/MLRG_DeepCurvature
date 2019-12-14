import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np
import pandas as pd


def plot_training(dir: str, max_epoch=100, prefix='stats-', swag=False, show_top_5=False):
    """
    Visualise the training process including train/test top1/top5 accuracy + loss
    :param dir: directories where the statistics files are saved
    :param max_epoch: maximum epoch allocated
    :param prefix: the prefix to the stats files (default: 'stats-')
    :param swag: whether SWAG is enabled
    :param show_top_5: whether show Top 5 accuracy in addition to the Top 1 accuracy
    :return:
    """
    stats = [
        'train_accuracy',
        'train_top5_accuracy',
        'test_accuracy',
        'test_top5_accuracy',
        'train_loss',
        'test_loss'
    ]
    x = np.arange(max_epoch)
    df = pd.DataFrame(np.nan, index=x, columns=stats)
    for i in range(max_epoch):
        a = np.load(dir + prefix + str(i) + ".npz", allow_pickle=True)
        for col in stats:
            if a[col] != [None]:
                df.loc[i, col] = a[col]
            else:
                df.loc[i, col] = np.nan
        if swag:
            try:
                df.loc[i, 'test_loss'] = a['swag_loss']
                df.loc[i, 'test_accuracy'] = a['swag_accuracy']
                df.loc[i, 'test_top5_accuracy'] = a['top5_accuracy']
            except KeyError:
                pass
    plt.subplot(121)
    sns.lineplot(x, df['train_accuracy'], label='Train Accuracy')
    sns.lineplot(x, df['test_accuracy'], label='Test Accuracy')
    if show_top_5:
        sns.lineplot(x, df['train_top5_accuracy'], label='Train Top5 Accuracy')
        sns.lineplot(x, df['test_top5_accuracy'], label='Test Top5 Accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.subplot(122)
    sns.lineplot(x, df['train_loss'], label='Train loss')
    sns.lineplot(x, df['test_loss'], label='Test loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
