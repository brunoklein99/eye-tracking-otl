import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def smooth(series, alpha=0.9):
    for i in range(1, len(series)):
        series[i] = (1 - alpha) * series[i - 1] + alpha * series[i]


def plot(pre_training, stage1_online, stage2_online, stage1_offline, stage2_offline):
    assert len(stage1_online) == len(stage1_offline)
    assert len(stage2_online) == len(stage2_offline)
    font_size = 40
    sns.set_style('darkgrid')
    plt.xlabel('Iterations', fontsize=font_size)
    plt.ylabel('Loss', fontsize=font_size)

    online = stage1_online + stage2_online
    offline = stage1_offline + stage2_offline

    offset = 0
    plt.plot(range(offset, offset + len(pre_training)), pre_training)
    offset += len(pre_training)
    plt.plot(range(offset, offset + len(online)), online)
    plt.plot(range(offset, offset + len(offline)), offline)
    plt.axvline(x=len(pre_training), color='r', linestyle=':')
    plt.axvline(x=len(pre_training) + len(stage1_online), color='r', linestyle=':')
    plt.legend(['Pre-training', 'w/ Online Learning', 'w/o Online Learning'], prop={'size': font_size})
    plt.rcParams.update({'font.size': font_size})
    plt.tick_params(labelsize=font_size)
    plt.text(len(pre_training) + 100, 0.05, 'Env 1', fontsize=font_size)
    plt.text(len(pre_training) + len(stage1_online) + 100, 0.05, 'Env 2', fontsize=font_size)
    plt.show()


def plot_smooth(pre_training, stage1_online, stage2_online, stage1_offline, stage2_offline, alpha=0.1):
    smooth(pre_training, alpha)
    smooth(stage1_online, alpha)
    smooth(stage2_online, alpha)
    smooth(stage1_offline, alpha)
    smooth(stage2_offline, alpha)
    plot(pre_training, stage1_online, stage2_online, stage1_offline, stage2_offline)


if __name__ == '__main__':
    with open('data-charts/losses_batch', 'rb') as f:
        losses_batch = pickle.load(f)
    with open('data-charts/stage1_online', 'rb') as f:
        stage1_online = pickle.load(f)
    with open('data-charts/stage2_online', 'rb') as f:
        stage2_online = pickle.load(f)
    with open('data-charts/stage1_offline', 'rb') as f:
        stage1_offline = pickle.load(f)
    with open('data-charts/stage2_offline', 'rb') as f:
        stage2_offline = pickle.load(f)
    plot_smooth(losses_batch, stage1_online, stage2_online, stage1_offline, stage2_offline)
    # offset = 0
    # pre_training = []
    # with open('data-charts/pre-training-losses-batch') as f:
    #     for line in f.readlines():
    #         pre_training.append(float(line))
    #
    # stage1 = []
    # with open('data-charts/stage1') as f:
    #     for line in f.readlines():
    #         stage1.append(float(line))
    #
    # stage2 = []
    # with open('data-charts/stage2') as f:
    #     for line in f.readlines():
    #         stage2.append(float(line))
    #
    # plot_smooth(pre_training, stage1, stage2)
