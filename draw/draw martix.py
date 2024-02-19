
import matplotlib.pyplot as plt
import numpy as np
import os
from draw.convergence_rate import plot_once


def plt_perfect_game_convergence_inline(game_name, logdir, is_x_log=True, is_y_log=True, y_num=3, x_num=0,
                                        log_interval_mode='node_touched'):
    # file_list = os.listdir(logdir)
    file_list = [f for f in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, f))]
    file_list.sort()

    plt.ylabel('log10(Exploitability)', fontsize=24)
    if log_interval_mode == 'node_touched':
        if is_x_log == True:
            is_y_log = True
            plt.xlabel('log10(Node touched)', fontsize=24)
        if is_x_log == False:
            is_y_log = True
            plt.xlabel('Node touched', fontsize=24)
    elif log_interval_mode == 'train_time':
        if is_x_log == True:
            is_y_log = True
            plt.xlabel('log10(Train Time)', fontsize=24)
        if is_x_log == False:
            is_y_log = True
            plt.xlabel('Train Time', fontsize=24)

    elif log_interval_mode == 'itr':
        if is_x_log == True:
            is_y_log = True
            plt.xlabel('log10(Iterations)', fontsize=24)
        if is_x_log == False:
            is_y_log = True
            plt.xlabel('Iterations', fontsize=24)

    for i_file in range(len(file_list)):
        plot_once(
            logdir + '/' + file_list[i_file],
            i_file,
            file_list[i_file],
            is_x_log=is_x_log,
            is_y_log=is_y_log,
            y_num=y_num,
            x_num=x_num
        )

    # plt.title(game_name, fontsize=24)











if __name__ == '__main__':
    plt.figure(figsize=(32, 10), dpi=60)

    game_name = 'Matrix game'
    file_path = 'XX'
    file_path1 = 'XX'
    file_path3 = 'XX'

    plt.subplot(121)
    plt_perfect_game_convergence_inline(
        game_name,
        file_path1,
        # is_x_log=False,
        is_x_log=True,
        x_num=0,
        y_num=2,
        log_interval_mode='itr'
    )

    # plt.subplot(132)
    # plt_perfect_game_convergence_inline(
    #     game_name,
    #     file_path2,
    #     # is_x_log=False,
    #     is_x_log=True,
    #     x_num=0,
    #     y_num=2,
    #     log_interval_mode='itr'
    # )

    plt.subplot(122)
    # plt.xlim(1, 4.8)
    plt_perfect_game_convergence_inline(
        game_name,
        file_path3,
        # is_x_log=False,
        is_x_log=True,
        x_num=1,
        y_num=2,
        log_interval_mode='train_time'
    )

    plt.suptitle("2-Player {}-Action Zero-Sum Matrix Games".format(file_path1[-2:]), fontsize=30)
    # plt.tight_layout(h_pad=4, w_pad=2, pad=2)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.2)
    # plt.axis()
    plt.legend(edgecolor='red', fontsize=20)
    plt.savefig(file_path + '/log_pic_{}_9_3.png'.format(file_path1[-2:]), dpi=240)
    plt.show()