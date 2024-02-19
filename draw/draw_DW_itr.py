import csv
import numpy as np
import matplotlib.pyplot as plt
import convergence_rate
import os

if __name__ == '__main__':


    plt.figure(figsize=(16, 10), dpi=240)
    base_logdir = os.getcwd() + '/data_backup/extensive'
    convergence_rate.plt_correspondence_extensive(
        base_logdir,
        x_num=0,
        y_num=7,

    )

    plt.tight_layout()
    # plt.axis()
    plt.legend(edgecolor='red', fontsize=20)
    plt.savefig(os.getcwd() + '/pic.png')
