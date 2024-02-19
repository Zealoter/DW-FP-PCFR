import numpy as np
import matplotlib.pyplot as plt
import optim
import os
import time
import csv
import draw.convergence_rate
from joblib import Parallel, delayed

N = 2
num_of_act = 10


def get_br_policy(bar_q1, bar_q2):
    return np.argmax(bar_q1), np.argmax(bar_q2)


def get_rm_policy(bar_q1, bar_q2):
    def regret_matching(r):
        copy_r = r.copy()
        copy_r[copy_r < 0] = 0
        if np.sum(copy_r) == 0:
            return np.ones(num_of_act) / num_of_act
        else:
            return copy_r / np.sum(copy_r)

    return regret_matching(bar_q1), regret_matching(bar_q2)


def no_regret_learning(u1, u2, itr, mode, log_interval, keys):
    def get_epsilon(bar_sigma1, bar_sigma2):
        tmp2 = np.matmul(bar_sigma1, u2)
        tmp1 = np.matmul(u1, bar_sigma2)
        return np.max(tmp1) + np.max(tmp2)

    def get_value(bar_sigma1, bar_sigma2):
        value2 = np.dot(np.matmul(bar_sigma1, u2), bar_sigma2)
        value1 = np.dot(bar_sigma1, np.matmul(u1, bar_sigma2))
        return value1, value2

    def save_epsilon(i_itr):
        tmp_epsilon = get_epsilon(bar_sigma1 / np.sum(bar_sigma1), bar_sigma2 / np.sum(bar_sigma2))
        epsilon_list.append(tmp_epsilon)
        x_list.append(i_itr)
        now_time_list.append(time.time())
        dw_list.append(overall_w)

    def get_min_change(max_q, max_v, a):
        max_num = np.max(max_q)
        gap = max_num - max_q
        change_num = max_v - max_v[a]
        change_ge_zero = np.where(change_num > 0.0)
        tmp_time = gap[change_ge_zero] // change_num[change_ge_zero] + 1
        if not list(tmp_time):
            return 99999999999999999999999.0
        else:
            return np.min(tmp_time)

    epsilon_list = []
    regret_list = []
    x_list = []

    bar_sigma1 = np.zeros(num_of_act)
    bar_sigma2 = np.zeros(num_of_act)

    bar_q1 = np.zeros(num_of_act)
    bar_q2 = np.zeros(num_of_act)

    step = []
    overall_w = 0
    now_time_list = []
    dw_list = []
    log_base = 10

    if mode in ['FP', 'DWFP', 'LinerFP']:
        sigma1 = 0
        sigma2 = 0

        for i in range(1, itr + 1):
            q1 = u1[:, sigma2]
            q2 = u2[sigma1, :]

            if mode == 'LinerFP':
                bar_q1 += i * (q1 - q1[sigma1])
                bar_q2 += i * (q2 - q2[sigma2])

                bar_sigma1[sigma1] += i
                bar_sigma2[sigma2] += i

                overall_w += i

            elif mode == 'FP':
                bar_q1 += (q1 - q1[sigma1])
                bar_q2 += (q2 - q2[sigma2])

                bar_sigma1[sigma1] += 1
                bar_sigma2[sigma2] += 1

                overall_w += 1

            elif mode == 'DWFP':
                w1 = get_min_change(bar_q1, q1, sigma1)
                w2 = get_min_change(bar_q2, q2, sigma2)
                w = min(w2, w1)

                bar_q1 += w * (q1 - q1[sigma1])
                bar_q2 += w * (q2 - q2[sigma2])

                bar_sigma1[sigma1] += w
                bar_sigma2[sigma2] += w

                overall_w += w

                step.append(np.log10(w))

            sigma1, sigma2 = get_br_policy(bar_q1, bar_q2)

            if log_base <= i:
                log_base *= log_interval
                save_epsilon(i)

        save_epsilon(itr)
    else:
        sigma1 = np.ones(num_of_act) / num_of_act
        sigma2 = np.ones(num_of_act) / num_of_act
        if mode in ['RM', 'RM+', 'LinerRM+']:
            for i in range(1, itr + 1):
                q1 = np.matmul(u1, sigma2)
                q2 = np.matmul(sigma1, u2)

                if keys[0]:
                    bar_q1 += i * (q1 - np.dot(sigma1, q1))
                    bar_q2 += i * (q2 - np.dot(sigma2, q2))

                    bar_sigma1 += (i * sigma1)
                    bar_sigma2 += (i * sigma2)

                    overall_w += i
                else:
                    bar_q1 += (q1 - np.dot(sigma1, q1))
                    bar_q2 += (q2 - np.dot(sigma2, q2))

                    bar_sigma1 += sigma1
                    bar_sigma2 += sigma2

                    overall_w += 1

                if mode in ['LinerRM+', 'RM+']:
                    bar_q1[bar_q1 < 0] = 0
                    bar_q2[bar_q2 < 0] = 0

                sigma1, sigma2 = get_rm_policy(bar_q1, bar_q2)

                if log_base <= i:
                    log_base *= log_interval
                    save_epsilon(i)

        else:
            for i in range(1, itr + 1):
                if keys[1]:
                    tmp_sigma1 = sigma1.copy()
                    sigma1 = np.zeros(num_of_act)
                    sigma1[np.random.choice(num_of_act, p=tmp_sigma1)] = 1.0

                    tmp_sigma2 = sigma2.copy()
                    sigma2 = np.zeros(num_of_act)
                    sigma2[np.random.choice(num_of_act, p=tmp_sigma2)] = 1.0

                q1 = np.matmul(u1, sigma2)
                q2 = np.matmul(sigma1, u2)
                r1 = q1 - np.dot(sigma1, q1)
                r2 = q2 - np.dot(sigma2, q2)
                if i == 1:
                    w = 1
                else:
                    tmp_bar_q = np.concatenate((bar_q1, bar_q2))
                    tmp_r = np.concatenate((r1, r2))
                    w, _ = optim.find_optimal_weight(tmp_bar_q, tmp_r, overall_w)
                if w == np.inf:
                    w = 1
                if keys[2]:
                    w = max(w, overall_w / (2 * i))
                overall_w += w

                bar_q1 += w * r1
                bar_q2 += w * r2

                bar_sigma1 += (w * sigma1)
                bar_sigma2 += (w * sigma2)

                sigma1, sigma2 = get_rm_policy(bar_q1, bar_q2)

                if log_base <= i:
                    log_base *= log_interval
                    save_epsilon(i)

        save_epsilon(itr)

    result_dict = {
        'bar_sigma1'   : bar_sigma1 / np.sum(bar_sigma1),
        'bar_sigma2'   : bar_sigma2 / np.sum(bar_sigma2),
        'epsilon_list' : np.array(epsilon_list),
        'regret_list'  : np.log10(np.array(regret_list)),
        'x_list'       : x_list,
        'step'         : step,
        'now_time_list': now_time_list,
        'dw_list'      : dw_list
    }
    return result_dict


def train_sec(i_itr):
    for i_key in train_dict.keys():
        print(i_key)
        u1 = np.random.randn(num_of_act, num_of_act)
        u2 = -u1

        result_file_path = ''.join(
            [
                logdir,
                '/',
                i_key,
                '/',
                str(i_itr),
            ]
        )
        os.makedirs(result_file_path)
        start_time = time.time()
        tmp_result_dict = no_regret_learning(u1, u2, 10000, i_key, 1.5, train_dict[i_key])

        with open(result_file_path + '/epsilon.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['itr', 'train_time(ms)', 'epsilon', 'DW_num'])
            for i_eps in range(len(tmp_result_dict['x_list'])):
                writer.writerow([
                    tmp_result_dict['x_list'][i_eps],
                    (tmp_result_dict['now_time_list'][i_eps] - start_time) * 1000,
                    tmp_result_dict['epsilon_list'][i_eps],
                    tmp_result_dict['dw_list'][i_eps]
                ])
    return


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)

    train_dict = {
        'FP'                    : [False],
        'LinerFP'               : [True],
        'DWFP'                  : [False],
        'RM'                    : [False],
        'RM+'                   : [False],
        'LinerRM+'              : [True],
        'vanilla GreedyRM'      : [False, False, False],
        'MC GreedyRM'           : [False, True, False],
        'MC GreedyRM with Floor': [False, True, True],
    }
    now_path_str = os.getcwd()

    now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    logdir = ''.join(
        [
            now_path_str,
            '/logGFSPSampling/Matrix/',
            now_time_str,

        ]
    )
    num_of_core = 1
    num_of_train = 1
    ans_list = Parallel(n_jobs=num_of_core)(
        delayed(train_sec)(i_itr) for i_itr in range(num_of_train)
    )

    plt.figure(figsize=(32, 10), dpi=180)
    plt.subplot(121)
    draw.convergence_rate.plt_perfect_game_convergence_inline(
        'XXXX',
        logdir,
        is_x_log=True,
        x_num=0,
        y_num=2,
        log_interval_mode='itr'
    )
    plt.subplot(122)
    draw.convergence_rate.plt_perfect_game_convergence_inline(
        'XXXX',
        logdir,
        is_x_log=True,
        x_num=1,
        y_num=2,
        log_interval_mode='train_time'
    )

    plt.tight_layout()
    # plt.axis()
    plt.legend(edgecolor='red')
    plt.savefig(logdir + '/pic.png')
