import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plot_color = [
    [
        (57 / 255, 81 / 255, 162 / 255),  # deep
        (202 / 255, 232 / 255, 242 / 255),  # middle
        (114 / 255, 170 / 255, 207 / 255),  # light

    ],
    [
        (168 / 255, 3 / 255, 38 / 255),
        (253 / 255, 185 / 255, 107 / 255),
        (236 / 255, 93 / 255, 59 / 255),
    ],
    [
        (0 / 255, 128 / 255, 51 / 255),
        (202 / 255, 222 / 255, 114 / 255),
        (226 / 255, 236 / 255, 179 / 255)
    ],
    [
        (128 / 255, 0 / 255, 128 / 255),
        (204 / 255, 153 / 255, 255 / 255),
        (128 / 255, 128 / 255, 128 / 255)
    ],
    [
        (255 / 255, 215 / 255, 0 / 255),  # yellow
        (255 / 255, 239 / 255, 213 / 255),
        (255 / 255, 250 / 255, 240 / 255),
    ],
    [
        (207 / 255, 145 / 255, 151 / 255),  # pink
        (231 / 255, 208 / 255, 211 / 255),
        (245 / 255, 239 / 255, 238 / 255),
    ],
    [
        (255 / 255, 165 / 255, 0 / 255),  # orange
        (255 / 255, 192 / 255, 128 / 255),
        (255 / 255, 224 / 255, 192 / 255),
    ],
    [
        (184 / 255, 146 / 255, 106 / 255),  # brown
        (210 / 255, 191 / 255, 166 / 255),
        (239 / 255, 237 / 255, 231 / 255),
    ],
    [
        (63 / 255, 55 / 255, 54 / 255),  # black
        (126 / 255, 127 / 255, 122 / 255),
        (234 / 255, 230 / 255, 223 / 255),
    ],
]

plot_marker = ['s', '^', '*', 'o', 'D', 'x', '+', '<', '>']


def get_file_name_list(path: str) -> list:
    file_list = os.listdir(path)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    csv_file = []
    for i_file in file_list:
        if i_file[-2:] == 'WS':
            continue
        csv_file.append(path + '/' + i_file + '/epsilon.csv')
        # csv_file.append(path + '/' + i_file + '/tmp.csv')
    return csv_file


def get_result(file_path: str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=',', skiprows=2)  # 提取file_path这个文件里面的内容，转换为一个数组形式  ，表示里面的分隔符

    return data


def plot_once(path, num, ex_name, is_x_log=True, is_y_log=True, y_num=3, x_num=0):
    # 如果是时间x_num=1,如果是itr x_num=0
    csv_file_list = get_file_name_list(path)  # 获取子文件中所有的epsilon文件
    _10_num = int(0.1 * len(csv_file_list))
    tmp_data = get_result(csv_file_list[0])

    if is_x_log:
        indices = np.where(np.log10(tmp_data[:, x_num]) < 1.1)
        tmp_data = np.delete(tmp_data, indices, axis=0)
        tmp_x = np.log10(tmp_data[:, x_num])  # 训练次数    从tmp_data中取第一列的所有元素，并进行以10为底的log运算
    else:
        tmp_x = tmp_data[:, x_num]  # 否则直接讲第一列元素赋给tmp_x
    tmp_min_x = tmp_data.shape[0]

    tmp_y = tmp_data[:, y_num]  # 将第三列的值赋值给tmpy  即epsilon

    y_matrix = np.zeros((len(csv_file_list), tmp_min_x))  # 创建二维数组

    y_matrix[0, :] = tmp_data[:, y_num]  # 将tem_y全部元素赋值给二维数组的第一行
    for i in range(1, len(csv_file_list)):
        tmp_data = get_result(csv_file_list[i])
        now_min_x = tmp_data.shape[0]
        if now_min_x < tmp_min_x:
            tmp_min_x = now_min_x
            y_matrix = y_matrix[:, -tmp_min_x:]
            tmp_y = tmp_y[-tmp_min_x:]
            tmp_x = tmp_x[-tmp_min_x:]
        tmp_data = tmp_data[-tmp_min_x:, :]

        tmp_y += tmp_data[:, y_num]  # 对第y-num列的所有数据求和 分别对应相加
        y_matrix[i, :] = tmp_data[:, y_num]
        if is_y_log:
            plt.scatter(tmp_x, np.log10(tmp_data[:, y_num]), s=1, color=plot_color[num][1], alpha=0.7)
        else:
            plt.scatter(tmp_x, tmp_data[:, y_num], s=1, marker=plot_marker[num], color=plot_color[num][1], alpha=0.3)
    y_matrix.sort(axis=0)  # 对y_matrix的第一列从小到大排序
    tmp_y = tmp_y / len(csv_file_list)  # 求平均
    if is_y_log:
        tmp_y = np.log10(tmp_y)
        y_matrix = np.log10(y_matrix)

    plt.plot(tmp_x, tmp_y, marker=plot_marker[num], markersize=10, c=plot_color[num][0], lw=2, label=ex_name)

    plt.fill_between(tmp_x, y_matrix[_10_num, :], y_matrix[-_10_num - 1, :], color=plot_color[num][2], alpha=0.5)
    plt.tick_params(axis='both', labelsize=20)


def plt_perfect_game_convergence_inline(game_name, logdir, is_x_log=True, is_y_log=True, y_num=3, x_num=0,
                                        log_interval_mode='node_touched'):
    file_list = os.listdir(logdir)
    file_list.sort()

    plt.ylabel('log10(Exploitability)')
    if log_interval_mode == 'node_touched':
        is_x_log = False
        is_y_log = True
        plt.xlabel('Node touched')
    elif log_interval_mode == 'train_time':
        is_x_log = False
        is_y_log = True
        plt.xlabel('Train Time')
    elif log_interval_mode == 'itr':
        is_x_log = False
        is_y_log = True
        plt.xlabel('Itr')
    else:
        plt.xlabel(log_interval_mode)

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

    plt.title(game_name)


def plt_correspondence(logdir, y_num=3, x_num=0):
    def compare(tmp):
        tmptmp=int(tmp.split('-')[0])
        return tmptmp

    file_list = os.listdir(logdir)
    file_list.sort(key=compare)
    plt.title('The correspondence between the number of iterations of DW-FP and FP',
              font={'size': 24})
    plt.ylabel('Corresponding Iterations in FP', font={'size': 24})
    plt.xlabel('Iterations in DW-FP', font={'size': 24})

    for i_file in range(len(file_list)):
        plot_once(
            logdir + '/' + file_list[i_file],
            i_file,
            file_list[i_file],
            is_x_log=True,
            is_y_log=True,
            y_num=y_num,
            x_num=x_num
        )

def plt_correspondence_extensive(logdir, y_num=3, x_num=0):
    def compare(tmp):
        return len(tmp)

    file_list = os.listdir(logdir)
    file_list.sort(key=compare)
    plt.title('The correspondence between the number of iterations of DW-PCFR and PCFR',
              font={'size': 24})
    plt.ylabel('Corresponding Iterations in PCFR', font={'size': 24})
    plt.xlabel('Iterations in DW-PCFP', font={'size': 24})

    for i_file in range(len(file_list)):
        plot_once(
            logdir + '/' + file_list[i_file],
            i_file,
            file_list[i_file],
            is_x_log=True,
            is_y_log=True,
            y_num=y_num,
            x_num=x_num
        )