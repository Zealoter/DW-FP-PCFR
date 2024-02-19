import numpy as np
import os
import time
import csv
import copy


class GFSPSamplingSolver(object):
    def __init__(self, config: dict):
        self.game = config.get('game')
        self.total_exp_name = config.get('total_exp_name')
        self.game_name = self.game.game_name
        self.game_info = config.get('game_info', '')
        self.is_show_policy = config.get('is_show_policy', False)
        now_path_str = os.getcwd()

        now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        self.result_file_path = ''.join(
            [
                now_path_str,
                '/logGFSPSampling/',
                self.total_exp_name,
                '/',
                self.game_info,
                '/',
                str(config.get('No.')),
                '_',
                now_time_str
            ]
        )

        self.is_rm_plus = config.get('is_rm_plus', False)
        self.node_touched = 0
        self.train_mode = config.get('train_mode', 'fix_itr')

        self.total_train_constraint = config.get('total_train_constraint', 1000)
        self.log_interval = config.get('log_interval', 100)
        self.log_interval_mode = config.get('log_interval_mode')
        self.ave_mode = config.get('ave_mode', 'vanilla')
        self.log_mode = config.get('log_mode', 'normal')
        self.log_state_start_num = config.get('log_state_start_num', 1000)
        self.start_time = 0
        self.itr_num = 0
        self.total_weight = 0
        self.total_reward = np.zeros(self.game.player_num)
        self.weight_list = []

        self.last_itr_node_touched = 0
        self.moving_ave_pure_policy_rate = 0
        self.first_log = True
        self.ave_weight = 0
        self.log_threshold = self.log_interval

    def save_model(self, episode):
        tmp_policy = self.game.get_his_mean_policy()
        np.save(self.result_file_path + '/' + str(episode) + '.npy', tmp_policy)

    def log_model(self, itr: int, train_time: float = 0.0):
        sum_imm_regret_per_player = self.game.get_sum_imm_regret()
        weighted_mean_imm_regret = sum_imm_regret_per_player / self.total_weight

        epsilon = 0
        log_info = {
            'itr'                   : itr,
            'train_time(ms)'        : train_time * 1000,
            'epsilon'               : epsilon,
            'mean_imm_regret'       : np.sum(weighted_mean_imm_regret),
            'node_touched'          : self.node_touched,
            'sum_imm_regret'        : np.sum(sum_imm_regret_per_player),
            'last_time_node_touched': self.node_touched - self.last_itr_node_touched,
            'total_weight'          : self.total_weight
        }
        for i_player in range(self.game.player_num):
            log_info['mean_reward' + str(i_player + 1)] = self.total_reward[i_player] / self.total_weight
        for i_player in range(self.game.player_num):
            log_info['perfect_BR' + str(i_player + 1)] = weighted_mean_imm_regret[i_player]

        self.save_model(self.itr_num)
        with open(self.result_file_path + '/tmp.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_info.keys())
            if self.first_log:
                writer.writeheader()
            writer.writerow(log_info)
        return

    def get_epsilon(self, appr_ne_policy: dict):
        # p1 BR value
        self.game.now_policy = copy.deepcopy(appr_ne_policy)
        for i_key in self.game.imm_regret.keys():
            self.game.imm_regret[i_key] *= 0

        game_value = self.epsilon_walk_tree('_', np.ones(self.game.player_num), 1.0)
        gap_value = np.zeros_like(game_value)
        fix_imm_regret = copy.deepcopy(self.game.imm_regret)

        for i_player in range(self.game.player_num):
            self.game.now_policy = copy.deepcopy(appr_ne_policy)
            for info in self.game.info_set_list[self.game.player_set[i_player]]:
                br_act = np.argmax(fix_imm_regret[info])
                self.game.now_policy[info] = np.zeros_like(fix_imm_regret[info])
                self.game.now_policy[info][br_act] = 1.0
            tmp_br_value = self.epsilon_walk_tree('_', np.ones(self.game.player_num), 1.0)
            gap_value[i_player] = tmp_br_value[i_player] - game_value[i_player]
        epsilon = np.sum(gap_value)
        return epsilon, game_value

    def walk_tree(self, his_feat, player_pi, pi_c):
        return self.vanilla_walk_tree(his_feat, player_pi, pi_c)

    def vanilla_walk_tree(self, his_feat, player_pi, pi_c):
        self.node_touched += 1
        if np.sum(player_pi) == 0.0 or pi_c == 0:
            return np.zeros(self.game.player_num)

        r = np.zeros(self.game.player_num)

        if self.game.get_now_player_from_his_feat(his_feat) == 'c':

            now_prob = self.game.get_chance_prob(his_feat)
            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
            for a_i in range(len(now_action_list)):
                tmp_r = self.vanilla_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    player_pi,
                    pi_c * now_prob[a_i]
                )
                r += tmp_r
        else:
            now_player = self.game.get_now_player_from_his_feat(his_feat)
            tmp_info = self.game.get_info_set(now_player, his_feat)
            now_player_index = self.game.player_set.index(now_player)
            now_prob = player_pi[now_player_index]

            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
            if len(now_action_list) == 0:
                tmp_reward = self.game.judge(his_feat)
                tmp_reward[0] *= player_pi[1]
                tmp_reward[1] *= player_pi[0]
                return tmp_reward * pi_c

            v = np.zeros(len(now_action_list))

            if tmp_info not in self.game.imm_regret.keys():
                self.game.generate_new_info_set(tmp_info, now_player, len(now_action_list))

            self.game.now_prob[tmp_info] += (now_prob * pi_c)

            for a_i in range(len(now_action_list)):
                new_player_pi = copy.deepcopy(player_pi)
                new_player_pi[now_player_index] = new_player_pi[now_player_index] * self.game.now_policy[tmp_info][a_i]
                tmp_r = self.vanilla_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    new_player_pi,
                    pi_c
                )
                v[a_i] = tmp_r[now_player_index]
                for i in range(self.game.player_num):
                    if i != now_player_index:
                        r[i] = r[i] + tmp_r[i]
                r[now_player_index] += tmp_r[now_player_index] * self.game.now_policy[tmp_info][a_i]

            tmp_regret = v - np.dot(self.game.now_policy[tmp_info], v)
            if self.is_rm_plus:
                pass
            else:
                tmp_regret = tmp_regret * self.ave_weight
            self.game.imm_regret[tmp_info] += tmp_regret

        return r

    def epsilon_walk_tree(self, his_feat, player_pi, pi_c):
        if self.game.player_num - np.count_nonzero(player_pi) == 2 or pi_c == 0:
            return np.zeros(self.game.player_num)
        now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
        if len(now_action_list) == 0:
            tmp_reward = self.game.judge(his_feat)
            tmp_reward[0] *= player_pi[1]
            tmp_reward[1] *= player_pi[0]
            return tmp_reward * pi_c

        r = np.zeros(self.game.player_num)

        if self.game.get_now_player_from_his_feat(his_feat) == 'c':
            now_prob = self.game.get_chance_prob(his_feat)
            for a_i in range(len(now_action_list)):
                tmp_r = self.epsilon_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    player_pi,
                    pi_c * now_prob[a_i]
                )
                r += tmp_r
        else:
            now_player = self.game.get_now_player_from_his_feat(his_feat)
            tmp_info = self.game.get_info_set(now_player, his_feat)
            now_player_index = self.game.player_set.index(now_player)

            v = np.zeros(len(now_action_list))

            if tmp_info not in self.game.now_policy.keys():
                self.game.now_policy[tmp_info] = np.ones(len(now_action_list)) / len(now_action_list)
            if tmp_info not in self.game.imm_regret.keys():
                self.game.imm_regret[tmp_info] = np.zeros(len(now_action_list))

            for a_i in range(len(now_action_list)):
                new_player_pi = copy.deepcopy(player_pi)
                new_player_pi[now_player_index] = new_player_pi[now_player_index] * self.game.now_policy[tmp_info][a_i]
                tmp_r = self.epsilon_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    new_player_pi,
                    pi_c
                )
                v[a_i] += tmp_r[now_player_index]
                for i in range(self.game.player_num):
                    if i != now_player_index:
                        r[i] = r[i] + tmp_r[i]
                r[now_player_index] += tmp_r[now_player_index] * self.game.now_policy[tmp_info][a_i]

            tmp_regret = v - np.sum(self.game.now_policy[tmp_info] * v)
            self.game.imm_regret[tmp_info] += tmp_regret
        return r

    def get_ave_weight(self):
        if self.ave_mode == 'vanilla':
            return 1
        elif self.ave_mode == 'log':
            return np.log10(self.itr_num + 1)
        elif self.ave_mode == 'liner':
            return self.itr_num
        elif self.ave_mode == 'square':
            return self.itr_num * self.itr_num
        else:
            print('error')
            return 0

    def regret_matching_strategy(self, info):
        tmp_r = copy.deepcopy(self.game.imm_regret[info])
        tmp_r[tmp_r < 0] = 0.0

        if np.sum(tmp_r) < 1.0e-20:
            tmp_act = np.random.rand(len(self.game.now_policy[info]))
            self.game.now_policy[info] = tmp_act / np.sum(tmp_act)
        else:
            self.game.now_policy[info] = tmp_r / np.sum(tmp_r)

    def all_state_regret_matching_strategy(self):
        # 在全遍历模式下的遗憾匹配
        for info in self.game.now_prob.keys():
            tmp_now_prob = self.game.now_prob[info] * self.game.now_policy[info]
            tmp_now_prob = tmp_now_prob * self.ave_weight
            self.game.w_his_policy[info] += tmp_now_prob

            if self.is_rm_plus:
                self.game.imm_regret[info][self.game.imm_regret[info] < 0] = 0.0

            self.regret_matching_strategy(info)

            # 到达这个信息集的概率
            self.game.now_prob[info] = 0

    def prepare_before_train(self, tmp_reward):
        self.last_itr_node_touched = self.node_touched
        self.itr_num += 1

        self.all_state_regret_matching_strategy()

        if self.game_info == 'syncPCFR':
            pass
        else:
            self.ave_weight = self.get_ave_weight()
            self.total_weight += self.ave_weight

        if self.game_info == 'syncPCFR':
            if self.itr_num == 1:
                self.last_weight = 1
            self.total_weight += self.last_weight
            self.weight_list.append(self.last_weight)
            self.total_reward += (tmp_reward * self.last_weight)
        else:
            self.total_reward += (tmp_reward * self.ave_weight)

    def is_log_func(self):
        def update_log_threshold(baseline):
            while baseline >= self.log_threshold:
                if self.log_mode == 'normal':  # 等差存储
                    self.log_threshold += self.log_interval
                elif self.log_mode == 'exponential':  # 等比存储
                    self.log_threshold *= self.log_interval
                print('node_touched:', baseline)

        now_time = time.time() - self.start_time

        if self.node_touched > self.log_state_start_num:  # 不然太小了，记录的误差太大
            # 训练存储条件
            if self.log_interval_mode == 'itr':
                flag_num = self.itr_num
            elif self.log_interval_mode == 'train_time':
                flag_num = now_time
            elif self.log_interval_mode == 'node_touched':
                flag_num = self.node_touched
            else:
                flag_num = 0
            if flag_num >= self.log_threshold:
                self.log_model(self.itr_num, now_time)
                self.first_log = False
                update_log_threshold(flag_num)

    def is_train_end_func(self):
        if self.train_mode == 'fix_itr':
            if self.itr_num >= self.total_train_constraint:
                return True
        elif self.train_mode == 'fix_node_touched':
            if self.node_touched >= self.total_train_constraint:
                return True
        elif self.train_mode == 'fix_train_time':
            if time.time() - self.start_time >= self.total_train_constraint:
                return True
        return False

    def log_epsilon(self):
        print('start cal epsilon')
        epsilon_start_time = time.time()
        file_list = os.listdir(self.result_file_path)
        file_list.remove('tmp.csv')
        file_list = [int(i.split('.')[0]) for i in file_list]
        file_list.sort()
        file_list = [str(i) + '.npy' for i in file_list]
        epsilon_list = []
        for i_file in file_list:
            sim_ne_policy = np.load(self.result_file_path + "/" + i_file, allow_pickle=True).item()
            epsilon, game_value = self.get_epsilon(sim_ne_policy)
            epsilon_list.append(epsilon)

        with open(self.result_file_path + '/tmp.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            rows = list(reader)
        tmp_i = 0
        with open(self.result_file_path + '/epsilon.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(rows[0])
            for row in rows[1:]:
                row[2] = epsilon_list[tmp_i]
                tmp_i += 1
                writer.writerow(row)

        print('cal epsilon completed, cost time:', time.time() - epsilon_start_time)

    def train(self):

        os.makedirs(self.result_file_path)
        self.start_time = time.time()
        tmp_reward = 0
        while True:
            self.prepare_before_train(tmp_reward)
            tmp_reward = self.walk_tree('_', np.ones(self.game.player_num), 1.0)

            # 训练结束条件
            if self.is_train_end_func():
                self.log_model(self.itr_num, time.time() - self.start_time)
                self.save_model(self.itr_num)
                break

            self.is_log_func()

        print('train completed:')
        print('cost time:', time.time() - self.start_time)

        print(
            'itr:', self.itr_num,
            'info_num:', len(self.game.get_his_mean_policy()),
            'node_touched:', self.node_touched
        )

        if self.is_show_policy:
            print(len(self.game.get_his_mean_policy()))
            for i_keys in sorted(self.game.get_his_mean_policy().keys()):
                print(i_keys, ': ', self.game.get_his_mean_policy()[i_keys])
            print()
            print(self.game.imm_regret)
            self.all_state_regret_matching_strategy()
            for i_keys in sorted(self.game.imm_regret.keys()):
                print(i_keys, ': ', self.game.imm_regret[i_keys])
            print()
        self.start_time = time.time()
        self.log_epsilon()

        with open(self.result_file_path + '/weight_list.csv', 'a', newline='') as ff:
            writer = csv.writer(ff)
            writer.writerow(self.weight_list[1:])
