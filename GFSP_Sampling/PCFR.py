
import numpy as np
from GFSP_Sampling.GFSP import GFSPSamplingSolver


class PCFRSolver(GFSPSamplingSolver):
    def __init__(self, config: dict):
        super().__init__(config)
        self.dynamic_now_list = []
        self.dynamic_opp_list = []
        self.dynamic_v = {}
        self.game.game_train_mode = 'PCFR'

    def walk_tree(self, his_feat, player_pi, pi_c):
        if self.game.game_train_mode == 'vanilla':
            return self.vanilla_walk_tree(his_feat, player_pi, pi_c)
        elif self.game.game_train_mode == 'PCFR':
            return self.PCFR_walk_tree(his_feat, player_pi, pi_c)
        return 0

    def PCFR_walk_tree(self, his_feat, player_pi, pi_c):
        self.node_touched += 1
        if np.sum(player_pi) == 0.0 or pi_c == 0:
            return np.zeros(self.game.player_num)

        now_player = self.game.get_now_player_from_his_feat(his_feat)
        if now_player == 'c':
            r = np.zeros(self.game.player_num)
            now_prob = self.game.get_chance_prob(his_feat)
            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
            for a_i in range(len(now_action_list)):
                tmp_r = self.PCFR_walk_tree(
                    self.game.get_next_his_feat(his_feat, now_action_list[a_i]),
                    player_pi,
                    pi_c * now_prob[a_i]
                )
                r += tmp_r
        else:
            now_action_list = self.game.get_legal_action_list_from_his_feat(his_feat)
            if len(now_action_list) == 0:
                tmp_reward = self.game.judge(his_feat)
                tmp_reward[0] *= player_pi[1]
                tmp_reward[1] *= player_pi[0]
                return tmp_reward * pi_c

            r = np.zeros(self.game.player_num)
            v = np.zeros(len(now_action_list))

            now_player_index = self.game.player_set.index(now_player)
            tmp_info = self.game.get_info_set(now_player, his_feat)
            now_prob = player_pi[now_player_index]
            opp_prob = player_pi[1 - now_player_index]
            if now_prob == 1:
                if tmp_info not in self.dynamic_now_list:
                    self.dynamic_now_list.append(tmp_info)

            if opp_prob == 1:
                if tmp_info not in self.dynamic_opp_list:
                    self.dynamic_opp_list.append(tmp_info)

            if tmp_info not in self.game.imm_regret.keys():
                self.game.generate_new_info_set(tmp_info, now_player, len(now_action_list))

            if opp_prob == 0:
                a_i = self.game.now_policy[tmp_info]
                r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                r[now_player_index] = 0

            elif now_prob == 0:
                for a_i in range(len(now_action_list)):
                    tmp_r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], player_pi, pi_c)
                    v[a_i] = tmp_r[now_player_index]
                    if self.game.now_policy[tmp_info] == a_i:
                        r[now_player_index] = tmp_r[now_player_index]

                if tmp_info in self.dynamic_v.keys():
                    self.dynamic_v[tmp_info] += v
                else:
                    self.dynamic_v[tmp_info] = v

            else:
                for a_i in range(len(now_action_list)):
                    if self.game.now_policy[tmp_info] == a_i:
                        prob = 1.0
                    else:
                        prob = 0.0

                    tmp_player_pi = np.ones(self.game.player_num)
                    tmp_player_pi[now_player_index] = prob

                    tmp_r = self.PCFR_walk_tree(his_feat + now_action_list[a_i], tmp_player_pi, pi_c)

                    v[a_i] = tmp_r[now_player_index]
                    r = r + tmp_r
                    if prob == 1:
                        pass
                    else:
                        r[now_player_index] = r[now_player_index] - tmp_r[now_player_index]

                if tmp_info in self.dynamic_v.keys():
                    self.dynamic_v[tmp_info] += v
                else:
                    self.dynamic_v[tmp_info] = v
        return r

    def update_now_policy_P(self, info):
        # tmp_pure_policy = np.random.randint(len(self.game.imm_regret[info]))
        self.game.now_policy[info] = np.argmax(self.game.imm_regret[info])
        # if self.game.imm_regret[info][self.game.now_policy[info]] == self.game.imm_regret[info][tmp_pure_policy]:
        #     self.game.now_policy[info] = tmp_pure_policy

    def all_state_regret_matching_strategy(self):
        for info in self.dynamic_now_list:
            self.game.w_his_policy[info][self.game.now_policy[info]] += 1

        for info in self.dynamic_opp_list:
            self.game.imm_regret[info] += self.dynamic_v[info]
            self.update_now_policy_P(info)

        self.dynamic_now_list = []
        self.dynamic_opp_list = []
        self.dynamic_v = {}

