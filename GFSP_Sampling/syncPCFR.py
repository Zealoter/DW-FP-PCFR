

import numpy as np
from GFSP_Sampling.PCFR import PCFRSolver


class syncPCFRSolver(PCFRSolver):
    def __init__(self, config: dict):
        super().__init__(config)
        self.last_weight = 10000000000000000

    def all_state_regret_matching_strategy(self):
        self.last_weight = 10000000000000000
        for info in self.dynamic_opp_list:
            q_max = np.max(self.game.imm_regret[info])
            q_gap = q_max - self.game.imm_regret[info]
            q_chasing = self.dynamic_v[info] - self.dynamic_v[info][self.game.now_policy[info]]
            if np.max(q_chasing) == 0:
                continue
            chasing_ge_zero = np.where(q_chasing > 0.0)
            tmp_weight = q_gap[chasing_ge_zero] // q_chasing[chasing_ge_zero] + 1
            self.last_weight = min(np.min(tmp_weight), self.last_weight)

            if self.last_weight <= 1:
                self.last_weight = 1
                break

        for info in self.dynamic_now_list:
            self.game.w_his_policy[info][self.game.now_policy[info]] += self.last_weight

        for info in self.dynamic_opp_list:
            self.game.imm_regret[info] += self.last_weight * self.dynamic_v[info]
            self.update_now_policy_P(info)

        # tmp_list = self.dynamic_now_list + self.dynamic_opp_list

        self.dynamic_now_list = []
        self.dynamic_opp_list = []
        self.dynamic_v = {}


