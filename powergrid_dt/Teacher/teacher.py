import os
import time
import random
import grid2op
import numpy as np
import pandas as pd
from grid2op import Environment


class Teacher_One:
    def __init__(self, env, config) -> None:
        self.env = env
        self.action = self.env.action_space()
        self.save_path = config['save_path']
        self.config = config

    def topology_search(self, dst_step):
        obs = self.env.get_obs()
        min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
        print("step-%s, line-%s(from bus-%d to bus-%d) overflows, max rho is %.5f" %
            (dst_step, overflow_id, self.env.line_or_to_subid[overflow_id],
            self.env.line_ex_to_subid[overflow_id], obs.rho.max()))
        all_actions = self.env.action_space.get_all_unitary_topologies_change(self.env.action_space)
        action_chosen = self.env.action_space({})
        tick = time.time()
        for action in all_actions:
            if not self.env._game_rules(action, self.env):
                continue
            obs_, _, done, _ = obs.simulate(action)
            if (not done) and (obs_.rho.max() < min_rho):
                min_rho = obs_.rho.max()
                action_chosen = action
        print("find a greedy action and max rho decreases to %.5f, search duration: %.2f" %
            (min_rho, time.time() - tick))
        return action_chosen


    def save_sample(self, dst_step, line_to_disconnect, obs, action, obs_, save_path):
        if action == self.env.action_space({}):
            return None  # not necessary to save a "do nothing" action
        
        act_or, act_ex, act_gen, act_load = [], [], [], []

        for key, val in action.as_dict()['change_bus_vect'][
            action.as_dict()['change_bus_vect']['modif_subs_id'][0]].items():
            if val['type'] == 'line (extremity)':
                act_ex.append(key)
            elif val['type'] == 'line (origin)':
                act_or.append(key)
            elif val['type'] == 'load':
                act_load.append(key)
            else:
                act_gen.append(key)
        pd.concat(
            (
                pd.DataFrame(
                    np.array(
                        [self.env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                        self.env.line_or_to_subid[line_to_disconnect],
                        self.env.line_ex_to_subid[line_to_disconnect], str(np.where(obs.rho > 1)[0].tolist()),
                        str([i for i in np.around(obs.rho[np.where(obs.rho > 1)], 2)]),
                        action.as_dict()['change_bus_vect']['modif_subs_id'][0], act_or, act_ex, act_gen, act_load,
                        obs.rho.max(), obs.rho.argmax(), obs_.rho.max(), obs_.rho.argmax()]).reshape([1, -1])),
                pd.DataFrame(np.concatenate((obs.to_vect(), obs_.to_vect(), action.to_vect())).reshape([1, -1]))
            ),
            axis=1
        ).to_csv(os.path.join(save_path, 'Experiences1.csv'), index=0, header=0, mode='a')


    def run(self, line2attack:list, num_episode:int):
        for episode in range(num_episode):
        # traverse all attacks
            for line_to_disconnect in line2attack:
                self.env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
                # traverse all scenarios
                for chronic in range(len(os.listdir(self.env.chronics_handler.path))):
                    self.env.reset()
                    dst_step = episode * 72 + random.randint(0, 72)  # a random sampling every 6 hours
                    print('\n\n' + '*' * 50 + '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                        self.env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                        self.env.line_or_to_subid[line_to_disconnect], self.env.line_ex_to_subid[line_to_disconnect]))
                    # to the destination time-step
                    self.env.fast_forward_chronics(dst_step - 1)
                    obs, reward, done, _ = self.env.step(self.env.action_space({}))
                    if done:
                        break
                    # disconnect the targeted line
                    new_line_status_array = np.zeros(obs.rho.shape, dtype=np.int)
                    new_line_status_array[line_to_disconnect] = -1
                    action = self.env.action_space({"set_line_status": new_line_status_array})
                    obs, reward, done, _ = self.env.step(action)
                    if obs.rho.max() < 1:
                        # not necessary to do a dispatch
                        continue
                    else:
                        # search a greedy action
                        action = self.topology_search(dst_step)
                        obs_, reward, done, _ = self.env.step(action)
                        self.save_sample(dst_step, line_to_disconnect, obs, action, obs_, self.save_path)
