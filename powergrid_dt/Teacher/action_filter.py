import os
import numpy as np
import pandas as pd


def read_data(tabs):
    data = None
    for tab in tabs:
        if data is None:
            data = pd.read_csv(tab, header=None)
        else:
            data = pd.concat((data, pd.read_csv(tab, header=None)))
    return data


def filter_data(data):
    data['del'] = False
    for row in range(len(data) - 1):
        if data.iloc[row, 12] - data.iloc[row, 14] < 0.02:
            # this action decreases rho less than 2%
            data.iloc[row, -1] = True
        if data.iloc[row, 7] == 'None':
            # this action is "do nothing"
            data.iloc[row, -1] = True
        
        # 3) NEW rule: high post-rho (unsafe even after action)
        #     if rho_after > 1.2, mark this sample for deletion
        if data.iloc[row, 14] > 1.2:
            # after applying this action, grid is still too stressed
            data.iloc[row, -1] = True
    return data


def save_action_space(data, save_path, threshold=0):
    actions = data.iloc[:, -495:-1].astype(np.int16)
    actions['action_list'] = actions.apply(lambda s: str([i for i in s.values]), axis=1)
    nums = pd.value_counts(actions['action_list'])
    # filter out actions that occur less frequently
    # the actions that occur times less than the threshold would be filtered out
    action_space = np.array([eval(item) for item in nums[nums >= threshold].index.values])
    file = os.path.join(save_path, 'actions%d.npy' % action_space.shape[0])
    np.save(file, action_space)
    print('generate an action space with the size of %d' % action_space.shape[0])


def run(tables, save_path, threshold=0):
    data = read_data(tables)
    data = filter_data(data)
    save_action_space(data, save_path, threshold)


# if __name__ == "__main__":
#     # hyper-parameters
#     TABLES = ["./Experiences1.csv", "./Experiences2.csv"]  # Use your own data
#     SAVE_PATH = "../ActionSpace"
#     THRESHOLD = 1

#     data = read_data(TABLES)
#     data = filter_data(data)
#     save_action_space(data, SAVE_PATH, THRESHOLD)