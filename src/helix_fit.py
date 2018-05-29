import numpy as np
from trackml.dataset import load_event
import pickle

from data_path import DATA_PATH


def get_helix_params(X):
    Z = np.zeros((X.shape[0], 10), np.float32)
    Z[:, 0] = X[:, 0] ** 2
    Z[:, 1] = 2 * X[:, 0] * X[:, 1]
    Z[:, 2] = 2 * X[:, 0] * X[:, 2]
    Z[:, 3] = 2 * X[:, 0]
    Z[:, 4] = X[:, 1] ** 2
    Z[:, 5] = 2 * X[:, 1] * X[:, 2]
    Z[:, 6] = 2 * X[:, 1]
    Z[:, 7] = X[:, 2] ** 2
    Z[:, 8] = 2 * X[:, 2]
    Z[:, 9] = 1
    v, s, t = np.linalg.svd(Z, full_matrices=True)
    smallest_index = np.argmin(np.array(s))
    T = np.array(t)
    T = T[smallest_index, :]
    S = np.zeros((4, 4), np.float32)
    S[0, 0] = T[0]
    S[0, 1] = S[1, 0] = T[1]
    S[0, 2] = S[2, 0] = T[2]
    S[0, 3] = S[3, 0] = T[3]
    S[1, 1] = T[4]
    S[1, 2] = S[2, 1] = T[5]
    S[1, 3] = S[3, 1] = T[6]
    S[2, 2] = T[7]
    S[2, 3] = S[3, 2] = T[8]
    S[3, 3] = T[9]
    return S


def check_helix_params(X, S):
    ones = np.reshape(np.transpose(np.repeat(1, len(X[:, 0]))), (len(X[:, 0]), 1))
    X_ap = np.append(X, ones, axis=1)
    for x_ap in X_ap:
        result = np.matmul(np.matmul(x_ap, S), x_ap)
        if result > 1e-3:
            return False
    return True


def prepare_train_data(evens_path):
    train_X_list = []
    train_Y_list = []
    hits, cells, particles, truth = load_event(evens_path)
    for idx, particle_row in particles.iterrows():
        if particle_row['nhits'] > 1:
            track = truth[truth['particle_id'] == particle_row['particle_id']]
            X = track[['tx', 'ty', 'tz']].as_matrix()
            S = get_helix_params(X)
            if check_helix_params(X, S):
                train_Y_list.append(np.concatenate([S[0], S[1, 0:3], S[2, 0:2], S[3, 0:1]]))
                train_X_list.append(np.concatenate([particle_row[['vx', 'vy', 'vz', 'px', 'py', 'pz', 'q']].as_matrix(),
                                                    particle_row[['vx', 'vy', 'vz', 'px', 'py', 'pz']].map(
                                                        np.square).as_matrix()]))
    return np.array(train_X_list, dtype=np.float64), np.array(train_Y_list, dtype=np.float64)


def main():
    EVENT_TITLE = '/train_100_events/event000001000'
    train_X_array, train_Y_array = prepare_train_data(DATA_PATH + EVENT_TITLE)
    pickle.dump(train_X_array, open(DATA_PATH + '/train_X.pkl', 'wb'))
    pickle.dump(train_Y_array, open(DATA_PATH + '/train_Y.pkl', 'wb'))


if __name__ == '__main__':
    main()
