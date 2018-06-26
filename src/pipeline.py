from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from numba import jit
from trackml.dataset import load_event
from trackml.score import score_event

from data_path import DATA_PATH

EVENTS = ['/train_100_events/event00000' + str(1000 + i) for i in range(0, 100)]


@jit
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
        if result > 1e-6:
            return False
    return True


def prepare_train_data(truth, particles):
    train_X_list = []
    train_Y_list = []
    for idx, particle_row in particles.iterrows():
        if particle_row['nhits'] > 1:
            track = truth[truth['particle_id'] == particle_row['particle_id']]
            X = track[['tx', 'ty', 'tz']].values
            S = get_helix_params(X)
            if check_helix_params(X, S):
                train_Y_list.append(np.concatenate([S[0], S[1, 0:3], S[2, 0:2], S[3, 0:1]]))
                train_X_list.append(np.concatenate([particle_row[['vx', 'vy', 'vz', 'px', 'py', 'pz', 'q']].values,
                                                    particle_row[['vx', 'vy', 'vz', 'px', 'py', 'pz']].map(
                                                        np.square).values,
                                                    np.array([particle_row['vx'] * particle_row['vy'],
                                                              particle_row['vx'] * particle_row['vz'],
                                                              particle_row['vy'] * particle_row['vz'],
                                                              particle_row['px'] * particle_row['py'],
                                                              particle_row['px'] * particle_row['pz'],
                                                              particle_row['py'] * particle_row['pz']])]))
    return np.array(train_X_list, dtype=np.float64), np.array(train_Y_list, dtype=np.float64)


def prepare_test_data(particles):
    test_X_list = []
    for idx, particle_row in particles.iterrows():
        test_X_list.append(np.concatenate([
            particle_row[['particle_id']],
            particle_row[['vx', 'vy', 'vz', 'px', 'py', 'pz', 'q']].values,
            particle_row[['vx', 'vy', 'vz', 'px', 'py', 'pz']].map(
                np.square).values,
            np.array([particle_row['vx'] * particle_row['vy'],
                      particle_row['vx'] * particle_row['vz'],
                      particle_row['vy'] * particle_row['vz'],
                      particle_row['px'] * particle_row['py'],
                      particle_row['px'] * particle_row['pz'],
                      particle_row['py'] * particle_row['pz']])]))
    return np.array(test_X_list, dtype=np.float64)


@jit
def find_helix(hit, helixes):
    S = np.zeros((4, 4), np.float64)
    for helix in helixes:
        S[0, 0] = helix[1]
        S[0, 1] = S[1, 0] = helix[2]
        S[0, 2] = S[2, 0] = helix[3]
        S[0, 3] = S[3, 0] = helix[4]
        S[1, 1] = helix[5]
        S[1, 2] = S[2, 1] = helix[6]
        S[1, 3] = S[3, 1] = helix[7]
        S[2, 2] = helix[8]
        S[3, 2] = S[2, 3] = helix[9]
        S[3, 3] = helix[10]
        temp = np.matmul(np.matmul(hit, S), hit)
        if temp < 1e-6:
            return int(helix[0])


for i in range(0, 1):
    EVENTS_TRAIN = ['/train_100_events/event000001001']
    EVENTS_TEST = ['/train_100_events/event000001002']
    #    EVENTS_TEST = EVENTS[(i * 20):(20 * (i + 1))]
    #    EVENTS_TRAIN = EVENTS[0:(i * 20)] + EVENTS[(20 * (i + 1)):100]
    X_train = None
    y_train = None
    for EVENT in EVENTS_TRAIN:
        hits, cells, particles, truth = load_event(DATA_PATH + EVENT)
        X_event, y_event = prepare_train_data(truth, particles)
        if (X_train is None and y_train is None):
            X_train = X_event
            y_train = y_event
        else:
            X_train = np.vstack((X_train, X_event))
            y_train = np.vstack((y_train, y_event))
    regressors = []
    for i in range(0, y_train.shape[1]):
        regressor = xgb.XGBRegressor()
        regressor = regressor.fit(X=X_train, y=y_train[:, i])
        regressors.append(regressor)
    for EVENT in EVENTS_TEST:
        hits, cells, particles, truth = load_event(DATA_PATH + EVENT)
        X_predict = prepare_test_data(particles)
        predicts = [X_predict[:, 0]]
        for regressor in regressors:
            predicts.append(regressor.predict(data=X_predict[:, 1:]))
        predicted_helix_params = np.transpose(np.vstack(tuple(predicts)))
        predicted_tracks = defaultdict(list)
        hits.insert(loc=4, column='1', value=1)
        for hit in hits.values:
            predicted_tracks[find_helix(hit[1:5], X_predict)].append(hit[0])
        predicted_tracks_dataframe = {'track_id': [], 'hit_id': []}
        for id, track in predicted_tracks.items():
            if not track:
                del predicted_tracks[id]
            else:
                predicted_tracks_dataframe['track_id'] += [id] * len(track)
                predicted_tracks_dataframe['hit_id'] += track
        print(score_event(truth, pd.DataFrame.from_dict(predicted_tracks_dataframe)))
