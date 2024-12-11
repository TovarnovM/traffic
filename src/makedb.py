import pandas as pd
import numpy as np
import os
import pickle
import multiprocessing
from tqdm import tqdm
from functools import partial


def extract_step(df, step):
    dfs = df[df['time'] == step]
    return dfs


def window_vehicle(dfs, vehicle_id, width_forward, width_back, max_pos = 999, x4test=None):
    inds = dfs['vehicle_id'] == vehicle_id
    if np.any(inds) == False:
        return None
    vrow = dfs[inds]
    x = vrow['pos_x'].values[0]
    y = vrow['pos_y'].values[0]
    if x4test is not None:
        x = x4test
    x1 = x - width_back
    x2 = x + width_forward 
    inds = np.logical_and(dfs['pos_x'] >= x1, dfs['pos_x'] < x2)

    if x1 < 0:
        inds = np.logical_or(inds, dfs['pos_x'] > max_pos + x1 )
    if x2 > max_pos:
        inds = np.logical_or(inds, dfs['pos_x'] < (x2 - max_pos) )
    
    df_window = dfs[inds].copy()

    if x1 < 0:
        df_window.loc[df_window['pos_x'] > max_pos + x1, 'pos_x'] -= max_pos+1
    if x2 > max_pos:
        df_window.loc[df_window['pos_x'] < (x2 - max_pos), 'pos_x'] += max_pos+1
    df_window.loc[:, 'pos_x'] -= x -  width_back
    
    return df_window, x, y


def npwindow(dfw, width):
    res = np.zeros((2, 2, width), dtype=np.int8)
    # [тип, скорость]
    for i, (rowi, row) in enumerate(dfw.iterrows()):
        vel = int(row['Velocity'])
        changed_line = int(row['changed_line'])  
        tp = int(row['type'])
        xi = int(row['pos_x'])
        yi = int(row['pos_y'])
        # print(xi, yi, vel)
        res[0, yi, xi] = tp
        res[1, yi, xi] = (vel + 1) * (-1 if changed_line else 1)
    return res


def foo(i):
    return i*2



def main():
    
    fpath = os.path.join(os.path.dirname(__file__), 'data', 'Results_NS_anomaly.csv')
    dbname = os.path.basename(fpath).split('.')[0]
    fname_dffs = os.path.join(os.path.dirname(__file__), 'data', f'dffs_{dbname}.bin')

    print('open csv...', end='')
    df = pd.read_csv(fpath)
    print('done')

    print('prepropess...', end='')
    df = df.fillna(0.0)
    ids = sorted(set(df['vehicle_id'].to_numpy()))
    steps = sorted(set(df['time'].to_numpy()))
    print('done')


    if os.path.exists(fname_dffs):
        print(f'read {fname_dffs}...', end='')
        with open(fname_dffs, 'rb') as f:
            dfss = pickle.load(f)
        print('done')
    else:
        
        print('create windows...', end='')
        dfss = [extract_step(df, step) for step in tqdm(steps)]
        
        print(f'dsve {fname_dffs}...', end='')
        with open(fname_dffs, 'wb') as f:
            pickle.dump(dfss, f)
        print('done')
   
    width_forward, width_back = 21, 13
    megadb = {
        'width_forward': width_forward, 
        'width_back': width_back,
        'vids': {}
    }

    with multiprocessing.Pool(32) as p:
        tps = [(fname_dffs, vid, width_forward, width_back) for vid in ids]
        for key, d in tqdm(p.imap(make_one_window, tps), total=len(tps)):
            megadb['vids'][key] = d
    
    megares_fname = os.path.join(os.path.dirname(__file__), 'data', f'megadb_{width_forward}_{width_back}_{dbname}.bin')
    with open(megares_fname, 'wb') as f:
        pickle.dump(megadb, f)


def make_one_window(tp):
    dffs_fname, vid, width_forward, width_back = tp
    with open(dffs_fname, 'rb') as f:
        dfss = pickle.load(f)
    
    winds = []
    ys = []
    for dfs in dfss:
        r = window_vehicle(dfs, vid, width_forward, width_back)
        if r is None:
            continue
        zzz = dfs
        df2, x, y = r
        npw = npwindow(df2, width_forward + width_back + 1)
        winds.append(npw)
        ys.append(y)
    vrow = zzz[zzz['vehicle_id'] == vid]
    tp = vrow['type'].values[0]
    
    winds = np.array(winds, dtype=npw.dtype)
    ys = np.array(ys, dtype=npw.dtype)
    key = f'{vid}'
    return key, {
        'vtype': tp,
        'winds': winds,
        'ys': ys
    }

if __name__ == "__main__":
    main()