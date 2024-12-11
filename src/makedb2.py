import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import pickle
from functools import partial

def clean_vehicle_ids(df, id0=0):
    dfres = df.copy()
    for vid in set(df['vehicle_id']):
        dfres.loc[dfres['vehicle_id'] == vid, 'vehicle_id'] = id0
        id0 += 1
    return dfres


def extract_step(df, step):
    dfs = df[df['time'] == step]
    return dfs


def window_vehicle(dfs, vehicle_id, width_forward, width_back, max_pos = 499, x4test=None):
    vrow = dfs[dfs['vehicle_id'] == vehicle_id]
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
        vel = int(row['velocity'])
        changed_line = int(row['changed_line'])  
        stp = row['type']
        tp = {'AV': 1, 'HDV': 2, 'HDV_DEF':3}[stp]
        xi = int(row['pos_x'])
        yi = int(row['pos_y'])
        # print(xi, yi, vel)
        res[0, yi, xi] = tp
        res[1, yi, xi] = (vel + 1) * (-1 if changed_line else 1)
    return res


def one_file2dict(tp):
    fname, width_forward, width_back = tp
    df = pd.read_csv(fname)
    df = clean_vehicle_ids(df)
    dfss = [extract_step(df, step) for step in df['time'].unique()]
    res = {
        'AV': [], 'HDV': [], 'HDV_DEF': []
    }

    for vid in df['vehicle_id'].unique():
        winds = []
        ys = []
        for dfs in dfss:
            r = window_vehicle(dfs, vid, width_forward, width_back)
            if r is None:
                break
            zzz = dfs
            df2, x, y = r
            npw = npwindow(df2, width_forward + width_back + 1)
            winds.append(npw)
            ys.append(y)
        else:
            vrow = zzz[zzz['vehicle_id'] == vid]
            tp = vrow['type'].values[0]
            winds = np.array(winds, dtype=npw.dtype)
            ys = np.array(ys, dtype=npw.dtype)
            res[tp].append((winds, ys))
    res['meta'] = extract_meta(fname)
    res['fname'] = fname
    return res

def extract_meta(fname):
    base = os.path.basename(fname)
    pure = base.split('.')[0]
    lst = pure.split('_')
    cars = int(lst[lst.index('cars') + 1])
    hdv = float(lst[lst.index('hdv') + 1])
    av = float(lst[lst.index('av') + 1])
    hdv_def = float(lst[lst.index('def') + 1])
    return cars, hdv, av, hdv_def


def main():
    all_csvs = glob.glob(os.path.join(os.path.dirname(__file__), 'data', 's_nfs_model_for_ml', '*.csv'))
    res1 = one_file2dict(all_csvs[0])
    print(len(res1['AV']))
    print(len(res1['HDV']))
    print(len(res1['HDV_DEF']))

def main1():
    all_csvs = glob.glob(os.path.join(os.path.dirname(__file__), 'data', 's_nfs_model_for_ml', '*.csv'))
    megadb = {
        'AV': [], 'HDV': [], 'HDV_DEF': []
    }
    width_forward= 13
    width_back=13
    tps = [(fname, width_forward, width_back) for fname in all_csvs]
    with multiprocessing.Pool(32) as p:
        for res in tqdm(p.imap(one_file2dict, tps), total=len(tps)):
            for key in megadb:
                megadb[key].extend(res[key])
    
    megares_fname = os.path.join(os.path.dirname(__file__), 'data', f'megadb_{width_forward}_{width_back}.bin')
    with open(megares_fname, 'wb') as f:
        pickle.dump(megadb, f)

def main2():
    all_csvs = glob.glob(os.path.join(os.path.dirname(__file__), 'data', 's_nfs_model_for_ml', '*.csv'))
    width_forward= 13
    width_back=13
    tps = [(fname, width_forward, width_back) for fname in all_csvs]
    with multiprocessing.Pool(32) as p:
        for res in tqdm(p.imap(one_file2dict, tps), total=len(tps)):
            fname = res['fname']
            basename = os.path.basename(fname)
            pure = basename.split('.')[0]
            with open(os.path.join(os.path.dirname(__file__), 'data', 'bins', pure + '.bin'), 'wb') as f:
                pickle.dump(res, f)



if __name__ == '__main__':
    main1()