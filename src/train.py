import argparse
import os
from pprint import pprint
import sys
import csv
import glob
import time
import optuna
from tqdm import tqdm
import pandas as pd
import pickle


from torch.utils.data import Dataset, DataLoader, random_split
import torch as T
import torch.nn as nn
import pytorch_lightning as pl 
import torchmetrics

import numpy as np

app_folder = os.path.dirname(os.path.abspath(__file__))


DB_FILE_PATH = os.path.join(app_folder, 'data', 'megadb_13_13.bin')
assert os.path.exists(DB_FILE_PATH)

import torch.nn.functional as F
from sklearn import model_selection
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from argparse import Namespace
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy


from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

T.set_float32_matmul_precision('medium')

@jit
def get_ind(xs_shape0, i_start, seq_len):
    i = i_start
    forslice = np.arange(i_start, i_start + xs_shape0)
    inds = np.zeros((xs_shape0-seq_len+1, seq_len), dtype=np.int32)
    for i in range(0, xs_shape0-seq_len+1):
        inds[i] = forslice[i:i+seq_len]
    return inds

def warm_jits():
    xs = np.array([1,2,3,4,5,6])
    ys = np.array([1,2,3,4,5,6])+10

    ind = get_ind(xs.shape[0], 0, 3)

class DatasetVehicleSeq(Dataset):
    def __init__(self, seq_len, db_path=None, assert_ys=False, allow_cache=True):
        super().__init__()
        self.Xs = None
        self.Ys = None
        self.inds = None
        self.seq_len = seq_len

        if db_path is None:
            db_path = DB_FILE_PATH
        
        folder = os.path.dirname(db_path)
        fname = os.path.basename(db_path).split('.')[0]
        cached_name = os.path.join(folder, f'{fname}_seqlen_{seq_len}.bin')

        if allow_cache and os.path.exists(cached_name):
            self._load_from_cached(cached_name)
        else:
            Xs, Ys, inds = self._load_from_db(db_path, seq_len, assert_ys=assert_ys)
            self.Xs = T.tensor(Xs, dtype=T.float32)
            self.Ys = T.tensor(Ys, dtype=T.long)
            self.inds = T.tensor(inds)            
            if allow_cache:
                with open(cached_name, 'wb') as f:
                    pickle.dump((Xs, Ys, inds), f) 


    def _load_from_cached(self, cached_name):
        with open(cached_name, 'rb') as f:
            Xs, Ys, inds = pickle.load(f)
        self.Xs = T.tensor(Xs, dtype=T.float32)
        self.Ys = T.tensor(Ys, dtype=T.long)
        self.inds = T.tensor(inds)  


    def _load_from_db(self, db_path, seq_len, assert_ys=False, allow_cache=True):
        with open(db_path, 'rb') as f:
            megadb = pickle.load(f)
        inds = []
        Xs = []
        Ys = []
        ni = 0
        shapex = None
        for k in {'AV', 'HDV', 'HDV_DEF'}:
            episodes = megadb[k]
            tp = {'AV': 0, 'HDV': 1, 'HDV_DEF':2}[k]
            for xs, ys in episodes:
                ind = get_ind(xs.shape[0], ni, seq_len)
                inds.append(ind)
                # берем только картинку-матрицу, где хранятся скорости
                xxs = xs[:, 1, :, :]
                shapex = xxs.shape
                # делаем из матрицы вектор и добавляем в начало координату Y агента (чтобы НС знала, где находится ТС, который мы отслеживаем) 
                row = np.hstack([ys.reshape(-1, 1), xxs.reshape(shapex[0], -1)])
                Xs.append(row)

                # Xs и Ys должны иметь одинаковую длину
                Ys.append(np.repeat(tp, ys.shape[0]))
                ni += xs.shape[0]
                if assert_ys:
                    assert xs[0, 0, :, :][ys[0], xs.shape[-1]//2] - 1 == tp, 'Какая-то шляпа с разметкой БД'
        return np.concatenate(Xs), np.concatenate(Ys), np.vstack(inds)
        # self.Xs = T.tensor(np.concatenate(Xs), dtype=T.float32)
        # self.Ys = T.tensor(np.concatenate(Ys), dtype=T.long)
        # self.inds = T.tensor(np.vstack(inds))
        # self.seq_len = seq_len

    def __getitem__(self, index):
        index2 = self.inds[index]
        return self.Xs[index2], self.Ys[index2[-1]]
    
    
    def __len__(self):
        return self.inds.shape[0]
    



class DataModuleSeq(pl.LightningDataModule):
    def __init__(self, batch_size, seq_len,num_workers, pin_mem=True, **kw) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_mem = pin_mem
        self.seq_len = seq_len
        self.ds_train, self.ds_val = None, None


    def setup(self, stage=None):
        data_all = DatasetVehicleSeq(self.seq_len)
        self.ds_train, self.ds_val = random_split(
            data_all, 
            (0.9, 0.1), 
            generator=T.Generator().manual_seed(42))


    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_mem,
            persistent_workers=self.num_workers>0
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size if self.batch_size < len(self.ds_val) else len(self.ds_val),
            num_workers=2,
            shuffle=False,
            pin_memory=self.pin_mem,
            persistent_workers=self.num_workers>0
        )   



class NN_lightning(pl.LightningModule):  
    def __init__(self, 
                 n_features, 
                 fc_size,
                 fc_layers,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 activation_fn_name='relu',
                 n_target=3, 
                 **kw):
        super(NN_lightning, self).__init__()
        self.n_features = n_features
        self.n_target = n_target
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.fc_size = fc_size

        lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)

        self.n_features = n_features
        self.n_target = n_target

        self.fc_layers = fc_layers

        self.learning_rate = learning_rate
        self.fc_size = fc_size
        self.lstm = lstm
        lrs = []
        inp_n = hidden_size
        outp_n = self.fc_size
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]
        for i in range(self.fc_layers-1):
            lrs.append(nn.Linear(inp_n, outp_n))
            lrs.append(activation_fn())

            inp_n = outp_n
        lrs.append(nn.Linear(inp_n, n_target))

        self.seq = nn.Sequential(*lrs)
        self.loss_fn = T.nn.NLLLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_target)
        self.t0 = 0
        self.t1 = 1
        self.trial = kw.get('trial', None)
        
        
    def forward(self, data):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(data)
        y_pred = self.seq(lstm_out[:,-1])
        return nn.functional.log_softmax( y_pred, dim=1 )
    
    
    def configure_optimizers(self):
        optimizer = T.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"}]
        

    
    
    def _common_step(self, batch, batch_idx):
        data, Y = batch
        y_pred = self.forward(data)
        loss = self.loss_fn(y_pred, Y)
        return loss, y_pred, Y
    
    
    def on_train_epoch_start(self) -> None:
        self.t0 = time.time_ns()
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.log('step', self.trainer.current_epoch)
        self.log_dict(
            {
                "train/loss": np.mean(loss.detach().cpu().numpy()),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch)
        )
        return {"loss": loss }
    
    
    def on_train_epoch_end(self):
        # outputs = self.training_step_outputs
        # self.on_epoch_end_template(outputs, "train")
        self.training_step_outputs.clear() 
        self.t1 = time.time_ns()
        self.log('step', self.trainer.current_epoch, sync_dist=True)
        sec_epoch = (self.t1 - self.t0)/1e9
        self.log('sec_epoch', sec_epoch, sync_dist=True)
        if self.trial is not None:
            self.trial.set_user_attr("sec_epoch", sec_epoch)
            

    def on_epoch_end_template(self, outputs, stage, on_step=False, on_epoch=True, prog_bar=False):
        accs = np.array(outputs["accs"])
        dy = np.concatenate(outputs["dy"])
        loss = np.array(outputs["loss"])
        d = {
            f"{stage}/loss": np.mean(loss),
            f"{stage}/acc": np.mean(accs),
            f"{stage}/acc_std": np.std(accs),
            f"{stage}/dy_max": np.max(np.abs(dy))
        }

        self.log('step', self.trainer.current_epoch, sync_dist=True)
        self.log_dict(d,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            sync_dist=True
        )
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.trial is not None:
            for k, v in d.items():
                self.trial.set_user_attr(k, float(v))
        return d
        
    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = {
            "loss": [], 
            "accs": [], 
            "dy": []
        }

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accs = self.accuracy(y_pred, y)
        lossnp = loss.detach().cpu().numpy()
        self.log('step', self.trainer.current_epoch, sync_dist=True)
        self.log("val/loss", np.mean(lossnp), prog_bar=True, sync_dist=True)

        self.validation_step_outputs['loss'].append(lossnp)
        self.validation_step_outputs['accs'].append(accs.detach().cpu().numpy())
        yn = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        self.validation_step_outputs['dy'].append(np.array(np.argmax(y_pred, axis=1) == yn, dtype=np.float32))
        return loss
    
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self.stats = self.on_epoch_end_template(outputs, "val")

    

def train1(**kw2update):
    kw = dict(
        n_features=55, 
        fc_size=256,
        fc_layers=2,
        hidden_size=128,
        seq_len=32,
        batch_size=2048,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.003,
        num_workers=8,

        # pin_mem=False
    )

    kw.update(kw2update)
    
    pprint(kw)
    ds = DatasetVehicleSeq(kw['seq_len'], assert_ys=True)
    del ds
    lstm_nn = NN_lightning(**kw)

    # NN_lightning.load_from_checkpoint()

    dm = DataModuleSeq(**kw)
    # fn = 'best_nn_maddness.bin'
    # fn = os.path.join(app_folder, fn)
    # with open(fn, 'rb') as f:
    #     state_dict = T.load(f)['state_dict']
    # lstm_nn.load_state_dict(state_dict)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(os.path.dirname(__file__), 'lightning_logs_new'), name=f'seq_len_{kw["seq_len"]}')
    early_stop_callback =  EarlyStopping(monitor="val/loss", min_delta=0.00, patience=7, verbose=False, mode="min")

    if kw2update.get('lrfind', False):
        tr = pl.Trainer(    
        # fast_dev_run=True,
        devices=[0],
        # strategy=strategy,
        # overfit_batches=1,
        max_epochs=310,
        logger=tb_logger,
        log_every_n_steps=8,
        
        enable_progress_bar=True,
        callbacks=[early_stop_callback]
    
        )
        try:
            tuner = Tuner(tr)
            res = tuner.lr_find(lstm_nn, dm)
            print(res)
            print(res.suggestion())
            print(lstm_nn.learning_rate)
            # fig = res.plot(show=True, suggest=True)
            # fig.show()
            return lstm_nn.learning_rate
        except Exception as e:
            print(e)
        
    strategy = DDPStrategy(process_group_backend="gloo")    
    tr = pl.Trainer(    
        # fast_dev_run=True,
        devices=[0, 1],
        strategy=strategy,
        # overfit_batches=1,
        max_epochs=120,
        logger=tb_logger,
        log_every_n_steps=8,
        
        enable_progress_bar=True,
        callbacks=[early_stop_callback]
    
    )
    
    tr.fit(lstm_nn, dm)#, ckpt_path=r'Z:\stoh_env\stohastic_env\lightning_logs\version_21\checkpoints\epoch=28-step=1081439.ckpt')


def _test_dataset():

    import time
    warm_jits()


    print('start')
    t1 = time.time()
    ds = DatasetVehicleSeq(128, assert_ys=False)
    print(len(ds))
    print(time.time() - t1)
    x, y = ds[999]
    print(x.shape, y.shape)    


if __name__ == '__main__':
    for seq_len in [5]:
        #lr = train1(seq_len=seq_len, lrfind=True)
        train1(seq_len=seq_len, learning_rate=0.01)
    # _test_dataset()


# if __name__ == '__main__':
#     import time
#     warm_jits()


#     print('start')
#     t1 = time.time()
#     ds = DatasetVehicleSeq(128)
#     print(len(ds))
#     print(time.time() - t1)
#     x, y = ds[999]
#     print(x.shape, y.shape)



# def objective(trial: optuna.Trial, gpu) -> float:
#     kw = dict(
#         n_features=8, 
#         n_target=14,
#         batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
#         fc_size=trial.suggest_int("fc_size", 32, 128, step=32),
#         fc_layers=trial.suggest_int("fc_layers", 1, 3),
#         activation_fn_name=trial.suggest_categorical("activation_fn_name", ["relu", "leaky_relu", "elu"]),
#         learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),

#         num_workers=4,
#         trial=trial
#     )
#     host = os.environ.get('ENV_HOST_NAME', 'unknown')
#     trial.set_user_attr("host", host)
#     lstm_nn = NN_lightning(**kw)
#     dm = BezDataModule(**kw)
#     early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=3, verbose=False, mode="min")
#     tr = pl.Trainer(    
#         devices=[gpu],
#         # fast_dev_run=True,
#         # overfit_batches=0.01,
#         max_epochs=200  ,
#         logger=False,
#         enable_progress_bar=False,
#         callbacks=[early_stop_callback, PyTorchLightningPruningCallback(trial, monitor="val/R2")]

#     )
#     tr.fit(lstm_nn, dm)#, ckpt_path=r'Z:\stoh_env\stohastic_env\lightning_logs\version_21\checkpoints\epoch=28-step=1081439.ckpt')
#     res = tr.test(lstm_nn, dm, verbose=True)[0]
#     return res["test/R2"]
    

# def optimize(gpu=0):
#     mysql_test(raise_=True, create_db_if_not_exists=True)
#     pruner1 = optuna.pruners.MedianPruner(n_warmup_steps=50, n_min_trials=4, n_startup_trials=20)
#     pruner = optuna.pruners.PatientPruner(pruner1, patience=30)
    
#     study_name = os.environ.get('OPTUNA_STUDY_NAME', 'inercia')
#     n_trials = int(os.environ.get('OPTUNA_N_TRIALS', 100))

#     study = optuna.create_study(
#         direction="maximize", 
#         pruner=pruner, 
#         storage=MYSQL_STORAGE_NAME, 
#         study_name=study_name, 
#         load_if_exists=True, 
#         sampler=optuna.samplers.TPESampler())
#     try:
#         study.optimize(lambda trial: objective(trial, gpu), n_trials=n_trials)
#     except KeyboardInterrupt:
#         pass








# def _tst1():
#     ds = DatasetBezier()
#     print(ds[0][0].shape)
#     print(ds[0][1].shape)


#     print(len(ds))

# if __name__ == '__main__':
#     T.set_float32_matmul_precision('high')
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--gpu", type=int, default=0)
#     # hparams = parser.parse_args()
#     # optimize(gpu=hparams.gpu)
#     train1()