"""Load numeric NPZ models and recognize one WAV without retraining."""
import argparse
from pathlib import Path
import numpy as np
from hmmlearn.hmm import GaussianHMM
from lab3 import LABELS,features,load_audio,check_topology


def load_models(directory):
    models={}
    for label in LABELS:
        with np.load(Path(directory)/f'{label}.npz',allow_pickle=False) as data:
            m=GaussianHMM(n_components=len(data['startprob']),covariance_type='diag',init_params='')
            m.startprob_=data['startprob']; m.transmat_=data['transmat']; m.means_=data['means']
            m.covars_=np.diagonal(data['covars'],axis1=1,axis2=2).copy()
            m.n_features=m.means_.shape[1]
        check_topology(m); models[label]=m
    return models


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('wav'); p.add_argument('--models',default='results_demo/models')
    args=p.parse_args()
    F=features(load_audio(args.wav))
    scores={w:m.score(F) for w,m in load_models(args.models).items()}
    for w,s in sorted(scores.items(),key=lambda item:item[1],reverse=True): print(w,round(s,3))
    print('Prediction:',max(scores,key=scores.get))
