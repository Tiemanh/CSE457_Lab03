"""Lab 3: reproducible whole-word Gaussian HMM experiments.

All audio in dataset_demo is SYNTHETIC. Replace --data for human recordings.
No test-driven model selection; 5-state/39D is fixed before test evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from hmmlearn.hmm import GaussianHMM
from numba import njit
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits

LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']
SR, HOP = 16000, 160
SEED = 457


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_audio(path):
    y, sr = sf.read(path, dtype='float64', always_2d=True)
    y = y.mean(axis=1)
    if len(y) == 0 or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-8:
        raise ValueError(f'Empty/silent/nonfinite waveform: {path}')
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def features(y, dim=39, return_parts=False):
    y = np.asarray(y, dtype=float)
    if dim not in (13, 39):
        raise ValueError('dim must be 13 or 39')
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-8:
        raise ValueError('Audio must be a finite, non-silent mono signal')
    trimmed, bounds = librosa.effects.trim(y, top_db=30, frame_length=400, hop_length=HOP)
    if len(trimmed) < 512:
        raise ValueError('Utterance too short after endpoint detection')
    emphasized = librosa.effects.preemphasis(trimmed, coef=0.97)
    mfcc = librosa.feature.mfcc(y=emphasized, sr=SR, n_mfcc=13,
                               n_fft=512, win_length=400, hop_length=HOP,
                               window='hamming', n_mels=26, fmin=20, fmax=SR/2)
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1]-1)
    if width < 3:
        raise ValueError('At least 3 frames needed for delta features')
    d1 = librosa.feature.delta(mfcc, width=width, order=1)
    d2 = librosa.feature.delta(mfcc, width=width, order=2)
    raw = np.vstack([mfcc, d1, d2]).T
    # Per-utterance CMVN is an inference-time operation, no shared test statistics.
    F = (raw - raw.mean(axis=0)) / np.maximum(raw.std(axis=0), 1e-6)
    if not np.isfinite(F).all():
        raise ValueError('Nonfinite features')
    return (F[:, :dim], (mfcc,d1,d2), bounds) if return_parts else F[:, :dim]


def discover(root):
    root = Path(root)
    records, seen = [], {}
    for split, minimum in [('train',6), ('test',2)]:
        for label in LABELS:
            paths = sorted((root/split/label).glob('*.wav'))
            if len(paths) < minimum:
                raise ValueError(f'{root/split/label}: need >= {minimum} WAV, got {len(paths)}')
            for path in paths:
                y = load_audio(path)
                # Hash decoded/resampled samples, so metadata-only copies are detected.
                digest = hashlib.sha256(np.round(y,7).astype('<f8').tobytes()).hexdigest()
                if digest in seen:
                    raise ValueError(f'Duplicate waveform: {path} and {seen[digest]}')
                seen[digest] = str(path)
                info = sf.info(path)
                records.append(dict(path=str(path), split=split, label=label,
                                    waveform_sha256=digest, source_sr=info.samplerate,
                                    source_channels=info.channels, source_subtype=info.subtype,
                                    seconds=len(y)/SR))
    return records


def check_topology(model):
    n = model.n_components
    allowed = np.eye(n, dtype=bool) | np.eye(n, k=1, dtype=bool)
    assert np.all(model.transmat_[~allowed] == 0), 'Forbidden transition'
    assert np.all(model.transmat_ >= 0)
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1)
    np.testing.assert_array_equal(model.startprob_, np.eye(n)[0])
    assert model.transmat_[-1,-1] == 1


class FlooredGaussianHMM(GaussianHMM):
    """Baum-Welch from hmmlearn; explicit variance floor after each M-step.

    A floor regularizes EM; record any objective decrease rather than promising
    strict unregularized-EM monotonicity.
    """
    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if self.covariance_type == 'diag':
            self._covars_ = np.maximum(self._covars_, 1e-2)


def train_hmm(sequences, states=5, dim=39):
    sequences = [x[:, :dim] for x in sequences]
    if any(len(x) < states for x in sequences):
        raise ValueError('Each sequence must have at least n_states frames')
    model = FlooredGaussianHMM(n_components=states, covariance_type='diag',
                              n_iter=80, tol=1e-3, init_params='', params='tmc',
                              random_state=SEED, min_covar=1e-2)
    model.startprob_ = np.eye(states)[0]
    model.transmat_ = np.eye(states)*0.7 + np.eye(states,k=1)*0.3
    model.transmat_[-1,-1] = 1
    # Deterministic temporal initialization uses only training utterances.
    chunks = [np.vstack([np.array_split(x,states)[s] for x in sequences]) for s in range(states)]
    model.means_ = np.vstack([x.mean(axis=0) for x in chunks])
    model.covars_ = np.vstack([np.maximum(x.var(axis=0),0.1) for x in chunks])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(np.vstack(sequences), [len(x) for x in sequences])
    check_topology(model)
    history = list(model.monitor_.history)
    diffs = np.diff(history)
    diagnostics = dict(iterations=len(history), history=history,
                       monitor_converged=bool(model.monitor_.converged),
                       stopped_before_limit=len(history)<model.n_iter,
                       final_gain=float(diffs[-1]) if len(diffs) else None,
                       objective_decreases=int(np.sum(diffs < -1e-5)),
                       warnings=[str(w.message) for w in caught])
    return model, diagnostics


def fit_models(train, states=5, dim=39):
    models, monitors = {}, {}
    start = time.perf_counter()
    for label in LABELS:
        models[label], monitors[label] = train_hmm([r['F'] for r in train if r['label']==label], states, dim)
    return models, monitors, time.perf_counter()-start


def predict_hmm(models, rows, dim=39):
    results=[]
    start=time.perf_counter()
    for row in rows:
        X=row['F'][:,:dim]
        scores={label:float(models[label].score(X)) for label in LABELS}
        if not np.isfinite(list(scores.values())).all():
            raise ValueError('HMM produced nonfinite likelihood')
        order=sorted(scores,key=scores.get,reverse=True)
        results.append(dict(file=Path(row['path']).name, truth=row['label'], prediction=order[0],
                            frames=len(X), margin_per_frame=(scores[order[0]]-scores[order[1]])/len(X),
                            **scores))
    return results,time.perf_counter()-start


@njit(cache=True)
def dtw_distance(x,y):
    """Minimum total Euclidean path cost / (T+U); same convention for all templates."""
    n,m=len(x),len(y)
    prev=np.full(m+1,np.inf)
    prev[0]=0.
    for i in range(n):
        cur=np.full(m+1,np.inf)
        for j in range(m):
            cost=np.sqrt(np.sum((x[i]-y[j])**2))
            cur[j+1]=cost+min(prev[j+1],cur[j],prev[j])
        prev=cur
    return prev[m]/(n+m)


def predict_dtw(train, rows, dim=13):
    results=[]
    start=time.perf_counter()
    for row in rows:
        distances={label:min(dtw_distance(row['F'][:,:dim],r['F'][:,:dim])
                             for r in train if r['label']==label) for label in LABELS}
        results.append(dict(file=Path(row['path']).name,truth=row['label'],
                            prediction=min(distances,key=distances.get),**distances))
    return results,time.perf_counter()-start


def accuracy(rows):
    return float(accuracy_score([r['truth'] for r in rows],[r['prediction'] for r in rows]))


def plot_cm(rows,path,title):
    cm=confusion_matrix([r['truth'] for r in rows],[r['prediction'] for r in rows],labels=LABELS)
    fig,ax=plt.subplots(figsize=(5.4,4.3))
    ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,colorbar=False,cmap='Blues')
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
    return cm.tolist()


def log_forward_viterbi(log_b,A,pi):
    """Reference DP, any end state accepted, matching hmmlearn.score/decode."""
    with np.errstate(divide='ignore'):
        log_A,log_pi=np.log(A),np.log(pi)
    T,N=log_b.shape
    alpha=log_pi+log_b[0]
    delta=alpha.copy()
    back=np.zeros((T,N),dtype=int)
    for t in range(1,T):
        alpha=log_b[t]+logsumexp(alpha[:,None]+log_A,axis=0)
        candidates=delta[:,None]+log_A
        back[t]=np.argmax(candidates,axis=0)
        delta=log_b[t]+np.max(candidates,axis=0)
    path=np.zeros(T,dtype=int); path[-1]=np.argmax(delta)
    for t in range(T-2,-1,-1):
        path[t]=back[t+1,path[t+1]]
    return float(logsumexp(alpha)),float(np.max(delta)),path


def gaussian_log_emissions(model,X):
    variances=np.diagonal(model.covars_,axis1=1,axis2=2)
    return -0.5*(np.log(2*np.pi*variances).sum(axis=1)[None,:]
                 + ((X[:,None,:]-model.means_[None,:,:])**2/variances[None,:,:]).sum(axis=2))


def parse_lexicon(path):
    lex={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields=line.split('#',1)[0].split()
        if not fields: continue
        if len(fields)<2 or fields[0] in lex:
            raise ValueError('Malformed or duplicate lexicon entry: '+line)
        lex[fields[0]]=fields[1:]
    missing=set(LABELS)-lex.keys()
    if missing: raise ValueError('Missing lexicon words: '+str(missing))
    return lex


def language_counts(path):
    unigram, bigram, outgoing=Counter(),Counter(),Counter()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        words=line.split('#',1)[0].split()
        if not words: continue
        if set(words)-set(LABELS): raise ValueError('Corpus contains out-of-vocabulary words')
        unigram.update(words)
        # No EOS category here. Denominator counts outgoing transitions, not all tokens.
        for prev,cur in zip(['<s>']+words[:-1],words):
            bigram[prev,cur]+=1; outgoing[prev]+=1
    return unigram,bigram,outgoing


def bigram_probability(prev,cur,bigram,outgoing,k=0.5):
    if k<=0 or cur not in LABELS: raise ValueError('Invalid k or target word')
    return (bigram[prev,cur]+k)/(outgoing[prev]+k*len(LABELS))


def add_noise(y,snr_db,rng):
    # Set SNR on active speech, not trailing silence; do not peak-clip.
    active,_=librosa.effects.trim(y,top_db=30,frame_length=400,hop_length=HOP)
    noise=rng.normal(size=len(y))
    noise*=np.sqrt(np.mean(active**2)/(10**(snr_db/10)*np.mean(noise**2)))
    return y+noise


def run(data='dataset_demo',output='results_demo'):
    root,out=Path(data),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    figs=out/'figures'; figs.mkdir(exist_ok=True)
    model_dir=out/'models'; model_dir.mkdir(exist_ok=True)
    provenance_path=root/'PROVENANCE.json'
    provenance=json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.exists() else {'kind':'user_supplied_unverified'}
    kind=provenance['kind']
    tag='SYNTHETIC DEMO' if kind.startswith('synthetic') else 'USER DATA'
    print('Dataset:',root,'|',tag,flush=True)
    records=discover(root)
    save_csv(out/'data_audit.csv',records)
    start=time.perf_counter()
    rows=[dict(r,F=features(load_audio(r['path']))) for r in records]
    feature_seconds=time.perf_counter()-start
    train=[r for r in rows if r['split']=='train']
    test=[r for r in rows if r['split']=='test']
    # Deterministic inner split: last 2 TRAIN files per class for validation.
    inner,valid=[],[]
    for label in LABELS:
        group=[r for r in train if r['label']==label]
        inner.extend(group[:-2]); valid.extend(group[-2:])
    save_json(out/'split.json',dict(train=[r['path'] for r in train],test=[r['path'] for r in test],
                                  inner_train=[r['path'] for r in inner],validation=[r['path'] for r in valid]))
    for row in [next(r for r in test if r['label']==lab) for lab in LABELS[:2]]:
        F,parts,bounds=features(load_audio(row['path']),return_parts=True)
        fig,axes=plt.subplots(3,1,figsize=(8,6),sharex=True)
        for ax,part,name in zip(axes,parts,['MFCC 13','Delta','Delta-delta']):
            im=ax.imshow(part,aspect='auto',origin='lower',extent=[0,part.shape[1]*.01,0,13])
            ax.set_ylabel(name); fig.colorbar(im,ax=ax,pad=.01)
        axes[-1].set_xlabel('Seconds after trimming'); axes[0].set_title(f'{tag}: {row["label"]}')
        fig.tight_layout(); fig.savefig(figs/f'features_{row["label"]}.png',dpi=150); plt.close(fig)
    print('Training E1 and E2 (fixed settings; validation separated)...',flush=True)
    experiments=[]; main_models=None
    main_monitor=None; main_train_time=0; validation_scores=None
    convergence={}
    for states,dim in [(3,39),(5,39),(7,39),(5,13)]:
        inner_models,_,_=fit_models(inner,states,dim)
        vrows,_=predict_hmm(inner_models,valid,dim)
        models,monitor,fit_seconds=fit_models(train,states,dim)
        pred,score_seconds=predict_hmm(models,test,dim)
        trpred,_=predict_hmm(models,train,dim)
        experiments.append(dict(states=states,dimensions=dim,train_accuracy=accuracy(trpred),
                                validation_accuracy=accuracy(vrows),test_accuracy=accuracy(pred),
                                train_seconds=fit_seconds,test_ms_per_trial=1000*score_seconds/len(test),
                                hit_iteration_limit=sum(not m['stopped_before_limit'] for m in monitor.values()),
                                objective_decreases=sum(m['objective_decreases'] for m in monitor.values())))
        key=f'{states}state_{dim}dim'; convergence[key]=monitor
        save_csv(out/f'predictions_{key}.csv',pred)
        plot_cm(pred,figs/f'cm_{key}.png',f'{tag}: HMM {states} states / {dim}D')
        if (states,dim)==(5,39):
            main_models,main_monitor,main_train_time=models,monitor,fit_seconds
            main_pred,validation_scores=pred,vrows
            main_test_seconds=score_seconds
    save_csv(out/'experiments.csv',experiments)
    save_json(out/'convergence.json',convergence)
    print('Main model topology:',flush=True)
    for label,m in main_models.items():
        print(label, '\n',np.round(m.transmat_,3),flush=True)
        np.savez_compressed(model_dir/f'{label}.npz',startprob=m.startprob_,transmat=m.transmat_,
                            means=m.means_,covars=m.covars_)
    save_json(model_dir/'metadata.json',dict(labels=LABELS,states=5,dimensions=39,covariance='diag',
                                            normalization='utterance_cmvn',kind=kind,sample_rate=SR))
    fig,ax=plt.subplots(figsize=(7,4))
    for label,monitor in main_monitor.items(): ax.plot(monitor['history'],label=label)
    ax.set(xlabel='EM iteration',ylabel='Training log-likelihood',title=f'{tag}: convergence')
    ax.legend(); fig.tight_layout(); fig.savefig(figs/'convergence.png',dpi=150); plt.close(fig)
    durations=[]; forward_checks=[]
    for label in LABELS[:2]:
        row=next(r for r in test if r['label']==label); X=row['F']; m=main_models[label]
        vscore,path=m.decode(X,algorithm='viterbi')
        forward,reference_v,reference_path=log_forward_viterbi(gaussian_log_emissions(m,X),m.transmat_,m.startprob_)
        np.testing.assert_allclose(forward,m.score(X),atol=1e-6)
        np.testing.assert_allclose(reference_v,vscore,atol=1e-6)
        np.testing.assert_array_equal(reference_path,path)
        assert np.all(np.isin(np.diff(path),[0,1]))
        assert forward>=vscore-1e-6
        forward_checks.append(dict(label=label,forward=forward,viterbi=float(vscore),gap=forward-vscore))
        fig,ax=plt.subplots(figsize=(7,3))
        ax.step(np.arange(len(path)),path,where='post'); ax.set_yticks(range(5))
        ax.set(xlabel='Frame (10 ms hop)',ylabel='State',title=f'{tag}: {Path(row["path"]).name}')
        fig.tight_layout(); fig.savefig(figs/f'viterbi_{label}.png',dpi=150); plt.close(fig)
        for s,count in enumerate(np.bincount(path,minlength=5)):
            durations.append(dict(label=label,state=s,frames=int(count),approx_duration_ms=int(count)*10))
    save_csv(out/'state_durations.csv',durations); save_csv(out/'forward_viterbi.csv',forward_checks)
    score_examples=[next(r for r in main_pred if r['truth']==label) for label in LABELS[:3]]
    save_csv(out/'likelihood_examples.csv',score_examples)
    print('DTW baseline and learning curves...',flush=True)
    # Warm JIT before timing; feature extraction excluded equally from HMM/DTW timers.
    dtw_distance(train[0]['F'][:,:13],train[0]['F'][:,:13])
    dtw_distance(train[0]['F'],train[0]['F'])
    dtw13,dtw13_seconds=predict_dtw(train,test,13)
    dtw39,dtw39_seconds=predict_dtw(train,test,39)
    save_csv(out/'predictions_dtw13.csv',dtw13); save_csv(out/'predictions_dtw39.csv',dtw39)
    plot_cm(dtw13,figs/'cm_dtw13.png',f'{tag}: DTW 13D')
    plot_cm(dtw39,figs/'cm_dtw39.png',f'{tag}: DTW 39D')
    comparison=[dict(method='HMM39',accuracy=accuracy(main_pred),train_samples_per_class=min(Counter(r['label'] for r in train).values()),
                     train_seconds=main_train_time,test_ms_per_trial=main_test_seconds*1000/len(test)),
                dict(method='DTW13',accuracy=accuracy(dtw13),train_samples_per_class=min(Counter(r['label'] for r in train).values()),
                     train_seconds=0.,test_ms_per_trial=dtw13_seconds*1000/len(test)),
                dict(method='DTW39',accuracy=accuracy(dtw39),train_samples_per_class=min(Counter(r['label'] for r in train).values()),
                     train_seconds=0.,test_ms_per_trial=dtw39_seconds*1000/len(test))]
    save_csv(out/'comparison.csv',comparison)
    learning=[]
    for n in [1,3,6]:
        subset=[r for label in LABELS for r in [x for x in train if x['label']==label][:n]]
        ms,_,_=fit_models(subset)
        hp,_=predict_hmm(ms,test); dp,_=predict_dtw(subset,test)
        learning.append(dict(samples_per_class=n,hmm_accuracy=accuracy(hp),dtw_accuracy=accuracy(dp)))
    save_csv(out/'learning_curve.csv',learning)
    rng=np.random.default_rng(SEED)
    noisy=[dict(r,F=features(add_noise(load_audio(r['path']),20,rng))) for r in test]
    noisy_hmm,_=predict_hmm(main_models,noisy)
    noisy_dtw,_=predict_dtw(train,noisy)
    save_csv(out/'predictions_noise_hmm.csv',noisy_hmm)
    save_csv(out/'predictions_noise_dtw.csv',noisy_dtw)
    plot_cm(noisy_hmm,figs/'cm_noise.png',f'{tag}: HMM, white noise 20 dB')
    plot_cm(noisy_dtw,figs/'cm_noise_dtw.png',f'{tag}: DTW, white noise 20 dB')
    errors=[]
    for condition,hrows,drows in [('clean',main_pred,dtw13),('noise20dB',noisy_hmm,noisy_dtw)]:
        for h,d in zip(hrows,drows):
            errors.append(dict(condition=condition,file=h['file'],truth=h['truth'],hmm=h['prediction'],dtw=d['prediction'],
                               hmm_correct=h['prediction']==h['truth'],dtw_correct=d['prediction']==d['truth']))
    save_csv(out/'error_analysis.csv',errors)
    correct_scores=[r[r['truth']]/r['frames'] for r in main_pred]
    other_scores=[r[label]/r['frames'] for r in main_pred for label in LABELS if label!=r['truth']]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(correct_scores,bins=8,alpha=.65,label='True class score')
    ax.hist(other_scores,bins=12,alpha=.65,label='Other class scores')
    ax.set(xlabel='Log likelihood / frame',ylabel='Count',title=f'{tag}: score distributions'); ax.legend()
    fig.tight_layout(); fig.savefig(figs/'score_distribution.png',dpi=150); plt.close(fig)
    # Calibrated using inner held-out train data, not test. Small-sample proposal only.
    threshold=float(np.quantile([r[r['truth']]/r['frames'] for r in validation_scores],.05))
    rejected=sum(max(r[label] for label in LABELS)/r['frames'] < threshold for r in main_pred)
    lex=parse_lexicon(Path(__file__).parent/'lexicon.txt')
    uni,bi,outgoing=language_counts(Path(__file__).parent/'corpus.txt')
    save_json(out/'language_model.json',dict(unigram=dict(uni),outgoing=dict(outgoing),
                bigram=[dict(previous=p,current=w,count=n) for (p,w),n in sorted(bi.items())],lexicon=lex))
    # Choose by acoustic uncertainty only, never use true label to choose candidate.
    uncertain=min(main_pred+noisy_hmm,key=lambda r:r['margin_per_frame'])
    uncertain_condition='noise20dB' if any(uncertain is r for r in noisy_hmm) else 'clean'
    reranked=[]
    for prev in ['<s>']+LABELS:
        probabilities={w:bigram_probability(prev,w,bi,outgoing) for w in LABELS}
        np.testing.assert_allclose(sum(probabilities.values()),1.)
        for weight in [0.,.1,.3,1.]:
            combined={w:uncertain[w]/uncertain['frames']+weight*np.log(probabilities[w]) for w in LABELS}
            pred=max(combined,key=combined.get)
            reranked.append(dict(file=uncertain['file'],condition=uncertain_condition,truth=uncertain['truth'],previous=prev,weight=weight,
                                 acoustic_prediction=uncertain['prediction'],prediction=pred,**combined))
    save_csv(out/'lm_reranking.csv',reranked)
    # Controlled didactic example, not measured audio: makes LM dominance explicit.
    toy_acoustic={w:-12. for w in LABELS}; toy_acoustic.update(len=-10.05,xuong=-10.)
    toy=[]
    for weight in [0.,.1,.3,1.]:
        scores={w:toy_acoustic[w]+weight*np.log(bigram_probability('mo',w,bi,outgoing)) for w in LABELS}
        toy.append(dict(weight=weight,prediction=max(scores,key=scores.get),**scores))
    save_csv(out/'lm_controlled_example.csv',toy)
    summary=dict(dataset_kind=kind,synthetic=kind.startswith('synthetic'),train_files=len(train),test_files=len(test),
                 feature_seconds=feature_seconds,main_configuration=dict(states=5,dimensions=39,selection='fixed_before_test'),
                 comparison=comparison,experiments=experiments,
                 noise=dict(snr_db=20,hmm_accuracy=accuracy(noisy_hmm),dtw_accuracy=accuracy(noisy_dtw)),
                 reject=dict(threshold=threshold,source='5th percentile of true-label scores on inner validation',
                             clean_known_rejected=rejected,unknown_false_accept_rate=None,
                             caveat='Needs independent calibration and unknown-word validation; final-model scale may drift.'),
                 disagreement_cases=[r for r in errors if r['hmm_correct']!=r['dtw_correct']],
                 full_covariance='Skipped: only 6 utterances/class; 780 covariance parameters/state at 39D vs 39 diagonal.',
                 lm_context_status='Hypothetical contexts, no observed previous-word labels; no LM accuracy claim.',
                 limitations=['Synthetic voice shared across train/test; no human-speaker generalization claim.',
                              'Lexicon and corpus are custom demonstrations, not instructor-provided.',
                              'DTW reimplemented because Lab 2 source was not supplied.',
                              'Test variants reported descriptively, never used for model selection.'],
                 versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','librosa','hmmlearn','scikit-learn','matplotlib']})
    save_json(out/'summary.json',summary)
    write_analysis(out,summary,main_pred,dtw13,main_monitor)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def write_analysis(out,summary,hmm,dtw,monitors):
    lines=['# Kết quả thực nghiệm Lab 3','',
           '**Dữ liệu:** '+summary['dataset_kind'],
           '**Không phải đánh giá trên giọng người thật** nếu dùng dataset_demo. Tất cả số liệu dưới đây được tính từ lần chạy hiện tại.','',
           '## E1–E2: số trạng thái và đặc trưng','',
           '| States | Chiều | Train | Validation | Test | Chạm 80 vòng |',
           '|---|---|---|---|---|---|']
    for r in summary['experiments']:
        lines.append(f'| {r["states"]} | {r["dimensions"]} | {r["train_accuracy"]:.0%} | {r["validation_accuracy"]:.0%} | {r["test_accuracy"]:.0%} | {r["hit_iteration_limit"]} |')
    lines+=['','Cấu hình chính 5-state/39D được ấn định trước. Validation lấy 2 file cuối mỗi lớp từ train; test chỉ dùng báo cáo. '
             'Với 10 test utterance, một lỗi tương ứng 10 điểm phần trăm; không đủ để kết luận một cấu hình tối ưu phổ quát.',
             'Nếu train cao hơn validation/test: có dấu hiệu overfit hoặc lệch phân bố. Nếu cả train lẫn validation thấp: xem lại đặc trưng, khởi tạo và khả năng underfit. '
             'Chạm giới hạn vòng lặp không đồng nghĩa hội tụ; xem final_gain và objective_decreases trong convergence.json.',
             '', '## E3: covariance','',summary['full_covariance'],'', '## E4: DTW và HMM','',
             '| Phương pháp | Accuracy | Fit (s) | Score/trial (ms) |','|---|---|---|---|']
    for r in summary['comparison']:
        lines.append(f'| {r["method"]} | {r["accuracy"]:.0%} | {r["train_seconds"]:.3f} | {r["test_ms_per_trial"]:.3f} |')
    lines+=['','Thời gian score đã loại trích đặc trưng và JIT warm-up cho cả hai phương pháp. DTW chỉ lưu template nên thời gian ước lượng tham số bằng 0; '
             'đây không phải tổng latency end-to-end. DTW13 là baseline MFCC; DTW39 giúp kiểm soát khác biệt số chiều.',
             'HMM nén mỗi lớp thành phân bố theo trạng thái; DTW đối sánh mọi template nên chi phí tăng theo số mẫu. Xem learning_curve.csv cho 1, 3, 6 mẫu/lớp.']
    cases=summary['disagreement_cases']
    if cases:
        c=cases[0]
        lines+=['',f'Ví dụ quan sát: {c["condition"]}, {c["file"]}; nhãn {c["truth"]}, HMM → {c["hmm"]}, DTW → {c["dtw"]}. '
                 'Khác biệt cho thấy mô hình phân bố và đối sánh template phản ứng khác nhau với chuỗi này. '
                 'Tác động của duration, delta và nhiễu là giả thuyết cần thêm ablation, không suy ra nguyên nhân duy nhất từ một mẫu.']
    else:
        lines+=['','Không quan sát thấy trường hợp một phương pháp đúng còn phương pháp kia sai trong clean/noise lần chạy này. '
                 'Không bịa một ví dụ thắng/thua; cần thêm ghi âm hoặc điều kiện khó hơn để đáp ứng mục này trên dữ liệu thật.']
    lines+=['','## E5: robustness','',f'Nhiễu trắng 20 dB: HMM {summary["noise"]["hmm_accuracy"]:.0%}; DTW {summary["noise"]["dtw_accuracy"]:.0%}. '
             'Nhiễu được thêm vào waveform trước endpoint/MFCC. Cùng nhiễu và seed cho hai phương pháp. '
             'Đây là một loại mismatch, không đại diện người nói mới hoặc phòng thu khác.',
             '', '## E6: LM và lexicon','',
             'lm_reranking.csv chọn utterance có acoustic margin nhỏ nhất; thử các context giả định và λ = 0, 0.1, 0.3, 1.0. '
             'Không có context thật nên không báo LM accuracy. Score/frame + λ log P là re-ranking có trọng số, không phải MAP nguyên bản nếu λ tùy chọn. '
             'lm_controlled_example.csv là điểm âm học giả lập riêng để phân tích cơ chế; nếu nhãn thật là len thì prior ủng hộ len giúp, '
             'nếu nhãn thật là xuong thì cùng prior có thể làm sai. Không tính ví dụ giả lập vào accuracy âm thanh.',
             'Đếm mẫu số bằng số bigram đi ra từ context; không dùng tổng unigram gồm cả từ cuối câu. Vì không mô hình EOS, '
             'tổng xác suất trên đúng 5 lệnh bằng 1 cả với context chưa gặp.',
             '', '## Viterbi, Forward và reject','',
             'forward_viterbi.csv đối chiếu DP viết riêng với thư viện trên hai utterance. Forward cộng mọi đường; Viterbi lấy đường tốt nhất. '
             'Duration ≈ số frame × 10 ms; state không có nhãn phone. Hai heatmap dùng chung tham số; cường độ và nhịp biến đổi khác nhau '
             'phản ánh các tín hiệu được tổng hợp khác nhau, không phải bằng chứng trực tiếp về khác biệt phone.',
             f'Ngưỡng reject đề xuất {summary["reject"]["threshold"]:.3f} log-likelihood/frame từ inner-validation. '
             f'Có {summary["reject"]["clean_known_rejected"]} test từ đã biết bị reject. Chưa có unknown test nên chưa đo FAR. '
             'Sau fit lại full train, cần hiệu chỉnh ngưỡng bằng tập calibration riêng vì thang score có thể đổi.',
             '', '## Kết luận','',
             'Pipeline đã chạy đủ từ WAV đến đặc trưng, HMM, Viterbi, DTW và LM minh họa. Để kết luận về nhận dạng tiếng Việt '
             'trên người thật cần thu âm độc lập và chạy lại. Lexicon/corpus cần thay bằng bản giảng viên nếu được yêu cầu.']
    (out/'ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='dataset_demo')
    parser.add_argument('--output',default='results_demo')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data,args.output)
