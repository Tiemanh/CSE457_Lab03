"""Independent correctness checks, run with python -m unittest -v."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from lab3 import (LABELS, features, log_forward_viterbi, dtw_distance, train_hmm,
                  check_topology, language_counts, bigram_probability, parse_lexicon,
                  gaussian_log_emissions, load_audio, discover)


class LabTests(unittest.TestCase):
    def test_forward_viterbi_against_exhaustive_enumeration(self):
        A=np.array([[.6,.4],[0.,1.]])
        pi=np.array([1.,0.])
        emissions=np.array([[.8,.2],[.3,.7],[.4,.6]])
        paths=list(itertools.product(range(2),repeat=3))
        probabilities=[]
        for q in paths:
            p=pi[q[0]]*emissions[0,q[0]]
            for t in range(1,3): p*=A[q[t-1],q[t]]*emissions[t,q[t]]
            probabilities.append(p)
        forward,viterbi,path=log_forward_viterbi(np.log(emissions),A,pi)
        self.assertAlmostEqual(np.exp(forward),sum(probabilities))
        self.assertAlmostEqual(np.exp(viterbi),max(probabilities))
        self.assertEqual(tuple(path),paths[np.argmax(probabilities)])

    def test_long_sequence_does_not_underflow(self):
        b=np.full((10000,2),-100.)
        f,v,_=log_forward_viterbi(b,np.array([[.7,.3],[0,1.]]),np.array([1.,0.]))
        self.assertTrue(np.isfinite([f,v]).all())
        self.assertGreaterEqual(f,v-1e-6)

    def test_dtw_identical_and_stretched(self):
        x=np.array([[0.],[1.],[2.]])
        self.assertEqual(dtw_distance(x,x),0.)
        self.assertEqual(dtw_distance(x,np.repeat(x,2,axis=0)),0.)
        self.assertGreater(dtw_distance(x,x+10),0.)

    def test_mfcc_cmvn_and_dimensionality(self):
        t=np.arange(16000)/16000
        y=np.sin(2*np.pi*(180*t+50*t*t)) * np.hanning(len(t))
        f=features(y)
        self.assertEqual(f.shape[1],39)
        self.assertTrue(np.isfinite(f).all())
        np.testing.assert_allclose(f.mean(axis=0),0,atol=1e-8)
        np.testing.assert_allclose(features(y,13),f[:,:13])

    def test_silence_and_short_audio_rejected(self):
        for y in [np.zeros(16000),np.ones(50),np.array([np.nan])]:
            with self.assertRaises(ValueError): features(y)

    def test_hmm_topology_and_score_reference(self):
        rng=np.random.default_rng(42)
        sequences=[np.vstack([rng.normal(s,.2,size=(12,3)) for s in range(3)]) for _ in range(4)]
        with threadpool_limits(limits=1):
            m,diag=train_hmm(sequences,states=3,dim=3)
        check_topology(m)
        f,v,path=log_forward_viterbi(gaussian_log_emissions(m,sequences[0]),m.transmat_,m.startprob_)
        self.assertAlmostEqual(f,m.score(sequences[0]),places=6)
        self.assertAlmostEqual(v,m.decode(sequences[0],algorithm='viterbi')[0],places=6)
        self.assertTrue(np.all(np.diff(path)>=0))
        self.assertTrue(np.all(m._covars_>=.01))

    def test_bigram_normalizes_including_terminal_word_context(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)/'corpus.txt'
            p.write_text('mo len\nmo\n',encoding='utf-8')
            uni,bi,out=language_counts(p)
            self.assertEqual(uni['mo'],2)
            self.assertEqual(out['mo'],1)
            self.assertAlmostEqual(bigram_probability('mo','len',bi,out),1.5/3.5)
            for prev in ['mo','len','unseen','<s>']:
                self.assertAlmostEqual(sum(bigram_probability(prev,w,bi,out) for w in LABELS),1.)

    def test_lexicon_duplicate_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)/'lex.txt'; p.write_text('mo m o\nmo m o\n')
            with self.assertRaises(ValueError): parse_lexicon(p)

    def test_demo_split_and_audio_format(self):
        root=Path(__file__).parent/'dataset_demo'
        rows=discover(root)
        self.assertEqual(len(rows),40)
        self.assertEqual(sum(r['split']=='train' for r in rows),30)
        self.assertTrue(all(r['source_sr']==16000 and r['source_channels']==1
                            and r['source_subtype']=='PCM_16' for r in rows))
        for r in rows[:3]: self.assertEqual(features(load_audio(r['path'])).shape[1],39)


if __name__=='__main__': unittest.main()
