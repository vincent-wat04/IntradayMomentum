"""Causality and architectural smoke checks for the new project."""
import unittest

import numpy as np
import pandas as pd

from multifrequency_models import matured_train_indices, screen_factors, target
from index_timing_framework import MarketPanel


class CausalModelTests(unittest.TestCase):
    def test_label_maturity_matches_executable_entry(self):
        dates=pd.Index([f'2025-01-{d:02d}' for d in range(1,11)])
        panel=MarketPanel(dates,np.arange(10.)+100,np.arange(10.)+101,
                          pd.DataFrame(index=dates),0)
        for ex,lag in [('next_open',0),('next_close',1)]:
            y,m=target(panel,5,ex)
            self.assertEqual(m[0],5+lag)
            self.assertTrue(np.isfinite(y[4-lag]))
            self.assertFalse(np.isfinite(y[5-lag]))

    def test_train_only_uses_exited_labels(self):
        y=np.arange(9.)
        maturity=np.arange(9)+5
        valid=np.ones(9,bool)
        self.assertEqual(matured_train_indices(7,maturity,y,valid,None).tolist(),[0,1,2])
        self.assertEqual(matured_train_indices(7,maturity,y,valid,2).tolist(),[1,2])
        self.assertEqual(matured_train_indices(7,maturity,y,valid,2,
                         positions=np.arange(9)+10).tolist(),[])

    def test_screen_cannot_read_post_split_labels(self):
        rng=np.random.default_rng(3);n=140;split=100
        x=rng.normal(size=n);y=.3*x+rng.normal(size=n)*.2
        frame=pd.DataFrame({('signal',5):x})
        a,_=screen_factors(frame,y,np.arange(n)+5,split)
        y[split:]=1e6*rng.normal(size=n-split)
        frame.iloc[split:,0]=1e6*rng.normal(size=n-split)
        b,_=screen_factors(frame,y,np.arange(n)+5,split)
        self.assertEqual(a,b)

    def test_paper_module_reports_missing_dependency(self):
        import raven_paper_model
        if raven_paper_model.torch is None:
            out=raven_paper_model.experiment('/nonexistent','SPY',1,'next_open')
            self.assertEqual(out['status'],'dependency_unavailable')
        else:
            net=raven_paper_model.RAVEN(channels=3)
            xx=raven_paper_model.torch.randn(2,64,3)
            yy,entropy,diversity,prefix=net(xx)
            self.assertEqual(tuple(yy.shape),(2,))
            self.assertEqual(tuple(prefix.shape),(2,3))
            self.assertTrue((prefix[:,1:]>=prefix[:,:-1]).all().item())
            self.assertTrue(np.isfinite([entropy.item(),diversity.item()]).all())


if __name__=='__main__':unittest.main()
