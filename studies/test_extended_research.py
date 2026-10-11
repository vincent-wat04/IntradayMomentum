"""Focused checks for new factor aggregation and model execution."""
import unittest
import numpy as np
import pandas as pd

from extended_timing_research import aggregate
from raven_lite_timing import adaptive_features,backtest
from index_timing_framework import MarketPanel


class ExtendedTests(unittest.TestCase):
    def test_stability_is_past_k_variance(self):
        daily=pd.DataFrame({'OPEN_VOL_SHARE':[.1,.2,.4,.8]},
                           index=pd.Index(['a','b','c','d']))
        out=aggregate(daily,'OPEN_VOL_SHARE_VAR',2)
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertAlmostEqual(out.iloc[2],.01)
        altered=daily.copy();altered.iloc[3,0]=100
        self.assertAlmostEqual(aggregate(altered,'OPEN_VOL_SHARE_VAR',2).iloc[2],.01)

    def test_routing_depends_only_on_supplied_history(self):
        rng=np.random.default_rng(13)
        x=rng.normal(size=(3,20,10))
        e,g,w,l=adaptive_features(x,.5)
        self.assertEqual(e.shape,(3,3,20))
        self.assertEqual(g.shape,(3,30))
        self.assertTrue(np.all(np.diff(l,axis=1)>=0))
        self.assertTrue(np.all(l>=1))
        self.assertTrue(np.all(l<=4))
        self.assertTrue(np.allclose(w.sum(axis=1),1))
        changed=x.copy();changed[2]+=10000
        other=adaptive_features(changed,.5)
        self.assertTrue(np.allclose(e[:2],other[0][:2]))

    def test_close_version_first_new_position_return_is_next_day(self):
        dates=pd.Index(['2025-06-02','2025-06-03','2025-06-04','2025-06-05'])
        panel=MarketPanel(dates,np.array([100.,105.,115.,125.]),
                          np.array([100.,110.,120.,130.]),pd.DataFrame(index=dates),0)
        open_=backtest(panel,np.array([0,1]),np.array([1.,1.]),'next_open',0)
        close_=backtest(panel,np.array([0,1]),np.array([1.,1.]),'next_close',0)
        self.assertAlmostEqual(open_['strategy']['total_return'],120/105-1)
        self.assertAlmostEqual(close_['strategy']['total_return'],120/110-1)


if __name__=='__main__':unittest.main()
