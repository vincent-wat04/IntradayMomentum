"""Timing regression checks for the reusable factor backtest.

Run: python -m unittest test_index_timing_framework.py
"""
import unittest

import numpy as np
import pandas as pd

from index_timing_framework import (MarketPanel,StudyConfig,event_prices,
                                    portfolio_study,orthogonalize_past_k,
                                    daily_factors_from_z,event_wins,FACTOR_NAMES,
                                    trailing_bucket_ids,nonoverlapping_returns)


class TimingTests(unittest.TestCase):
    def setUp(self):
        dates=pd.Index(["2025-01-02","2025-01-03","2025-01-06","2025-01-07"],name="day")
        factors=pd.DataFrame({"MID_MED":[1.0]},index=dates[:1])
        self.panel=MarketPanel(dates,np.array([100.,105.,115.,125.]),
                               np.array([100.,110.,120.,130.]),factors,0)

    def test_first_realized_return_is_at_correct_close(self):
        p=np.array([0])
        _,entry,exit_=event_prices(self.panel,p,1,"next_open")
        self.assertEqual((entry[0],exit_[0]),(105.,110.))
        _,entry,exit_=event_prices(self.panel,p,1,"next_close")
        self.assertEqual((entry[0],exit_[0]),(110.,120.))
        _,entry,exit_=event_prices(self.panel,p,2,"next_open")
        self.assertEqual((entry[0],exit_[0]),(105.,120.))
        _,entry,exit_=event_prices(self.panel,p,2,"next_close")
        self.assertEqual((entry[0],exit_[0]),(110.,130.))

    def test_skew_proxy_is_checkpoint_mean_minus_median(self):
        z=np.array([[0.,0.,0.,0.,5.],[-5.,0.,0.,0.,0.]])
        result=daily_factors_from_z(z,pd.Index(["a","b"]))
        self.assertAlmostEqual(result.SKEW_DIFF.iloc[0],1.0)
        self.assertAlmostEqual(result.SKEW_DIFF.iloc[1],-1.0)
        self.assertEqual(event_wins("SKEW_DIFF",False,np.array([0.1]))[0],True)
        self.assertEqual(event_wins("SKEW_DIFF",True,np.array([-0.1]))[0],True)

    def test_open_and_close_portfolio_entitlements(self):
        s=self.panel.factors.MID_MED
        nan=pd.Series(np.nan,index=s.index)
        cache={(fid,1,version):(s,{10:nan,90:nan})
               for fid in FACTOR_NAMES for version in ("raw","raw_matched","orthog")}
        results=portfolio_study("SPY",self.panel,
                  StudyConfig(windows=(1,),fee_bp=0),cache)
        by_mode={r["执行代码"]:r for r in results if r["规则"]=="正值持有" and r["因子版本"]=="原始" and r["因子ID"]=="MID_MED"}
        self.assertAlmostEqual(by_mode["next_open"]["组合收益率"],130/105-1)
        self.assertAlmostEqual(by_mode["next_close"]["组合收益率"],130/110-1)
        self.assertEqual(by_mode["next_open"]["收益日数"],3)
        self.assertEqual(by_mode["next_close"]["收益日数"],3)

    def test_orthogonalization_fits_only_prior_pairs(self):
        dates=pd.Index([f"2025-01-{d:02d}" for d in range(1,11)],name="day")
        close=np.array([100.,101.,103.,102.,107.,108.,112.,109.,115.,120.])
        past=pd.Series(close,index=dates).div(pd.Series(close,index=dates).shift(2)).sub(1)
        factor=2+3*past
        factor.iloc[7]+=1.0
        cfg=StudyConfig(orthog_lookback=5,orthog_min_periods=3)
        panel=MarketPanel(dates,close.copy(),close.copy(),pd.DataFrame(index=dates),0)
        residual=orthogonalize_past_k(factor,panel,2,cfg)
        self.assertTrue(np.isnan(residual.iloc[4]))
        self.assertAlmostEqual(residual.iloc[7],1.0,places=11)
        # Later observations cannot revise the historical residual.
        modified=factor.copy();modified.iloc[8:]+=1000
        panel.close[8:]+=500
        other=orthogonalize_past_k(modified,panel,2,cfg)
        self.assertAlmostEqual(other.iloc[7],residual.iloc[7],places=11)

    def test_bucket_assignment_uses_only_past_and_handles_ties(self):
        cfg=StudyConfig(threshold_lookback=4,threshold_min_periods=2)
        s=pd.Series([1.,2.,3.,4.,5.,6.,7.])
        original=trailing_bucket_ids(s,cfg,5)
        self.assertEqual(list(original[:2]),[0,0])
        self.assertEqual(list(original[2:]),[5]*5)
        changed=s.copy();changed.iloc[5:]=-1000
        revised=trailing_bucket_ids(changed,cfg,5)
        self.assertEqual(list(revised[:5]),list(original[:5]))
        ties=trailing_bucket_ids(pd.Series([0.,0.,0.,0.]),cfg,5)
        self.assertEqual(list(ties[2:]),[3,3])

    def test_bucket_nonoverlap_respects_exit_date(self):
        positions=np.arange(5)
        returns=np.arange(5,dtype=float)
        self.assertEqual(list(nonoverlapping_returns(positions,returns,2,"next_open")),[0.,1.,2.,3.,4.])
        self.assertEqual(list(nonoverlapping_returns(positions,returns,2,"next_close")),[0.,2.,4.])


if __name__=="__main__":
    unittest.main()
