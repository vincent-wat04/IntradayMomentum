"""Slippage of one-shot market orders against the recorded TQQQ book (needs data/mdrec)."""
import sys
sys.path.insert(0, ".")
from quantkit.data import mdrec
from quantkit.execution import book

snaps = mdrec.book_snapshots("US.TQQQ")
print(len(snaps), "uncrossed snapshots over", snaps.day.nunique(), "days")
print((book.slippage_table(snaps, sizes=(1000, 2000, 5000, 10000, 20000))).round(4))
print("assumption in the backtest: half-spread 0.0050 $/share")
