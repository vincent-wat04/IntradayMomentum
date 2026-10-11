"""Generate auditable Markdown and CSV from the frozen six-horizon model outputs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def fmt(x):return '—' if x is None else f'{x:+.3f}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True);p.add_argument('--lite',required=True)
    p.add_argument('--paper-status',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    baseline=json.load(open(a.baseline));lite=json.load(open(a.lite))
    paper=json.load(open(a.paper_status));results=baseline+lite
    table=[];selections=[]
    for z in results:
        if z['status']!='ok':continue
        perf=z['daily_long_cash'];met=z['metric']
        table.append({'symbol':z['symbol'],'horizon':z['horizon'],
            'execution':z['execution'],'model':z['model'],
            'training_window':z['training_window'],'n_test':z['n_test'],
            'train_min':z['n_train_min'],'train_max':z['n_train_max'],
            'pearson':met['pearson'],'rank_ic':met['rank_ic'],
            'direction_hit':met['direction_hit'],
            'greedy_nonoverlap_count':met['greedy_nonoverlap_count'],
            'quintile_means':json.dumps(met['test_quintile_means']),
            'monotone_up_steps':met['monotone_up_steps'],
            'strategy_return':perf['strategy']['total_return'],
            'buy_hold_return':perf['buy_hold_price']['total_return'],
            'strategy_sharpe':perf['strategy']['sharpe'],
            'active_coef_mean':z.get('active_coef_mean'),
            'features':','.join(f"{q['name']}:{q['k']}" for q in z['selected']),
            'first_signal':z['first_signal'],'last_signal':z['last_signal']})
        if z['model']=='elastic_net' and z['training_window']==60:
            for q in z['selected']:
                selections.append({'symbol':z['symbol'],'horizon':z['horizon'],
                    'execution':z['execution'],'factor':q['name'],'K':q['k'],
                    'n_pretest':q['n'],'direction':q['direction'],
                    'pearson_early':q['ic_halves'][0],
                    'pearson_late':q['ic_halves'][1],
                    'rank_early':q['rank_ic_halves'][0],
                    'rank_late':q['rank_ic_halves'][1],
                    'signed_bucket_spread':q['signed_spread'],
                    'monotone_steps_of_two':q['monotone']})
    for name,rows in [('model_metrics.csv',table),('selected_factors.csv',selections)]:
        with open(out/name,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 六期限指数择时模型复核（2026-10-04）','',
    '## 结论','',
    '这份结果属于探索性研究。SPY/QQQ 每只仅有 397 个交易日（2024-06 至 2025-12）；没有 2026 样本。',
    '滚动线性和 RAVEN-lite 已实跑。完整 RAVEN 的论文结构代码已编写，但本环境 PyTorch 缺失，',
    '标准及官方 CPU 安装源均无法提供 wheel，**其 24 组训练与测试均未运行**，status JSON 如实记录 dependency_unavailable。',
    '不能以当前结果宣称复现论文收益或可部署 alpha。','',
    '## 数据切分与防前视','',
    '- 固定测试信号从 2025-06-02 开始。对每只 ETF、每个期限和执行时点单独筛选因子；筛选仅用测试开始前已经退出的标签。',
    '- 168 个候选（29 个因子 × 6 个 K，6 个方差因子不取 K=1）；按 pretest 时间均分两段，Pearson 和 Spearman 必须同号，按二者较弱绝对值及三桶价差/单调性排序，每个因子只留一个 K，剔除历史高度相关项，最多选四项。',
    '- 输入值是对应 K 的历史日因子聚合，并用截至该日之前最多 120 对样本的 K 日价格收益作滚动正交；信号形成当日收盘才可用。缺失仅向前填充至多 5 日，模型上下文不允许未来填充。',
    '- T+0：t 收盘观察，t+1 开盘入场，h 日收益为 log(C[t+h]/O[t+1])，标签在 t+h 收盘成熟；T+1：t+1 收盘入场，log(C[t+1+h]/C[t+1])，标签在 t+1+h 收盘成熟。',
    '- 每个测试信号 t 的训练条件是 `signal_index < t` 且 `label_maturity <= t`。测试日期上的因子和已成熟标签允许后续滚动拟合；未来标签不能进入当前拟合。',
    '- 60/120 个**已成熟且因子齐全**的最近观察作为固定窗口，或使用全部既往成熟样本扩张；标准化器每次只在该训练片段拟合。ElasticNet l1_ratio=.5、alpha=.05（目标按训练标准差缩放），Ridge alpha=10，参数预设，不按测试期挑模型。',
    '- lite 使用相同逐期限选中的最多四因子、20 日上下文和 5 日 patch，CIT 阈值 .3/.6/.9，岭回归专家与全窗口支路，每日以所有成熟样本重新拟合。它**不是**论文原结构。',
    '- 报表中的测试期五桶是**事后按预测值排名的描述性分层**，不是可在线获得的五桶阈值；日度 long/cash 始终按预测值是否大于 0 定仓，按执行价建仓并计 1 bp/次单边换仓。',
    '', '## 测试期预测 Pearson IC', '',
    '表中 `EN60/EN120/EN扩张` 是三个事先指定的训练方案；**不要按测试 IC 事后选最优一列**。',
    'Rank IC、五桶、逐行预测、持仓收益、最大回撤和训练样本数在 CSV/JSON。','',
    '|执行|ETF|h|n|非重叠约数|EN60|EN120|EN扩张|RAVEN-lite|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lookup={(z['symbol'],z['execution'],z['horizon'],z['model'],str(z['training_window'])):z
            for z in results if z['status']=='ok'}
    for ex in ['next_open','next_close']:
        for sym in ['SPY','QQQ']:
            for h in [1,5,10,20,30,60]:
                zs=[lookup[(sym,ex,h,'elastic_net',str(w))] for w in [60,120,'expanding']]
                l=lookup[(sym,ex,h,'raven_lite','expanding')]
                lines.append(f"|{ex}|{sym}|{h}|{zs[0]['n_test']}|"
                    f"{zs[0]['metric']['greedy_nonoverlap_count']}|"
                    + '|'.join(fmt(z['metric']['pearson']) for z in zs+[l])+'|')
    lines+=['','## 容量与稳健性解释','',
    '6 个预测期限 × 2 个执行 × 2 个标的，线性基线各三种窗口共 144 组，lite 24 组；多重选择使单个高 IC 的证据价值很低。',
    'h=20/30/60 目标高度重叠；60 日测试虽有约 86–87 个相邻信号，真正能连续安排的非重叠持有期通常只有 2 个。',
    'SPY 60 日最早的 60 窗口甚至只有 48 个成熟有效训练点；四个筛选特征加截距尚可做正则线性拟合，但对 3 个三层 Transformer 专家明显不够。',
    '同样，样本跨越的市场状态和年度过少，不能用年内滚动相关证明 regime 稳定。',
    '几乎所有测试 long/cash 方案跑输相同日期的 ETF 买入持有；少数表面胜出仅约 0.05–1.85 个百分点，经过 168 组对照后不足以确立优势。',
    '', '## 完整 RAVEN 的实现边界','',
    '`raven_paper_model.py` 用 64 日实例标准化，16 日 patch、128 维共享逐通道投影、sinusoidal 位置编码、',
    '学习 patch 重要性并从近到远按 .3/.6/.9 取 CIT 前缀、三个独立的三层 Transformer encoder、',
    '形状对齐均值池化、余弦相似度冗余惩罚和可训练 CAW、全窗口 Self-Attention、MLP 回归头；',
    '目标 MSE 加 .1 负熵、.01 多样性，AdamW 和 cosine 调度 60 epochs。',
    '原文没有完整开源代码和所有层的明确参数；raw gate confidence 的参数化、64 日上下文、ETF 特征输入是此处明示的实现选择。',
    '论文使用多年、数千股票的面板。即使安装 PyTorch，这个仅两只 ETF 的实验也只是**架构移植**，不是同数据的论文指标复现。',
    '', '## 可重复运行','', '```bash',
    'python -m pip install -r requirements.txt',
    'python -m unittest test_multifrequency_models.py test_extended_research.py test_index_timing_framework.py',
    'python multifrequency_models.py --spy /path/SPY_5min.csv --qqq /path/QQQ_5min.csv --output baseline.json --models ridge elastic_net raven_lite',
    'python -m pip install torch  # full model optional dependency',
    'python raven_paper_model.py --spy /path/SPY_5min.csv --qqq /path/QQQ_5min.csv --output raven_full.json',
    '```','',
    '纸面结构出处：He 等，RAVEN, arXiv:2606.24062，尤其式 (1)–(17) 和实验设置。',
    '行情为未经 ETF 分红复权的公开五分钟价，收盘 bar 并未核验为真实集合竞价成交价；组合成交、滑点与实际执行能力尚未确认。','']
    (out/'MODEL_RESEARCH.md').write_text('\n'.join(lines))
    print('wrote',out,'model rows',len(table),'selection rows',len(selections),
          'full paper statuses',set(z['status'] for z in paper))

if __name__=='__main__':main()
