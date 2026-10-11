"""中文字体解析：沙箱会定期清掉 /usr/share/fonts，所以必须自己兜底。

用法（所有画图脚本统一这样写）：

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cjkfont import use_cjk
    use_cjk(plt)

查找顺序：项目内 assets/ -> 系统目录 -> 自动下载到 assets/。
全都失败时**显式告警**，而不是静默画出一堆方框 ——
之前就吃过这个亏：图生成"成功"了，但所有中文是 tofu，直到看图才发现。
"""
from __future__ import annotations

import sys
from pathlib import Path

FONT = "NotoSansCJKsc-Regular.otf"
URL = ("https://raw.githubusercontent.com/googlefonts/noto-cjk/main/"
       "Sans/OTF/SimplifiedChinese/" + FONT)


def _candidates():
    here = Path(__file__).resolve().parent
    return [here / "assets" / FONT,
            here.parent / "assets" / FONT,
            Path("/usr/share/fonts/cjk") / FONT,
            Path("/usr/share/fonts/truetype/noto") / FONT]


def find_font(download: bool = True) -> Path | None:
    for p in _candidates():
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    if not download:
        return None
    tgt = Path(__file__).resolve().parent / "assets" / FONT
    tgt.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        urllib.request.urlretrieve(URL, tgt)
        if tgt.exists() and tgt.stat().st_size > 1_000_000:
            return tgt
    except Exception as ex:                      # 离线环境
        print(f"[cjkfont] 下载失败：{ex}", file=sys.stderr)
    return None


def use_cjk(plt, download: bool = True) -> bool:
    """注册中文字体并设好 rcParams。返回是否成功。"""
    import matplotlib.font_manager as fm
    p = find_font(download=download)
    plt.rcParams["axes.unicode_minus"] = False
    if p is None:
        print("[cjkfont] **找不到中文字体，图中的中文会显示为方框**。"
              f"\n  请手动下载 {URL}\n  放到 {Path(__file__).resolve().parent/'assets'}/",
              file=sys.stderr)
        return False
    fm.fontManager.addfont(str(p))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    return True
