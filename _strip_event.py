# -*- coding: utf-8 -*-
"""去掉公开站点里的赛事标识，只保留项目定位。逐条替换，命中数打印出来。"""
import pathlib

root = pathlib.Path(r"D:\WorkBuddy\personal-workspace\failbound-final")

edits = {
    "tech-report.html": [
        ("<strong>科研 Agent 的能力边界标定</strong>　|　赛道 04 AI Agent　|　验证场：材料信息学",
         "<strong>科研 Agent 的能力边界标定</strong>　|　材料信息学 · 贝叶斯优化"),
        ("<dt>参赛形式</dt><dd>单人 · 团队 Verity　·　2026-10-23 定稿　·　赛道 04 AI Agent</dd>",
         "<dt>项目形式</dt><dd>单人研究　·　2026-10-23 定稿</dd>"),
        ("人（Verity）", "人（研究者）"),
        ("由参赛者手工完成", "由研究者手工完成"),
        ("所有科学判断与实验由参赛者本人做出", "所有科学判断与实验由研究者本人做出"),
    ],
    "sandbox-log.html": [
        ("| 赛道 04 AI Agent | 2026-10-23", "| 材料信息学 | 2026-10-23"),
    ],
}

for name, pairs in edits.items():
    p = root / name
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        n = t.count(old)
        print(f"{name}: {n} × {old[:36]}")
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")

print("done")
