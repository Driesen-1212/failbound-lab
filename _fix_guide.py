# -*- coding: utf-8 -*-
"""guide.html：本地私有路径换成公开仓库路径，页脚去掉赛事材料名。"""
import pathlib

p = pathlib.Path(r"D:\WorkBuddy\personal-workspace\failbound-final\guide.html")
t = p.read_text(encoding="utf-8")

pairs = [
    ("cd /d D:\\WorkBuddy\\personal-workspace\\AI4S_Hackathon\\_brainstorm_share",
     "cd /d D:\\failbound-lab"),
    ("set FAILBOUND_RESULTS=D:\\WorkBuddy\\personal-workspace\\AI4S_Hackathon\\unzipped\\agent_lab\\results",
     "set FAILBOUND_RESULTS=agent_lab\\results"),
    ("<footer>FailBound-Lab 讲解册 · HTML 图文版 · 2026-11-17 · 配套：《决赛答辩台本》《决赛文案包_终版》《一页总账》</footer>",
     "<footer>FailBound-Lab 讲解册 · HTML 图文版 · 2026-11-17</footer>"),
]
for old, new in pairs:
    print(t.count(old), "×", old[:40])
    t = t.replace(old, new)

p.write_text(t, encoding="utf-8")
print("left AI4S_Hackathon:", t.count("AI4S_Hackathon"), "| 决赛:", t.count("决赛"))
