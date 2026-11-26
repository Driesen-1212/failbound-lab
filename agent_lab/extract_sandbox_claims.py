# -*- coding: utf-8 -*-
"""从 Replay 全量消息中固化沙箱承诺物 → results/sandbox_claims.json"""
import json
import os
import re

ROWS = os.environ.get("REPLAY_JSON", "C:/Users/86137/AppData/Local/Temp/replay_all.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sandbox_claims.json")

rows = json.load(open(ROWS, encoding="utf-8"))

# 1. [done] 行观测（任务参数：n_pool 与 thr，门控/无门控各自出现）
pat = re.compile(r"\[done\]\s+(\S+)\s+n_pool=\s*(\d+)\s+thr=([+\-0-9.naninf]+)")
done_obs = {}
for r in rows:
    c = r.get("content")
    txt = ""
    if isinstance(c, dict):
        txt = str(c.get("output", "")) + str(c.get("content", ""))
    elif isinstance(c, str):
        txt = c
    for m in pat.finditer(txt):
        t, np_, thr = m.group(1), int(m.group(2)), float(m.group(3))
        done_obs.setdefault(t, set()).add((np_, thr))
done_obs = {t: sorted(v) for t, v in done_obs.items()}

# 2. 交付承诺（12:17:25 交付消息 + 12:15:32 sha256sum 原始输出）
claims = {
    "thread_id": "1006bac1-87b2-40ab-8203-f86a542df976",
    "share_url": "https://sciencepal.ai/share/1006bac1-87b2-40ab-8203-f86a542df976",
    "n_messages": len(rows),
    "files": {
        "runs_gate.jsonl": {"lines": 2179, "runs": 2160, "task_meta": 18, "arm": "main", "gate": True, "max_spent": 205.0, "seconds": 2251, "sha256": "816326b73982c4889549bbe72334ceff6dea78c7de5e0cc0dbf29b475d417a41"},
        "runs_nogate.jsonl": {"lines": 2179, "runs": 2160, "task_meta": 18, "arm": "main", "gate": False, "max_spent": 200.0, "seconds": 4686, "sha256": "1b1e201090f75b8dc1602f64279f8dd079287dbaea37a9d89e1cc28670c64966"},
        "runs_gate_ablate.jsonl": {"lines": 2179, "runs": 2160, "task_meta": 18, "arm": "ablate", "gate": True, "max_spent": 204.0, "seconds": 2138, "sha256": "cd7b343409bb6e7307492e0f8afe637ba061749b76db6e566e20c190466dd365"},
        "runs_nogate_ablate.jsonl": {"lines": 2179, "runs": 2160, "task_meta": 18, "arm": "ablate", "gate": False, "max_spent": 200.0, "seconds": 3786, "sha256": "d056ae2ad2da59c1337b0ca21fc49ac7889962967735d28191ee650ebc32255b"},
    },
    "package_sha256": "69c79a06826ce7bd08e49686e7291114327b9667737cb4d8357c0190cb7cc2d1",
    "run_logs_sha256": {
        "run1_nogate_main.log": "9c186ec88867154ffe8bc23e65244b93b788bdb19673de393606f8b5701dfaac",
        "run2_gate_main.log": "e8b8ce74ee7a3a4ba4c53d22a5a874f5d059fb506b30a9a016dce6de0f3dfcdc",
        "run3_nogate_ablate.log": "5d997d07edd0020bbe132c2c49ddcabcf3d785b2f01e0a6c75e166c73892229b",
        "run4_gate_ablate.log": "b0e77a59aefd3278adb44281513f687281c44f782100456a86340e8f75d5ba6f",
    },
    "done_line_observations": done_obs,
    "provenance": {
        "claims_message": "Replay assistant message 2026-10-17T12:17:25Z（沙箱最终交付，含 4 个 SHA256）",
        "sha256sum_output": "Replay tool output 2026-10-17T12:15:32Z（沙箱执行 sha256sum 的原始输出，未截断）",
        "done_lines": "Replay 各工具输出中 run_experiments.py 打印的 [done] 行（含 n_pool 与 thr）",
        "integrity_gate": "Replay tool output 2026-10-16T07:16:16Z（sha256sum -c SHA256SUMS.txt，31 文件全 OK）",
        "config_fix": "Replay 2026-10-16T07:17:26Z（沙箱对第一个包内 config.py 的路径修复；四组全量改用第二个包，其 config.py 与本地当前版本逐比特相同）",
    },
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(claims, f, ensure_ascii=False, indent=2)
print("written:", OUT)
print("done_line 观测任务数:", len(done_obs), "| 观测点总数:", sum(len(v) for v in done_obs.values()))
