# WitnessFlow 计算版本 v1 · 冻结说明（2026-10-09）

- 主张定稿见 `design/witnessflow/claims_evidence_final.md`。
- 冻结后不再追加实验。湿实验（真实分管染色）不在本版本范围内。

## 1. 文件清单与哈希

- `MANIFEST.tsv`：代码、协议、测试、结果、图表数据、外部输入（原始 FCS 与 La Jolla 快照库 `ljcr_snapshot_db`）的 sha256。由 `witnessflow/tools/freeze_manifest.py` 生成。
- `ENV.txt`：Python 3.13.11（Anaconda）、numpy 2.4.2、scipy 1.16.3、pandas 2.2.3、scikit-learn 1.9.0、numba 0.64.0、matplotlib 3.10.8；Windows 11。

## 2. 输入数据

| 数据 | 来源 | 校验 |
|---|---|---|
| Levine-32（`data/raw/Levine32/`） | GitHub lmweber/benchmark-data-Levine-32-dim | sha256 记录在协议与 MANIFEST 中 |
| WINGMEN（`data/raw/WINGMEN/`，54 个 FCS） | figshare 30174547，CC BY 4.0 | MD5 54/54 与 figshare 一致（`tools/verify_md5.py`），sha256 记录在 MANIFEST 中 |
| CytoNorm 示例（`data/raw/CytoNorm/`） | GitHub saeyslab/CytoNorm inst/extdata | sha256 记录在 MANIFEST 中 |
| La Jolla 覆盖表 | `ljcr_snapshot_db`（只读，La Jolla v1.2 快照） | sha256 记录在 MANIFEST 中 |

## 3. 协议与偏离记录

| 协议 | 说明 |
|---|---|
| `levine32_v1` | 冻结于揭盲前；阈值规则有缺陷，保留用于复现 |
| `levine32_v2` | 揭盲后修订了阈值规则，已注明 |
| `wingmen_v1` | 冻结于看到表达值之前；`e3` 节是事后添加的，属于探索性 |
| 压力测试变体 | bb6 / bb4、rareB_*、shift* 系列 |
| 补实验规格 | `design/witnessflow/exp_protocols_v1.md` 的 v2 部分 |

已知偏离：
- e5b 逐供者网格用 180 而不是 195（规格 v2 晚于运行启动），所以逐供者结果属于探索性；
- e5b 的 ε 是机制展示阈值，事后选定；
- e5c 的家族是人为的 5 槽位约束。

## 4. 版本身份

- **校准**：当前版本是固定整数种子的重跑（覆盖率 0.992–1.000）。旧版本用 hash 种子，覆盖率 0.989–1.000，只作历史记录。
- **e5d**：当前版本是固定检验家族的汇总（`e5d_fixed_family_summary.tsv`）。`e5d_unified_summary.tsv`（按当次可检查项目数校正）以及各 worker 的原始汇总，都是历史中间版本。
- **v2 回放**：`results/levine32_v2/initial_audit.tsv` 与 `followup.tsv` 是随机数分离修复之后的版本。第 1–2 天日志里的数字来自更早的运行。
- **e3b / e3c / e3d**：早于随机数修复，只作探索记录；对应结论已被 e3e、e5f、e5i 取代。
- **bb6 / bb4 压力测试**：早于随机数修复和逐管检查。

## 5. 逐目标账本的实际保存范围（`results/*/ledger/`）

已保存：
- 需求向量：`*_requirements.npz`，包含 v_lp、lp_true 和 3 份打乱；
- 全部优化后面板：`*_panels.json`，共 72 份；
- 全部配对汇总：`ledger_summary.tsv`；
- 逐目标明细：**只有真实需求、alloc 0** 的 `*_targets.tsv`，共 18 张。

其余分配和打乱对照的逐目标明细**没有保存**，可以确定性重建：
1. 读取对应的 `*_panels.json`；
2. 在 `e5j_ledger.py` 里用相同的 `ALLOC` 种子（`1000*a + r`）重算 `well_counts`；
3. 再用 `per_target()` 得到逐目标结果。

## 6. 复现

见 `REPRODUCE.md`。

## 7. 图表

见 `FIGURE_MAP.md`。源数据由 `witnessflow/src/make_figures.py` 生成，放在 `witnessflow/figures/data/`。

2026-10-09：暂不作图，已删除 4 张草图（`figures/png/`）。脚本默认只输出数据表；需要作图时运行 `python src/make_figures.py --png`。
