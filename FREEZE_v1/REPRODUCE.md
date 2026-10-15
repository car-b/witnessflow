# 复现命令（按顺序；在 `witnessflow/` 下执行；Windows PowerShell 5.1，路径含中文时一律加引号）

0. 测试：`python tests/test_toy.py`，应显示 26 条 PASS、`FAILS: 0`。之前日志里写的"25/27 项"是计数错误，以本条为准。
   - 依赖：除 ENV.txt 所列之外，同构检查（`tools/scan/lat02_structure.py`）还需要 networkx 3.6.1。
   - `src/design.py` 里的 La Jolla 数据库路径 `LJCR_DB` 在冻结版中是本机绝对路径，换机复现时要修改这一行（公开版已改为可配置，见英文 README）。

1. 数据（已在本地，哈希见 MANIFEST）：
   - Levine：`python tools/fetch.py <url> data/raw/Levine32/Levine_32dim_notransform.fcs`
   - WINGMEN：`python tools/fetch_figshare.py 30174547 data/raw/WINGMEN`，然后 `python tools/verify_md5.py data/raw/WINGMEN`
   - CytoNorm：`tools/fetch.py` 逐个下载 inst/extdata 下的 6 个 FCS

2. 主回放（会生成 thresholds.json；后续所有脚本都依赖它）：
   - `python src/replay.py protocols/levine32_v2.json`
   - `python src/replay.py protocols/wingmen_v1.json`
   - 压力测试：`protocols/levine32_v2_rareB_{1,0p3,0p1}.json`、`levine32_v2_shift{0,Allp0p5,NonBBp0p5,NonBBm0p5}.json`
   - 汇总：`python src/summarize.py results/<protocol_id>`、`python src/compare_rare.py`、`python src/compare_shift.py`

3. 探索：`python src/explore.py <protocol> e1 e2`

4. 预算设计：
   - `python src/e3e.py <protocol>`（含 hybrid_LPaudit）
   - `python src/e5f_attrib.py <protocol>`

5. 补实验：
   - `python src/e5b.py <protocol>`（两个数据集）
   - `python src/e5b_supp.py protocols/wingmen_v1.json`
   - `python src/e5c.py protocols/wingmen_v1.json`
   - e5d：`python src/e5d.py <protocol> 1000 200 <k> 8`，k = 0..7 并行；然后依次运行 `python src/sum_e5d.py results/<pid>`、`python src/sum_e5d_fixed.py results/<pid> <r>`（Levine r=71，WINGMEN r=130）
   - e5e：`python src/e5e.py <anneal|breakout> 141 <seed> <seconds> <out_dir>`（种子 0–4 跑 600 s，种子 10–14 跑 1800 s）
   - 校准：`python src/e5h_calib.py <pairs|singles|pairs+direct|mixed|direct_only> 1000`
   - 阈值敏感性：`python src/e5g_threshold.py <protocol>`

6. 收尾实验（e5i–e5l）：
   - `python src/e5i_shuffle.py <protocol>`
   - `python src/e5j_ledger.py <protocol>`
   - `python src/e5k_boundary.py <protocol>`
   - `python src/e5l_act5_threshold.py <protocol>`

7. 证书：
   - `python tools/verify_s3626.py`
   - `tools/scan/lat02_structure.py`（同构）、`lat05*.py`（正交表 / 横截）、`lat06_derived.py`（导出设计）

8. 图表：`python src/make_figures.py`

9. 冻结清单：`python tools/freeze_manifest.py`

**说明**：
- 所有随机种子都写在各脚本顶部或调用参数里。
- e5d 的结果与并行进程数无关（每个重复各自使用 RNG）。
- e5e 按墙钟时间停止，结果可能随机器负载变化，不保证可逐位复现；完整块表与缺口已保存。
- 运行时长是本机墙钟时间（104 个逻辑核，进程默认在处理器组 0），不是 CPU 核时。
