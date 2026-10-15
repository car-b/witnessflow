# 图表 × 源数据 × 指标定义（v1）

**通用定义**：
- **半宽达标**：一个三元组的 8 个模式各算一个 95% Clopper–Pearson 边际区间，取半宽最大值；≤ ε 记为达标。分母是全部目标三元组。这不是联合 95%。
- **实际误差达标**：8 个模式中与参照（reference 切分）频率之差的最大值 ≤ ε。参照本身也有抽样误差。
- **误差条**：除非另注，都是分配种子 × 设计种子之间的标准差。这些重复只是随机化重复，不是独立的生物学复现。

| 面板 | 内容 | 源数据（`figures/data/`） | 原始结果 | 备注 |
|---|---|---|---|---|
| Fig 1A | 格子问题与探索记录 | —（示意图，待绘制） | `design/witnessflow/scan_*.md` | 纪录数字取自 `tools/scan/main_table.tsv` |
| Fig 1B | 关联矩阵转置为码字，三个码字按位 AND | —（示意图） | — | 数学 |
| Fig 2A | PG(1,25) 上的有限几何圆，130×20 = 2600 | —（示意图） | `tools/scan/out_lat/lat02_structure.txt`；`tools/verify_s3626.py` | 要画关联示意，不要画欧氏圆 |
| Fig 2B | F₅ 二次插值的构造示意 | —（示意图） | `src/e5c.py` 的 `oa_design` | — |
| Fig 2C | 正交表 vs 退火 vs 均衡随机 vs 合法随机 | `fig2C_orthogonal_array.tsv` | `results/wingmen_v1/e5c_*.tsv` | 10 个分配种子；退火和随机各 3 个设计种子；人为约束 |
| Fig 2D | 低阶重复与门控降阶 | —（示意图） | `lat06_derived.txt` | — |
| Fig 3A | 密集节点上的半宽达标（锯齿） | `fig3A_sawtooth.tsv` | `results/*/e5b_steiner_nodes.tsv`、`e5b_r180.tsv` | 机制展示阈值：Levine T 0.06 / B 0.08，WINGMEN CD4T 0.05 / CD8T 0.08；10 个分配种子 |
| Fig 3B | 各节点上 λ_T 的分布（WINGMEN 经典叠加） | `fig3B_multiplicity_wingmen.tsv` | 组合计算，精确值 | — |
| Fig 3C | 逐供者：两端最低值 − 180 节点 | `fig3C_per_donor_shape.tsv` | `results/wingmen_v1/e5b_per_donor.tsv` | 探索性（规格偏离）；ε_d = 1.1·ε*_d |
| Fig 3D | 拆管后计数差为零 | `fig3D_tube_split.tsv` | `results/wingmen_v1/e5b_tube_split.tsv` | 2 个父群体 × 10 个分配种子 |
| Fig 4A | 一条目标的证据账本（共测管 → 计数 → 区间 → 检查 → 需求） | —（示意图，取 `ledger/*_targets.tsv` 中的实例） | `results/*/ledger/` | — |
| Fig 4B | 一致性检查：只做单点 vs 单点 + 点对 | `fig4B_qc_fixed_family.tsv` | `results/*/e5d_fixed_family_summary.tsv` | 固定家族 M = r×21；空白 1000 次，其余每类 200 次 |
| Fig 4C | 经典 / 统一 / 打乱 / 真实需求 | `fig4C_attribution_shuffle.tsv` | `results/*/e5i_shuffle.tsv` | 真实与统一需求各 18 条记录（2 个初始设计 × 3 个优化种子 × 3 个分配）；打乱需求 54 条；经典基线取 3 个唯一分配（原表里重复记录 6 条，已去重）。误差条是这些随机化重复的标准差，不是独立重复 |
| Fig 4D | 分诊账本：新达标 / 失去，以及实际误差的取舍 | `fig4D_ledger.tsv` | `results/*/ledger/ledger_summary.tsv` | — |
| 补充 S1 | 阈值敏感性（设计） | `supp_threshold_design.tsv` | `e5g_threshold_design.tsv` | 两个节点 |
| 补充 S2 | 第五幕阈值敏感性 | `supp_act5_threshold.tsv` | `e5l_act5_threshold.tsv` | 偏移 ±0.25 |
| 补充 S3 | 管大小不均 | `supp_boundary_unequal.tsv` | `e5k_boundary.tsv`（A 部分） | 单一管数节点 |
| 补充 S4 | 组成偏移 | `supp_boundary_composition.tsv` | `e5k_boundary.tsv`（B 部分） | 每种模式 20 次 |
| 补充 S5 | 区间覆盖校准 | `supp_calibration.tsv` | `results/calibration/` | 40 个场景 × 1000 次 |
| 补充 S6 | e5e 阴性搜索 | — | `results/e5e_lambda2*/`（含完整块表） | — |
| 补充 S7 | 跨管偏移与真实批次幅度 | — | `results/levine32_v2_shift*/`、`results/cytonorm_real_batch/` | 模拟 + 幅度参照 |

> 2026-10-09：草图已删除，暂不作图；本表只保留面板与源数据的对应关系。
