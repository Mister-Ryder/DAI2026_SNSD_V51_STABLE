# SNSD v5.1 冻结版交接说明

## 1. 本轮冻结结论

本轮方法开发冻结在 **SNSD v5.1**。不再继续 v5.2，也不再追加新的搜索或训练分支。论文建议把方法写成一个统一的三图结构化联合优化协议，并把 `AlternatingConflictBenefit` 称为 **SNSD-v5.1 phased / best-performing configuration**。

这个名称只表示它在 original 8 seeds 和独立 fresh 8 seeds 上都取得最高平均增益，不表示它在每个 seed 上都优于 simultaneous Full 或 Shuffled。

## 2. 冻结方法包含什么

1. **子图层**保留智能体的真实候选、局部冲突、删除和回填响应，不把局部搜索空间再次压缩为少量标量。
2. **边界图层**交换候选 ID、边界弧、冲突/替代/互补关系、收益、代价和可执行响应书。
3. **全图层**计算残差、验证完整可行性，并在 `current/J/R0/Rm/C0/CA` 的精确候选并集中作最终裁决。
4. **同状态反事实保护**在相同输入状态下分别生成 zero-semantic `C0` 和 treated-semantic `CA`。`C0` 始终作为 B-enabled 的安全候选；`CA` 只有严格优于所有安全候选时才进入后续域。
5. **反事实微轮**对已经使用的边界端点和 refill 施加 no-good/排序反馈，使每个 coordination cycle 产生多个不同响应，而不是重复同一个候选。
6. **Phased communication**在持续协商中交替使用 conflict-only 和 benefit-only 语义，避免每一步同时混合两种信号。

精确候选并保证加入通信候选不会降低当前已知最好目标。全部 v5.1 协调均不增加 paid worker calls。

## 3. 核心代码

- `SNSD_V51_FINAL/src/snsd_v51/boundary_semantic.py`：C0/CA 同状态反事实、差异化微轮和结构化响应 portfolio。
- `SNSD_V51_FINAL/src/snsd_v51/semantic_safeguard_protocol.py`：八种方法协议、strict semantic gate、exact union 和事件记录。
- `SNSD_V51_FINAL/run.py`：正式运行入口。
- `SNSD_V51_ABLATIONS/scripts/launch_oracle_64.sh`：Oracle v2 完整矩阵。
- `SNSD_V51_ABLATIONS/scripts/launch_learned_original_64.sh`：Learned v2 完整矩阵。
- `SNSD_V51_ABLATIONS/scripts/launch_learned_fresh_64.sh`：独立 fresh-seed Learned 完整矩阵。
- `SNSD_V51_ABLATIONS/analysis/summarize.py`：事件级调度、预算、污染和完整性验证。
- `SNSD_V51_ABLATIONS/analysis/finalize_audit.py`：Learned checkpoint 最终合同审计。
- `SNSD_V51_ABLATIONS/analysis/paired_version_analysis.py`：逐 seed 配对、exact sign test 和 sign-flip mean test。

## 4. 已完成正式实验

| Matrix | Seeds × methods | Runs | Calls/run | Status |
|---|---:|---:|---:|---|
| v5.1 Oracle v2 | 8 × 8 | 64 | 288 | audited pass |
| v5.1 Learned v2 | 8 × 8 | 64 | 288 | audited pass |
| v5.1 Learned fresh confirmation | 8 × 8 | 64 | 288 | audited pass |

合计 192 个 v5.1 正式 runs、55,296 次 paid worker calls。三个矩阵均为完整 C3、`d12 × m4`；通信 schedule、预算、未接受 CA 污染和额外 worker calls 的违规数均为 0。两套 Learned 矩阵均严格加载同一个 validation-selected checkpoint，`oracle=false`。

## 5. 最终 Learned 结果

| Configuration | Original 8 gain | Fresh 8 gain | Pooled 16 gain |
|---|---:|---:|---:|
| **Phased / Alternating** | **38,805.5 ± 5,752.8** | **32,963.8 ± 5,022.8** | **35,884.6 ± 6,026.4** |
| Full simultaneous | 36,630.4 ± 7,956.1 | 31,149.0 ± 5,115.7 | 33,889.7 ± 7,054.4 |
| Shuffled | 35,894.3 ± 8,015.7 | 32,725.3 ± 6,467.9 | 34,309.8 ± 7,223.9 |
| OneShot | 34,777.9 ± 7,994.1 | 27,370.0 ± 5,702.9 | 31,073.9 ± 7,722.3 |
| SelectionOnly ($A=0,B=1$) | 34,133.1 ± 5,945.2 | 27,569.0 ± 7,530.3 | 30,851.1 ± 7,378.8 |
| A0B0 | 16,416.1 ± 4,987.8 | 12,437.6 ± 5,277.3 | 14,426.9 ± 5,369.1 |

Phased pooled mean final objective 为 **6,118,671.3**。

## 6. 最有说服力的多智能体贡献

- `B given A = +20,214.1 ± 5,685.9`，16/16 seeds 正向。
- `B without A = +16,424.2 ± 5,095.9`，16/16 seeds 正向。
- Phased − SelectionOnly = `+5,033.6 ± 5,072.7`，14/16 正向。
- Phased − OneShot = `+4,810.7 ± 4,742.8`，14/16 正向。
- `A given B = +3,038.6 ± 5,773.6`，12/16 正向，是方向性而非决定性证据。

因此论文的核心论点应是：**高维边界结构、联合联盟执行和持续的分时通信共同产生有效的跨智能体影响传播。** 不应把贡献缩减为 learned scalar credit 的精确排序。

## 7. Oracle 的正确位置

Oracle Full gain 为 `35,589.1 ± 4,433.3`；`B given A` 和 `B without A` 分别为 `+17,428.5` 与 `+17,994.8`，均 8/8 正向。

Oracle 是 analytic mechanism-diagnostic scaffold，不掌握未查询候选，也不是全局 upper bound。它只能用于说明机制是否具备潜在作用，不能用于宣称 Learned 距离最优解的差距。

## 8. 权威结果文件

- `SNSD_V51_ABLATIONS/results/formal/V51_ORACLE_V2_FORMAL_AUDITED_SUMMARY.json`
- `SNSD_V51_ABLATIONS/results/formal/V51_ORACLE_V2_FORMAL_AUDITED_REPORT.md`
- `SNSD_V51_ABLATIONS/results/formal/V51_LEARNED_V2_FORMAL_AUDITED_SUMMARY.json`
- `SNSD_V51_ABLATIONS/results/formal/V51_LEARNED_V2_FORMAL_AUDITED_REPORT.md`
- `SNSD_V51_ABLATIONS/results/formal/V51_LEARNED_CONFIRMATORY_FRESH_SEEDS101_808_SUMMARY.json`
- `SNSD_V51_ABLATIONS/results/formal/V51_LEARNED_CONFIRMATORY_FRESH_SEEDS101_808_REPORT.md`
- `SNSD_V51_ABLATIONS/results/formal/V51_LEARNED_CONFIRMATORY_FRESH_SEEDS101_808_PREREGISTRATION.tsv`
- `SNSD_V51_ABLATIONS/results/formal/v51_final_formal_paired_analysis.json`
- `SNSD_V51_ABLATIONS/results/formal/V51_FINAL_FORMAL_PAIRED_ANALYSIS.md`
- `docs/V51_FORMAL_VALIDATION_REPORT_ZH.md`
- `paper/V51_PAPER_READY_TABLES.tex`

第一版无 `_V2_` 标识的 v5.1 Oracle 文件继承了错误的 OneShot 调度，只能用于保留开发历史，**不得引用其 OneShot、Full−OneShot 或方法排名**。

## 9. 服务器权威运行目录

- Oracle v2：`/root/autodl-tmp/DAI2026_SNSD_V15/runs/v51_semantic_safeguard_oracle_scale_d12m4_v2`
- Learned v2：`/root/autodl-tmp/DAI2026_SNSD_V15/runs/v51_semantic_safeguard_learned_selected_formal_d12m4_v2`
- Fresh confirmation：`/root/autodl-tmp/DAI2026_SNSD_V15/runs/v51_learned_confirmatory_seeds101_808_d12m4`

如需复现，使用上述三个 launcher，并将输出目录改成一个全新的目录；现有 launcher 会拒绝覆盖正式结果。

## 10. 留给后续版本而不属于本轮的事项

- 针对 `CA-C0` 或 coalition delta 重新端到端训练 message model。
- 在 W3 或其他独立实例上验证跨实例泛化。
- 设计稳定击败 Shuffled 的语义对齐训练目标。
- 继续提升最终目标值或对接更强的局部 worker。

这些事项不影响 v5.1 当前作为一篇分布式多智能体联合优化论文方法与实验主体的完整性，但必须在论文 limitations 中承认单一 C3 实例的范围限制。
