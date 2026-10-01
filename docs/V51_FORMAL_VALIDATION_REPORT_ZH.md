# v5.1 正式结果验证报告

## 总体判断：可在明确限制条件下写入论文

截至 2026-08-02，v5.1 的实现、原始 8-seed 矩阵和独立 fresh-seed 矩阵已经形成完整、可追溯的证据链。最稳定的结论是：**结构化联盟 delete--replace--refill 执行器（factor B）带来大幅且逐 seed 一致的增益；持续的冲突--收益分时通信相对零语义和 OneShot 也有稳定增益。**

当前不应声称精确 learned semantic ordering 已稳定击败 Shuffled，也不应把 analytic oracle 描述为性能上界。v5.1 AlternatingConflictBenefit 是两个 seed 集上均值最高的配置，可作为论文中的 best-performing configuration，但不能表述为在所有 seeds 或所有通信对照上显著占优。

## 验证对象与实验口径

- 数据：完整 C3 单实例。
- Learned 原始矩阵：seeds `11,22,33,44,55,66,77,88`，8 methods，共 64 runs。
- Learned 独立确认矩阵：实验启动前固定 seeds `101,202,303,404,505,606,707,808`，8 methods，共 64 runs。
- Oracle 机制诊断矩阵：原始 8 seeds，8 methods，共 64 runs。
- 每个 run：4 个宏轮、每轮 12 步、共 288 次 paid worker calls。
- Learned checkpoint：64/64 严格加载，456,075 parameters，`missing_keys=[]`，`unexpected_keys=[]`，`oracle=false`。
- 事件级协议审计：两套 Learned 矩阵和 Oracle v2 均满足 `schedule/budget/unaccepted-CA-pollution/additional-worker-call violations = 0`。

权威输入：

- `V51_LEARNED_V2_FORMAL_AUDITED_SUMMARY.json`
- `SNSD_V51_ABLATIONS/results/formal/V51_LEARNED_CONFIRMATORY_FRESH_SEEDS101_808_SUMMARY.json`
- `V51_ORACLE_V2_FORMAL_AUDITED_SUMMARY.json`
- `v51_final_formal_paired_analysis.json`
- `SNSD_V51_ABLATIONS/results/formal/V51_LEARNED_CONFIRMATORY_FRESH_SEEDS101_808_PREREGISTRATION.tsv`

## 关键计算复核

所有差值均从 `per_seed` 行重新配对计算，而不是用两个汇总均值相减。复核脚本为 `SNSD_V51_ABLATIONS/analysis/paired_version_analysis.py`，输出为 `v51_final_formal_paired_analysis.json` 和 `V51_FINAL_FORMAL_PAIRED_ANALYSIS.md`。

### 1. Learned 最优配置

| Seed set | Alternating gain | Full gain | Shuffled gain |
|---|---:|---:|---:|
| Original 8 | 38,805.5 ± 5,752.8 | 36,630.4 ± 7,956.1 | 35,894.3 ± 8,015.7 |
| Fresh 8 | 32,963.8 ± 5,022.8 | 31,149.0 ± 5,115.7 | 32,725.3 ± 6,467.9 |
| Descriptive pooled 16 | **35,884.6 ± 6,026.4** | 33,889.7 ± 7,054.4 | 34,309.8 ± 7,223.9 |

Alternating 在两个独立 seed 集上均取得最高平均增益，因此可标记为 best-performing configuration。它相对 Full 的 pooled 配对差为 `+1,994.9 ± 4,759.3`，但仅 10/16 seeds 为正；相对 Shuffled 为 `+1,574.9 ± 5,799.1`，12/16 seeds 为正。两者均不足以支持“普遍优于”或“稳定显著优于”的表述。

### 2. 联盟联合执行（factor B）

| Pooled 16-seed effect | Mean ± SD | Positive seeds | Exact two-sided sign p |
|---|---:|---:|---:|
| `B given A` | **+20,214.1 ± 5,685.9** | **16/16** | 0.0000305 |
| `B without A` | **+16,424.2 ± 5,095.9** | **16/16** | 0.0000305 |

这是当前最强、最无歧义的实验证据。B 在有无 semantic factor A 时均逐 seed 为正，说明联合 delete--replace--refill 不是由某个 learned seed 或偶然排序造成的。

### 3. 持续分时通信

| Pooled 16-seed contrast | Mean ± SD | Positive seeds | Exact sign p |
|---|---:|---:|---:|
| Alternating − SelectionOnly | **+5,033.6 ± 5,072.7** | **14/16** | 0.0042 |
| Alternating − OneShot | **+4,810.7 ± 4,742.8** | **14/16** | 0.0042 |
| Alternating − Full simultaneous | +1,994.9 ± 4,759.3 | 10/16 | 0.4545 |
| Alternating − Shuffled | +1,574.9 ± 5,799.1 | 12/16 | 0.0768 |

因此可以写：持续的 conflict--benefit phased communication 明显优于零语义和一次性通信。不能写：精确 learned 语义已经稳定优于置乱语义，或 Alternating 在所有 seeds 上优于 simultaneous Full。

### 4. Factor A 与交互

- `A given B = +3,038.6 ± 5,773.6`，12/16 为正，exact sign `p=0.0768`。
- `A without B = -751.3 ± 5,489.2`，6/16 为正。
- `A × B interaction = +3,789.9 ± 7,581.2`，12/16 为正，exact sign `p=0.0768`。

这些结果支持“语义通信需要结构化联盟执行器才能发挥作用”的方向性解释，但证据强度低于 factor B，不宜使用确定性因果语言。Alternating 相对 SelectionOnly 的 14/16 结果比 simultaneous Full 的 A 主效应更适合支撑分时通信贡献。

### 5. Oracle 机制诊断

- Oracle Full gain：`35,589.1 ± 4,433.3`。
- Oracle `B given A = +17,428.5`，8/8 为正。
- Oracle `B without A = +17,994.8`，8/8 为正。
- Oracle OneShot gain：`36,860.0 ± 3,965.6`，高于 Oracle Full。

Oracle 只使用已揭示候选域上的解析因子、压力和机会成本，是 mechanism-diagnostic scaffold，不知道未查询候选，也不是全局 upper bound。它验证 B 的执行机制，但同时说明解析语义持续注入仍可能扰动搜索路径。

## 发现的问题及影响

1. **高：单一 C3 实例。** 16 seeds 是同一实例上的随机性重复，不等于 16 个独立问题实例。因此论文结论必须限定为 full C3 case study，不作跨实例普适性宣称。
2. **中：Shuffled 接近 Alternating。** fresh seeds 上二者只差 `+238.5`，5/8 为正；这说明主要收益更可能来自高维结构化边界载荷、持续交互和联盟执行，而非 learned credit 的精确排序。
3. **中：最佳配置选择。** Alternating 在 original 和 fresh 两组均值都最高，但其相对 Full/Shuffled 的逐 seed优势不稳定。正文可将其作为 best-performing configuration，不能写 statistically dominant。
4. **中：现有 checkpoint 不是针对 v5.1 的 `CA-C0` 直接训练。** 当前结果证明已有 learned message model 可以接入新执行器，不证明端到端训练已经最优。
5. **低：Oracle 与 Learned 数值不可作上下界比较。** 两种策略源不同，应分表或分段报告。

## 论文可用结论

- Structure-preserving boundary coalitions yield large and seed-consistent improvements under both semantic conditions.
- Safeguarded same-state `C0/CA` evaluation prevents harmful semantic responses from polluting subsequent agent domains without additional worker calls.
- Sustained phased conflict--benefit communication outperforms zero-semantic and one-shot coordination across the pooled seeds.
- The phased configuration attains the highest mean gain in both the original and fresh seed sets, while its advantage over simultaneous or shuffled communication remains non-uniform.

## 论文禁止或必须弱化的结论

- 不得称 analytic oracle 为 upper bound。
- 不得声称 learned semantic ordering 显著击败 Shuffled。
- 不得声称 Alternating 对 Full 在所有 seeds 上更好。
- 不得把同一 C3 上的 seeds 写成多个独立 datasets/instances。
- 不得声称当前 checkpoint 是针对 v5.1 coalition delta 端到端训练得到的。

## 最终置信度

**Share with caveats。** 代码协议、预算、矩阵完整性和关键计算已经达到论文引用要求；factor B 与持续 phased-vs-zero/one-shot 的结论可信。方法的跨实例泛化和精确 learned semantic alignment 仍未验证，必须在论文限制段中显式说明。
