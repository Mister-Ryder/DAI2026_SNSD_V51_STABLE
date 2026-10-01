# v5.1 冻结版最终结果综合（C3）

## 一句话结论

v5.1 的 **Phased/Alternating（AlternatingConflictBenefit）** 是当前冻结协议下
平均表现最好的 learned 配置：原 8 seeds 平均增益 **38,805.5**，独立 fresh
8 seeds 为 **32,963.8**，描述性 pooled 16 seeds 为
**35,884.6 ± 6,026.4**。不过它相对 Full 仅在 **10/16** seeds 获胜，
相对 Shuffled 仅在 **12/16** seeds 获胜，因此应写成“平均表现最佳配置”，
不能写成“对所有通信协议普遍占优”。

当前最强且最稳定的证据不是协议排序，而是结构化联盟联合动作 B：
`B|A = +20,214.1 ± 5,685.9`、`B|¬A = +16,424.2 ± 5,095.9`，
两者均为 **16/16 seeds 正向**。这直接支持论文的核心论点：通信必须通过
边界图联合删弧—替换—子图回填来改变可行联合动作域，而不能只改变下一轮
局部搜索锚点。

## 证据层级

1. **主要性能证据：v5.1 Learned。** 使用 validation-selected checkpoint，
   正式评估时冻结；所有 learned 运行均为 `oracle_mode=false`。
2. **确认性证据：fresh 8 seeds。** seeds 为
   `101/202/303/404/505/606/707/808`，在该矩阵启动前写入清单并固定；
   它是确认性判断来源。
3. **描述性证据：pooled 16 seeds。** 合并原始 seeds
   `11/22/33/44/55/66/77/88` 与 fresh seeds，用于报告效应大小与配对符号，
   不能替代 fresh confirmatory 结论。
4. **机制证据：独立 Oracle 矩阵。** Oracle 仅用于诊断机制是否具有产生新解
   的能力，不是上界、不是 learned 策略结果，也没有参与 learned 运行时决策。

## 完整实验协议与冻结审计

| 项目 | 冻结协议 |
|---|---|
| 数据与规模 | 完整 C3，non-smoke；6 个分组智能体 |
| 每次运行 | 12 个 negotiation steps × 4 个 macro rounds × 6 agents = **288 次付费 worker 调用** |
| 方法矩阵 | 8 个配置；原 learned、fresh learned、Oracle 各 **64/64** 完成 |
| Learned checkpoint | 456,075 参数；SHA-256 `a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59` |
| Checkpoint 使用 | 原 learned 与 fresh learned 均为 strict load；各 64/64 hash 一致、64/64 `oracle=false` |
| 配对公平性 | C0 与 CA 从同一状态配对评估；相同 seeds、worker 预算和正式选择器 |
| 语义日程 | Full/FrozenComm：24 次 real；A0 cells：24 次 zero；Phased：12 conflict-only + 12 benefit-only；OneShot：4 real + 20 zero；Shuffled：24 shuffled |
| 审计结果 | 缺失、意外、重复 cell 均 0；预算与额外 worker 调用违规均 0；语义日程违规 0；未接受 CA 污染 0 |

因此三套正式矩阵的已注册审计均为 **0 违规**。论文中可以明确写“所有矩阵
使用相同 288-query 预算并通过冻结协议审计”，但不能把 Oracle 的运行结果
混入 learned 性能表。

## Learned 主结果

| 配置 | 原 8 seeds：gain ± SD | Fresh 8 seeds：gain ± SD | Pooled 16 seeds：gain ± SD |
|---|---:|---:|---:|
| **Phased/Alternating** | **38,805.5 ± 5,752.8** | **32,963.8 ± 5,022.8** | **35,884.6 ± 6,026.4** |
| Full simultaneous | 36,630.4 ± 7,956.1 | 31,149.0 ± 5,115.7 | 33,889.7 ± 7,054.4 |
| Shuffled | 35,894.3 ± 8,015.7 | 32,725.3 ± 6,467.9 | 34,309.8 ± 7,223.9 |
| OneShot | 34,777.9 ± 7,994.1 | 27,370.0 ± 5,702.9 | 31,073.9 ± 7,722.3 |
| SelectionOnly（A0B1） | 34,133.1 ± 5,945.2 | 27,569.0 ± 7,530.3 | 30,851.1 ± 7,378.8 |
| FrozenAnchorComm（A1B0） | 15,106.4 ± 6,545.1 | 12,244.9 ± 4,849.4 | 13,675.6 ± 5,757.6 |
| FrozenAnchorSelectionOnly（A0B0） | 16,416.1 ± 4,987.8 | 12,437.6 ± 5,277.3 | 14,426.9 ± 5,369.1 |
| NoPeerFinalShield | 16,423.0 ± 4,983.1 | 12,437.6 ± 5,277.3 | 14,430.3 ± 5,368.4 |

Phased/Alternating 的原 8 seeds 平均最终目标为 **6,121,118.6**，fresh
8 seeds 为 **6,116,224.0**。Fresh 上其均值仍是第一，但只比 Shuffled 高
238.5，因此“确认平均排名”与“确认普遍显著优于 Shuffled”是两件不同的事；
后者当前不成立。

## 配对效应：论文最重要的结果表

| Learned 配对量 | Fresh confirmatory | Pooled 16 seeds | 正确解释 |
|---|---:|---:|---|
| B｜A | +18,904.1 ± 4,094.6，8/8 | **+20,214.1 ± 5,685.9，16/16** | 最强主效应；结构化联盟在有语义时稳定有效 |
| B｜¬A | +15,131.4 ± 6,837.8，8/8 | **+16,424.2 ± 5,095.9，16/16** | 最强主效应；无语义时仍稳定有效 |
| A｜B | +3,580.0 ± 3,858.2，7/8 | **+3,038.6 ± 5,773.6，12/16** | 边缘/补充证据，不是普遍正向 |
| A｜¬B | -192.8 ± 5,395.7，4/8 | -751.3 ± 5,489.2，6/16 | A 脱离 B 后无独立效果 |
| Phased − SelectionOnly | +5,394.8 ± 5,278.1，7/8 | **+5,033.6 ± 5,072.7，14/16** | 支持语义调度相对 A0B1 的贡献 |
| Phased − OneShot | **+5,593.8 ± 3,918.3，8/8** | **+4,810.7 ± 4,742.8，14/16** | 支持持续分阶段通信，不只是一次通信 |
| Phased − Full | +1,814.8 ± 4,908.2，5/8 | +1,994.9 ± 4,759.3，**10/16** | 仅能说平均更高，不能说普遍优越 |
| Phased − Shuffled | +238.5 ± 6,402.8，5/8 | +1,574.9 ± 5,799.1，**12/16** | 方向为正但证据不充分 |

Pooled 中，B 的两个精确双侧 sign test 与 sign-flip mean test 均为
`p=3.05e-5`。Phased−SelectionOnly 的两个检验分别为 `p=.0042/.0017`，
Phased−OneShot 为 `p=.0042/.0019`。相比之下，`A|B` 为
`p=.0768/.0531`，因此应称为 **marginal complementary evidence**；
Phased−Full 为 `p=.4545/.1171`，Phased−Shuffled 为 `p=.0768/.2962`，
均不支持“普遍优越”。

## Oracle 机制诊断（必须与 Learned 分开）

| Oracle 指标，原 8 seeds | mean ± SD | 正/负 seeds |
|---|---:|---:|
| Phased/Alternating gain | 34,202.0 ± 5,874.2 | -- |
| Full gain | 35,589.1 ± 4,433.3 | -- |
| OneShot gain | 36,860.0 ± 3,965.6 | -- |
| B｜A | +17,428.5 ± 5,358.2 | 8/0 |
| B｜¬A | +17,994.8 ± 4,135.2 | 8/0 |
| A｜B | +2,307.0 ± 6,822.9 | 4/4 |

Oracle 同样复现了 B 在两个条件下 8/8 正向，因而支持“联盟联合动作本身
具备稳定机制价值”。但是 Oracle 的最高配置是 OneShot，而 Learned 的最高
配置是 Phased；这不是矛盾，因为 Oracle 是解析式机制脚手架，**不是全局最优
上界**。论文不能把 Oracle 目标值称为 upper bound，也不能把 Oracle 排名用于
证明 learned 排名。

## 结果解释

### 创新点 1：结构化三图联合动作得到强验证

分组会压缩全图搜索空间，标量通信又会继续丢失边界关系。v5.1 不再让通信
只产生一个低维搜索锚点，而是让子图保留私有候选，让边界联系图组织跨智能体
删弧—替换联盟，再由子图执行回填、全图执行精确可行性与目标验证。
`B|A` 与 `B|¬A` 均为 16/16 正向，说明收益并不依赖某一种消息打分器，
而来自联合边界动作对可行候选域的实质扩张。这应是论文最核心的贡献与结果。

### 创新点 2：分阶段冲突—收益通信具有增量价值，但需要限定

Phased/Alternating 将冲突与收益信息分阶段消费，使联合优化先处理可行化压力，
再处理价值回填。它相对 OneShot 与 SelectionOnly 的 pooled 优势均超过
4,800，且均为 14/16 正向；fresh 上相对 OneShot 更是 8/8 正向。这支持
“持续的分阶段通信优于一次性通信或无语义选择”。但其相对 Full 和 Shuffled
的证据较弱，因此创新点应写成 **phased semantic coordination complements
the coalition operator**，不能写成所有 seed 上都能识别最优语义顺序。

### A 因子的边界

`A|B=+3,038.6`、12/16 表明语义信息在联盟动作存在时有边缘增益；
`A|¬B=-751.3` 表明语义本身不会自动转化为改进。这个结果反而强化方法动机：
通信表示必须和可执行的联合边界动作耦合。A 的作用应表述为“依赖 B 的补充
贡献”，而非独立、稳定的主效应。

## 论文可写与不可写边界

### 可以写

- v5.1 使用冻结的 validation-selected learned checkpoint，并在启动前固定的
  fresh seeds 上完成独立确认性测试。
- Phased/Alternating 在原始、fresh、pooled 三个视图中均取得最高平均增益。
- 联盟联合动作 B 在有无 learned semantics 时均产生大幅且 16/16 正向的
  配对提升，是最稳定的核心贡献。
- Phased 相对 SelectionOnly 和 OneShot 分别提升 5,033.6 和 4,810.7，
  两者均为 14/16 正向。
- A 在 B 下提供 +3,038.6、12/16 的边缘补充证据。
- 三套正式矩阵均使用 288 次 worker 调用，且注册审计 0 违规。

### 不可以写

- 不能写 Phased 普遍优于 Full；实际为 **10/16**。
- 不能写 Phased 普遍优于 Shuffled；实际为 **12/16**，且 fresh 均值差仅
  238.5。
- 不能写 A 是独立稳定主效应；`A|¬B` 的 pooled 均值为负。
- 不能把 Oracle 写成 upper bound、learned performance 或运行时教师。
- 不能把 pooled 16 seeds 代替 fresh confirmatory 的证据层级。
- 不能从单一 C3 实例推断跨数据集、跨场景泛化，也不能宣称全局最优。

## 推荐论文主线

主结果顺序建议固定为：

1. 先报告 fresh confirmatory 与 pooled performance，说明 Phased 是平均最优配置；
2. 再用 `B|A` 与 `B|¬A` 的 16/16 结果证明结构化联合边界动作；
3. 用 Phased−OneShot、Phased−SelectionOnly 证明持续分阶段通信的附加价值；
4. 主动报告 Phased−Full 10/16、Phased−Shuffled 12/16 和 `A|B` 的边缘性；
5. 最后单列 Oracle 机制诊断和 0 违规审计，避免任何 learned/Oracle 混淆。

这种叙事既能突出两个实质创新点，也不会让审稿人通过 Shuffled、OneShot 或
Oracle 口径反向质疑论文结论。
