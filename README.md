# DAI 2026 / SNSD-v5.1 稳定复现包

本目录是论文方法、正式证据和复现材料的冻结容器。当前算法为
**Semantic-Safeguarded Tri-Graph Joint Negotiation (SNSD-v5.1)**，论文主配置固定为
**AlternatingConflictBenefit（Phased）**。

## 项目边界

~~~text
DAI2026_SNSD_V51_STABLE/
├── SNSD_V51_FINAL/          # 唯一最终 Learned 方法、GNN 训练、C3、正式 checkpoint
├── SNSD_V51_ABLATIONS/      # 八方法、Oracle、正式矩阵与配对分析
├── paper/                   # 中文方法母稿与 ACM sigconf LaTeX
├── docs/                    # 正式验证、交接和重组报告
├── output/pdf/              # 已编译论文材料
├── tools/                   # 跨项目审计与 manifest 生成
└── manifest/                # 稳定包 SHA-256 清单
~~~

过程版本源码已移到：

~~~text
../snsd_v15/archive/DAI2026_SNSD_V51_STABLE_MIXED_SRC_20260802/
~~~

该归档只用于追溯，不参与正式运行。

## 最终方法

[SNSD_V51_FINAL](SNSD_V51_FINAL/README.md) 只公开 Learned + Phased：

~~~bash
cd SNSD_V51_FINAL
python run.py --data data/C3.csv --checkpoint checkpoints/checkpoint_message_v4_5_strict.pt --output runs/c3_seed11 --seed 11 --device cuda:0
~~~

入口固定为 6 智能体、4 个宏轮、每轮 12 步、288 次 paid local-worker 调用；不存在
method 或 oracle 开关。最终包中的 protocol 文件是 v5.1 当前运行所需的内部协议层，
不是可选择的历史版本；experiment_v*.py 与 run_v15_v*.py 已全部移出。

## 消融与正式证据

[SNSD_V51_ABLATIONS](SNSD_V51_ABLATIONS/README.md) 包含八种方法、Oracle 机制诊断、
Original/Fresh Learned 与 Oracle 三套 64-cell 矩阵。它导入最终项目核心，不保存第二份算法。

完整 C3 上共 192 runs、55,296 次 paid worker calls。Phased pooled 16-seed 平均增益为
35,884.6 ± 6,026.4；结构化联合动作因子 B 在有无语义消息时分别贡献
+20,214.1 和 +16,424.2，均为 16/16 正向；Phased 相对零语义和 OneShot 分别为
+5,033.6 和 +4,810.7，均为 14/16 正向。

## GNN 训练与正式模型

2026-08-03 服务器重新开启后，以下材料已全部下载：

- 正式 selected checkpoint；
- selected run 的 model manifest、history、split、dataset、candidate、teacher 与 pressure 回执；
- v3 warm-start checkpoint 及完整训练回执；
- 8 个 strict warm-up 目录；
- 8-candidate validation-only selector 与 sweep manifests；
- 三套正式实验 launch manifests；
- cplexpy37 的 Conda、pip、PyTorch、CUDA 和 GPU 环境快照。

正式 checkpoint 已在本地严格验证：

~~~text
file SHA-256 = a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59
state hash    = e2fbc9aa02b879ac9cd0bbe55f00838b9668d193933427e822bc00ff910505f8
parameters    = 456075
missing/unexpected keys = [] / []
~~~

训练源码、三个训练入口和精确 selected-cell 重跑脚本均在
[SNSD_V51_FINAL](SNSD_V51_FINAL/README.md)。详细合同见
[TRAINING_REPRODUCTION.md](SNSD_V51_FINAL/TRAINING_REPRODUCTION.md)。

唯一仍无法补出的历史材料是“训练进程启动瞬间”的源码字节快照：服务器上的 trainer
在该训练进程运行期间被后续兼容性开发更新过。正式 checkpoint、数据、配置、history、
选择回执和环境均已冻结，因此推理复现完整；但不应宣称能够逐字节重建同一训练过程。
准确状态为 **PASS_WITH_TRAINING_SOURCE_SNAPSHOT_GAP**。

## 已完成验收

- 46/46 测试通过；
- 最终入口及三个 GNN 训练入口通过；
- 完整 C3 preflight：69,923 nodes、979,960 edges、8,980 cut edges；
- 6 agents、12 × 4、288 calls、same-state C0/CA；
- 正式 checkpoint 文件哈希与严格 state-dict 加载通过。

清单生成：

~~~bash
/home/jiayan/anaconda3/envs/cplexpy37/bin/python -B tools/build_manifest.py
~~~

方法与证据见 [论文母稿](paper/SNSD_V51_METHOD_AND_EVIDENCE_ZH.md)，重组与恢复记录见
[V51_REORGANIZATION_REPORT_ZH.md](docs/V51_REORGANIZATION_REPORT_ZH.md)。
