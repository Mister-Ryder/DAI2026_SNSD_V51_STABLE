# SNSD-v5.1 稳定包重组报告

## 结论

本次重组解决了两个问题：

1. 原 DAI2026_SNSD_V51_STABLE 同时保存最终入口、v2--v5.0 过程继承链、消融脚本与
   正式结果，项目边界不清晰；
2. 原稳定包只有 GNN 推理模块，缺少离线数据、teacher、trainer 和多卡训练入口，
   Learned 启动链还导入了不存在的训练模块。

重组后形成“最终方法、消融证据、历史过程”三条互不混淆的路径。

## 新结构

### A. 纯净最终项目

目录：SNSD_V51_FINAL/

职责：

- 唯一公开求解入口 run.py；
- 固定 Learned + AlternatingConflictBenefit；
- 固定 6 agents、12 steps × 4 macro rounds、288 paid calls；
- 完整当前算法运行闭包；
- C3 数据与正式 checkpoint 合同；
- 完整 GNN 训练链和测试。

最终 CLI 不存在 --method 与 --oracle 参数。历史 experiment_v*.py 和
run_v15_v*.py 不在该项目中。

最终算法仍由若干内部 protocol 层组成，因为 v5.1 的已验证行为通过逐层 override 和
super 调用实现。它们已按职责命名，是当前单一实现的运行依赖，不是多个对外版本。
贸然合并为一个巨型文件会改变方法分派与结果协议，因此本轮没有用未经正式实验验证的
重写替换已冻结行为。

### B. 消融与对比项目

目录：SNSD_V51_ABLATIONS/

职责：

- 八种正式方法 cell；
- Original Learned、Fresh Learned 和 Oracle 矩阵；
- 64-cell 启动器；
- 严格汇总、checkpoint 合同审计和逐 seed 配对分析；
- 192 个正式 run 的冻结结果与日志。

消融项目通过路径导入 SNSD_V51_FINAL/src，不复制模型或协议核心。这样每个 ablation
切换的是同一最终实现中的明确机制，而不是另一份漂移代码。

### C. 过程历史

目录：

~~~text
../snsd_v15/archive/DAI2026_SNSD_V51_STABLE_MIXED_SRC_20260802/
~~~

原稳定包的 src/、旧 requirements 和 V15 方法演进文档已经移入这里。原 snsd_v15
主项目也仍保留这些过程源码的权威副本。归档仅用于追溯，不参与最终或消融入口。

## GNN 训练恢复范围

已补入：

- graph_layers.py 与 model.py：三图编码和 pre-query acquisition；
- runtime.py：Learned policy 运行时；
- training/offline_dataset.py：leakage-safe 数据闭包；
- message_teacher.py、rolling_teacher.py、strict_teacher.py；
- warmstart_trainer.py、message_warmstart_trainer.py、gnn_trainer.py；
- train_warmstart.py、train_message_warmstart.py、train_gnn.py；
- scripts/train_gnn_8gpu.sh、select_checkpoint.py、check_checkpoint.py；
- 模型、消息、runtime、teacher/trainer 与启动脚本测试。

恢复的是严格 rolling Real/Zero message GNN 训练链。它不等于已经针对 v5.1 的
C_A-C_0 联盟差值重新训练。

## 保留的兼容字段

正式 JSON 和事件回执中仍存在 v4.5、v50、v51 等 schema 字符串与私有字段名。这些字段
是既有 192-run 证据的解析合同。删除或重命名会使新代码无法审计旧结果，因此予以保留；
它们不对应公开旧版本入口，也不允许用户选择旧算法。

同理，正式结果 JSON 内记录的服务器绝对路径是原始 provenance，不能为了目录美观而
改写。新运行入口和说明文档均已改用新项目路径。

## 服务器材料恢复状态

服务器于 2026-08-03 重新开启后，已下载：

- 正式 checkpoint_message_v4_5_strict.pt；
- selected run 的 model manifest、training history、split/dataset receipt、
  candidate、teacher label 与 pressure；
- v3 warm-start checkpoint 及完整训练目录；
- 8 个 strict warm-up manifest 与 teacher config；
- validation-only sweep、selector 和训练日志；
- Original Learned、Fresh Learned、Oracle 三套 launch manifests；
- cplexpy37 的 Conda、pip、PyTorch/CUDA/GPU/OS 环境快照。

正式 checkpoint 已在本地通过一次文件哈希与 strict state-dict 加载：

~~~text
file SHA-256:
a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59

state hash:
e2fbc9aa02b879ac9cd0bbe55f00838b9668d193933427e822bc00ff910505f8

parameters: 456075
missing/unexpected keys: [] / []
~~~

当前只剩一个历史边界：正式训练进程启动后，服务器 trainer 文件在进程仍运行时被后续
late-stage 开发更新，启动瞬间已加载的源码字节快照没有单独保存。当前源码与普通
非-late 正式 manifest 兼容，但不能声称原训练过程可逐字节重建。稳定包状态因此记为
PASS_WITH_TRAINING_SOURCE_SNAPSHOT_GAP；正式推理与 checkpoint 复现已经完整。

## 验收标准

- 最终入口不暴露方法或 Oracle 开关；
- 消融入口准确暴露 8 methods 与 Oracle；
- TriGraphModel 参数量为 456,075；
- GNN 三个训练 CLI 可导入并显示帮助；
- Python 3.7 正式测试套件全部通过；
- 完整 C3 preflight 报告 6 agents、12 × 4、288 calls、same-state C0/CA；
- 三套正式矩阵均为完整 non-smoke 协议且无预算/日程/污染违规；
- C3 文件大小和 SHA-256 与冻结值一致；
- 稳定包 manifest 在所有修改后重新生成。
