# 正式 Learned checkpoint 状态

正式 checkpoint 与完整训练溯源已于 2026-08-03 从服务器下载到本项目。

## 正式文件

~~~text
checkpoints/checkpoint_message_v4_5_strict.pt
~~~

本地一次性严格校验结果：

~~~text
file SHA-256:
a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59

state-dict hash:
e2fbc9aa02b879ac9cd0bbe55f00838b9668d193933427e822bc00ff910505f8

parameter_count: 456075
strict_load: true
missing_keys: []
unexpected_keys: []
~~~

Original Learned 与 Fresh Learned 两套正式矩阵共 128/128 runs 使用该文件，
oracle_mode=false。

## 已恢复的训练链

checkpoints/provenance/ 下现有：

- formal_selected_run/：正式 checkpoint 的 model manifest、history、split、dataset、
  candidate、teacher label 和 pressure 回执；
- formal_warmstart_message_v3/：v3 warm-start checkpoint 及完整训练回执；
- warmup_teacher_strict/：seeds 11/22/33/44/55/66/77/88 的 manifest 与 teacher config；
- formal_training_root/：fast sweep、selector、补充 manifest 和训练日志；
- server_environment/：Conda explicit、pip freeze、PyTorch/CUDA/GPU/OS 信息。

三套正式实验的 launch manifests 位于：

~~~text
../SNSD_V51_ABLATIONS/results/formal/provenance/launch_manifests/
~~~

精确文件清单与哈希见 provenance/README.md。

## 校验命令

~~~bash
sha256sum checkpoints/checkpoint_message_v4_5_strict.pt
python scripts/check_checkpoint.py checkpoints/checkpoint_message_v4_5_strict.pt
~~~

## 论文声明边界

现在可以从本地严格加载并重跑 Learned 方法，不再存在 checkpoint 二进制缺口。

仍不能声称当前模型针对 v5.1 的 C_A-C_0 或 coalition-delta 目标端到端训练。另一个
历史限制是训练启动瞬间的 trainer 源码字节快照没有保留；当前源码是兼容的后续版本。
因此可以声称推理复现完整、训练合同和回执完整，但不能声称原训练过程可逐字节重建。
