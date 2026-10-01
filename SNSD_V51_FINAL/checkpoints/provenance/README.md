# 正式 GNN provenance 清单

本目录于 2026-08-03 从服务器原始路径直接下载，文件名和内容未改写。

## 1. formal_selected_run

正式 selected trial：
warmstart_e2e_lr3e4_eps05_d12_e12_fast。

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| checkpoint_message_v4_5_strict.pt | 1,905,167 | a003225adfdd8f4c7c8d00ca0580847dc79586f67c569eb76c6dfda28ebf0e59 |
| model_manifest_message_v4_5_strict.json | 19,921 | 19ddf482c6f8c44734dfc8c98050c84460f799256bfc3fe5bebd38ceb2d6bbc3 |
| training_history_message_v4_5_strict.jsonl | 93,290 | 0de072655d5b1e0111a9e9ff6e5c971c633e05ecaa43ee5d26fa47b7759c0783 |
| dataset_receipt.json | 10,707 | d312e3311d036e34f3b322faf2381681bf8f85bfb0a06263e88281c75b2d1656 |
| split_manifest.json | 689 | 3162f32ff290b41113d10e7b50658c94e23894c0904e4502a2ecac5025d51813 |
| prequery_candidates.jsonl | 6,727,272 | b8967c2e32d4db69ceaf029d1bbc38ca4a5472bd2f65c7d719dbd8e5b8380a2e |
| teacher_labels.jsonl | 863,938 | 74df86ccc109ec2af78b0a16ff889236f64326666c11c4cefd75d8de40872098 |
| arc_pressures.jsonl | 9,246,265 | a8a63a4dd258c26169ddd4b8d26f02383f629e5c2bed8f760ab6d50729447cda |

## 2. formal_training_root

关键选择回执：

- selected_fast_main.json：8 个 eligible candidates，validation-only；
- fast_sweep_manifest.tsv：候选训练配置；
- fast_remaining_manifest.tsv 与 fast_corrected_additions_manifest.tsv：追加运行记录；
- selected trial log；
- 同目录保留其他根级小型 log/manifest，作为过程 provenance。

selected_fast_main.json 的 SHA-256：
fc9f5626cf797e1dd137e233431dbace59c192851f27a05a943289fe0c12c6fe。

fast_sweep_manifest.tsv 的 SHA-256：
82c8e3c43c3eeac82a65df8c2ac6470d7121e205086c43ddbfd5bd447dcb77ff。

## 3. warmup_teacher_strict

包含 seeds 11、22、33、44、55、66、77、88；每个目录均包含
warmup_manifest.json 与 teacher_config.json。dataset_receipt.json 中记录的八个
manifest hash 已与服务器文件身份对应。

八个 warmup manifest 合计约 15.27 MB。C3 SHA-256 为
ec95f50c11d800f051e218aa1e414df873ddd12e1f71ce911da3ba28adff647e。

## 4. formal_warmstart_message_v3

完整 v3 warm-start trial，包含 checkpoint、model manifest、training history、
cross-fit teacher manifest、dataset/split receipt、candidate、teacher label 和 pressure。

checkpoint_message_v3.pt：

- 文件 SHA-256：
  6e9c72018d5341e04326076c22a26debc8a070d8e5ef5d29ac136cf17ac58ea8
- manifest state hash：
  46dea02bdc1fb656dcde6beaef52174cb49d6952140715f351e51a4bebbd34a6

## 5. server_environment

- conda_explicit.txt
- pip_freeze.txt
- system_and_gpu.txt

正式环境为 /root/miniconda3/envs/cplexpy37，Python 3.7，PyTorch 1.12.1，
PyTorch CUDA build 11.3。GPU 与 OS 明细保留在 system_and_gpu.txt。

## 6. 正式实验 launch manifests

位于稳定包的：

~~~text
SNSD_V51_ABLATIONS/results/formal/provenance/launch_manifests/
~~~

分别保存 Original Learned、Fresh Learned 和 Oracle v2。

## 7. 源码快照边界

训练进程运行期间，服务器上的 trainer 源文件被后续 late-stage 开发更新。正在运行的
Python 进程使用的是启动时已加载的旧代码，但该字节级源文件快照没有单独保存。当前源码
与正式 manifest 的普通非-late 合同兼容；正式数值身份以 checkpoint、model manifest、
history、dataset hashes 和 selector receipt 为准。
