# COMP3314 Group Project — CSA 复现

目标论文：Xu et al., *Reconstruction and Recognition of Tensor-Based Objects With Concurrent Subspaces Analysis*, IEEE TCSVT 2008。

## 从这里开始

| 文档 | 用途 |
|---|---|
| [同伴报告／展示材料](./docs/TEAM_BRIEF.md) | 按研究问题整理：论文与课程要求、实现、TA符合范围、全部实验数据与比较、批判分析、折叠附录及原始结果索引 |
| [完整交接文档](./docs/HANDOFF.md) | 新会话先读：项目来源、论文、课程截图要求、教师/TA邮件、数据与实现、关键结果、验证边界及下一步 |
| [当前复现报告](./docs/REPORT.md) | 三个数据集的当前结果、确认协议、本组约定、差距分析、PIE 说明及提交要点 |
| [全部实验与数据总览](./docs/REPORT.md#experiment-overview) | 已执行实验批次、数据、论文差距、扩展方向及结果口径，含负结果和历史协议区别 |
| [历史档案](./docs/HISTORY.md) | 整理前全部 13 份文档的内容；选题、旧计划、回信和历次实验供追溯 |
| [FERET 数据说明](./FERET/README.md) | 官方配对、标注、下载完整性与读取约定 |

## 当前状态

- **2026-10-08 DATER固定监督扩展已完成并核查，负结果保留。** 一个2π裁剪内输入、二十相同划分、24维：DATER直接NN为**71.64%±3.18%**，加普通LDA23为**67.11%±3.59%**；CSA同条件74.33%/80.13%，PCA82.70%/88.81%。20新DATER模型，复用40对照，120评分／38,400预测通过核查；19/20达到20轮上限，不声称全部收敛。DATER是方法扩展，未知设置未获TA／作者确认，不替换CSA共享／选参／扩展／probe极值。[协议、比较、失败记录与重跑](./docs/REPORT.md#orl-dater)。
- **2026-10-08三项受控分析全部完成并核查。** 错例/成本零拟合，σ×上下文四输入×20预定划分×四方法320模型，probe眼点四向扰动1280评分零拟合。固定π上下文CSA24＋普通LDA23为**87.08%±2.30%**，相对本轮2π裁剪内80.13%配对+6.95pp（20/20提高），距论文90.9%仍差3.82pp；同预算PCA为91.08%。CSA390的上下文2π/π为91.16%/90.09%，没有四方法共同最优输入。低维CSAπ上下文四向原图眼点±1像素扰动平均−0.08pp，范围局限公开；实际投影权重830/618792，本机预计算特征投影＋LDA＋NN配对约2.1–3.3倍速度差，不含Gabor/训练。320模型及512000条清洁/扰动预测检查通过，旧共享/选参/扩展/探索极值保留。该轮当时保留DATER规划；后续DATER已单独完成，正式提交材料仍保留规划。[全部结果、证据、局限与后续规划](./docs/REPORT.md#orl-controlled)。
- **2026-10-08裁剪内、原维数有限优化已完成并核查，未实质改善。** 四个固定输入σ=2π/π×zero/reflect，gallery内选输入/起点/NN后，CSA24＋普通LDA23为**81.41%±2.41%**，相对配对基线81.25%仅+0.16pp（2升4平4降），距论文90.9%差9.49pp。共享CSA390/PCA24→23/PCA33→25为88.41%/88.41%/91.69%，均下降。960内层模型、40新外层拟合＋40历史对照、独立核查通过。满足已明确TA数量、尺寸、模型与共享条件，未知设置仍是本组约定；此前上下文扩展87.84%、原共享79.66%及probe极值分别保留。[前后/论文比较、冻结参数与证据边界](./docs/REPORT.md#orl-ta-crop)。
- **ORL原维数未知参数的有限迭代已完成并核查。** gallery内三阶段选择后，CSA24＋普通LDA23为**87.84%±3.28%**，相对固定基线81.25%提高6.59pp、相对上一轮上下文84.31%提高3.53pp；距论文90.9%仍差3.06pp，2/10划分达到论文数字。输入阶段87.81%贡献主要增益，后续幂次／起点／NN净增约0.03pp；八组选择σ=π上下文，十组均为单位列／居中余弦。共享方案的纯CSA390为91.03%，低于上一轮91.56%，不能称四方法共同最优。3520内层模型、900候选、120新外层拟合、独立预测／更新核查通过；上下文及幂次为本组扩展，未知设置未获作者确认，不替换原共享主表。[参数、全部前后比较与TA／原文边界](./docs/REPORT.md#orl-unknown-parameters)。
- **ORL三套固定输入对照已完成并核查。** 基线/TT＋均衡/上下文，原维数、G2/P8、十个相同种子、共享居中余弦：CSA24＋LDA23为81.25%/78.31%/**84.31%**，纯CSA390为90.00%/89.56%/**91.56%**。上下文分别提高3.06/1.56pp，胜平负9/0/1及8/1/1；PCA等预算24→23为91.00%、论文维数33→25为94.38%。新输入均为本组扩展，不替换原共享主表或训练内选参；120模型、38,400预测和60个独立CSA三轮核查通过，光照负结果保留。[完整比较、诊断与协议边界](./docs/REPORT.md#orl-next-directions)。
- **MNIST四步有限优化已完成并核查，CSA未改善。** 相同20–29十种子，G30/G40原CSA为89.66%/90.50%，gallery内共享输入/NN选参后为88.89%/90.18%（−0.77/−0.32pp）；PCA为86.67%/88.03%→86.94%/88.26%。原20–22三种子CSA也由90.34%/91.04%降为89.33%/90.34%。保留原主表及全部负结果；满足教师的数据量、尺寸、划分、5×8 Gabor、NN与Tmax=3约束，保留Table IV维数，未知输入/距离/初始化与统计细节仍属本组约定，不能称原作者确认或精确复现。[比较、TA/原文范围与重跑](./docs/REPORT.md#mnist-bounded-optimization)。
- **200次最佳结果探索已完成并核查。** G2/P8、Tmax=3、普通CSA32+LDA24最高 **95.00%**；纯CSA560最高 **97.50%**；保留390维最高 **97.19%**。它们对应固定配置的均值分别为86.42%/91.43%/92.00%，按probe选最高值，不能当作稳定均值或精确原协议复现。全部6,003个配置、1,200,600条评分及6,432条接近/超过论文的索引保留。[配置、英文表述与重跑](./docs/REPORT.md#orl-best-search)。
- **此前训练内选参结果保留。** 普通Gabor CSA+LDA十次均值 **83.28%±3.30%**、最高 **89.06%**；收缩扩展均值 **85.47%±4.57%**。比上一轮训练内优化分别提高0.81/0.72pp；普通流程纯CSA390均值 **89.97%**。这组未按probe挑参数，与新增探索最高值分别报告。[参数、对照与分析](./docs/REPORT.md#orl-csa-refinement)。
- **固定probe的样本量扩展已完成并核查。** 同一P5、嵌套G2/G3/G5、10种子：普通Gabor CSA+LDA **81.30%/88.10%/94.50%**，纯CSA390 **89.80%/93.95%/97.45%**。同维PCA对照与独立CSA更新复算完成，共核查90个模型、64,400条预测记录；更多训练不是原G2/P8复现成功。[完整结果与差距证据](./docs/REPORT.md#orl-sample-size)。
- **CSA+LDA后续优化已完成并核查。** 十次训练内选参后，普通LDA灰度/Gabor为 **78.56%±3.98% / 82.47%±3.14%**；协方差收缩扩展为 **84.13%±2.38% / 84.75%±4.66%**。扩展相对前轮论文维数锚点提高6.66/5.09pp；同配置移除收缩损失7.28/4.53pp。正则化单列为扩展，不宣称原普通LDA已恢复论文数值。[本轮结果与差异](./docs/REPORT.md#csa-lda-optimization)。
- **CSA+LDA专项排查已完成。** 同NN原点下，灰度22维CSA加LDA由71.84%提高到77.47%，Gabor24维由77.00%提高到79.66%；低秩表示和小样本LDA适配是当前薄弱环节。独立广义特征分解、NN、空间轴与通道排列未发现足以解释差距的实现错误。针对两CSA目标的训练内再选参反而下降，保留负结果及原主表。详见 [专项分析](./docs/REPORT.md#csa-lda)。
- **2026-10-07 已重新审计原论文、TA协议和代码，并修正ORL眼点/裁剪与训练内选参。** 本轮240次外层模型运行、76,800条预测，首轮、固定参数对照与追加秩搜索均通过核查。详见 [报告第3节](./docs/REPORT.md#orl)。
- 已实现并运行 **ORL、MNIST、原始灰度 FERET**；PIE 按教师许可省略，尚不声明论文全部实验已完整复现。
- ORL新共享输入＋论文维数、10种子：纯Gabor CSA **90.19%±2.37%**，CSA＋LDA **79.66%±2.77%**；较旧冻结流程提高7.22/7.97个百分点。首轮训练选参为 **89.72% / 80.34%**；追加秩搜索未提高均值，完整保留。仍未精确恢复论文94.4%/90.9%。
- MNIST 共享流程复用并核查：PCA **87.52% / 88.58%**，CSA **90.34% / 91.04%**，对应 G30/P470、G40/P460。旧逐方法选参探索单列。
- FERET 共享固定余弦对照复用并核查：CSA **88.73% / 94.25% / 95.38%**，对应 50/100/200 训练身份；旧验证选定结果单列。
- 旧眼点的影响已由只换坐标的固定对照确认；独立CSA/LDA公式核对未发现核心更新错误。此前gallery内选参不使用本轮probe分数；新增200次探索使用probe分数选择配置与划分。历史probe已观察，因此不称新盲测。

## 目录导航

```text
COMP3314GP/
├─ README.md              # 唯一入口
├─ docs/
│  ├─ TEAM_BRIEF.md       # 同伴写报告／展示：主题正文、完整数据折叠附录
│  ├─ REPORT.md           # 当前结果与提交材料
│  ├─ HISTORY.md          # 原文档统一档案
│  └─ HANDOFF.md          # 新会话完整交接
├─ csa.py / gabor.py      # 核心算法与原固定 Gabor
├─ audit_orl_protocol.py  # 当前ORL训练内优化、固定对照与核查
├─ diagnose_csa_lda.py    # 灰度/Gabor CSA+LDA专项定位与核查
├─ optimize_csa_lda.py    # 普通LDA/收缩扩展训练内优化与消融
├─ refine_orl_csa.py      # Gabor专项选参、HOSVD起点、秩分配与划分敏感性
├─ search_orl_best.py     # 200次探索搜索、完整网格及最佳观测结果重放
├─ optimize_mnist_protocol.py # MNIST共享输入/NN、论文维数、gallery内有限优化与独立核查
├─ run_orl_fixed_inputs.py # ORL三套固定输入、120次原维数/等预算配对对照与独立核查
├─ optimize_orl_unknowns.py # 原维数下未知输入/幅值/初始化/NN的gallery内有限迭代
├─ optimize_orl_ta_crop.py # 裁剪内四输入、原维数共享参数的gallery内有限优化与核查
├─ dater.py / run_orl_dater.py # 固定DATER监督扩展，20模型、六分支比较
├─ verify_orl_dater.py / evaluate_orl_dater.py # 独立模型核查与评分读取修正入口
├─ run_orl_controlled.py # σ×上下文固定20划分、错例/成本与probe扰动，独立核查
├─ run_orl_sample_size.py # 固定P5的G2/G3/G5扩展与同维PCA诊断
├─ run_shared_protocol.py / verify_shared_protocol.py  # 此前四步对照
├─ *.py                  # 数据处理、实验与核查入口
├─ data/                 # ORL/MNIST 数据、眼点与特征
├─ FERET/                # 授权原始数据与官方配对
└─ results/              # 逐次实验、图表、协议与源码快照
```

## 环境与重跑

从 `COMP3314GP` 执行。依赖锁定在 [requirements.txt](./requirements.txt)，OpenCV 预处理依赖在 [requirements-preprocess.txt](./requirements-preprocess.txt)。首次准备自己的 Python 环境：

```powershell
python -m pip install -r requirements.txt
python -m pip install --no-deps --target .deps -r requirements-preprocess.txt
```

已有结果可直接查看。需要重跑时使用尚不存在的输出目录；各入口及核查命令见 [报告运行说明](./docs/REPORT.md#outputs)。

## 结果入口

| 内容 | 保存位置 |
|---|---|
| 最新ORL DATER方法扩展 | [六组结果](./results/orl_dater/summary.csv)；[配对变化](./results/orl_dater/paired_changes.csv)；[固定协议](./results/orl_dater/protocol.json)；[核查](./results/orl_dater/verification.json)；[数值敏感性](./results/orl_dater/numerical_sensitivity.csv)；[结果图](./results/orl_dater/comparison.png)；[报告与重跑](./docs/REPORT.md#orl-dater) |
| 最新ORL固定受控证据 | [四输入结果](./results/orl_controlled/summary.csv)；[因素及交互](./results/orl_controlled/factorial_effects.csv)；[错例](./results/orl_controlled/historical_aligned_error_montage.png)；[实测成本](./results/orl_controlled/cost_benchmark.csv)；[扰动汇总](./results/orl_controlled/robustness_summary.csv)；[清洁核查](./results/orl_controlled/verification.json)；[扰动核查](./results/orl_controlled/robustness_verification.json)；[对比图](./results/orl_controlled/comparison.png)；[报告/规划](./docs/REPORT.md#orl-controlled) |
| 最新ORL裁剪内优化、负结果与符合范围 | [前后数据](./results/orl_ta_crop_optimized/comparison.csv)；[汇总](./results/orl_ta_crop_optimized/summary.json)；[逐组参数](./results/orl_ta_crop_optimized/selected_settings.csv)；[协议](./results/orl_ta_crop_optimized/protocol.json)；[核查](./results/orl_ta_crop_optimized/verification.json)；[对比图](./results/orl_ta_crop_optimized/comparison.png)；[报告](./docs/REPORT.md#orl-ta-crop) |
| ORL未知参数的有限迭代 | [前后及论文比较](./results/orl_unknown_parameters/comparison.csv)；[逐组参数](./results/orl_unknown_parameters/selected_settings.csv)；[冻结协议](./results/orl_unknown_parameters/protocol.json)；[核查](./results/orl_unknown_parameters/verification.json)；[迭代图](./results/orl_unknown_parameters/iteration_comparison.png)；[报告与范围](./docs/REPORT.md#orl-unknown-parameters) |
| MNIST有限优化、负结果与符合范围 | [十次前后/论文对照](./results/mnist_protocol_optimized/comparison.csv)；[原三次配对](./results/mnist_protocol_optimized/original_three_seed_comparison.csv)；[协议](./results/mnist_protocol_optimized/protocol.json)；[核查](./results/mnist_protocol_optimized/verification.json)；[对比图](./results/mnist_protocol_optimized/comparison.png) |
| 200次最佳观测结果探索 | [最高值与参数](./results/orl_best_search/best_results.json)；[接近/超过论文的全部运行](./results/orl_best_search/near_or_exceed.csv)；[全部配置统计](./results/orl_best_search/configuration_summary.csv)；[分布图](./results/orl_best_search/best_distribution.png) |
| 最新G2/P8针对性迭代 | [orl_csa_refined](./results/orl_csa_refined/)；[对比图](./results/orl_csa_refined/refinement_comparison.png)；[全部参数](./results/orl_csa_refined/selected_settings.csv) |
| 固定配置100次划分诊断 | [全部分数](./results/orl_csa_refined/randomness/results.csv)；[分布图](./results/orl_csa_refined/randomness_distribution.png)；[核查](./results/orl_csa_refined/randomness/verification.json) |
| 固定probe的样本量扩展与核查 | [orl_sample_size](./results/orl_sample_size/)；[学习曲线](./results/orl_sample_size/learning_curve.png)；[配对变化](./results/orl_sample_size/paired_changes.csv) |
| 普通LDA / 收缩扩展优化与核查 | [csa_lda_optimized](./results/csa_lda_optimized/)；[结果图](./results/csa_lda_optimized/optimization_comparison.png)；[全部选定参数](./results/csa_lda_optimized/selected_settings.csv) |
| CSA+LDA专项诊断、选择负结果与核查 | [csa_lda_focus](./results/csa_lda_focus/)；[专项对比图](./results/csa_lda_focus/diagnosis_comparison.png) |
| 当前ORL审计/优化、全部预测与核查 | [orl_protocol_audit](./results/orl_protocol_audit/)；[当前三数据集完整CSV](./results/orl_protocol_audit/shared_comparison.csv) |
| 当前ORL十种子与论文比较 | [comparison.png](./results/orl_protocol_audit/comparison.png) |
| 眼点/裁剪固定对照及图像 | [原因对照图](./results/orl_protocol_audit/contrasts/comparison.png)；[裁剪对照](./results/orl_protocol_audit/crop_comparison.png) |
| 此前四步结果与核查 | [shared_protocol_final](./results/shared_protocol_final/) |
| 等系数与逐种子配对差 | [paired_differences.png](./results/shared_protocol_final/step4/paired_differences.png) |
| 当前 ORL 固定分支 | [orl_branches_v3](./results/orl_branches_v3/) |
| ORL Gabor 单因素诊断 | [orl_gabor_diagnosis](./results/orl_gabor_diagnosis/) |
| 历史 MNIST 与早期 ORL 选参探索 | [available_tuned](./results/available_tuned/) |
| 当前 FERET 验证选定及固定对照 | [feret_validation_selected_v3](./results/feret_validation_selected_v3/) |
| ORL 重建、收敛与早期曲线 | [available_reproduction_v2](./results/available_reproduction_v2/) |

## 提交要求

提交前仍需整理 3–4 页 LaTeX 报告、9 分钟展示与 3 分钟 Q&A，并发布公共代码仓库及运行说明。代码由 Codex 按论文公式辅助编写，小组应审阅、理解并披露辅助来源。FERET 原始数据受许可约束，不随公共代码发布；完整评分与复现 bonus 要点见 [提交说明](./docs/REPORT.md#submission)。
