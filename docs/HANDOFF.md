# COMP3314 CSA 项目完整交接文档

更新：2026-10-08。工作区：`D:\HKU\COURSE\y2s1\COMP3314\COMP3314GP`。

**用途：让新会话在不读取全部聊天的情况下，了解项目为什么做、课程如何评分、原文和教师要求是什么、已经实现和核查了什么、哪些结果可报告，以及接下来该做什么。** 本文是状态与决策交接；完整历次讲解和邮件档案在 [HISTORY.md](./HISTORY.md)，详细当前报告在 [REPORT.md](./REPORT.md)。历史文件中的“最终”“待办”“不可取得”可能已过时，应先按本文和当前结果文件判断。

## 0. 新会话先读什么

**同伴准备报告／展示时优先读 [TEAM_BRIEF.md](./TEAM_BRIEF.md)**：按论文研究、实现／协议、数据比较、批判分析和交付组织；全部实验族及统计放入折叠附录，完整结果目录可追溯。REPORT保留详细历史解释。

1. 本文：完整来源、约束、当前结果和下一步。
2. [README.md](../README.md)：目录、依赖、结果导航。
3. [REPORT.md](./REPORT.md)：结果解释、全部协议差异、运行命令。
   其中[第10节：全部实验与数据总览](./REPORT.md#experiment-overview)按批次汇总共享锚点、gallery内选参、固定对照、扩展、probe探索最高值及论文差距；含早期G2/V2/P6与后续G2/P8区别、失败／退步、重建／数值核查／错例／成本／扰动。不从总览反推未公开的原作者设置，具体运行仍须回读协议和原始结果文件。
4. 需要修改某项实验时，再读该结果目录的 `protocol.json`、参数记录、`source_snapshot/` 和对应脚本；不要从旧报告数字猜当前配置。
5. 争议涉及原文时，回到原始 PDF。涉及邮件时，回到本文第3节及 HISTORY 对应原句。

### 当前一句话结论

已自行实现并实际运行 **ORL、MNIST、原始灰度 FERET** 的 CSA/PCA/LDA 实验，完成大量协议、数值与输入排查；PIE 经教师许可省略。ORL最新固定σ×上下文二十划分，π上下文CSA24→23为87.08%±2.30%，相对本轮2π裁剪内80.13%配对+6.95pp。此前裁剪内gallery选参81.41%、上下文/幂次扩展gallery选参87.84%继续分别保留，均低于论文90.9%。另有超过论文数字的probe选择探索最高值。**不能宣称原论文全部实验已 full reproduction，也不能把探索最大值写成稳定平均表现。** 课程允许明确说明无法匹配的细节，数值精确相等不是评分要求。后续DATER方法扩展已完成但未改善（直接71.64%、加LDA67.11%），见下段，不能拼入CSA复现主表。

**最新2026-10-08 DATER扩展按后续授权完成并停止：** 用户现要求完成DATER执行；[完整结果、协议与局限](./REPORT.md#orl-dater)，源`dater.py`/`run_orl_dater.py`，独立核查`verify_orl_dater.py`，评分读取修正`evaluate_orl_dater.py`，结果`results/orl_dater/`。固定2π裁剪内、4200–4219、G2/P8、(4,2,3)=24维；DATER直接NN71.64±3.18%，加普通LDA23为67.11±3.59%，同条件CSA74.33/80.13、PCA82.70/88.81；相对CSA分别−2.69/−13.02pp，未改善。20新DATER模型、40复用对照、1194模式更新和38,400预测核查通过。19模型达到20轮上限，仅4219第18轮满足阈值，165次联合DTC下降，不称全部收敛／全局最优。初次自由路径严格核查失败记录保留；改以独立实际轨迹核查加20自由参考重算敏感性报告，最大自由最终方向差约1.01e−5，不宣称逐轮完全相等。首次评分JSON误读失败在写评分前发生，新增读取修正入口，拟合源码／模型／历史全部不变；两个修正核查入口均在成功评分前冻结，失败和敏感性不可省略。DATER完整单位起点及广义特征更新按Figure3；单位列／符号、Tmax20、ε.001是本组固定约定，没有ridge／QR／奇异fallback。TA的Tmax3适用于CSA；DATER为新方法扩展，TA未确认其参数，原文也没有匹配ORL24D行；CSA论文90.9只作不同方法参考。原共享、gallery选参、上下文扩展和probe极值保持独立；正式提交材料仍待实施，MNIST/FERET未改。重跑须按REPORT使用补充核查和读取修正入口，不调用原驱动已知失败的verify_models/evaluate路径。

**最新2026-10-08受控实验，三项完成并停止：** 用户要求全面规划并执行此前建议，随后选择“先完成受控实验，DATER与提交材料保留规划”。源码`run_orl_controlled.py`，结果`results/orl_controlled/`，[完整证据和保留规划](./REPORT.md#orl-controlled)。评分前冻结种子4200–4219、σ2π/π×crop/context四输入、原维数四方法、identity/Tmax3/普通LDA/单位列居中余弦及全部源/输入/索引/所用历史证据哈希。320新实验模型另160CSA独立数值复算、160PCA特征方程和102400NN预测通过。CSA24→23四输入为80.13±3.49/84.56±2.44/81.25±3.03/87.08±2.30%，π上下文相对2π裁剪内+6.95pp（20/0/0），距论文仍差3.82pp，同预算PCAπ上下文91.08%；σ裁剪内效应+1.13、上下文效应+2.52，差中之差+1.39pp，仅描述性证据。CSA390四输入88.66/91.16/88.70/90.09%，上下文减σ反而下降1.06pp。复用320模型，probe双眼共同沿原图x/y±1像素，1280评分零拟合、409600独立NN预测和全部1280保存文件回读通过；低维CSAπ上下文平均−0.08pp、最差方向均值−0.52pp、最差单划分/方向−1.875pp，不覆盖单眼误检/旋转/大偏移，不证明普遍鲁棒。已有stage_C错例3200记录覆盖400张不同照片，修正224/新增111、CSAonly158/PCAonly249，原图/对齐图/最近邻已核查；无人工眼点真值，不归因于唯一原因。实测830/618792个权重，所计部署数组228272/5171968字节，本机CoreUltra9 275HX两线程预计算特征320probe投影＋LDA＋NN约21–29/60–82ms，配对2.08–3.29倍，不含Gabor/训练，不是峰值内存。协议与代码未改，环境信息/全部负对照保留；手工视觉/汇总回读及扰动图不属于标准stage自动输出。上下文是扩展、未知设置未获作者确认；历史79.66/81.41/87.84及探索最高值不替换，不能跨不同划分计算配对改善。该受控轮当时未实施DATER；后续DATER已单独完成，报告/展示及公共仓库发布仍未实施；详细20模型扩展、3–4页/9分钟和仓库规划已记在REPORT，MNIST/FERET没有调参。

**最新2026-10-08裁剪内有限优化，已按用户授权完成并停止：** [前后比较、参数与证据](./REPORT.md#orl-ta-crop)，源码`optimize_orl_ta_crop.py`，结果`results/orl_ta_crop_optimized/`。固定原400张40类、G2/P8、种子20–29、双眼对齐/均衡56×46、5×8 Gabor输出28×23×40、Tmax3、普通LDA和论文维数。四输入σ=2π/π×zero/reflect，39×39核、λ₀=4、裁剪内滤波后stride00；没有上下文/幂变换/收缩/新秩。自己的80张gallery完整16折选前二输入，再两起点×六NN规则；960内层模型、280候选，十组选择冻结并独立核查后，40新外层拟合＋40历史基线。CSA24→23为**81.25→81.41±2.41%**（+0.16pp，2升4平4降，最高85.63%，0/10达到90.9%），CSA390为90.00→88.41%，PCA24→23为88.63→88.41%，PCA33→25为92.72→91.69%（论文87.2%）。内层508/800→566/800→568/800提升未转化为实质外层收益，原因尚未单独证实。最终π/reflect六组、π/zero一组、2π/zero三组，identity/HOSVD=8/2，全部单位列/居中余弦；不能按probe再选一个统一配置。304被选内层CSA独立复算、22,400内层预测、20外层CSA独立复算、20PCA核查、40基线NN重放及25,600外层预测通过；源码/输入/选择哈希未变，另有评分前和补充统计手工回读JSON。满足明确TA数量、尺寸、模型及共享条件；TA未逐项确认σ/补边/NN/列尺度/起点，未知设置不能写成作者协议。旧共享79.66%不同配置，81.41−79.66=1.75pp不是本轮配对增益；此前87.84%是扩展而非本轮裁剪内成绩。负结果、旧主表和probe极值完整保留；本轮不再自动扩大搜索。

**此前ORL未知参数有限迭代，用户已授权执行并完成：** [三阶段证据与参数](./REPORT.md#orl-unknown-parameters)，源码`optimize_orl_unknowns.py`，输出`results/orl_unknown_parameters/`。保留G2/P8、种子20–29、Tmax3、CSA24→普通LDA23／CSA390／PCA24→23／PCA33→25。每组自己的80张gallery完整16折：12输入选前二→三个幂次→两初值和六NN规则；3520内层模型、900候选，十组选择冻结且验证后才评分，120新外层拟合加80历史对照。CSA24→23为81.25→84.31→A87.81→B87.78→C**87.84±3.28%**，最终相对上下文+3.53pp（8/1/1），论文差距3.06pp，2/10达到90.9%，最高91.56%。最终八组σ=π上下文，1组原上下文、1组CDF_min；幂次1/0.75为7/3，identity/HOSVD为6/4，全部单位列/居中余弦。共享C纯CSA390为91.03±1.43%，比上下文91.56下降0.53pp；PCA24／33为90.69／94.16%，也略降，不称四方法共同最优。416个被选内层CSA独立复算、72,000验证预测、120新外层模型／64,000外层预测核查通过，源码冻结哈希未变。保留所有负阶段与旧表，扩展及未知作者协议继续单列。差距不能断定一定是协议错误；此次仍非盲测，也不是全局最优证明。整轮已停止，不自动扩大搜索。

**本次后续MNIST更新：** 用户授权的四步有限优化已完成、独立核查通过；没有缩小CSA差距。十个相同种子G30/G40：原CSA89.66%/90.50%→新gallery内选参88.89%/90.18%；PCA86.67%/88.03%→86.94%/88.26%。原20–22三种子CSA90.34%/91.04%→89.33%/90.34%。保留旧共享主表及全部负结果；不能混合三次与十次均值作优化增益。详见[REPORT的MNIST最新段](./REPORT.md#mnist-bounded-optimization)，结果根`results/mnist_protocol_optimized/`。

**MNIST收尾与ORL三输入最新执行：** REPORT已记录MNIST可接受的部分复现证据/局限及欧氏距离下降消融，教师/TA没有指定欧氏或余弦。用户随后授权[三套固定输入](./REPORT.md#orl-next-directions)，120次拟合已全部完成并核查：十个相同G2/P8、原维数/共享居中余弦，CSA24＋普通LDA23基线81.25%→光照组合78.31%/上下文84.31%（上下文+3.06pp，9/10提高）；纯CSA390为90.00%→89.56%/91.56%（上下文+1.56pp，8提高1平1降）。上下文PCA24→23/PCA33→25为91.00%/94.38%。两新输入作为扩展单列，不替换共享主表/训练内选参/探索最高值。源码`run_orl_fixed_inputs.py`，输出`results/orl_fixed_input_extensions/`；核查120模型、38,400预测、60个独立CSA三轮复算、30项历史基线正确数。首次画布插值核查失败在评分前修正并保留，后续仅核查路径修正记录于`verifier_amendment.json`；未改变拟合代码或分数。光照负结果保留，该三输入轮已停止；当时DATER未实现，后续单独执行见最新段；不自动合并输入、扩大网格或增加种子。

---

## 1. 项目来源、用户目标与工作约定

### 1.1 来源

- HKU COMP3314，工作目录属于 `y2s1/COMP3314`。
- 课程介绍：[Lec0 Introduction.pdf](<../../Lec0 Introduction.pdf>)，用户最初要求研究第6–7页列出的12篇论文，并按 group project 的评分、讲解难度和可复现性排名。
- 之后选定 **Concurrent Subspaces Analysis，CSA**，从选题评估逐步转为按原论文做实验复现。
- 12篇论文排名、早期中英文教学、复现规划及最初 full reproduction 实验清单均保存在 [HISTORY 的历史01/02](./HISTORY.md#history-01)。不应把历史选题排名当作本轮实验结论。
- 本文第2节的课程要求来自用户上传截图，而不是 CSA 论文。第3节是用户提供的教师/TA邮件；邮件未通过连接邮箱读取，记录日期不等于可验证的邮件发送日期。

### 1.2 用户目标如何演变

1. 初始目标是完整研究论文并尽量 full reproduction，要求先查原代码，再规划自行实现、下载必需数据。
2. 教師回复允许无法取得的数据集省略；FERET 随后取得，PIE 最终仍未取得，实际范围变为三个数据集。
3. 用户希望对作者未公布的细节采用合理、能够提高准确率的“本组约定”，并注明差异。
4. ORL，特别是普通 Gabor CSA+LDA，出现较大差距；多轮请求要求对照论文、TA要求与代码定位原因，不能只解释而不执行已授权排查。
5. 后续授权样本量扩展、收缩正则化扩展、低维秩和初始化优化、固定配置多划分诊断。
6. 最近授权更细参数搜索、更多划分，并采用原文的 best-results 句式报告接近/超过论文的结果。已执行200划分有限网格搜索，保留全部低分记录，并明确这是使用 probe 分数选择的探索极值。
7. 本次要求：评估离论文复现还有多少，提出有价值的补充，并创建包含**项目来源、论文、教师/TA要求、复现要求、课程截图全部核心要求**的全面交接文档。没有要求本次再启动新实验、发表仓库或创建新聊天。
8. 交接后的新请求：用户重申课程截图要求，明确授权执行MNIST四步有限优化并给出前后/论文比较及TA/原文符合范围。该实验现已完成并核查，CSA未改善，结果单列保留；没有扩大为新全网格搜索。
9. 随后授权完成ORL三输入固定对照，再授权列出未知参数并实际有限迭代；两轮均已完成并核查。该轮原维数CSA24→23为87.84%，只将输入/参数敏感性结果单列，不改写旧共享主表或探索最高值。
10. 再授权按TA明确条件做四个裁剪内输入、原维数共享参数有限优化；已完成并核查，CSA24→23为81.41%（相对配对基线仅+0.16pp），其他对照下降。87.84%作为此前上下文扩展保留；没有以负结果为由自动扩大搜索。
11. 授权执行此前规划，随后明确仅完成受控实验、DATER/提交材料保留规划。错例/成本、320模型固定σ×上下文和1280零拟合probe扰动均完成核查；π上下文87.08%为固定二十划分结果，不改称训练内选参赢家或原作者设置。
12. 随后明确授权完成DATER：一个固定输入、20监督张量模型，直接24维与23维LDA、复用CSA/PCA对照，核查后负结果单列；不扩大搜索或制作提交材料。

### 1.3 用户工作原则

- 用简洁自然的中文；教学内容应讲清背景、形状、公式与代码关系。此前要求过中英文对照，正式英文报告/展示可继续据此准备。
- 影响结论的关键假设必须说明。合理且可逆的细节可说明后执行；任务范围显著变化时再询问。
- 精准修改，保留现有目录、历史结果、失败实验，不顺手重构或删除无关文件。
- 用户嫌 Markdown 太多，已合并为 `REPORT.md`、`HISTORY.md` 和入口 README。本次明确要求才新增本文；后续优先更新现有文档。
- 修改后核查实际输出和预测，不把“脚本退出成功”当作实验已验证。
- 不未经授权发送邮件、发布仓库、上传数据或更改系统设置。

---

## 2. 课程 Group Project 要求与评分：按上传截图整理

### 2.1 选题要求 / Paper Selection

- 选择并深入研究一篇与课程主题有关的机器学习论文。
- 优先来自主要会议/期刊：ICML、ICLR、NeurIPS；AAAI、CVPR、ACL；JMLR、Machine Learning、TMLR；IEEE TNNLS、TPAMI、TIP、TCSVT 等。
- 对是否合适不确定时，向教师/TA咨询。所选 CSA 发表在 TCSVT，与 PCA、降维、特征表示和线性代数关联明确。

### 2.2 必须完成的论文研究 / Paper Study and Critical Analysis

- 解释问题定义、动机、方法和主要贡献。
- 将论文方法与课程概念联系起来。
- 解释实验设置和结果，判断哪些证据支持哪些主张。
- 批判分析假设、优点、限制，提出合理改进/扩展。
- 体现小组自己的理解、推理和学习过程；只转述摘要或展示高准确率不足以覆盖要求。

### 2.3 可选复现加分 / Optional Bonus: Experiment Reproduction

- 可以复现数值、图形或实验；**完整和部分复现都可计入 bonus**。
- 说明工作范围，与原文结果比较，并分析差异和困难。
- 从零实现，尤其在没有公开官方代码时，会在 bonus 评估中得到额外考虑。
- 提交代码时，要求清晰组织的**公共 GitHub 仓库**及 `README.md`：论文信息、环境/setup、运行方法、复现结果。
- 明确本组贡献和复用代码，致谢原始来源。
- Bonus 根据工作质量、范围、文档清晰度、分析深度评定；不是“运行次数多”或“超过一个数字”自动得满分。
- 可使用 CS department GPU farm；本项目目前主要用本机CPU完成，无强制GPU要求。

### 2.4 交付物 / Deliverables

- **Written Report：约3–4页，LaTeX编写。** 推荐包含：问题/动机/贡献；关键方法、算法、理论及课程联系；实验设置与结果解释；假设、优缺点和限制；小组见解、改进或扩展；复现设置、与原结果比较、困难、代码链接。
- 正确引用论文和其他来源；来自原文的结果/图与本组产生的结果/图区分标注。
- 复现提交需公共代码仓库、运行说明和文档。
- 现有 `docs/REPORT.md` 是详细项目材料，**不是已排版的3–4页最终LaTeX报告**。

### 2.5 展示与问答 / Presentation & Q&A

- **9分钟展示 + 3分钟Q&A**；所有组员应参加展示和问答。
- 必需内容：overview（问题/动机/贡献）、methods（核心算法和课程联系）、evidence/results（设置、主要发现及证据是否支持主张）、critical analysis（假设、优缺点、限制、扩展）、team insights（学到什么及克服的概念困难）。
- 若做复现，在规定时间内简明展示方法、结果、与原文比较、经验。
- 使用清晰图表并注明来源；幻灯片简洁聚焦；强调团队理解与推理。

### 2.6 评分 / Evaluation Criteria

| 项目 | 权重 | 核心要求 |
|---|---:|---|
| Paper Understanding | 40% | 准确、深入解释问题、方法、贡献和课程关联 |
| Critical Analysis & Insight | 30% | 解读证据，评估假设与限制，提供有支持的见解和扩展 |
| Written Report | 10% | 清晰、组织和连贯性，正确图表与引用 |
| Presentation & Q&A | 20% | 结构、视觉表达、答问质量 |
| Optional reproduction bonus | 最多额外10个百分点 | 与基础评分分开；质量、范围、文档、分析深度 |

**决策影响：70%的基础分集中在理解和批判分析。已有复现材料很丰富，但仍需把它转化成清晰报告和9分钟叙述。不能承诺最终成绩或满额bonus。**

---

## 3. 教师、TA与作者通信：原句及实施含义

### 3.1 教师确认的协议（用户提供的邮件）

> FERET and CMU PIE are not held by the teaching team. Request them here:
>
> - FERET Fa/Fb, 1,195 subjects: https://www.nist.gov/itl/products-and-services/color-feret-database
> - CMU PIE: https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html
>
> Use 63 subjects, poses C27, C05, C29, C09 and C07, illuminations 08/11 for PIE-1 and 10/13 for PIE-2.
>
> **If either set cannot be obtained in time, skip it. ORL and MNIST are enough.**
>
> Align FERET, ORL and PIE by fixing the two eyes, then normalize and apply histogram equalization: FERET 41×27, ORL 56×46, PIE 64×64. MNIST stays 28×28; use 500 images per digit.
>
> Gabor uses five scales and eight orientations. Reconstruction keeps 56×46×40 (ORL) and 64×64×40 (PIE-1). Recognition downsamples to 28×23×40 (ORL) and 16×16×40 (PIE); MNIST stays 28×28×40.
>
> FERET: first 50/100/200 subjects for training, the rest for testing; Fa is gallery, Fb is probe. ORL: 2 images per subject at random for training/gallery, the other 8 as probe. PIE: 3 per subject at random as gallery, the other 7 as probe. MNIST: G30/P470 and G40/P460. Classify with nearest neighbour.
>
> Tmax is the iteration limit of CSA. The paper sets Tmax = 3 for the recognition experiments. N is the number of training images and Nc is the number of classes; before LDA, keep the PCA or CSA dimension at most N − Nc, so the within-class scatter is not singular.
>
> If a dataset or setting cannot be matched exactly, **state the difference in the report. That will not affect the final grade**.

完整邮件及申请经过：[HISTORY 历史03](./HISTORY.md#history-03)。图像尺寸在代码中采用数组高×宽；眼坐标为(x,y)，不能颠倒。

实施说明：FERET本组用前q身份的Fa/Fb学习，其他身份Fa做gallery、Fb做probe；按官方配对表保留的顺序定义“前q”，顺序属于记录的约定。LDA前维数≤N−Nc是必要的秩上限控制，**不保证所有实际输入的类内散度一定满秩**，仍需数值零空间处理。

### 3.2 TA Qiyuan 最新回复

> Dear Zhengwei,
>
> This is a 2008 project and is no longer maintained. The teaching team does not have the original code or hyperparameter records, so please set the remaining ORL alignment and Gabor choices from the public literature and standard practice. Keep them fixed across methods, and state each difference in the report. An exact match to the published numbers is not required and will not affect the grade.
>
> Best regards,
> Qiyuan
> TA, COMP3314

原句：[HISTORY 后续通信](./HISTORY.md#history-14)。含义：剩余ORL对齐/Gabor可按公开文献和通行实践确定，但**同一比较条件跨方法固定一致**，逐项说明差异。没有取消双眼对齐、训练量、Tmax或NN要求，也没有确认本组的新设置等于作者原设置。

### 3.3 PIE获取结论：两封均来自Terence Sim

> Unfortunately, I do not have the rights to distribute the CMU PIE dataset. The rights belong to CMU.

> I do not have Simon's current contact. But he left cmu a long time ago.
>
> He also does not have the rights to distribute.

用户确认Simon指Simon Baker。记录：[HISTORY 历史12](./HISTORY.md#history-12)。

- 本组在期限内未取得**原始CMU PIE**，按教师许可省略。
- 正确原因是所联系作者无分发权，权利属于CMU；回信说Baker已离开CMU且也没有分发权。
- **不能改写成作者没有访问权、数据已销毁或全世界都无法取得。** 不附未经提供的邮件头/发送日期。
- Multi-PIE不是原始PIE的等价替代。所给官网链接不能证明已取得原论文那63人子集。
- 不填造PIE的Table III、Fig.5(b)、Fig.8。

可用于正式英文报告：

> We were unable to obtain the original CMU PIE dataset within the project timeframe. Terence Sim informed us that he does not have distribution rights, which belong to CMU. He also explained that Simon Baker had left CMU and likewise did not have distribution rights. We therefore omit the PIE experiments, as permitted by the instructor, and report our reproduction scope as ORL, MNIST, and grayscale FERET.

---

## 4. 目标论文、背景知识与科学主张

### 4.1 论文身份与来源

**Dong Xu, Shuicheng Yan, Lei Zhang, Stephen Lin, Hong-Jiang Zhang, Thomas S. Huang.** *Reconstruction and Recognition of Tensor-Based Objects With Concurrent Subspaces Analysis*. IEEE Transactions on Circuits and Systems for Video Technology, 18(1),36–47,January2008. DOI:10.1109/TCSVT.2007.903317。

- [本地原始PDF](../Reconstruction_and_Recognition_of_Tensor-Based_Objects_With_Concurrent_Subspaces_Analysis.pdf)
- [用户给的HKU代理IEEE页面](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/document/4358655)
- [DOI页面](https://doi.org/10.1109/TCSVT.2007.903317)
- 原文文本已提取在 `tmp/pdfs/csa_audit_text.txt`。抽取的表格/公式可能缺字符，关键数值要看PDF页图。
- 页图 `tmp/pdfs/csa_audit-08.png` 为印刷页43：Table I/II及best-results说明；`-09.png` 为页44：Fig.7/8与Table III/IV。以PDF实际内容为准，不混淆文件页与印刷页。

### 4.2 它提出什么问题、做什么

- 图像本来是二维矩阵；Gabor图像是高×宽×通道的三阶张量。传统PCA把所有元素拉成向量，投影矩阵可能很大；高维小样本下学习到的方向受到样本数限制。
- CSA保持对象的多个模式（mode），分别学习行、列、通道等投影基，把对象变成更小的核心张量。
- 目标是**最小化重建误差**，不是直接最大化分类准确率。利用列正交投影，等价于最大化投影后核心张量总能量。
- 交替固定其他模式，更新当前模式的领先特征子空间；更新后的基立即用于后续模式。
- 识别时将核心张量展开成特征，直接NN，或接普通Fisher LDA再NN。无监督CSA使用学习图像；LDA还使用学习身份标签。
- 二维CSA用于灰度人脸；三维CSA用于40通道Gabor。样本轴不是对象模式，**不能把N张图的样本轴一并压缩**。

### 4.3 关键数学与英文术语

| 概念 | 英文 | 本项目中的含义 |
|---|---|---|
| 张量 | tensor | 多维数组，例如ORL识别张量28×23×40 |
| 模式/展开 | mode / unfolding | 固定某一维，其他维排列成列，便于求模式协方差 |
| 核心张量 | core tensor | 投影后的r₁×r₂×r₃表示，不是分类标签 |
| 正交基 | orthonormal basis | UₖᵀUₖ=I，保持投影/重建等价关系 |
| 并行子空间分析 | concurrent subspaces analysis | 多个模式子空间联合优化，实际算法逐模式交替更新 |
| 类内/类间散度 | within-/between-class scatter | Fisher LDA希望类内紧凑、类间分离 |
| 小样本问题 | small sample size problem | N相对维度很小，协方差/散度估计不稳 |
| 画廊/探针 | gallery / probe | NN参考图像与待识别图像 |
| 收缩正则化 | covariance shrinkage | 本组扩展，抑制类内散度小特征值过度放大 |

核心公式（示意，×ₖ为第k模式乘积）：

```text
Gᵢ = Xᵢ ×₁ U₁ᵀ ×₂ U₂ᵀ … ×K U_Kᵀ
Xhatᵢ = Gᵢ ×₁ U₁ ×₂ U₂ … ×K U_K
min Σᵢ ||Xᵢ − Xhatᵢ||²_F, subject to UₖᵀUₖ=I
```

当前模式的其他模式投影为Yᵢ后，构造 `Σᵢ Yᵢ(k)Yᵢ(k)ᵀ`，取领先特征向量作Uₖ。PCA是向量空间的一次特征分解；CSA施加可分离的多模式结构。两者不能仅用“特征维数相同”就认定参数/表达能力完全一样。

普通LDA解 `S_b w = λ S_w w`；ORL G2：N=80，Nc=40，故LDA前维数≤40，最终维数还受有效Sw秩与Nc−1限制。小类内特征值即使不为零，也可能放大采样噪声。

### 4.4 贡献、时代背景与限制

- 在2008年前后的传统视觉表示与小样本降维研究中，强调保留张量结构、减少投影参数、拓展可用投影方向，是本论文的重要贡献。
- 表示方法不依赖像Tensorface那样完整的身份×姿态×光照集合；重建实验每人仅一张图也可学。
- “小样本优势”是重要主张，但不是只研究小样本；论文也用MNIST考察高维数字表示。不能把CSA说成专门为人脸小样本分类优化的监督算法。
- 重建能量大不一定保留更有判别力的信息；交替优化有结构限制，有限迭代、初始化、模式秩、输入细节可影响结果。
- 原文的CSA390与PCA76，以及纯CSA390与CSA24+LDA23，并非等特征/等存储预算。
- 旧小数据集结果不能直接推断现代大规模人脸或数字识别优越性；本项目目标是复现与分析该论文，不是建立现代SOTA。

### 4.5 原论文“best results”的准确解释

原文IV-C（印刷页43）明确：

> we report the best results, with the optimal parameter configurations given in brackets.

随后解释括号主要表示最优降维维数/前后级维数。原文没有交代重复划分次数、随机种子、均值/SD或是否在很多随机划分中选最大值。

**可确认它报告最佳参数结果；不能确认它报告多次平均，也不能确认它像本组最新实验一样跨200划分选最大值。没有证据指控作者挑种子或造假。**

---

## 5. 数据获取、现有文件与具体协议

### 5.1 数据清单

| 数据 | 当前状态 | 位置与来源 |
|---|---|---|
| ORL / AT&T faces | 已下载核查，40人×10张=400，原始112×92 | `data/raw/orl/`；Cambridge AT&T archive |
| MNIST | 官方测试IDX10000张已下载；每类前500张组成5000样本 | `data/raw/mnist/`；OSSci MNIST镜像，已对官方常用MD5 |
| 原始灰度FERET | 已授权下载所需子集，3680张TIFF，官方1195Fa/Fb对 | `FERET/`，NIST授权归档中附带gray_feret |
| 原始CMU PIE | 未取得，按教师许可省略 | 获取经过见第3.3节 |

ORL下载源：[att_faces.zip](https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip)。MNIST来源与哈希见 `data/dataset_manifest.json`。**该清单是早期下载阶段快照，里面“未取得FERET”“尚未实现”的scope现已过时，不得据此覆盖后续FERET事实。**

MNIST论文同时说“测试集前半”和“每类500张”，实际官方前5000张并不均衡。本组按教师“每类500张”要求，选择每个数字的前500张；具体原作者样本索引仍未知。

### 5.2 FERET必须遵守的文件约定

- 先读 [FERET/README.md](../FERET/README.md)、`download_report.json`、`verification.json`。
- **优先使用 `official_fa_fb_pairs.csv`**：1195身份2390图，全部有效眼坐标；按官方gallery名单匹配身份。不能重新从额外候选挑易识别图。
- 全部3680张TIFF，每张原始宽256、高384，附`.gnd`及XML；2390标准配对包含在其中。
- 官方gallery名单1196人，Fb1195人；独有00989不进入配对。不要随便增为1196对。
- TIFF存在L和灰度调色板P模式，读取 `Image.open(path).convert('L')`。
- 官方左右眼指人体左右，按x排序转换为画面左右，坐标(x,y)。额外317候选缺眼坐标，不属于标准配对。
- 来源不是把Color FERET彩色图转灰，而是授权大归档附带的原始灰度库。
- 下载说明中“尚未预处理”只是下载完成时状态；后续已有 `data/processed/feret/` 和实验输出。
- 不公开上传原始FERET或受限特征缓存；获取说明与脚本可与受限数据分开整理。无账号、密码写入交接或公开文件。

### 5.3 已确认协议与本组未知细节

| 项目 | 确认要求 | 本组约定/未知内容 |
|---|---|---|
| ORL灰度 | 56×46，双眼对齐、归一化、histeq，G2/P8 | YuNet/Haar近似眼点、目标眼距、裁剪补边、插值；原手工眼点未知 |
| ORL Gabor识别 | 28×23×40，5尺度8方向 | 核大小、边界、幅值处理、通道顺序、采样；逐协议保存 |
| MNIST | 28×28，每类500，G30/P470或G40/P460，Gabor28×28×40 | 每类前500索引、具体Gabor数值参数、随机种子 |
| FERET | 41×27，q50/100/200，其他身份Fa/Fb识别 | 官方配对顺序作为训练身份顺序、眼目标、强度处理、距离 |
| CSA | 识别Tmax=3，正交交替更新 | identity/random/HOSVD初值、可选中心化、epsilon=.001 |
| LDA | 前维数≤N−Nc，普通LDA基线 | 数值零空间阈值、方向列归一化、最终维数 |
| 分类 | 最近邻 | 居中/原点余弦、欧氏；同条件跨方法一致 |

旧Gabor基本式λ=4√2^s（s=0…4），θ=oπ/8（o=0…7），σ=2π，DC-free复核、响应幅值；`gabor.py`保留旧reflect流程。后续ORL试验采用过理论支持零边界和39×39零边界，不替换MNIST/旧重建的缓存。

最新探索中d20目标眼点为(13,24)/(33,24)、仿射replicate；d18为(14,25)/(32,25)、仿射zero。Gabor均39×39核、零补边，先56×46滤波，再`[::2,::2]`采样。**仿射裁剪补边与Gabor卷积补边是不同设置。**

公开依据已在此前查询并用于文档：

- [CSU官方归一化说明](https://www.cs.colostate.edu/evalfacerec/data/normalization.html)：固定眼点、裁剪与强度处理；不是作者ORL协议的证明。
- [Wiskott等1997论文](https://cenl.ucsd.edu/pubs/ieee-1997.pdf)：多尺度方向Gabor及幅值表示；不能把本组流程称为其完整弹性图匹配。
- [Kolda–Bader张量综述](https://www.mathsci.ai/publication/TensorReview.pdf)：HOSVD与交替更新；HOSVD是本组起点，不是已恢复作者代码。

---

## 6. 代码实现、复现类型与验证边界

### 6.1 是否有官方代码

此前已搜索原作者/论文相关页面，没有取得**已核验可运行的官方CSA复现代码**；教学团队明确没有原代码/超参数记录。不应把“本组没找到”写成“证明从未有任何代码”。当前为按论文公式自行实现，基础运算由NumPy/SciPy提供，没有调用现成CSA/Tucker/PCA/LDA学习算法。

代码由Codex辅助编写，小组应阅读、理解、核查，并按课程规定如实披露辅助来源及成员工作。不要把AI生成/复用内容包装为完全独立手写。

### 6.2 核心实现

- `csa.py`：一般n阶CSA、投影/重建、PCA、普通Fisher LDA、NN。模式依次更新；不压缩样本轴。
- 新增可选`initial_bases`供训练生成的HOSVD基输入，验证形状、有限性及正交性，复制传入数组。无该参数的identity/random路径已与历史实现逐元素一致回归。
- PCA中心化；CSA原基线不中心化，后来可显式比较中心化。两种处理必须记录。
- 普通LDA使用Sw正特征子空间白化与数值零空间剔除，α=0，无新增ridge。
- LDA列尺度可为单位列或Sw度量尺度；NN可为gallery均值居中余弦、原点余弦、居中欧氏。线性映射方向的任意尺度会影响实际距离，不能当作无关细节。
- HOSVD模式协方差和训练均值仅由当次学习图像生成。

### 6.3 收缩、样本量与大预算属于扩展

```text
Sw(α) = (1−α)Sw + α trace(Sw)/p · I
```

α候选0/.01/.05/.15/.35/.65/1，通过gallery内验证选择。正α改变普通LDA目标，因此单列为本组扩展；不是论文原算法，也不是自动Ledoit–Wolf估计。固定样本量实验还有α=.15对照。

- G3/G5增加训练图是扩展，不能写成G2/P8复现成功。
- 纯CSA560/720/1120不接LDA，故不受LDA前≤N−Nc限制，但比原纯CSA390预算大，单列。
- 当前NN/Gabor约定可在教师许可下探索；不同方法使用不同输入的各自最大值不能拿来证明方法更强。

### 6.4 四类结果必须分开

1. **共享输入、论文维数固定对照**：用于跨方法比較，保留原表维数。
2. **gallery内选参**：选择仅用gallery折；外层probe未用于本轮选择。不过历史probe已开发观察，不能称全新盲测。
3. **probe选择的探索最高值**：最近200划分网格；学习模型只拟合gallery，但probe分数选择配置和种子，选择阶段已经使用测试信息。
4. **本组扩展**：正α收缩、更大纯CSA预算、G3/G5训练量等。

后验赢家固定配置的200均值也经过配置选择，不能称独立无偏泛化估计。新随机种子仍复用同400图，不能通过换seed把已观察数据变成独立盲测。

### 6.5 已排查什么

- 原始眼点/裁剪有影响，已修正，且有只换眼坐标的固定参数对照。
- 独立核对CSA逐模式协方差、投影、重建、LDA散度和广义特征方程、NN；未发现足以解释剩余差距的核心计算错误。
- 核对空间轴、通道置换、重新对齐、距离原点/方向尺度、核支持/取整和增加迭代；没有单个设置恢复所有普通CSA+LDA数值。
- FFT Gabor与直接二维卷积核对，误差约3.1×10⁻⁷。
- 完整保存划分、预测、配置、源码/特征哈希、均值/SD；验证通过说明记录和计算内部一致，**不证明未知作者协议已匹配，也不代表所有历史运行都被另一个程序独立重新拟合**。

---

## 7. 当前结果与离论文的距离

数值单位%，差值单位百分点(pp)。以下并列不同口径，各行已标注；不能合成一张“本组全面超过论文”的排行榜。

### 7.1 三数据集主要差距

| 条件/方法 | 论文 | 当前结果 | 数字差距与判断 |
|---|---:|---:|---|
| FERET q50 CSA+LDA，共享固定余弦 | 88.2 | 88.73 | +0.53pp，数值已接近 |
| FERET q100 CSA+LDA，同上 | 92.2 | 94.25 | +2.05pp |
| FERET q200 CSA+LDA，同上 | 95.2 | 95.38 | +0.18pp |
| MNIST G30/P470 CSA，3种子 | 92.8 | 90.34±0.15 | −2.46pp；CSA优于同输入PCA的趋势保留 |
| MNIST G40/P460 CSA，3种子 | 92.4 | 91.04±0.25 | −1.36pp |
| ORL纯Gabor CSA390，共享论文维数，10种子 | 94.4 | 90.19±2.37 | −4.21pp |
| ORL普通Gabor CSA24→LDA23，同上 | 90.9 | 79.66±2.77 | −11.24pp，原维数配置的主要缺口 |
| ORL普通Gabor CSA+LDA，最新gallery内选参，10种子 | 90.9 | 83.28±3.30 | −7.62pp；最高89.06，但维数按gallery调整 |
| ORL灰度CSA+普通LDA，后续优化，10种子 | 84.1 | 78.56±3.98 | −5.54pp，仍未恢复原数值 |
| ORL灰度CSA+收缩LDA扩展，10种子 | 原普通LDA84.1 | 84.13±2.38 | 数字接近，但算法改变，不能声称原普通LDA匹配 |
| ORL Gabor CSA+收缩LDA最新扩展，10种子 | 原普通LDA90.9 | 85.47±4.57 | 仍低5.43pp，且为扩展 |

灰度后续78.56不是全部历史结果中最大的均值；此前ORL审计gallery选择灰度CSA为79.75±4.33。不要以“最新”暗示每次调参都更好。本文只列明确命名流程，保留负结果。

FERET固定余弦PCA+LDA：89.08/90.87/93.87；原文84.6/90.6/93.4。q50本组PCA高于CSA，q100/q200 CSA较高，**没有完整恢复论文全部相对优劣**。FERET为一次固定身份拆分，没有随机SD；旧表SD=0是占位，现主表留空。

MNIST固定PCA：G30=87.52±.40（论文88.5）；G40=88.58±.37（论文90.5）。历史逐方法训练内选参：PCA90.12/90.75，CSA90.22/91.22，因方法可用不同输入，单列探索，不替换共享主表。

后续MNIST共享输入有限优化另列：四种Gabor支持/边界，各外层划分独立5折gallery内预筛最多两个输入，再选择共享NN和identity/HOSVD；论文PCA60/50、CSA810/975、Tmax3不变，不接LDA、不增加训练图。20组选择全部冻结且200个被选内层模型重放后才评分；20组均选择identity。160评分、744000预测、78个不同外层模型的投影抽样/正交性及全部NN核查通过，两种G的CSA更新独立复算最大子空间误差4.13×10⁻¹⁰。新CSA十次均值88.89%/90.18%是负结果，**不替换此前90.34/91.04三种子主表**。输入和距离仅是本组约定：满足已确认教师约束、保留Table IV维数，不表示未知作者设置已确认；本轮未扩大秩网格或按probe另选输入/种子。G40只换输入的90.62%为事后消融，不改称验证赢家。

权威定位：当前共享表为 `results/orl_protocol_audit/shared_comparison.csv`。`results/shared_protocol_final/shared_comparison.csv`是**更早版本**，旧ORL分数和FERET SD占位不能当当前最终表。

### 7.2 200划分最佳观测搜索：已完成、已核查

范围：种子3000–3199，共200；3输入分支×15CSA案例，每划分45拟合，共**9000次拟合**；6003有效固定配置×200，**1,200,600评分格**。同400图被重复使用，不是百万独立实验。

| 记录 | 最高值/正确数 | seed | 固定赢家配置200均值±SD | 达到论文次数 |
|---|---:|---:|---:|---:|
| 普通CSA32→LDA24 | **95.00 / 304 of 320** | 3158 | 86.415625±2.446007 | ≥90.9：5/200 |
| 纯CSA560 | **97.50 / 312 of 320** | 3188 | 91.43125±1.860908 | ≥94.4：8/200 |
| 纯CSA390，原维数预算 | **97.1875 / 311 of 320** | 3025 | 91.998438±1.915624 | ≥94.4：15/200 |
| 普通CSA24→LDA23，原维数配置 | **88.4375 / 283 of 320** | 3166 | 80.159375±2.901389 | ≥90.9：0/200 |

完整赢家配置：

| 参数 | 普通95.00 | 纯560：97.50 | 纯390：97.1875 |
|---|---|---|---|
| 输入 | d20、replicate仿射、fixed39零Gabor | d18、zero仿射、fixed39零Gabor | 同纯560 |
| 幅值处理 | x^0.75 | sqrt(x) | sqrt(x) |
| CSA秩 | 4×1×8=32 | 14×2×20=560 | 15×1×26=390 |
| 初始基 | 学习集HOSVD | 学习集HOSVD | identity |
| CSA中心化 | false | false | false |
| LDA | 普通α=0，24维，单位列方向 | 无 | 无 |
| NN | gallery均值居中欧氏 | gallery均值居中余弦 | 原点余弦 |
| Tmax | 3 | 3 | 3 |

原24→23最高配置：d18/sqrt/identity/noncentered，单位列LDA，gallery均值居中余弦。

- 输出根：`results/orl_best_search/`。
- `best_results.json`：两方法全局赢家；`paper_dimension_best.json`：原390和24→23子组。
- `configuration_summary.csv`：6003配置统计；`scores/*.npz`：全部200评分；`near_or_exceed.csv`：6432条距论文不低于−1pp的记录。
- 普通CSA+LDA有2306格≥90.9，涉及62种子；纯CSA133格≥94.4，涉及28种子。这是**整个网格**有成功配置的种子数，不能当固定配置成功率。
- `verification.json`通过：20个不同高分模型重新拟合，独立核对6400条预测；全部CSV与评分格对应。不等于9000模型全部独立重拟合。
- `statistical_readback.json`通过：独立回读6003配置统计，原维数两赢家重放，6个共享输入对照，冻结源码hash一致。
- `verified_best/`保留高分嵌入和重放记录，`source_snapshot/`保留源码，`best_distribution.png`为对应赢家分布图。
- `statistical_readback.json`及`same_input_controls.json`由补充回读代码产生，不应未经检查声称标准stage自动生成它们。

同赢家划分、输入与距离的事后对照：seed3158普通CSA32+LDA24=95，原CSA24+LDA23=84.0625，纯CSA390=87.1875，PCA33+LDA25=85.3125。seed3188纯CSA560=97.5，原CSA24+LDA23=81.875，纯CSA390与PCA33+LDA25均96.5625。这些不是全划分优势，也非等预算比较。

用户要求保留的英文句式及必须伴随的说明：

> we report the best results, with the optimal parameter configurations given in brackets.
>
> In this reproduction, “best” denotes the maximum probe accuracy observed across 200 predefined random splits and the finite configuration grid. “Optimal” is restricted to that grid. Probe scores were used to select the configuration and split, so these results are exploratory maxima, not an independently validated estimate of generalization. We also report the corresponding fixed configuration's mean and standard deviation over all 200 splits and retain the complete search records.

**判断：高分运行已经可重放；保留原CSA390预算也能出现超过94.4的单次值。但原CSA24→LDA23在当前网格最高仍低2.4625pp，不能说原表所有配置已恢复。**

### 7.3 gallery内针对性迭代与100划分诊断

`results/orl_csa_refined/`，种子20–29，普通仅α=0；收缩单列。16内层折，每折75学习/5验证，40类保留，每张gallery验证一次；内层前维数≤35，外层≤40。候选初筛在完整gallery拟合，属于近似剪枝，不能称无偏LOO评分。

- 普通Gabor CSA+LDA83.28125±3.298154，max89.0625；此前82.46875，均值增加.8125pp，3提高/4相同/3下降。
- 收缩85.46875±4.574447，max90；此前84.75，4提高/4相同/2下降。
- 同普通输入纯CSA39089.96875±2.226585，max93.4375；原24→23锚点80.8125±3.261598；同输入PCA33+普通LDA92.09375±2.357253。
- 选定普通CSA保留训练总变化能量22.60%、类间变化26.07%，原24锚点19.11/22.11%。这不是身份信息百分比。
- 收缩赢家移除α后77±8.68；配置按收缩选出，不能拿这数替代重新选择的普通83.28。
- `evaluation/verification.json`：80评分、25600预测，重算60被选内层目标，Fisher相对残差最大约2.45×10⁻¹³，通过。

100固定配置诊断在本轮probe评分前冻结，seeds1000–1099，无幅值幂处理，d20/fixed39，identity/noncentered，单位列居中余弦：

| 固定配置 | 均值±SD | 最高 |
|---|---:|---:|
| 原CSA24→LDA23 | 80.925±2.97656 | 87.50 |
| CSA32(4×2×4)→LDA28 | 84.35625±2.84157 | 90.3125 |
| 纯CSA390 | 89.26875±1.99095 | 93.75 |

原24→23在100次也未达90.9，说明在该输入下“只换随机划分”没有恢复原数字。新200次又同时改变幅值、秩、起点、距离，不能把差异全部归因于多跑100次。100次核对300评分、96000预测记录；不表示96000独立图。

### 7.4 增加训练图的扩展：已经做完

`results/orl_sample_size/`：种子20–29。每人前5随机图作训练池，后5固定probe；取前2/3/5形成G2⊂G3⊂G5。较小G未使用的训练池图不进入P5；G2另有P8原协议锚点。

| 相同固定P5，10次均值 | G2 | G3 | G5 |
|---|---:|---:|---:|
| 纯CSA390 | 89.80 | 93.95 | 97.45 |
| 纯CSA24 | 76.40 | 82.95 | 90.60 |
| CSA24+普通LDA23 | 81.30 | 88.10 | 94.50 |
| CSA24+固定α=.15 LDA23 | 81.90 | 88.40 | 93.90 |
| PCA24+普通LDA23 | 88.45 | 94.05 | 97.45 |
| 纯PCA24 | 82.70 | 88.00 | 93.45 |

所有四种CSA配置均在所有10种子随训练量增大而改善。90模型、280评分、64400预测核查通过，独立3轮模式投影差约1.31×10⁻¹²。支持样本量/小样本估计影响的解释，但**不是G2/P8的原实验成功**。

也说明：同24维纯CSA→加LDA由76.4升81.3，所以“纯390高于低维+LDA”不能简单解释成LDA一定有害；前级压缩预算差别很大。

### 7.5 重建、收敛与图形覆盖

- 已有ORL Fig.4/5(a)对应定性收敛/重建实验，以及早期识别维数曲线，位于 `results/available_reproduction_v2/`。
- 每人第一图，N=40；Gabor56×46×40；重建等模式秩r×r×r；这是**学习样本Gabor张量重建**，不是Gabor逆变换灰度脸或未见图泛化。
- `RMSE=sqrt(SSE/N)`，不是除以ND的每像素RMSE。D为单张输入全部元素。
- PCA存储量d(N+D)；等秩CSA存储Nr³+r(56+46+40)；CR=ND/存储浮点参数数量，含共享基，不是文件字节压缩率。
- `orl_convergence_reconstruction.png`、`orl_reconstruction.csv`、`orl_convergence.json`、`dimension_curves.csv`、`orl_lda_curves.png`保留。
- 它们仍用**早期Gabor/眼点流程**，没有随新识别优化重算；原文Gabor绝对幅值不明，不声称纵轴精确相等，不做事后乘系数来贴曲线。
- FERET有 `paper_comparison.png`、`lda_curves.png` 对应表格/维数曲线。不能把既有图都标成“同一最终输入版本”。
- PIE Fig.5(b)/8/Table III省略；其余图表有对应工作不等于每个原曲线点和配置已逐项匹配。

---

## 8. 当前差距的解释：证据强弱分开

### 已有支持

1. 输入对齐和裁剪会改变数值，固定对照已验证；旧流程低分不能再当最新结论。
2. 低维模式秩结构、初始化与后级距离组合影响普通CSA+LDA。32维调节有改善，但当前不能分离每个因素的贡献。
3. G2的Sw估计更不稳定，增加训练图明显改善；收缩可改善部分配置，但没有恢复原普通LDA。
4. 小预算CSA最大化总投影能量，可能压掉判别差异；同输入PCA低维对照往往更强。
5. 选参和划分口径会扩大最高值与平均值的差距；200次最大值超过论文不表示均值超过。

### 仍不能唯一确定

- 作者完整裁剪/眼点、Gabor有限支持和幅值归一化、具体距离与LDA方向尺度、初始化、MNIST索引、FERET排序，以及随机划分/聚合口径。
- 没有作者原代码或超参可逐行匹配。TA允许按文献约定并报告差异，这些未知项不再是课程完成的阻塞条件。
- 未找到核心数值错误不等于证明作者一定采用同输入；现有结果也不能证明作者选了很多次最好结果。
- 同400张ORL重复划分相互重叠。SD是划分分布描述，不能把所有probe记录视为独立观测宣称统计显著。

### 为什么CSA未必比PCA强

CSA的可分离结构带来更少基参数和不同归纳偏置，也限制允许的投影空间；优势依赖数据与秩结构。当前低维4×2×3对输入可能不适配，且CSA目标不是监督判别目标。纯390与低维24+LDA不公平等预算；本组等24维实验也没有支持CSA普遍更强。应把“没有稳定恢复原相对优势”作为批判分析结果，而不是先验认定代码必错。

---

## 9. 实现导航、环境与重跑

### 9.1 关键目录/脚本

| 文件/结果 | 用途与现状 |
|---|---|
| `csa.py`, `gabor.py` | 核心CSA/PCA/LDA/NN与旧Gabor实现 |
| `prepare_available_data.py`, `run_available_reproduction.py` | 早期ORL/MNIST准备、固定实验、重建/曲线 |
| `prepare_feret.py`, `run_feret.py`, `select_feret.py`, `verify_feret.py` | FERET官方配对处理、初始实验、训练内选择/核查 |
| `audit_orl_protocol.py` → `orl_protocol_audit` | 修正眼点/裁剪、共同输入比较、gallery选择、固定对照、秩补测 |
| `diagnose_csa_lda.py` → `csa_lda_focus` | 普通LDA数值/原点/轴与通道诊断，保留失败选择 |
| `optimize_csa_lda.py` → `csa_lda_optimized` | 灰度/Gabor普通与收缩训练内优化、去收缩消融 |
| `refine_orl_csa.py` → `orl_csa_refined` | Gabor专项、HOSVD、秩、归一化、十次结果及100固定划分 |
| `search_orl_best.py` → `orl_best_search` | 200次探索网格、完整评分统计、最高值重放 |
| `run_orl_sample_size.py` → `orl_sample_size` | 同P5 G2/G3/G5及PCA等24维对照 |
| `optimize_mnist_protocol.py` → `mnist_protocol_optimized` | MNIST四输入、5折gallery内共享选择、十次负结果及TA/原文符合范围 |
| `run_shared_protocol.py`, `verify_shared_protocol.py` | 此前四步共享协议对照；后续ORL有更新 |
| `docs/HISTORY.md` | 合并的早期13份文档，另附TA后续通信 |

旧结果目录多是有证据价值的探索，不应为“清爽”删除原始CSV、协议或快照。用户之前授权的是合并Markdown，不是删除实验数据。

### 9.2 环境

Windows PowerShell，Python3.11（现有绝对解释器 `C:\Users\gwdsd\AppData\Local\Programs\Python\Python311\python.exe`）。

锁定依赖：NumPy2.4.6、SciPy1.17.1、Pillow12.3.0、Matplotlib3.11.1、threadpoolctl3.6.0；预处理OpenCV-headless4.12.0.88装在项目`.deps/`，见两份requirements。

```powershell
Set-Location 'D:\HKU\COURSE\y2s1\COMP3314\COMP3314GP'
# 若环境尚未准备，再安装；已有环境不必重复安装。
python -m pip install -r requirements.txt
python -m pip install --no-deps --target .deps -r requirements-preprocess.txt
```

本机CPU已实际完成这些实验，无GPU要求。最近搜索用3进程各2BLAS线程，约23分钟；时间依赖缓存和机器，峰值内存未测，不给其他笔记本资源保证。当前交接无待续实验任务。

### 9.3 最简高分重放

```powershell
python search_orl_best.py --stage replay --family 'CSA+LDA'
python search_orl_best.py --stage replay --family CSA
```

分别重放已保存95%和97.5%赢家，不重新选参，不是随机典型运行。390维赢家和原24→23赢家的完整记录在 `paper_dimension_best.json`，该CLI的family选择没有单独390选项。

### 9.4 全部200划分重跑：只有需要时才做

```powershell
python search_orl_best.py --stage prepare --output results/orl_best_search_rerun
python search_orl_best.py --stage run --output results/orl_best_search_rerun
python search_orl_best.py --stage summarize --output results/orl_best_search_rerun
python search_orl_best.py --stage verify --output results/orl_best_search_rerun
python search_orl_best.py --stage plot --output results/orl_best_search_rerun
```

每步成功、结果检查通过后再下一步；prepare输出目录必须尚不存在。run可互不重叠地分批`--start/--stop`，stop不含；完整范围3000–3199。**依赖已有ORL特征缓存，不是从空仓库下载/生成全部数据的单入口。**

协议冻结包括源码和特征hash；修改搜索/core后，不要直接续旧评分，使用新输出根并说明版本差异。不要改源码以绕过hash检查。统计回读时不要覆写既有赢家索引。

### 9.5 gallery内迭代重跑

```powershell
python refine_orl_csa.py --stage prepare --output results/orl_csa_refined_rerun
python refine_orl_csa.py --stage select --output results/orl_csa_refined_rerun
python refine_orl_csa.py --stage randomness-plan --output results/orl_csa_refined_rerun
python refine_orl_csa.py --stage evaluate --output results/orl_csa_refined_rerun
python refine_orl_csa.py --stage randomness --output results/orl_csa_refined_rerun
python refine_orl_csa.py --stage analyze --output results/orl_csa_refined_rerun
```

十个种子20–29选择全部冻结后评分；randomness-plan在评分前。evaluate内置核查。现有结果仅核查：`python refine_orl_csa.py --stage verify`。依赖先前csa_lda_optimized输入/眼点和锚点，不是空数据入口。

### 9.6 样本量扩展重跑

```powershell
python run_orl_sample_size.py --output results/orl_sample_size_rerun
python run_orl_sample_size.py --diagnostic --output results/orl_sample_size_rerun
python run_orl_sample_size.py --analyze --output results/orl_sample_size_rerun
```

同样用新输出根、逐步核查；仅复查现有输出：`python run_orl_sample_size.py --verify`。依赖ORL审计catalogue/缓存。

其他准备与运行入口的依赖、默认输出可能是早期版本，先查`--help`和REPORT第9节，不要盲跑默认值覆盖。准备干净公共仓库时需要另做真正从原始数据到最终固定实验的运行链，不能只提供依赖未发布results缓存的命令。

---

## 10. 验证和证据入口：优先读这些

| 证据 | 位置 | 已有核查含义 |
|---|---|---|
| 当前三数据集共享表 | `results/orl_protocol_audit/shared_comparison.csv` | 当前ORL及复用MNIST/FERET，不混用旧共享表 |
| 最新探索高分 | `results/orl_best_search/best_results.json` | 正确计数、参数、均值SD、选择口径 |
| 原论文维数高分 | `results/orl_best_search/paper_dimension_best.json` | 原390与24→23子组重放 |
| 探索模型验证 | `results/orl_best_search/verification.json` | 200完整评分、20模型/6400预测独立核查 |
| 额外统计回读 | `results/orl_best_search/statistical_readback.json` | 6003配置统计、原维数2重放、6共享条件对照、源码hash |
| gallery内迭代 | `results/orl_csa_refined/evaluation/results.csv`、`summary.json`、`verification.json` | 80评分、25600预测、普通/扩展分开 |
| 参数与能量 | `results/orl_csa_refined/selected_settings.csv`、`training_energy.csv` | 每个种子实际选择、训练诊断 |
| 固定100诊断 | `results/orl_csa_refined/randomness/results.csv`及`verification.json` | 所有100保留，300评分/96000记录 |
| 样本量扩展 | `results/orl_sample_size/summary.json`、`analysis.json`、`verification.json` | 90模型和固定P5真实配对 |
| FERET | `results/feret_validation_selected_v3/fixed_results.csv`、`selected_configs.json`、`verification.json` | 区分validation_selected与cosine_control |
| MNIST/早期重建 | `results/available_reproduction_v2/fixed_results.csv`、`verification.json` | 旧输入和三种子，未伪装成最新重跑 |
| MNIST最新有限优化 | `results/mnist_protocol_optimized/comparison.csv`、`original_three_seed_comparison.csv`、`protocol.json`、`verification.json` | 十次基线与选参配对、原三次对照，负结果完整保留；补充统计/符合范围见`supplementary_readback.json`，非标准stage自动生成 |

数值真实性的核查不能排除选择偏差；数据统计hash正确也不等于原作者图像索引相同。遇到CSV和文档冲突，核对流程标签、protocol与逐预测，不静默挑最大值。

---

## 11. 现在最值得补充什么：优先级与验收

以下为初次交接时的建议，不代表已执行。后续MNIST有限优化已单列完成，当前无待续实验；新实验仍按用户下一项明确任务决定。

### P0：课程正式交付和结果口径（收益最高）

1. 冻结一张最终论文对照表：共享/原维数主结果、gallery内选择结果、探索best、扩展分别标识；原文没有SD，不擅自补SD或称其均值。
2. 3–4页LaTeX报告：明确范围ORL/MNIST/灰度FERET，PIE许可省略；说明输入/距离/排序/索引和统计差异。
3. 9分钟展示与3分钟Q&A：讲清张量→CSA更新→NN/LDA→数据证据→限制/小组洞见，避免展示历次搜索流水账。
4. 公共GitHub与README：环境、数据获取、从原始数据生成最终输入、固定协议运行、绘图、预期结果；引用与AI辅助/成员贡献说明。现有缓存依赖是需要整理的实际交付缺口。

验收：干净环境按README能运行选定最终实验或明确授权数据前提；表格可追溯正确计数；报告页数和时间符合要求；原图/本组图标识清楚。**当前项目内未发现已完成的.tex/.pptx正式材料，没有已确认发布的公共仓库。**

### P1：统一图形与一个有解释力的补充实验

- 整理Table I/II/IV对应对照，以及Fig.4/5(a)/6/7的已有曲线。若要声称同一最终流程，重建/收敛图需按最终固定输入重新生成，保留早期图，不任意缩放纵轴。
- 最有价值的新增实验是**固定相同划分和输入的消融**：24→32秩、identity→HOSVD、原幅值→0.75幂、距离余弦→欧氏、LDA23→24，分离因素及必要交互。目前95%赢家同时改变多项，不能解释究竟哪项造成改善。
- 与PCA共享输入、距离、划分，做相同前级/后级维数或相同存储预算的对照；已有等24维对照可先整理，不必重新全网格搜索。

验收：预先冻结小规模对照，报告全部划分及失败案例，增益必须来自明确的配对比较。新seed不自动构成独立盲测，不按测试得分删低分。

### P2：有助于讲解的附加材料

- 错例展示：同probe在CSA/PCA的最近邻、对应gallery图，讨论姿态、光照、眼点误差；仅作为事后诊断。
- 准确率分布与范围：现有200分布已足够展示best与mean不同，不需要为了最大值无休止增加seed。
- 计算量/压缩效率：记录同机器/线程/输入的时间、基和系数存储，若测内存则注明测量方式；没有实测峰值不报猜测值。
- 如果时间有限，先用已有固定P5样本量实验、同24维LDA对照及分布图做批判分析，通常比再调几百次参数更有内容。

### 推荐9分钟叙述结构

| 时间 | 内容 |
|---|---|
| 0:00–1:00 | 向量化与张量：问题/动机 |
| 1:00–3:00 | CSA投影、交替更新、与PCA/LDA课程联系 |
| 3:00–4:00 | 论文实验和本组范围、教师协议、PIE省略 |
| 4:00–6:00 | 三数据集主结果，ORL原维数与探索最高值分开 |
| 6:00–8:00 | 小样本、模式预算、LDA和输入差异的证据 |
| 8:00–9:00 | 贡献、限制、团队学习与结论 |

Q&A应能回答：为什么不压样本轴；为何重建好不保证识别好；390与24+LDA能否公平比較；Sw秩为什么≤N−Nc；正α为何是扩展；95%为什么不能当平均；PIE为何省略；哪些代码/分析是本组贡献。

---

## 12. 新会话注意事项与可直接复制的开场

### 不要重复踩的坑

- 不把95/97.5探索值替代83.28等训练内均值，不把图像拟合隔离等同于选参未用测试。
- 不说原文是多次平均或证实作者跨多次挑最好；原文未明确统计规则。
- 不说只剩Gabor有差距：灰度普通CSA+LDA仍低，MNIST有小差距，PIE缺失，部分原维数相对趋势不同。
- 不把正α、G5/P5、560维纯CSA叫原协议精确复现。
- 不把TA允许差异理解为取消共享输入/训练量/Tmax要求。
- 不把数据清单旧scope、旧共享表、旧图和download时状态当当前状态。
- 不因旧结果更多而删除证据，不额外生成大量计划MD。
- 不未经授权上传FERET/缓存，或披露下载凭据。
- 不先重跑全部实验；先从当前已验证记录和用户具体下一项任务着手。

### 可复制的新会话提示

> 请先阅读 `D:\HKU\COURSE\y2s1\COMP3314\COMP3314GP\docs\HANDOFF.md`，再按其中导航读取 `README.md`、`docs/REPORT.md`及相关结果协议。这里包含项目来源、CSA原论文、课程截图评分、教师/TA与作者回信、数据、代码、全部关键结果和排查结论。请保留共享原维数结果、gallery内选参、probe选择的探索最高值以及扩展实验之间的区别，不把未知协议写成原作者已确认设置。PIE经教师许可省略。先简短说明你接手后的理解，再按我接下来指定的任务执行；不要自动再做全网格搜索或新增大量Markdown。

初次交接已完成资料与结果回读，并检查当时本文全部本地链接；当时新增本文、README导航，澄清probe选参范围及FERET流程标签，未改变算法、参数、数据或成绩。后续已新增MNIST独立实验脚本及结果，并更新现有三份导航文档；原算法、原数据、历史成绩及其他数据集结果均保留。
