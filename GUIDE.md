# COMP 559 项目完整讲解（给团队看）

> Rice University — Machine Learning with Graphs — Spring 2026
> Team: Jingwen Chen, Jingtao Wang, Shouyu Wang
> 本文档是给全组成员的技术总览和操作手册。读完这一份就能开工。

---

## 目录

0. [一分钟版本](#0-一分钟版本)
0.5. [当前进度 & 下一步](#05-当前进度--下一步)
1. [项目是什么](#1-项目是什么)
2. [从原提案到当前方案](#2-从原提案到当前方案)
3. [数据集 —— Hetionet 详解](#3-数据集--hetionet-详解)
4. [核心知识点（必读）](#4-核心知识点必读)
5. [系统架构](#5-系统架构)
6. [代码地图（每个文件在干什么）](#6-代码地图每个文件在干什么)
7. [环境部署](#7-环境部署)
8. [完整跑通流程](#8-完整跑通流程)
9. [实验设计（给 Report 用）](#9-实验设计给-report-用)
10. [Report 写作骨架（8 页）](#10-report-写作骨架8-页)
11. [分工建议](#11-分工建议)
12. [常见问题与坑](#12-常见问题与坑)

---

## 0. 一分钟版本

我们在做的事情：
- 在 **Hetionet**（一张包含 47,031 个生物医学实体的异构知识图谱）上做 **drug repurposing**：预测哪个化合物能治哪个疾病。
- 方法分四层：**异构 GNN** 学节点嵌入 → **元路径检索** 找到 Compound 到 Disease 的可解释路径 → **自然语言化** → **LLM (Claude)** 基于这些路径生成判别 + 解释。
- 对比四个方法：GNN-only、DistMult(KGE)、LLM-only（不给路径）、GNN-RAG+LLM（我们的方法）。
- 指标：AUROC / AUPRC / Hits@K / 配对准确率 / LLM 解释忠实度。

原提案里"闭环 LLM-as-judge + RL"的部分砍掉了，放 future work。简化换来 6 周内可交付。

**我们这个项目属于研究生期末项目的中上水平（A-/A）**，novelty 不冲顶会但工程扎实、评测规范。

---

## 0.5 当前进度 & 下一步

### 已完成（截至 2026-04-19）

- [x] 方案设计 + 文档（本文件）
- [x] 环境部署 + Hetionet 数据加载（`scripts/inspect_data.py` 已跑通）
- [x] GNN 训练栈代码 + smoke test（6 epochs → AUROC **0.897**）
- [x] DistMult KGE 基线代码 + smoke test（3 epochs → AUROC **0.8615**）
- [x] 检索栈代码（已验证 Cholecalciferol → osteoporosis 的路径挖掘正确）
- [x] LLM 接入层代码（Anthropic SDK 封装 + prompt caching）
- [x] 主实验脚本（`main_results.py`，四方法对照 + McNemar + bootstrap CI）
- [x] 错误分析脚本 + DDR1 case study + top-K ablation

### 下一步按顺序执行

| 步骤 | 命令 | 耗时 | API key? | 产出 |
|---|---|---|---|---|
| 1 | `python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt` | ~10 分钟 | 否 | `gnn.pt` + 真实 AUROC |
| 2 | `python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt` | ~10 分钟 | 否 | `distmult.pt` + baseline AUROC |
| 3 | 注册 Anthropic 账号 + 充 $20 + 拿 API key | ~5 分钟 + $20 | — | env var 配好 |
| 4 | `python experiments/main_results.py --gnn-ckpt checkpoints/gnn.pt --kge-ckpt checkpoints/distmult.pt --n-pos 75 --n-neg 75 --judge` | ~15 分钟 + ~$5-8 | 是 | **主结果表** + JSONL |
| 5 | `python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10 --n-test 20 --n-neg 20` | ~10 分钟 + ~$2 | 是 | **top-K 曲线** |
| 6 | `python experiments/error_analysis.py --file runs/main_results.jsonl` | 几秒 | 否 | **错误分桶表** + 定性 case |
| 7 | `python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt` | ~30 秒 + 1 次 LLM 调用 | 是 | **DDR1 定性分析** |
| 8 | 全组按 §10 骨架写 8 页 report | ~1-2 周 | — | 最终交付（May 5）|

### 执行时的注意事项

**步骤 1-2（训练）**：
- 不需要 API key，不花钱，可以放着跑
- 训练完看 stdout 最后一行的 `TEST AUROC`。正式训练应该比 smoke test 高 3-5 个点（GNN 0.93+，DistMult 0.85+）
- 如果数字没涨反而掉，检查是否过拟合（early stop 已经在代码里：保留 val AUROC 最高的 checkpoint）

**步骤 3（API key）**：
- 去 https://console.anthropic.com 注册
- 充值最少 $5（够主实验用），保险起见充 $20
- 拿到 `sk-ant-xxxxx`，放到环境变量 `ANTHROPIC_API_KEY`
- Windows cmd: `set ANTHROPIC_API_KEY=sk-ant-xxx`（只在当前会话有效）
- 永久保存：Windows 系统属性 → 环境变量

**步骤 4（主实验）**：
- **最关键的一步**，这产生 report 里表 1 的所有数字
- 跑之前先小跑一次验证：`--n-pos 5 --n-neg 5 --judge`，花不到 $1 看看有没有报错
- 没问题再跑全量 N=150
- 如果钱紧：加 `--model claude-haiku-4-5`，成本降到 1/10 但质量会下降

**步骤 5（ablation）**：
- 对每个 K 跑一次 N=40（20 正 + 20 负），总共 5 × 40 = 200 次 LLM 调用
- K=0 是 LLM-only（不给路径），K=1/3/5/10 是 GNN-RAG

**步骤 6-7（分析）**：
- 便宜且必须跑
- 错误分桶产出的 case + DDR1 case 是 report §5 Case Study / Error Analysis 的直接素材

**步骤 8（写 report）**：
- 按本文档 §10 的 8 页骨架来
- 分工见 §11
- 用 Overleaf 协作，GitHub 放代码
- 每跑完一个实验，立刻把数字和图填进 Overleaf，别积压

### 预算总控

| 项 | 估算 |
|---|---|
| Claude Sonnet 4.6 主实验 | ~$6 |
| ablation_k | ~$2 |
| DDR1 case + 小实验 | ~$1 |
| 重跑补数据缓冲 | ~$3 |
| **合计** | **~$12** |

每人分摊 $4，或者用一个人的账号统一跑。

### 如果时间紧（最后一周）最低交付清单

按优先级排，砍到这个程度仍然能交：

1. 步骤 1（GNN 训练）—— 保 §4 主结果中 GNN-only 一行
2. 步骤 4（主实验 N=30）—— 最小可信的四方法对比
3. 步骤 7（DDR1 case）—— 一个漂亮的定性例子
4. 报告 §3 Method + §4 主结果表 + §6 DDR1

砍掉：DistMult（§4.3）、ablation K（§4.2）、error analysis（§5.1）、faithfulness（§4.4）。
这样最低能保 B+。但按完整方案跑完至少 A-。

---

## 1. 项目是什么

### 1.1 一句话定位

> 结合 **异构图神经网络** 与 **大语言模型**，在生物医学知识图谱上做可解释的 drug-repurposing 推理。

### 1.2 要解决的问题 —— Drug Repurposing

**Drug repurposing（药物再利用）**：找一个已批准的老药，看它有没有治别的病的潜力。

为什么重要：
- 新药研发平均 10 年、20 亿美元；老药已通过安全性验证，推广速度快几十倍
- 2020 年瑞德西韦 (remdesivir) 原本是抗埃博拉药，被 repurpose 到 COVID-19

在图上做这件事的动机：
- 药物 A 和疾病 B 之间如果有 "A 结合 蛋白 G → G 参与 通路 P → P 在疾病 B 中失调" 这样的路径，就有药理学依据
- 图神经网络可以学到节点和邻域的表示，帮我们算"A 离 B 有多近"

### 1.3 我们的技术路径

![pipeline](architecture 见第 5 章)

四层：
1. **GNN 编码器**：学每个节点（化合物、基因、疾病 ...）的 128 维向量
2. **子图检索**：给一对 (compound, disease)，枚举所有长度 ≤3 的预定义元路径，按 GNN 嵌入相似度取 top-K
3. **自然语言化**：把路径翻成英文，"Aspirin binds PTGS2, PTGS2 is associated with inflammation"
4. **LLM 推理**：Claude 读路径，输出 JSON `{prediction, confidence, rationale}`

---

## 2. 从原提案到当前方案

### 2.1 原提案要做什么

**标题**：Learning to Retrieve: Closed-loop GNN Subgraph Selection Optimized by LLM Feedback

**原计划**：
- 数据：STRING 蛋白互作数据库的癌症通路子集，379 个蛋白、3498 条边
- LLM 当 **judge**（裁判），给检索到的子图打分
- 用 **强化学习（PPO / REINFORCE）** 把 LLM 分数作为奖励信号，训练一个 "Learning to Retrieve" 策略网络
- DDR1 激酶作为 case study

### 2.2 为什么简化

| 原计划 | 问题 | 简化后 |
|---|---|---|
| 闭环 RL 检索策略 | PPO 调参难，奖励稀疏，6 周出不来结果 | 开环 + 静态 top-K |
| LLM 当 judge | 多次调用 + RL 训练，API 预算炸 | LLM 当 generator，一次 forward |
| STRING 379 蛋白子集 | 规模太小，关系单一（只有 PPI）| Hetionet 47K 节点，11 类节点，24 种关系 |
| 自建文献 corpus | 工程量大 | 图中实体名 + 关系本身就有语义 |

### 2.3 简化后丢失了什么，留下了什么

**丢失**（写进 Limitations + Future Work）：
- 自适应检索（现在靠固定元路径 + cosine 相似度）
- 闭环优化（RL 反馈）
- 文献 corpus 融合（可以后续加 BioBERT）

**留下**（核心卖点）：
- GNN + LLM 协同做生物医学推理
- 可解释性：每条预测都附带一条 "为什么" 的自然语言链
- Faithfulness 评测：LLM 是不是瞎编
- 完整对比：4 个方法配对显著性检验

---

## 3. 数据集 —— Hetionet 详解

### 3.1 什么是 Hetionet

- 2017 年 Himmelstein 等人在 *eLife* 发表的**开源生物医学异构知识图谱**
- 整合了 29 个公共数据库（Entrez, DrugBank, MeSH, GO, SIDER, Reactome, ...）
- **47,031 个节点，2,250,197 条边**
- 2017 年后被广泛用作 benchmark（Rephetio 项目）

官方仓库：https://github.com/hetio/hetionet

### 3.2 节点类型和数量（我们实验确认过的数字）

| 节点类型 | 数量 | 举例 |
|---|---|---|
| Gene | 20,945 | DDR1, KRAS, TP53, BRCA1 |
| Biological Process | 11,381 | cell cycle, apoptosis |
| Side Effect | 5,734 | headache, nausea |
| Molecular Function | 2,884 | kinase activity |
| Pathway | 1,822 | PI3K signaling |
| **Compound** | **1,552** | Aspirin, Caffeine, Cholecalciferol |
| Cellular Component | 1,391 | nucleus, mitochondrion |
| Symptom | 438 | fever, fatigue |
| Anatomy | 402 | liver, brain |
| Pharmacologic Class | 345 | NSAIDs |
| **Disease** | **137** | type 2 diabetes, Alzheimer's, osteoporosis |

**重点关注 Compound 和 Disease**，它们是我们的主任务 `Compound-treats-Disease` 的两端。

### 3.3 关系类型

我们的主任务 **`Compound-treats-Disease`（CtD）**只有 **755 条边**，这是正样本数量。
因为数量少，训练集仅约 605 条，要小心过拟合。

其他有用的关系（在检索和辅助任务里能用到）：

| 关系 | 数量 | 解读 |
|---|---|---|
| Compound-binds-Gene | 11,571 | 药物结合靶点 |
| Compound-upregulates-Gene | 18,756 | 药物上调基因表达 |
| Compound-downregulates-Gene | 21,102 | 药物下调基因表达 |
| Gene-associates-Disease | 12,623 | 基因与疾病相关（GWAS 等） |
| Gene-interacts-Gene | 294,328 | 蛋白互作 |
| Compound-resembles-Compound | 12,972 | 化合物结构相似 |
| Compound-palliates-Disease | 390 | 药物缓解症状（轻度治疗） |
| Disease-localizes-Anatomy | 3,602 | 疾病发生在哪个器官 |

### 3.4 为什么选 CtD 作为主任务

1. **语义清晰**：treats 就是"能治"，没有歧义
2. **有现成 benchmark**：Rephetio 项目已报告 AUROC ~0.97，我们有参考值
3. **标签质量高**：来自 DrugCentral + MEDI，临床确认过的
4. **样本极少（755 条）**：这是个 feature 不是 bug —— 低资源 + 小图 正好适合 GNN + RAG 互补

### 3.5 我们能做什么 case

- **DDR1 + KRAS 癌症 case**（呼应原提案）：DDR1 是 Gene[17750]，KRAS 是 Gene[2523]，Hetionet 里有它们与多种癌症的 association 边
- **维生素 D3 → 骨质疏松**：我们已经验证 top-1 检索路径生物学正确
- **老药新用 case**：找 gnn 预测分数高但实际没有 CtD 边的 pair，人工 check 是否合理

---

## 4. 核心知识点（必读）

### 4.1 Graph Neural Network (GNN)

**一句话**：GNN 就是让每个节点通过"和邻居聊天"来更新自己的表示。

**标准消息传递（message passing）公式**：

```
h_v^(l+1) = UPDATE( h_v^(l),  AGGREGATE({ h_u^(l) : u ∈ 邻居(v) }) )
```

- `h_v^(l)`: 节点 v 在第 l 层的向量表示
- `AGGREGATE`: 怎么汇总邻居信息（mean / sum / max / attention）
- `UPDATE`: 把自己的 + 邻居的合起来（通常是 MLP）

**两层 GNN 的效果**：每个节点"看到"了 2 跳内的所有信息。

我们用的 **GraphSAGE**：
```
h_v^(l+1) = σ(W · CONCAT(h_v^(l), MEAN({h_u^(l) : u ∈ N(v)})))
```

### 4.2 异构图与元路径

**同构图 vs 异构图**：
- 同构：所有节点/边一个类型（比如社交网络，全是 user-friends-user）
- 异构：多个类型（Compound / Gene / Disease + 各种关系）—— Hetionet 就是

**异构 GNN 的处理方式**（PyG 的 `to_hetero`）：
- 给每种关系类型单独训练一组消息传递权重
- 一个节点从 N 种邻居分别收集消息，再 sum/attention 合并

**元路径（Meta-path）**：
> 一条在节点类型之间切换的固定模板，定义了一种"语义走法"。

例子（我们在 `retrieval/metapath.py` 中定义了 9 条）：
- `CbGaD`: Compound-binds-Gene-associates-Disease （药物结合的基因与该病相关）
- `CrCtD`: Compound-resembles-Compound-treats-Disease （结构相似的药物能治这病）
- `CbGiGaD`: Compound-binds-Gene-interacts-Gene-associates-Disease（药物靶点的互作基因与病相关）

元路径 = 生物学假设。用它来枚举子图，保证找出的路径都有医学解读。

### 4.3 链接预测任务（Link Prediction）

**任务**：给图里两个节点 (u, v)，预测它们之间有没有某种关系。

我们的例子：给 (Aspirin, Headache)，预测 Aspirin-treats-Headache 这条边存不存在。

**训练方式**：
1. 把 CtD 边分成 train/val/test 三份（605/75/75）
2. 训练图里**只保留 train 边**，val/test 边被"藏起来"
3. 模型 forward 算所有节点的 embedding
4. 对每条 train 正样本 (c, d) 算分数 `score(c, d) = h_c · h_d`（点积）
5. 随机采样等量的负样本 (c, d')，分数目标是低的
6. BCE 损失（cross entropy between sigmoid(score) and label）

**评测**：
- **AUROC**：排序好坏，1.0 完美，0.5 随机
- **AUPRC**：正样本稀疏时更有意义（我们 positive : negative ≈ 1 : 10，偏不平衡）
- **Hits@K**：给定疾病，预测 top-K 候选化合物里有没有真正的正样本
- **McNemar test**：两个模型在同批样本上的配对显著性检验

### 4.4 节点嵌入（Embedding）

**Embedding** = 把离散的东西（单词、节点、用户）映射成连续向量，让相似的东西在向量空间里靠近。

Hetionet 节点没有自带特征（不像图像有像素、文本有 BERT 表示），所以我们用 **可学习的 `nn.Embedding`**：
- 每个节点初始化一个随机 128 维向量
- GNN 训练时这些向量作为输入，经过消息传递后更新
- 训练结束后，每个节点有了一个"学出来的"表示

### 4.5 知识图谱嵌入（KGE）—— DistMult 和 ComplEx

KGE 是 GNN 出现前主流的图表示方法，不做消息传递，只学 (head, relation, tail) 三元组的分数。

**DistMult**（我们的 baseline）：
```
score(h, r, t) = h · diag(r) · t  = Σ_i h_i · r_i · t_i
```
- 每个节点、每个关系都是一个向量
- 训练目标：正样本分数高，负样本分数低

**为什么加 DistMult 作为 baseline**：
- 证明"GNN 的 message passing 比简单的节点-关系嵌入好"（或者"好不多"）
- 是图表示学习的经典对照组
- 审稿人/评委看到没 KGE 基线会质疑

### 4.6 Retrieval-Augmented Generation (RAG)

**RAG 原始思想**（Lewis et al. 2020）：
1. 用户提问
2. 用 dense retriever 从文档库里检索 top-K 相关文档
3. 把文档拼到 prompt 里，LLM 基于文档生成回答
4. 好处：LLM 不用"记住"所有知识，减少幻觉，可更新知识库

**GraphRAG / 我们的变体**：
- 文档库换成**图结构**
- 检索不是语义相似度 top-K，而是**从图里挖子图/路径**
- LLM 基于子图生成（也可以做分类、打分）

**为什么 RAG 对生物医学特别有用**：
- 基础 LLM 对小众药物/基因的知识不全
- 图检索提供"可追溯的证据链"，每个结论能回溯到具体边
- 减少 LLM 瞎编（hallucination）

### 4.7 LLM 在系统里的角色（三选一）

| 角色 | 做什么 | 我们选哪个 |
|---|---|---|
| **Generator** | 读子图，生成答案+解释 | **是**（主线）|
| **Judge** | 给模型输出打分，反馈给训练 | 仅作 faithfulness 评测 |
| **Retriever** | 选哪些子图喂给下游 | 没用（原提案 RL 部分被砍） |

Prompt 结构（见 [llm/prompts.py](llm/prompts.py)）：
- System：角色设定 + 输出 schema + "只用给定路径，不要瞎编"的约束
- User：compound 名字 + disease 名字 + 路径列表
- 输出：严格 JSON `{prediction, confidence, rationale}`

### 4.8 评测指标汇总

| 指标 | 用在哪 | 解读 |
|---|---|---|
| AUROC | GNN / KGE 链接预测 | 排序好坏，0.9+ 算好 |
| AUPRC | 同上 | 不平衡场景下更诚实 |
| Hits@K | GNN 链接预测 | 给一个 disease，预测 top-K 里有正样本的概率 |
| Accuracy | LLM 方法 | 正例判 yes / 负例判 no 的比例 |
| 95% CI（bootstrap） | 所有指标 | 估计指标稳定性 |
| **McNemar** | 两两方法对比 | p < 0.05 说明方法 A 显著比 B 强 |
| **Faithfulness rate** | LLM 解释质量 | LLM 的 rationale 是否只用了提供的路径 |

---

## 5. 系统架构

### 5.1 Pipeline 总览

```
┌─────────────────────────────────────────────────────────────┐
│            阶段 0: 数据（Hetionet JSON → PyG HeteroData）    │
│  11 类节点 47K 个 / 24 种关系 225 万条边                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 1: GNN 训练（SAGEConv + to_hetero）                   │
│  输入: HeteroData                                           │
│  输出: 每个节点 128 维 embedding + 链接预测分数              │
│  训练目标: CtD 正样本分数高，负样本分数低                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 2: 子图检索                                            │
│  输入: (compound_idx, disease_idx), GNN embedding            │
│  对每条预定义元路径枚举所有匹配的路径                        │
│  按 GNN embedding cosine 相似度打分, 取 top-K                │
│  输出: K 条 (nodes, edges, score) 三元组                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 3: 自然语言化                                          │
│  每条边按模板翻成英文：                                      │
│  "Aspirin binds PTGS2 → PTGS2 is associated with ..."       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 4: Claude API                                         │
│  输入: compound name, disease name, paths 文本              │
│  输出: JSON {prediction: yes/no, confidence, rationale}     │
│  可选: 再调一次 Claude 做 faithfulness judge                │
└─────────────────────────────────────────────────────────────┘
                           ↓
                     最终预测 + 自然语言解释
```

### 5.2 四个对照方法一次性跑

| 方法 | 流程 | 期待效果 |
|---|---|---|
| GNN-only | 只用 sigmoid(h_c · h_d) | 基线 |
| DistMult (KGE) | 只用 h · r · t | 传统方法基线 |
| LLM-only | 只给药名和病名，LLM 靠自己知识 | 显示"LLM 单吃不够" |
| **GNN-RAG+LLM** | GNN 检索 + LLM 推理（我们的方法） | 应该最好 |

---

## 6. 代码地图（每个文件在干什么）

### 6.1 数据层

| 文件 | 作用 |
|---|---|
| [data/load_hetionet.py](data/load_hetionet.py) | 下载 Hetionet JSON，解析成 PyG HeteroData。节点存 name 和 identifier |
| [data/splits.py](data/splits.py) | 把 CtD 边随机划分 train/val/test（80/10/10），**从训练图里移除 val/test 边**（避免泄漏），提供负采样 |

### 6.2 模型层

| 文件 | 作用 |
|---|---|
| [models/hetero_gnn.py](models/hetero_gnn.py) | 异构 GNN。每个节点类型一个 `nn.Embedding`，然后 2 层 SAGEConv 经 `to_hetero` 包装成异构版本 |
| [models/link_predictor.py](models/link_predictor.py) | 链接预测头。`DotLinkPredictor`（点积）和 `MLPLinkPredictor`（拼接+MLP）两种 |
| [models/kge.py](models/kge.py) | 把 HeteroData 压平成 (head, rel, tail) 三元组给 DistMult/ComplEx 用 |

### 6.3 检索层

| 文件 | 作用 |
|---|---|
| [retrieval/metapath.py](retrieval/metapath.py) | 9 条预定义元路径（CpD / CbGaD / CuGuD / CdGdD / CbGiGaD 等）|
| [retrieval/subgraph_extractor.py](retrieval/subgraph_extractor.py) | `build_adjacency`: 把 edge_index 转成 dict for 快速查邻居。`extract_paths`: 给定 (c, d) 枚举匹配每条元路径的具体路径，按 GNN cosine 打分，取 top-K |
| [retrieval/verbalizer.py](retrieval/verbalizer.py) | 每种关系的自然语言模板（"X binds Y", "X is associated with Y"），拼成完整路径文本 |

### 6.4 LLM 层

| 文件 | 作用 |
|---|---|
| [llm/prompts.py](llm/prompts.py) | System prompt（角色 + 输出 schema + 约束）、User prompt 模板、Judge prompt |
| [llm/client.py](llm/client.py) | Anthropic SDK 封装。开启 prompt caching 省钱。提供 `predict()` 和 `judge_faithfulness()` |

### 6.5 脚本层

| 文件 | 作用 |
|---|---|
| [scripts/inspect_data.py](scripts/inspect_data.py) | 下数据 + 打印所有节点/边统计。第一次跑它确认数据对齐 |
| [scripts/train_gnn.py](scripts/train_gnn.py) | 训练异构 GNN 做 CtD 链接预测。`--epochs 100` 约 5-10 分钟 GPU |
| [scripts/train_kge.py](scripts/train_kge.py) | 训练 DistMult 或 ComplEx baseline。`--epochs 30` 约 10 分钟 |
| [scripts/eval_linkpred.py](scripts/eval_linkpred.py) | 只做 GNN 评测：AUROC / AUPRC / Hits@1/3/10 |
| [scripts/run_rag_pipeline.py](scripts/run_rag_pipeline.py) | 早期版本的端到端：GNN + 检索 + Claude。主要实验用 main_results.py |

### 6.6 实验层

| 文件 | 作用 |
|---|---|
| [experiments/main_results.py](experiments/main_results.py) | **主实验**。同一批测试对上跑 GNN / DistMult / LLM-only / GNN-RAG+LLM 四种方法，输出 bootstrap CI + McNemar p 值 |
| [experiments/ablation_k.py](experiments/ablation_k.py) | top-K 消融：K ∈ {0, 1, 3, 5, 10}，K=0 就是 LLM-only |
| [experiments/case_study_ddr1.py](experiments/case_study_ddr1.py) | 定性分析。给定基因（默认 DDR1），找最相似的化合物和疾病，挖路径，让 LLM 解读 |
| [experiments/error_analysis.py](experiments/error_analysis.py) | 读 main_results 输出，分桶（GNN-对-LLM-错 等）打印典型 case |

---

## 7. 环境部署

### 7.1 硬件要求

| 组件 | 最低 | 推荐 |
|---|---|---|
| GPU | 可以 CPU 跑，但慢 10 倍 | ≥ 8GB VRAM（RTX 3060 以上都够）|
| RAM | 8GB | 16GB |
| 磁盘 | 1GB（数据 + 依赖） | 2GB |

Hetionet 图全量放进 GPU 显存 < 2GB，很友好。

### 7.2 软件安装（Windows + Anaconda）

```bash
cd "c:/Users/10249/OneDrive/桌面/RAG/rag-gnn-hetionet"
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
torch_geometric>=2.5.0
networkx>=3.0
requests>=2.31
tqdm>=4.66
anthropic>=0.39.0
```

**注意**：不要重装 `torch`。Anaconda base 里已经是 PyTorch 2.8 dev + CUDA 12.8，够用了。

Mac/Linux 也是 `pip install -r requirements.txt`，没区别。

### 7.3 API Key 配置

注册 Anthropic 账号拿 API key：https://console.anthropic.com

Windows cmd:
```cmd
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

Windows PowerShell:
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxx"
```

macOS/Linux bash:
```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

永久保存：Windows 在"系统属性 → 环境变量"里加。macOS 写 `~/.zshrc`。

也可以创建 `.env` 文件（参照 `.env.example`），但代码里现在只读环境变量。

### 7.4 数据下载

第一次跑 `scripts/inspect_data.py` 时自动从 GitHub 下载 Hetionet JSON（16.1 MB），存在 `cache/hetionet-v1.0.json.bz2`。之后直接用缓存。

如果网络不通：
- VPN 或者换到墙外机器跑
- 或者手动 curl/wget 存到 `cache/` 下

---

## 8. 完整跑通流程

### 8.1 训练阶段（一次性，~20 分钟）

```bash
# 确认环境
python scripts/inspect_data.py

# 训练 GNN
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt

# 训练 KGE baseline（DistMult 必须有，ComplEx 可选）
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt
python scripts/train_kge.py --kge complex  --epochs 30 --ckpt checkpoints/complex.pt
```

训练完 `checkpoints/` 下应该有 `gnn.pt`、`distmult.pt`（可选 `complex.pt`）。

### 8.2 评测阶段

```bash
# GNN 指标
python scripts/eval_linkpred.py --ckpt checkpoints/gnn.pt

# 预期输出:
# AUROC  0.93xx
# Hits@1  0.xx
# Hits@3  0.xx
# Hits@10 0.xx
```

### 8.3 实验阶段（要 API key）

```bash
set ANTHROPIC_API_KEY=sk-ant-xxx

# 主实验: 四方法对比, N=150 (75 正 + 75 负), 带 faithfulness judge
python experiments/main_results.py ^
    --gnn-ckpt checkpoints/gnn.pt ^
    --kge-ckpt checkpoints/distmult.pt ^
    --n-pos 75 --n-neg 75 --judge

# 输出到 runs/main_results.jsonl, stdout 给出四方法 accuracy 表 + McNemar p 值

# top-K 消融 (中等规模 N=40)
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10 --n-test 20 --n-neg 20

# DDR1 case study
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt

# 错误分析 (读 main_results.jsonl)
python experiments/error_analysis.py --file runs/main_results.jsonl --per-bucket 5
```

### 8.4 完整 shell 脚本（一键跑通）

保存为 `run_all.sh`（Windows 下 Git Bash 运行）：

```bash
#!/bin/bash
set -e

echo "== Step 1: inspect data =="
python scripts/inspect_data.py | tail -5

echo "== Step 2: train GNN (100 epochs) =="
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt

echo "== Step 3: train DistMult (30 epochs) =="
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt

echo "== Step 4: main experiment (needs API key) =="
python experiments/main_results.py \
    --gnn-ckpt checkpoints/gnn.pt \
    --kge-ckpt checkpoints/distmult.pt \
    --n-pos 75 --n-neg 75 --judge

echo "== Step 5: ablation K =="
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10 --n-test 20 --n-neg 20

echo "== Step 6: error analysis =="
python experiments/error_analysis.py --file runs/main_results.jsonl --per-bucket 5

echo "== Step 7: DDR1 case =="
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt
```

---

## 9. 实验设计（给 Report 用）

### 9.1 实验清单（按 report 对应的章节）

| Report 对应节 | 实验 | 脚本 | 预期产出 |
|---|---|---|---|
| §4.1 主结果表 | N=150 四方法 | `main_results.py` | 表 1 with accuracy ± CI + 星号显著性 |
| §4.2 top-K 消融 | K ∈ {0,1,3,5,10} | `ablation_k.py` | 折线图 + 饱和点分析 |
| §4.3 KGE 对照 | DistMult / ComplEx | `train_kge.py` | 两行加到主表 |
| §4.4 Faithfulness | `--judge` 开关 | `main_results.py` | 单个数字 + 几个 invented entities 例子 |
| §5.1 错误分析 | 四个 bucket | `error_analysis.py` | 表格 + 定性 case 描述 |
| §5.2 DDR1 case | 定性分析 | `case_study_ddr1.py` | 3 条路径 + LLM rationale 截图 |

### 9.2 预期结果

| 方法 | 预期 accuracy | 备注 |
|---|---|---|
| GNN-only | 0.80-0.85 | 小样本链接预测上限 |
| DistMult | 0.75-0.82 | 纯 KGE 通常略弱于 GNN |
| LLM-only (no retrieval) | 0.55-0.65 | Claude 对小众 compound 有限 |
| **GNN-RAG+LLM** | **0.85-0.92** | 应该是最高的 |

McNemar：GNN-RAG+LLM vs LLM-only 应该 p < 0.001（检索显著有用）
faithfulness：应该 > 80%（prompt 约束起作用）

### 9.3 预算估算

| 项 | 次数 | 单价估算 | 小计 |
|---|---|---|---|
| 主实验（N=150 × 2 次 LLM 调用 + 可选 judge 150 次）| ≈ 450 | Claude Sonnet 4.6 ~$0.01/call (带 caching) | ~$5 |
| top-K 消融（5 个 K × 40 样本）| 200 | 同上 | ~$2 |
| 重跑一次补数据 | - | - | ~$3 |
| **合计** | | | **$10** |

如果预算紧张：
- 把 main_results.py 里的 Sonnet 换成 Haiku（`--model claude-haiku-4-5`），成本降 1/10
- N 从 150 降到 100

---

## 10. Report 写作骨架（8 页）

### 10.1 Abstract（半页）
- 1 句 motivation：GNN 擅长拓扑但缺语义解释
- 2 句 method：GNN 检索 + LLM 推理 + faithfulness 验证
- 2 句 results：四方法对比 + McNemar 显著 + case study
- 1 句 contribution

### 10.2 Introduction（1 页）
- drug repurposing 的重要性
- 图结构 + 文献语义都很重要
- 现有 RAG-GNN 是 open-loop（引 [3] Hays & Richardson）
- 我们的贡献：三点（就是砍了闭环之后还能讲的）
  1. 在 Hetionet 上的 GraphRAG 完整 pipeline
  2. 四方法对照 + 配对显著性评测
  3. Faithfulness 评测（RAG 热门方向）

### 10.3 Related Work（0.5 页）
- GraphRAG / GNN-RAG（cite cmavro 2024）
- KG-aware LLM（ToG, KG-GPT）
- Drug repurposing on Hetionet（Himmelstein 2017, Rephetio）
- LLM-as-judge（Zheng 2023）

### 10.4 Method（2 页）
- §3.1 Problem formulation
- §3.2 Hetero GNN encoder（SAGEConv + to_hetero，公式）
- §3.3 Link predictor（dot product）
- §3.4 Meta-path subgraph retrieval（列出 9 条元路径，打分公式）
- §3.5 Path verbalization（模板）
- §3.6 LLM reasoning（prompt 结构 + JSON schema）
- §3.7 Faithfulness judge

### 10.5 Experiments（2.5 页）
- §4.1 Dataset（Hetionet stats 表）
- §4.2 Setup（hyperparams 表：hidden=128, layers=2, lr=1e-3, epochs=100）
- §4.3 Main results（主表 + McNemar）
- §4.4 Ablation K
- §4.5 Faithfulness
- §4.6 Error analysis（分桶表 + 2 个定性 case）

### 10.6 Case Study: DDR1（0.5 页）
- 接原提案 motivation：DDR1 kinase + KRAS + cancer
- 展示 3 条检索到的路径 + LLM 的解读

### 10.7 Discussion & Limitations（0.5 页）
- 元路径手工定义（未自动挖掘）
- 没有文献特征（未用 BioBERT）
- LLM 成本限制 N=150

### 10.8 Future Work（0.25 页）
- 闭环 LLM-as-judge with RL
- 加 BioBERT 文献特征
- 元路径自动挖掘
- 信息论分解量化 GNN/LLM 互补信息（呼应提案 [1][3]）

### 10.9 Conclusion（0.25 页）
- 3 句话总结

### 10.10 References
- Hetionet (Himmelstein 2017)
- GNN-RAG (Mavromatis 2024)
- RAG-GNN (Hays & Richardson 2026) — 原提案的 [3]
- DistMult (Yang 2015), ComplEx (Trouillon 2016)
- GraphSAGE (Hamilton 2017)
- Anthropic Claude (technical report)
- Bertschinger et al. (2014) — 提案 [1]
- Zheng et al. (2023) — 提案 [6]

---

## 11. 分工建议（3 人 team）

假设还有 ~3 周到 May 5 ddl：

### Jingwen Chen（一作，论文主笔）
- 跑全部训练和主实验（train_gnn / train_kge / main_results）
- 写 Abstract, Intro, Method 章节
- 最后整合三人笔记

### Jingtao Wang（实验分析）
- 跑 ablation_k 和 error_analysis，整理图表
- 写 Experiments 章节
- 设计 DDR1 case study 的具体 narrative

### Shouyu Wang（方法细节 + future work）
- 研究 metapath 和 verbalizer，可以加 1-2 条更复杂的 metapath
- 跑 case_study_ddr1 并人工 curate 最好的 3 条路径
- 写 Related Work, Discussion, Future Work 章节

### 协作节奏
- 每周一次 sync，过一下结果和 report 进度
- GitHub 私有 repo 放代码，Overleaf 写 report
- 截 Week -2 前跑完所有实验，最后一周专注写作和画图

---

## 12. 常见问题与坑

### 12.1 安装坑

**Q: `pip install torch_geometric` 报错**
A: Anaconda base 如果 torch 是老版本可能不兼容，检查 `python -c "import torch; print(torch.__version__)"`。需要 >= 2.0。

**Q: CUDA 不可用**
A: `python -c "import torch; print(torch.cuda.is_available())"` 如果 False，可能要重装带 CUDA 的 PyTorch。或者就用 CPU，训练慢 5-10 倍但能跑。

**Q: `ModuleNotFoundError: data.load_hetionet`**
A: 必须在项目根目录（`rag-gnn-hetionet/`）执行 `python scripts/xxx.py`，不能 `cd scripts && python xxx.py`。每个脚本顶部已经加了 `sys.path.insert` 处理这个问题。

### 12.2 训练坑

**Q: `UserWarning: The type 'Molecular Function' contains invalid characters`**
A: 不影响训练。PyG 抱怨节点类型名有空格。想消除就在 `load_hetionet.py` 里把 "Molecular Function" 改成 "MolecularFunction"，但要同步改所有引用。

**Q: 训练 loss 不下降 / AUROC 没涨**
A: 检查 `split.train_pos.size(1)` 应该是 605。如果是 0，说明数据加载有问题。

**Q: 显存爆了**
A: 改 `--hidden 64`，或者用 CPU（`--device cpu`）。

### 12.3 API 坑

**Q: `anthropic.AuthenticationError`**
A: API key 没配对。`echo %ANTHROPIC_API_KEY%`（Windows cmd）确认能看到值。

**Q: `rate_limit_error`**
A: Claude Sonnet 默认 5 req/min。脚本里已经是串行调用，一般没问题。如果还超，加 `time.sleep(1)` 在 llm.predict 调用之间。

**Q: JSON 解析失败**
A: llm/client.py 里 `_parse_json` 用正则抓第一个 `{...}` 块。如果 LLM 输出多了杂话，捕获异常后返回 "no" + confidence 0。偶尔失败几次不影响整体。

**Q: API 花销失控**
A: 改到 Haiku：`--model claude-haiku-4-5`，便宜 10 倍。或者减少 N。

### 12.4 报告坑

**Q: 主实验 N=150，正负样本不平衡怎么办**
A: 我们的主实验默认 75 正 + 75 负是 1:1 平衡的。accuracy 是合理的指标。如果想做更真实的 imbalanced 场景，加 `--n-neg 750` 做 1:10，改报 AUPRC。

**Q: LLM-only accuracy 比预期高（不是 0.55 而是 0.80）**
A: Claude 对常见药/病知识充足，可能对 drug name 就能识别出来。这反而能讲故事：小众化合物上 retrieval 的增益更大。在 report 里按 compound 的频次分层分析，"rare compound 上 retrieval 收益最大"。

**Q: McNemar p 值很大（不显著）**
A: 要么 N 不够（加到 N=200），要么方法之间本来就没差别（是信号，不是 bug，要在 Discussion 里讨论）。

**Q: faithfulness 率很低（< 70%）**
A: 加强 system prompt 的 "only use entities from provided paths" 约束。可以加一个 self-correction 步骤：让 LLM 先列出用到的实体，再检查都在 paths 里。

---

## 附录 A: 快速命令索引

```bash
# 环境
pip install -r requirements.txt

# 数据
python scripts/inspect_data.py

# 训练
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt

# 评测
python scripts/eval_linkpred.py --ckpt checkpoints/gnn.pt

# 主实验
python experiments/main_results.py --gnn-ckpt checkpoints/gnn.pt --kge-ckpt checkpoints/distmult.pt --n-pos 75 --n-neg 75 --judge

# 消融
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10

# 分析
python experiments/error_analysis.py --file runs/main_results.jsonl
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt
```

## 附录 B: 关键参数速查

| 参数 | 默认值 | 在哪配 |
|---|---|---|
| GNN hidden dim | 128 | `train_gnn.py --hidden` |
| GNN layers | 2 | `train_gnn.py --layers` |
| GNN epochs | 100 | `train_gnn.py --epochs` |
| 学习率 | 1e-3 | `train_gnn.py --lr` |
| 负样本比例 | 1:1 | `train_gnn.py --neg-ratio` |
| 检索 top-K | 5 | `main_results.py --top-k` |
| 主实验 N | 75+75 | `main_results.py --n-pos --n-neg` |
| LLM 模型 | claude-sonnet-4-6 | `--model` |

---

**有任何问题先查 §12，再问群。**
