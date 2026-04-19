# RAG-GNN on Hetionet

COMP 559 课程项目：异构 GNN 检索子图 + LLM 生成推理。

Pipeline：`Compound ←→ Disease` 链接预测，GNN 负责从 Hetionet 挖出 top-K 条元路径，语言化后喂给 Claude，LLM 输出判别 + 自然语言解释。

---

## 快速开始

```bash
cd "c:/Users/10249/OneDrive/桌面/RAG/rag-gnn-hetionet"
pip install -r requirements.txt

# 1. 下载数据 + 打印统计
python scripts/inspect_data.py

# 2. 训练 GNN（CPU 慢，GPU 快）
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt

# 3. 训练 KGE baseline（DistMult / ComplEx）
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt
python scripts/train_kge.py --kge complex  --epochs 30 --ckpt checkpoints/complex.pt

# 4. 评测 Hits@K（GNN）
python scripts/eval_linkpred.py --ckpt checkpoints/gnn.pt

# 5. 设 API key（Windows cmd）
set ANTHROPIC_API_KEY=sk-ant-xxx

# 6. 主实验：GNN-only vs LLM-only vs GNN-RAG+LLM (+ DistMult) 同批测试对，带 McNemar
python experiments/main_results.py --gnn-ckpt checkpoints/gnn.pt --kge-ckpt checkpoints/distmult.pt --n-pos 75 --n-neg 75 --judge

# 7. 错误分析：读 runs/main_results.jsonl，分桶打印
python experiments/error_analysis.py --file runs/main_results.jsonl

# 8. top-K 消融
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10

# 9. DDR1 case study
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt
```

---

## 目录结构

```
rag-gnn-hetionet/
├── cache/                        # Hetionet JSON 缓存
├── checkpoints/                  # 训好的 GNN
├── runs/                         # RAG pipeline 输出
├── data/
│   ├── load_hetionet.py          # 下载 + 构图 (PyG HeteroData)
│   └── splits.py                 # CtD 边训/验/测划分 + 负采样
├── models/
│   ├── hetero_gnn.py             # SAGEConv + to_hetero 异构 GNN
│   └── link_predictor.py         # 点积 / MLP 打分器
├── retrieval/
│   ├── metapath.py               # 8 条预定义 Compound→Disease 元路径
│   ├── subgraph_extractor.py     # 路径枚举 + GNN 嵌入打分
│   └── verbalizer.py             # 路径 → 自然语言
├── llm/
│   ├── prompts.py                # 系统 prompt + 判别 / 忠实度 prompt
│   └── client.py                 # Anthropic SDK 封装（带 prompt caching）
├── scripts/
│   ├── inspect_data.py
│   ├── train_gnn.py              # 主训练
│   ├── eval_linkpred.py          # AUROC / AUPRC / Hits@K
│   └── run_rag_pipeline.py       # 端到端 GNN-RAG + LLM
├── experiments/
│   ├── ablation_k.py             # top-K 影响
│   └── case_study_ddr1.py        # DDR1 定性分析
└── requirements.txt
```

---

## 关键技术选择

| 项 | 选择 | 理由 |
|---|---|---|
| GNN backbone | `SAGEConv + to_hetero` | 简单、PyG 官方范式、支持 Hetionet 11 类节点 / 24 类关系 |
| 节点特征 | 可学习 `nn.Embedding` | Hetionet 没有节点属性，embedding 从头学 |
| 主任务 | `Compound-treats-Disease` 链接预测 | 755 条正样本，是经典 drug-repurposing benchmark |
| 评测 | AUROC / AUPRC / Hits@1/3/10 | 全部跟 Rephetio 基线可比 |
| 子图检索 | 8 条元路径枚举 + GNN cosine 相似度打分 | 避免路径爆炸，保留生物学语义 |
| LLM | Claude Sonnet 4.6（可换） | 生物医学知识足够，带 prompt caching 省钱 |
| Prompt 结构 | 系统 + JSON schema 输出 | 可解析，`prediction / confidence / rationale` 三字段 |
| Faithfulness 评测 | LLM-as-judge（第二次调用） | 检查 rationale 是否幻觉出了路径里没有的实体 |

---

## 实验清单（report 能用的数据）

| 实验 | 脚本 | 输出 |
|---|---|---|
| 主结果表（4 个方法 × N=150，带 95% CI + McNemar p 值）| `experiments/main_results.py --n-pos 75 --n-neg 75 --judge` | `runs/main_results.jsonl` + stdout 表 |
| KGE 基线 AUROC/AUPRC | `scripts/train_kge.py --kge distmult` / `--kge complex` | stdout |
| GNN Hits@K | `scripts/eval_linkpred.py` | stdout |
| top-K ablation | `experiments/ablation_k.py --ks 0 1 3 5 10` | `runs/ablation_k.json` |
| Faithfulness 评测 | `main_results.py --judge` 自动带 | jsonl 里 `judge` 字段 |
| 错误分析分桶 | `experiments/error_analysis.py` | stdout（按 GNN-对/LLM-错 等分桶）|
| DDR1 case study | `experiments/case_study_ddr1.py` | stdout，定性 |

**写 report 的 Experiments 章节建议结构：**

1. **Main results table**（`main_results.py` stdout 直接贴）—— 包含 GNN-only / LLM-only / GNN-RAG+LLM / DistMult 四行，accuracy ± 95% CI
2. **McNemar 显著性**—— 证明 GNN-RAG+LLM 比 LLM-only 显著更好（支撑"检索有用"）
3. **top-K ablation 曲线**——  K=0 就是 LLM-only，能看到 K 增加收益在哪里饱和
4. **Faithfulness**—— 证明 LLM 不瞎编
5. **错误分析定性**—— GNN-对-LLM-错 的典型 case + paths（说明 pipeline 的短板）
6. **DDR1 case study**—— 呼应原提案 motivation

---

## Smoke test 结果

- 训练/验证/测试划分：605 / 75 / 75 条 `Compound-treats-Disease` 边
- **GNN**（SAGEConv + to_hetero，6 epochs）test AUROC = **0.897**
- **DistMult**（3 epochs）test AUROC = **0.8615**（有意义的基线差距）
- 检索示例：Cholecalciferol (维生素 D3) → 骨质疏松，top-1 路径走 Ergocalciferol 结构相似链（生物学正确）

正式训练（100 轮 GNN + 30 轮 KGE）AUROC 应能再涨几个点到 0.93-0.95。

---

## 已知限制 / 写进 report Limitations

- **闭环 RL 未实现**：原提案的 LLM-as-judge + RL 检索策略没做（工程量超预算）。已实现"静态 top-K + LLM 忠实度检查"作为 future work 的准备
- **元路径手工枚举**：8 条预定义，没有自动学习元路径
- **LLM 评测样本小**：每次 pipeline 只跑 ~60 对（API 成本），方差大
- **节点特征没用文献信息**：原 HasiHays 论文用了 BioBERT 编码 PubMed 文档作为节点特征，本项目没有这一层（future work）
