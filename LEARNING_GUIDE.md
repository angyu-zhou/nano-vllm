# Nano-vLLM 学习指南

这份指南面向准备学习 vLLM、Nano-vLLM 以及大语言模型推理系统的读者。目标不是逐行背诵源码，而是借助 Nano-vLLM 建立一套完整的推理框架认知：请求如何进入系统、如何被调度、模型如何执行、KV Cache 如何管理，以及各种性能优化分别解决什么问题。

> Nano-vLLM 是一个精简的 vLLM 风格推理框架。它保留了 Prefill/Decode、Continuous Batching、Paged KV Cache、Prefix Cache、Tensor Parallel、FlashAttention 和 CUDA Graph 等核心机制，同时减少了多模型和生产级工程代码，非常适合作为进入正式 vLLM 前的学习项目。

---

## 1. 学习目标

完成本指南后，你应该能够独立解释：

1. `LLM.generate()` 为什么是一个反复调度和执行的循环；
2. 一条请求从 Prompt 到生成结果经历了哪些模块；
3. Prefill 和 Decode 在输入形状、缓存行为和性能特征上的区别；
4. Continuous Batching 如何动态加入和移除请求；
5. KV Cache 为什么需要分页管理；
6. `block_table`、`slot_mapping` 和物理 KV Cache 之间的关系；
7. Prefix Cache 如何识别并共享相同前缀；
8. FlashAttention、CUDA Graph 和 Tensor Parallel 分别优化了什么；
9. Nano-vLLM 的实现如何映射到正式 vLLM。

学习过程中应始终围绕以下五个问题：

1. 请求是怎样流动的？
2. 模型是怎样执行的？
3. 多个请求是怎样被调度的？
4. KV Cache 是怎样分配、写入和读取的？
5. 每项性能优化消除了什么开销？

---

## 2. 项目概览

### 2.1 顶层文件

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目简介、安装方法、功能和性能数据 |
| `example.py` | 最小推理示例，适合作为第一个入口 |
| `bench.py` | 吞吐量基准测试 |
| `pyproject.toml` | Python 版本、依赖和包配置 |
| `nanovllm/` | 推理引擎、模型、算子和工具代码 |
| `assets/logo.png` | 项目 Logo，不是技术架构图 |

### 2.2 核心模块

| 问题 | 对应文件 |
| --- | --- |
| 用户怎样调用框架？ | `example.py` |
| 完整生成循环在哪里？ | `nanovllm/engine/llm_engine.py` |
| 一条请求保存哪些状态？ | `nanovllm/engine/sequence.py` |
| 每一轮应该运行哪些请求？ | `nanovllm/engine/scheduler.py` |
| KV Cache block 怎样分配？ | `nanovllm/engine/block_manager.py` |
| GPU 输入怎样准备和执行？ | `nanovllm/engine/model_runner.py` |
| Attention 怎样写入和读取 KV Cache？ | `nanovllm/layers/attention.py` |
| Qwen3 模型怎样完成前向计算？ | `nanovllm/models/qwen3.py` |
| 张量并行怎样切分线性层？ | `nanovllm/layers/linear.py` |
| logits 怎样变成 token？ | `nanovllm/layers/sampler.py` |
| 模型权重怎样加载和切分？ | `nanovllm/utils/loader.py` |
| Attention 元数据从哪里获得？ | `nanovllm/utils/context.py` |

### 2.3 依赖与运行环境

项目的主要依赖包括：

- PyTorch；
- Transformers；
- Triton；
- FlashAttention；
- xxHash。

完整运行依赖 CUDA、NCCL、Triton CUDA kernel 和 FlashAttention，因此推荐使用 Linux + NVIDIA GPU。macOS 或纯 CPU 环境适合阅读源码、手推数据结构和编写调度模拟实验，不适合直接运行完整推理流程。

不要把大量学习时间花在给本项目增加 CPU 或 MPS 支持上，这会偏离学习推理框架核心机制的目标。

---

## 3. 先建立总体运行模型

把 Nano-vLLM 暂时看成下面这条流水线：

```text
用户输入 Prompt
       │
       ▼
Tokenizer
       │
       ▼
Sequence：保存一条请求的运行状态
       │
       ▼
Scheduler：选择本轮执行的请求
       │
       ├── Prefill：处理 Prompt token
       │
       └── Decode：为每条请求生成一个新 token
       │
       ▼
BlockManager：分配和回收 KV Cache block
       │
       ▼
ModelRunner：准备张量并执行 GPU 模型
       │
       ▼
Qwen3 Transformer
       │
       ▼
Sampler：从 logits 中采样 token
       │
       ▼
Scheduler 更新 Sequence 状态
       │
       ├── 已结束：返回结果并释放 block
       │
       └── 未结束：进入下一轮调度
```

完整调用链可以概括为：

```text
LLM.generate()
│
├── add_request() × N
│   ├── tokenizer.encode()
│   ├── 创建 Sequence
│   └── scheduler.add()
│
└── while not is_finished():
    └── step()
        ├── scheduler.schedule()
        │   ├── 选择 Prefill 请求
        │   ├── 选择 Decode 请求
        │   └── 必要时抢占请求
        │
        ├── model_runner.call("run", ...)
        │   ├── prepare_prefill()
        │   │   或 prepare_decode()
        │   ├── set_context()
        │   ├── run_model()
        │   │   └── Qwen3ForCausalLM.forward()
        │   └── Sampler.forward()
        │
        └── scheduler.postprocess()
            ├── append_token()
            ├── 检查 EOS 和最大生成长度
            └── 释放已完成请求的 KV Cache block
```

第一遍阅读时只要能跟踪这条主链路，不必立即理解所有张量形状和 CUDA 细节。

---

## 4. 前置知识

不需要把所有前置知识学完后才开始读代码。建议在源码中遇到概念时按需补充。

### 4.1 必需知识

- Python 类、属性、生成循环和基本数据结构；
- PyTorch Tensor 的 shape、dtype、device 和常用张量操作；
- Decoder-only Transformer；
- Self-Attention；
- Tokenizer、token ID、logits 和采样；
- 自回归生成。

### 4.2 阅读模型代码时补充

- RMSNorm；
- RoPE；
- SwiGLU；
- Grouped Query Attention，简称 GQA；
- Causal Attention。

### 4.3 阅读性能代码时补充

- KV Cache；
- PagedAttention；
- FlashAttention varlen 接口；
- Triton kernel 基础；
- `torch.compile`；
- CUDA Graph；
- Tensor Parallel；
- `torch.distributed` 和 NCCL。

---

## 5. 推荐源码阅读顺序

不要按照目录顺序阅读。下面的顺序按照认知依赖组织，每一轮只解决一类问题。

---

## 第一阶段：API 与请求生命周期

### 目标

理解一条 Prompt 如何进入系统，以及生成循环为什么需要反复执行。

### 阅读顺序

1. `README.md`
2. `example.py`
3. `nanovllm/sampling_params.py`
4. `nanovllm/config.py`
5. `nanovllm/llm.py`
6. `nanovllm/engine/llm_engine.py`

### 重点问题

- 用户怎样创建 `LLM`？
- Prompt 在哪里被 tokenizer 编码？
- `SamplingParams` 控制哪些行为？
- `generate()` 为什么要调用多次 `step()`？
- 每个 step 从调度器获得什么？
- 模型输出的 token 怎样写回请求？
- 请求结束后结果怎样恢复到原始输入顺序？

### 第一遍可以忽略

- Triton kernel；
- CUDA Graph；
- Tensor Parallel；
- KV Cache 的物理地址计算；
- FlashAttention 的参数细节。

### 阶段产出

自己画出以下流程，不参考本文也能复述每一步：

```text
prompt
  → tokenize
  → Sequence
  → Scheduler
  → ModelRunner
  → model forward
  → Sampler
  → Sequence
  → result
```

### 验收清单

- [ ] 能找到请求创建的位置；
- [ ] 能找到生成主循环；
- [ ] 能找到单步执行入口；
- [ ] 能找到模型采样位置；
- [ ] 能解释为什么生成 100 个 token 通常需要多轮执行。

---

## 第二阶段：Sequence 与请求状态

### 目标

理解框架如何表示一条请求，以及调度、缓存和生成状态分别保存在什么地方。

### 阅读文件

- `nanovllm/engine/sequence.py`

### 重点概念

`Sequence` 可以理解为“一条生成请求的全部运行状态”，通常包括：

- 原始 Prompt token；
- 当前所有 token；
- 已生成 token；
- 当前请求状态；
- 本轮计划处理的 token 数；
- 已缓存的 token 数；
- KV Cache block table；
- 最大生成长度；
- EOS 等停止条件。

请求状态通常经历：

```text
WAITING → RUNNING → FINISHED
```

### Block 数学练习

假设：

```text
block_size = 256
prompt_tokens = 600
```

则：

```text
num_blocks = ceil(600 / 256) = 3

block 0: 256 token
block 1: 256 token
block 2:  88 token
```

继续思考：

1. 如果再生成 168 个 token，是否需要新 block？
2. 如果再生成第 169 个 token，会发生什么？
3. `last_block_num_tokens` 在边界处应该是多少？
4. 空请求、整块边界和只差一个 token 的情况分别怎样处理？

### 验收清单

- [ ] 能解释 `WAITING`、`RUNNING` 和 `FINISHED`；
- [ ] 能区分 Prompt token 与 Completion token；
- [ ] 能根据序列长度和 block size 计算 block 数；
- [ ] 能解释 `num_cached_tokens` 与 `num_scheduled_tokens` 的区别；
- [ ] 能说明 `block_table` 为什么属于 Sequence 状态。

---

## 第三阶段：Qwen3 模型前向

### 目标

理解一个 token ID 怎样经过模型得到下一个 token 的 logits。

### 阅读顺序

1. `nanovllm/models/qwen3.py` 中的 `Qwen3ForCausalLM`
2. `Qwen3Model`
3. `Qwen3DecoderLayer`
4. `Qwen3Attention`
5. `Qwen3MLP`
6. `nanovllm/layers/embed_head.py`
7. `nanovllm/layers/layernorm.py`
8. `nanovllm/layers/rotary_embedding.py`
9. `nanovllm/layers/activation.py`

### 模型结构

```text
input_ids
   │
   ▼
Embedding
   │
   ▼
N × DecoderLayer
   ├── RMSNorm
   ├── Self-Attention
   │   ├── Q/K/V projection
   │   ├── Q/K Norm
   │   ├── RoPE
   │   ├── KV Cache
   │   └── FlashAttention
   ├── Residual
   ├── RMSNorm
   ├── MLP
   │   ├── Gate projection
   │   ├── Up projection
   │   ├── SiLU
   │   └── Down projection
   └── Residual
   │
   ▼
Final RMSNorm
   │
   ▼
LM Head
   │
   ▼
logits
```

### 建议记录的张量形状

阅读时在纸上维护一张 shape 表。具体数值取决于模型配置，但语义通常如下：

```text
input_ids:    [num_tokens]
positions:    [num_tokens]
hidden_states:[num_tokens, hidden_size]
q:            [num_tokens, num_attention_heads, head_dim]
k/v:          [num_tokens, num_kv_heads, head_dim]
logits:       [num_sequences, vocab_size]
```

注意：Prefill 可能输入很多 token，但生成时通常只需要每条序列最后一个有效位置的 logits。因此输出 logits 的第一维更接近序列数，而不是本轮所有输入 token 的总数。

### 重点问题

- Embedding 怎样把 token ID 变成 hidden state？
- Qwen3 为什么分别设置 Q head 和 KV head？
- Q/K Norm 在哪里执行？
- RoPE 使用的 position 从哪里来？
- Attention 输出怎样与 residual 合并？
- MLP 为什么使用 gate 和 up 两个投影？
- LM Head 怎样把 hidden state 转换成词表 logits？

### 验收清单

- [ ] 能画出一个 DecoderLayer；
- [ ] 能区分 attention heads 和 KV heads；
- [ ] 能说明 RoPE 的作用；
- [ ] 能解释 SwiGLU 的 gate/up/down 结构；
- [ ] 能从 `input_ids` 跟踪到 `logits`。

---

## 第四阶段：Prefill、Decode 与 Scheduler

### 目标

理解推理框架如何处理多条不同长度的请求，以及为什么每一轮都要重新调度。

### 阅读文件

- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/model_runner.py` 中的 `prepare_prefill`
- `nanovllm/engine/model_runner.py` 中的 `prepare_decode`
- `nanovllm/engine/llm_engine.py` 中的 `step`

### 4.1 Prefill

Prompt 被 tokenize 后，第一次模型执行需要处理 Prompt token 并建立 KV Cache：

```text
输入：一条或多条请求的多个 Prompt token
输出：每条序列最后位置的 logits + 所有输入位置的 KV Cache
```

Prefill 的典型特点：

- 单次处理 token 较多；
- 并行度较高；
- 计算较密集；
- 主要影响首 token 延迟，即 TTFT；
- 长 Prompt 可能被 Chunked Prefill 拆成多个调度轮次。

### 4.2 Decode

Prefill 完成后，每轮为每条运行中的序列生成一个 token：

```text
第 1 轮：输入最后一个 Prompt token，生成 A
第 2 轮：输入 A，生成 B
第 3 轮：输入 B，生成 C
```

Decode 的典型特点：

- 每条序列每轮通常只有一个新 token；
- 需要重复执行很多轮；
- 会读取大量历史 KV Cache；
- 更容易受到显存带宽和 kernel launch 开销影响；
- 主要影响每 token 延迟，即 TPOT。

### 4.3 Continuous Batching

静态 batching 中，短请求结束后可能需要等待整个 batch 完成。Continuous Batching 在每一步重新选择请求：

```text
step 1: A B C
step 2: A B C
step 3: A D C   # B 完成，D 立即加入
step 4: A D C
```

这样可以提高 GPU 利用率和系统吞吐。

### 4.4 Chunked Prefill

如果一个 Prompt 很长，一次 Prefill 可能占满本轮 token budget，导致 Decode 请求长时间等待。Chunked Prefill 把长 Prompt 分块调度：

```text
长 Prompt：1000 token
每轮预算：256 token

round 1: prefill 256
round 2: prefill 256
round 3: prefill 256
round 4: prefill 232
```

需要重点观察：

- 本轮剩余 token budget；
- 当前请求已缓存和已处理的 token；
- chunk 是否刚好跨越 block 边界；
- chunk 尚未完成时是否应该采样；
- Prefill 与 Decode 怎样避免互相长期阻塞。

### 4.5 抢占

KV Cache 空间不足时，调度器可能暂时移出某条运行中的请求并释放其资源，以保证当前批次可以继续执行。阅读时重点跟踪：

- 什么条件触发抢占；
- 被抢占的请求回到什么状态；
- 已完成 token 是否保留；
- KV Cache block 怎样释放；
- 请求恢复后需要重新计算哪些部分。

### 手工调度练习

构造三个请求：

```text
A：Prompt 长度 500，生成 20 token
B：Prompt 长度 100，生成 5 token
C：Prompt 长度 1000，生成 10 token
```

假设：

```text
max_num_seqs = 2
max_num_batched_tokens = 512
```

逐轮记录：

| Step | 阶段 | 请求 | 本轮 token 数 | 缓存变化 | 完成情况 |
| --- | --- | --- | --- | --- | --- |
| 1 | Prefill |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

不要只写最终答案，应对照 `Scheduler.schedule()` 的实际分支逐步推演。

### 验收清单

- [ ] 能解释 Prefill 和 Decode 的区别；
- [ ] 能解释 TTFT 和 TPOT 分别受什么影响；
- [ ] 能说明 Continuous Batching 为什么提高吞吐；
- [ ] 能说明 Chunked Prefill 解决什么问题；
- [ ] 能根据 token budget 手工模拟调度；
- [ ] 能找到抢占与调度后处理代码。

---

## 第五阶段：Paged KV Cache 与 Prefix Cache

### 目标

理解 vLLM 风格推理系统最核心的缓存管理机制。

### 阅读顺序

1. `nanovllm/engine/block_manager.py`
2. `nanovllm/utils/context.py`
3. `nanovllm/engine/model_runner.py` 中的输入准备逻辑
4. `nanovllm/layers/attention.py`

### 5.1 为什么需要 KV Cache

如果每次生成新 token 都重新计算全部历史：

```text
第 1 轮：计算 token 1..100
第 2 轮：计算 token 1..101
第 3 轮：计算 token 1..102
```

会产生大量重复计算。KV Cache 保存历史 token 的 Key 和 Value，后续 Decode 只需：

1. 计算新 token 的 Q/K/V；
2. 把新 K/V 写入缓存；
3. 让新 Q 与历史 K/V 做 Attention。

### 5.2 为什么需要分页

如果每条请求都申请一块连续、按最大长度预留的 KV Cache，会出现：

- 请求长度不同；
- 最终生成长度未知；
- 大量内部碎片；
- 请求结束后的空间不易灵活复用。

Paged KV Cache 把物理缓存切成固定大小的 block：

```text
物理 KV Cache：

Block 0
Block 1
Block 2
Block 3
Block 4
...

请求 A 的逻辑序列：
[Block 2, Block 4, Block 1]

请求 B 的逻辑序列：
[Block 0, Block 3]
```

逻辑序列不必占用连续物理显存。

### 5.3 `block_table`

`block_table` 描述一条 Sequence 的逻辑 block 到物理 block 的映射：

```text
logical block 0 → physical block 7
logical block 1 → physical block 2
logical block 2 → physical block 9
```

例如一个 token 的逻辑位置是 600，且 `block_size = 256`：

```text
logical_block_index = 600 // 256 = 2
block_offset        = 600 % 256  = 88
physical_block      = block_table[2] = 9
physical_slot       = 9 * 256 + 88
```

### 5.4 `slot_mapping`

`slot_mapping` 告诉 KV Cache 写入 kernel：本轮每一个输入 token 的 K/V 应写到哪个物理 slot。

```text
slot = physical_block_id * block_size + offset_in_block
```

Prefill 一轮可能输入很多 token，因此会有很多 slot；Decode 中每条序列通常只写入一个新 slot。

### 5.5 Attention 读取历史缓存

写入和读取需要区分：

- `slot_mapping`：本轮新 K/V 写到哪里；
- `block_tables`：某条请求的历史 K/V 分布在哪些 block；
- `context_lens` 或相关长度信息：每条请求有多少历史 token；
- `cu_seqlens`：varlen Prefill 中不同序列的边界。

### 5.6 Prefix Cache

两个请求可能拥有相同的完整前缀：

```text
请求 A：共同长文档 + 问题 A
请求 B：共同长文档 + 问题 B
```

共同长文档对应的完整 KV Cache block 可以被复用，避免重复 Prefill。

本项目使用链式 block hash 表示前缀关系：

```text
hash_0 = hash(tokens_of_block_0)
hash_1 = hash(hash_0 + tokens_of_block_1)
hash_2 = hash(hash_1 + tokens_of_block_2)
```

阅读 `BlockManager` 时同时跟踪：

- `block_id`：物理 block 编号；
- `hash`：前缀链哈希；
- `token_ids`：block 的真实 token，用于避免哈希碰撞误命中；
- `ref_count`：有多少 Sequence 正在引用该 block。

### Prefix Cache 手工练习

给定：

```text
block_size = 4
A = [1,2,3,4, 5,6,7,8, 9]
B = [1,2,3,4, 5,6,7,8, 10]
```

回答：

1. A 和 B 有几个完全相同的 block？
2. 哪些 block 可以缓存和共享？
3. 最后一个不完整 block 是否应该按相同方式共享？
4. A 完成后，共享 block 的 `ref_count` 怎样变化？
5. B 完成后，哪些 block 可以回到空闲池？

### 验收清单

- [ ] 能说明 KV Cache 避免了哪些重复计算；
- [ ] 能说明分页怎样减少显存碎片；
- [ ] 能区分逻辑 block 与物理 block；
- [ ] 能根据 `block_table` 手算物理 slot；
- [ ] 能区分 `block_table` 和 `slot_mapping`；
- [ ] 能解释 Prefix Cache 的链式哈希；
- [ ] 能解释为什么命中 hash 后仍需比较 token；
- [ ] 能跟踪共享 block 的 `ref_count`。

---

## 第六阶段：全局 Context

### 目标

理解模型 `forward()` 参数看起来很少，却仍能获得 Attention 所需元数据的原因。

### 阅读文件

- `nanovllm/utils/context.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/attention.py`

### 核心设计

Nano-vLLM 使用全局 Context 在 `ModelRunner` 与 Attention 层之间传递运行元数据：

```text
ModelRunner.prepare_prefill/decode
       │
       ├── 构造 slot_mapping
       ├── 构造 cu_seqlens
       ├── 构造 block_tables
       └── 构造 context lengths
       │
       ▼
set_context(...)
       │
       ▼
Qwen3 model forward
       │
       ▼
Attention.get_context()
       │
       ▼
执行 Prefill 或 Decode Attention
       │
       ▼
reset_context()
```

因此，模型 `forward(input_ids, positions)` 虽然没有显式接收全部 Attention 元数据，Attention 层仍可以从全局 Context 获得它们。

### 注意事项

- 脱离 `ModelRunner.prepare_*` 单独调用模型时，Context 可能没有正确初始化；
- 忘记 reset 可能让后续执行读到过期状态；
- 调试 Attention 时必须同时检查当前是 Prefill 还是 Decode；
- 全局状态使模型签名简洁，但增加了隐式依赖。

### 验收清单

- [ ] 能说明 Context 由谁设置；
- [ ] 能说明 Context 由谁读取；
- [ ] 能列出 Context 中的主要 Attention 元数据；
- [ ] 能解释为什么模型前向签名中没有这些参数；
- [ ] 能说明全局 Context 的优点和调试风险。

---

## 第七阶段：Sampling

### 目标

理解 logits 怎样变成最终 token，并认识当前极简实现的能力边界。

### 阅读文件

- `nanovllm/sampling_params.py`
- `nanovllm/layers/sampler.py`
- `nanovllm/engine/scheduler.py` 中的结果后处理

### 重点问题

- temperature 怎样影响输出分布？
- 为什么 temperature 越低，输出越稳定？
- Sampler 的输出数量为什么与本轮序列数对应？
- EOS 在什么位置判断？
- 达到最大生成长度后怎样结束？
- 当前实现是否支持 greedy、top-k、top-p 和 repetition penalty？

本项目的采样实现较精简。不要默认它具备正式 vLLM 或 Transformers 中所有采样功能。

### 扩展练习

在理解当前实现后，可以尝试增加：

1. Greedy decoding；
2. Top-k；
3. Top-p；
4. 固定随机种子；
5. 每条 Sequence 独立的采样配置。

修改前应先为每种策略写小规模 logits 单元测试，避免把采样错误误认为模型错误。

### 验收清单

- [ ] 能从 logits 跟踪到 token ID；
- [ ] 能解释 temperature；
- [ ] 能找到停止条件处理位置；
- [ ] 能列出当前采样实现不支持的常用能力。

---

## 第八阶段：性能优化

只有在理解主链路、调度与 KV Cache 后再进入这一阶段。

### 8.1 FlashAttention

阅读：

- `nanovllm/layers/attention.py`

重点区分两类路径：

```text
Prefill：varlen FlashAttention
Decode：从 paged KV Cache 读取历史 K/V
```

先理解以下问题：

- Prefill 为什么需要 `cu_seqlens`？
- Decode 为什么需要 `block_tables`？
- 为什么两者适合不同的 Attention 接口？
- FlashAttention 主要减少什么内存访问和中间张量？

### 8.2 Triton KV Cache 写入

阅读：

- `nanovllm/layers/attention.py` 中的 KV Cache 写入 kernel

第一遍只需要把它理解为：

```text
根据 slot_mapping，
把本轮产生的 K/V，
写入对应的物理 KV Cache slot。
```

建议按以下顺序阅读 kernel：

1. 每个 program 负责哪个 token；
2. 怎样从 `slot_mapping` 得到 cache 位置；
3. K/V 输入和 cache 输出的 shape；
4. block 内偏移怎样计算；
5. 边界 mask 在哪里使用。

暂时不要一开始就研究 tile size 或性能调参。

### 8.3 Tensor Parallel

阅读：

- `nanovllm/layers/linear.py`
- `nanovllm/layers/embed_head.py`
- `nanovllm/engine/model_runner.py` 中的多进程和通信逻辑

核心模型：

```text
Column Parallel：
按输出维度切分权重，
每张 GPU 得到一部分输出。

Row Parallel：
按输入维度切分权重，
每张 GPU 得到部分和，
最后执行 all_reduce。
```

需要区分：

- Q/K/V 投影怎样切；
- Gate/Up 合并投影怎样切；
- Down projection 为什么需要归并；
- Vocabulary embedding 和 LM Head 怎样按词表切分；
- 权重加载时怎样只读取当前 rank 对应的 shard。

### 8.4 CUDA Graph

阅读：

- `nanovllm/engine/model_runner.py` 中的模型执行与 graph capture 逻辑

CUDA Graph 尤其适合 Decode：Decode 会重复执行形状相似的小计算，Python 调度和 kernel launch 的固定开销占比更明显。

重点理解：

- 为什么 CUDA Graph 通常不用于动态形状很强的 Prefill；
- 为什么需要预先捕获若干 batch size；
- 实际 batch 为什么可能 padding 到已捕获的 graph batch size；
- `enforce_eager` 对执行路径有什么影响；
- 第一次捕获和后续 replay 的成本有什么区别。

### 验收清单

- [ ] 能说明 FlashAttention 优化的主要瓶颈；
- [ ] 能解释 KV Cache 写入 kernel 的输入和输出；
- [ ] 能区分 Column Parallel 与 Row Parallel；
- [ ] 能说明为什么 Row Parallel 通常需要 all-reduce；
- [ ] 能说明 CUDA Graph 为什么更适合 Decode；
- [ ] 能区分吞吐、TTFT 和 TPOT。

---

## 6. 四周执行计划

以下安排假设每周学习 5 天、每天 1～2 小时。时间不足时可以把每周扩展为两周，不建议跳过手工练习。

### 第一周：API、请求与 Transformer

#### 阅读

- [ ] `README.md`
- [ ] `example.py`
- [ ] `nanovllm/config.py`
- [ ] `nanovllm/sampling_params.py`
- [ ] `nanovllm/engine/llm_engine.py`
- [ ] `nanovllm/engine/sequence.py`
- [ ] `nanovllm/models/qwen3.py`

#### 产出

- [ ] 画出 `generate()` 调用链；
- [ ] 画出 Qwen3 DecoderLayer；
- [ ] 整理模型前向的主要张量 shape；
- [ ] 手算不同序列长度需要的 block 数。

#### 周末验收

不看源码回答：

1. `Sequence` 保存了什么？
2. 为什么 `generate()` 中有循环？
3. logits 在哪里计算？
4. token 在哪里采样？
5. 一个 DecoderLayer 包含哪些主要模块？

### 第二周：Prefill、Decode 与调度

#### 阅读

- [ ] `nanovllm/engine/scheduler.py`
- [ ] `ModelRunner.prepare_prefill`
- [ ] `ModelRunner.prepare_decode`
- [ ] `LLMEngine.step`

#### 产出

- [ ] 手工模拟三个不同长度的请求；
- [ ] 记录每轮 batch 中的请求和 token 数；
- [ ] 解释 Chunked Prefill；
- [ ] 找到抢占和完成后处理代码。

#### 周末验收

不看源码回答：

1. Prefill 和 Decode 的输入有什么区别？
2. 为什么每一轮都要重新调度？
3. token budget 限制的是什么？
4. Chunked Prefill 为什么能改善调度公平性？
5. KV Cache 不足时系统怎样继续运行？

### 第三周：KV Cache、分页与前缀复用

#### 阅读

- [ ] `nanovllm/engine/block_manager.py`
- [ ] `nanovllm/utils/context.py`
- [ ] `nanovllm/layers/attention.py`
- [ ] `ModelRunner` 的 slot 和 block table 准备逻辑

#### 产出

- [ ] 画出逻辑 block 到物理 block 的映射；
- [ ] 手算多个 token 的 slot；
- [ ] 画出 KV Cache 写入和读取路径；
- [ ] 手工模拟两个相同前缀请求的 block 共享；
- [ ] 跟踪共享 block 的 `ref_count`。

#### 周末验收

不看源码回答：

1. KV Cache 避免了什么计算？
2. Paged KV Cache 为什么比连续预分配灵活？
3. `block_table` 与 `slot_mapping` 的区别是什么？
4. Prefix Cache 为什么使用链式 hash？
5. 为什么仅比较 hash 不够？

### 第四周：并行与性能

#### 阅读

- [ ] `nanovllm/layers/linear.py`
- [ ] `nanovllm/layers/embed_head.py`
- [ ] `nanovllm/layers/sampler.py`
- [ ] `ModelRunner` 的 KV Cache 分配逻辑
- [ ] `ModelRunner` 的 CUDA Graph 逻辑
- [ ] `bench.py`

#### 产出

- [ ] 画出 QKV 投影的 TP 切分方式；
- [ ] 画出 Row Parallel 的 all-reduce；
- [ ] 对比 eager 与 CUDA Graph 执行路径；
- [ ] 解释吞吐、TTFT、TPOT 三个指标；
- [ ] 设计一组 benchmark 参数矩阵。

#### 周末验收

不看源码回答：

1. FlashAttention 优化了什么？
2. CUDA Graph 为什么对 Decode 特别有价值？
3. Column Parallel 和 Row Parallel 有什么区别？
4. 增大 batch 为什么可能提高吞吐却增加延迟？
5. 增大 KV Cache block size 会带来什么权衡？

---

## 7. 动手实验

## 7.1 无 GPU 实验

这些实验适合在 macOS 或 CPU 环境完成。

### 实验 A：Sequence 边界测试

构造不同长度的 token 列表，检查或手算：

- `num_tokens`；
- `num_prompt_tokens`；
- `num_completion_tokens`；
- `num_blocks`；
- `last_block_num_tokens`。

重点覆盖：

```text
length = 0
length = 1
length = block_size - 1
length = block_size
length = block_size + 1
length = 2 * block_size
```

### 实验 B：调度器模拟器

不加载模型，只构造 Sequence 和调度状态，打印：

```text
step=1 type=prefill seqs=[0] scheduled_tokens=[256]
step=2 type=prefill seqs=[0] scheduled_tokens=[244]
step=3 type=decode  seqs=[0] scheduled_tokens=[1]
```

观察：

- WAITING 到 RUNNING；
- Chunked Prefill；
- token budget；
- 最大序列数；
- 完成后的移除；
- 缓存不足时的抢占。

### 实验 C：BlockManager 可视化日志

为 block 分配和释放过程增加临时调试输出：

```text
allocate   seq=0 blocks=[5, 8]
allocate   seq=1 blocks=[3]
cache-hit seq=2 blocks=[5]
deallocate seq=0 blocks=[5, 8]
```

重点观察：

- 空闲 block 数；
- block 所有者；
- hash 与 token；
- `ref_count`；
- shared block 何时可以真正回收。

### 实验 D：纯 Python Slot 计算器

写一个小函数，输入：

- token 的逻辑位置；
- `block_size`；
- `block_table`。

输出：

- logical block index；
- block offset；
- physical block ID；
- physical slot。

用边界值测试它，并与 `ModelRunner` 中的构造方式对照。

---

## 7.2 NVIDIA GPU 实验

建议先使用 Qwen3-0.6B，在 Linux + NVIDIA GPU 环境运行。

### 实验 A：跑通最小示例

目标：

- 模型正确加载；
- 多个 Prompt 可以生成；
- Completion 可以解码；
- 显存分配正常；
- 请求能够正常结束和释放缓存。

### 实验 B：观察每轮调度

在 `LLMEngine.step()` 附近临时打印：

```python
print(
    "prefill=", is_prefill,
    "num_seqs=", len(seqs),
    "tokens=", [seq.num_scheduled_tokens for seq in seqs],
)
```

使用短 Prompt 和超长 Prompt，观察：

```text
短 Prompt：prefill → decode → decode → ...
长 Prompt：chunked prefill → ... → decode → ...
```

### 实验 C：验证 Prefix Cache

连续提交两条具有相同长前缀的请求：

```text
共同长前缀 + 问题 A
共同长前缀 + 问题 B
```

在 `BlockManager` 缓存命中路径打印：

- 命中的 block 数；
- 命中的 token 数；
- block ID；
- `ref_count`。

确认第二条请求减少了需要实际 Prefill 的 token 数。

### 实验 D：比较 eager 与 CUDA Graph

分别测试：

```python
enforce_eager = True
enforce_eager = False
```

至少记录：

- 首次运行总耗时；
- 预热后总耗时；
- 总输出 token 数；
- tokens/s；
- 不同 batch size 下的变化。

不要只运行一次就下结论，CUDA Graph 捕获和模型预热会影响首次结果。

### 实验 E：修改 KV Cache 配置

逐个调整：

- `gpu_memory_utilization`；
- `kvcache_block_size`；
- `max_num_seqs`；
- `max_num_batched_tokens`。

记录：

- 可分配 KV Cache block 数；
- 最大并发量；
- 是否发生抢占；
- 吞吐变化；
- TTFT 和 TPOT 变化；
- 是否出现显存不足或明显浪费。

### 实验 F：Tensor Parallel

在至少两张 GPU 上设置 `tensor_parallel_size=2`，跟踪：

- 每个 rank 加载的权重 shape；
- QKV projection 的切分；
- Row Parallel 的 all-reduce；
- 主进程与 worker 的通信；
- 单卡和双卡吞吐差异。

模型很小时，多卡通信成本可能抵消并行收益，这也是正常实验结果。

---

## 8. 推荐进阶任务

完成主线学习后，可以按以下顺序扩展项目。

### 任务 1：增加 Greedy、Top-k 和 Top-p

涉及：

- `nanovllm/sampling_params.py`
- `nanovllm/layers/sampler.py`
- 相应采样测试

学习价值：理解 logits processing、每请求采样参数和数值边界。

### 任务 2：增加流式输出

当前生成接口主要在全部完成后返回结果。可以设计 iterator 或 callback，让每个 step 产生的新 token 被及时输出。

学习价值：理解引擎 step 与服务层之间的接口，以及完成顺序和请求顺序的区别。

### 任务 3：增加调度观测工具

记录每一步：

- Prefill/Decode；
- batch 内请求；
- token 数；
- KV block 占用；
- Prefix Cache 命中；
- 抢占次数。

学习价值：把隐式调度过程变成可观察的数据，为性能分析建立基础。

### 任务 4：支持另一种模型架构

例如增加与 Qwen3 结构相近的 Llama 类模型，需要处理：

- 模型定义；
- 配置字段；
- 权重名称映射；
- packed modules；
- TP 权重切分；
- RoPE 与 Attention 差异。

学习价值：区分“模型相关逻辑”和“推理引擎通用逻辑”。

### 任务 5：系统分析调度策略

尝试加入或比较：

- Prefill 优先；
- Decode 优先；
- 最短请求优先；
- aging；
- 不同 Chunked Prefill 大小；
- 不同抢占策略。

学习价值：理解吞吐、TTFT、TPOT、公平性和尾延迟之间的权衡。

---

## 9. 常见误区

### 9.1 一开始就阅读 Triton

先理解：

```text
token position
  → logical block
  → physical block
  → slot_mapping
  → KV Cache address
```

然后再读 kernel。否则即使看懂 Triton 语法，也很难理解索引设计。

### 9.2 把 Prefill 和 Decode 当作同一种计算

二者虽然调用同一个模型，但输入形状、Attention 路径、KV Cache 行为、性能瓶颈和调度目标都不同。

### 9.3 只读 Qwen3，不读引擎

Transformer 模型结构可以在许多项目中学到。Nano-vLLM 最值得学习的是：

- Scheduler；
- BlockManager；
- ModelRunner；
- KV Cache；
- 批处理和性能执行路径。

### 9.4 忽略张量 shape

遇到核心操作时，至少记录：

```text
input_ids
positions
q / k / v
slot_mapping
block_tables
cu_seqlens
logits
```

很多推理框架 bug 本质上都是长度、offset、padding、block 边界或 shape 不一致。

### 9.5 忽略全局 Context

Attention 的关键参数不一定出现在模型 `forward()` 签名中。调试时应检查 `set_context()` 与 `get_context()` 之间的数据是否正确。

### 9.6 只关注 tokens/s

性能测试至少区分：

- 吞吐量：单位时间处理的总 token；
- TTFT：请求到达后生成第一个 token 的时间；
- TPOT：首 token 之后每个 token 的平均时间；
- 尾延迟：较慢请求的 P95/P99 延迟。

某项优化可能提高总吞吐，却让单请求延迟或公平性变差。

### 9.7 在不支持的环境里反复安装

如果当前机器没有 NVIDIA CUDA 环境，应把重点放在：

- 阅读；
- 手算；
- CPU 可运行的数据结构实验；
- 调度器模拟；
- 远程 GPU 环境中的集中验证。

---

## 10. 从 Nano-vLLM 进入正式 vLLM

当你可以不看代码解释以下概念时，就可以开始阅读正式 vLLM：

- 生成循环；
- Prefill 与 Decode；
- Continuous Batching；
- Chunked Prefill；
- KV Cache 分页；
- `block_table` 与 `slot_mapping`；
- Prefix Cache；
- 抢占；
- Tensor Parallel；
- CUDA Graph。

进入正式 vLLM 时，不要从仓库第一行开始读。按照已经掌握的概念寻找对应组件：

```text
Nano-vLLM Sequence
        ↓
vLLM Request / Sequence state

Nano-vLLM Scheduler
        ↓
vLLM Scheduler

Nano-vLLM BlockManager
        ↓
vLLM KV Cache Manager

Nano-vLLM ModelRunner
        ↓
vLLM GPU Model Runner

Nano-vLLM Attention
        ↓
vLLM Attention Backend
```

正式 vLLM 的基本问题并没有改变，只是加入了更多生产能力：

- 多模型架构；
- 多种 Attention backend；
- 多机多卡；
- 异步执行；
- 更复杂的 Prefix Cache；
- Speculative Decoding；
- LoRA；
- 多模态；
- 服务层与 OpenAI 兼容 API；
- 更完整的调度、指标、容错和部署能力。

建议仍然沿用同一个分析框架：

```text
请求状态
  → 调度决策
  → KV Cache 决策
  → GPU 输入准备
  → 模型执行
  → 采样与后处理
```

---

## 11. 最终自测题

完成学习后，尝试在不运行代码的情况下回答：

1. 一个包含 600 个 Prompt token 的请求，在 block size 为 256 时需要几个 block？
2. Prefill 输入 600 个 token，为什么可能只返回一行 logits？
3. Decode 为什么不需要重新计算全部历史 token 的 K/V？
4. 一条逻辑连续的 Sequence 为什么可以使用不连续的物理 block？
5. `slot_mapping` 与 `block_table` 分别服务于什么操作？
6. Prefix Cache 为什么通常只复用完整 block？
7. 两条 Sequence 共享同一个 block 时，如何避免其中一条结束后误回收缓存？
8. 长 Prompt 为什么可能影响其他请求的 Decode 延迟？
9. Chunked Prefill 怎样缓解这一问题？
10. KV Cache 不够时为什么需要抢占？
11. CUDA Graph 为什么更适合形状较稳定的 Decode？
12. Tensor Parallel 中为什么某些层需要 all-reduce？
13. 增大 `max_num_batched_tokens` 可能怎样影响吞吐和 TTFT？
14. 增大 block size 为什么可能减少元数据开销，却增加内部碎片？
15. 如果 Attention 输出异常，你会按什么顺序检查 Sequence、Context、slot、block table 和 cache？

如果这些问题都能结合代码路径回答，你已经具备进入正式 vLLM 源码的基础。

---

## 12. 学习记录模板

建议每次阅读一个模块时填写以下模板：

```markdown
## 模块名称

### 它解决什么问题？

### 谁调用它？

### 它调用谁？

### 主要输入及 shape

### 主要输出及 shape

### 它读写哪些状态？

### Prefill 与 Decode 是否走不同分支？

### 有哪些 block/token 边界条件？

### 有哪些性能权衡？

### 我还不理解的问题

### 一个可以验证理解的小实验
```

每读完一个文件，都应该能够用一句话概括它的职责。例如：

- `LLMEngine`：组织请求、调度、执行和结果返回；
- `Scheduler`：在序列数、token budget 和缓存容量约束下选择本轮请求；
- `BlockManager`：管理物理 KV Cache block 的分配、共享和回收；
- `ModelRunner`：把 Sequence 状态转换为 GPU 模型需要的张量和 Context；
- `Attention`：写入新 K/V，并根据 Prefill 或 Decode 路径计算 Attention；
- `Sampler`：把每条序列的 logits 转换成下一个 token。

能稳定地做出这种概括，比记忆具体实现细节更重要。
