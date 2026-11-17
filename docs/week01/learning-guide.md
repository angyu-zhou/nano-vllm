# 第一周学习文档：跑通 Nano-vLLM，理解一次生成

适用对象：熟悉 Python 和大模型应用，刚开始学习 AI Infra 的工程师。预计投入 8～10 小时；环境安装耗时另受机器和依赖兼容性影响。

本文依据仓库提交 `cb37cd0` 的实现编写。配套材料：

- [第一周源码分析文档](source-analysis.md)：解释调用链、关键函数和请求状态。
- [观察脚本 observe.py](observe.py)：支持单请求、双请求和逐轮状态观察。
- [12 周总计划](../learning-plan-zh.md)：定位本周与后续知识的关系。

这些材料来自源码核对。文中的手工轨迹是注明条件的教学推演，耗时和生成文本需要你在自己的 GPU 环境中实测。配套脚本不代表已经在你的环境中完成兼容性验证。

## 1. 本周学习目标

本周结束时，你应该能够用自己的话回答：

1. 一个字符串怎样变成模型输入，模型输出又怎样变成字符串？
2. `LLM(...)` 初始化时做了什么，为什么它可能很慢？
3. 为什么生成 3 个 token 通常需要多轮执行？第一轮和后续轮次有什么区别？
4. `Sequence`、`Scheduler` 和 `ModelRunner` 分别管理什么？
5. 一个请求何时结束，多个请求何时全部结束？
6. 哪些状态在 CPU 上维护，哪些计算和缓存放在 GPU 上？

你将交付四项成果：环境记录、一组单/双请求运行日志、一张调用图、一份学习复盘。GPU 暂不可用时，先完成手工推演和源码分析，并把真实运行明确标为待完成。

### 1.1 本周知识边界

| 本周需要掌握 | 本周只需认识用途，后续深入 |
| --- | --- |
| token IDs、logits、采样、EOS | 采样算法的概率推导 |
| Prefill 与 Decode 的区别 | FlashAttention 的内核实现 |
| KV Cache 保存历史 K/V | 分页、哈希、引用计数和缓存淘汰 |
| 请求排队、执行、结束 | 抢占策略与调度公平性 |
| 模型初始化和生成分开计时 | CUDA Graph、编译优化与严谨性能基准 |
| 单卡运行和 CPU/GPU 分工 | 多卡张量并行与 NCCL 通信细节 |

遇到右列内容时，写一句“它解决什么问题”，然后回到主流程。

## 2. 开始前先建立基本概念

### 2.1 从模型调用者转向引擎开发者

你可能已经熟悉这样的应用流程：准备 prompt → 调用模型 → 处理回答。推理引擎还需要回答：哪些请求一起执行，模型本轮接收哪些 token，历史结果放在哪里，以及请求完成后哪些资源能复用。

学习本项目时，对每个函数固定问三个问题：**它接收什么、修改什么、返回什么？** 对每个重要变量再问：它在 CPU 还是 GPU 上？

### 2.2 本周术语表

| 术语 | 含义 | 本项目中的观察位置 |
| --- | --- | --- |
| Tokenizer | 文本与 token IDs 之间的编码/解码工具；一个 token 不一定对应一个字或一个词 | `LLMEngine.tokenizer` |
| Token ID | 词表中的整数编号 | `Sequence.token_ids` |
| Logits | 模型对下一 token 候选给出的未归一化分数 | `ModelRunner.run` 中的 `logits` |
| Sampling | 根据 logits 和温度等设置选择一个 token | `Sampler` |
| EOS | 用于表示生成终止的特殊 token | `config.eos`、`Scheduler.postprocess` |
| Prefill | 处理本轮需要计算的输入上下文，建立相应 KV；完整输入处理完后产生首个输出 token | `prepare_prefill` |
| Decode 阶段 | 输入上轮生成的 token，使用历史 KV，预测下一个 token | `prepare_decode` |
| Tokenizer decode | 把 token IDs 转回文本，与上面的 Decode 计算阶段是不同概念 | `tokenizer.decode` |
| KV Cache | 每层对已处理 token 计算出的 Key/Value 张量，后续注意力可以复用 | `allocate_kv_cache` |
| Batch | 一轮中一起执行的多个请求或其 token | `Scheduler.schedule` 返回的 `seqs` |
| Step | 一轮调度、模型执行和请求状态更新 | `LLMEngine.step` |
| 显存 | GPU 可使用的设备内存，保存模型权重、KV 和中间张量等 | `nvidia-smi`、PyTorch 显存统计 |

### 2.3 先在纸上完成一个例子

假设输入是 4 个 token，要求生成 3 个 token，不提前遇到 EOS：

```text
输入：[p0, p1, p2, p3]
第 1 轮：处理 p0..p3，产生 g0                 ← Prefill
第 2 轮：处理 g0，结合历史 KV，产生 g1        ← Decode
第 3 轮：处理 g1，结合历史 KV，产生 g2，结束  ← Decode
```

思考：产生 `g0` 的那一刻，模型是否已经把 `g0` 作为输入计算过？答案是没有。它刚被预测出来，在下一轮输入模型时才计算其 KV。这个区别会贯穿整套源码分析。

## 3. 环境准备：先判断自己属于哪种情况

| 当前条件 | 本周安排 |
| --- | --- |
| 已有 Linux/NVIDIA GPU 和能用的 PyTorch 环境 | 优先复用独立的开发环境，检查版本后安装本项目 |
| 有 GPU，但环境需要新建 | 准备虚拟环境，匹配 PyTorch、CUDA 与 FlashAttention，再安装项目 |
| 当前只有 Mac 或 CPU | 先读文档、做调用链和状态表；完整引擎运行在后续可用的 GPU 环境中完成 |

项目使用 CUDA、NCCL、Triton 和 FlashAttention，当前代码没有 CPU/MPS 推理后端。`enforce_eager=True` 也不会把它变成 CPU 程序。GPU 需满足所选 FlashAttention 等依赖的支持条件。

下面的运行命令均在 **Linux GPU 机器、项目根目录、独立 Python 环境** 中执行。配套脚本使用 Qwen3，建议从 Qwen3-0.6B 开始。当前文档不替你执行安装或下载。

### 3.1 检查系统和 GPU

```bash
pwd
git rev-parse HEAD
python --version
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

记录机器和版本，不需要追求某个固定吞吐。`nvidia-smi` 中显示的 CUDA 版本主要反映驱动可支持的 CUDA 版本范围，不能据此断定当前 Python 使用哪个 CUDA 构建。

如果 PyTorch 已安装，运行：

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
print("NCCL backend available:", torch.distributed.is_nccl_available())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

`CUDA available` 应为 `True`。NCCL 后端可用不代表其通信初始化一定成功，只说明该 PyTorch 构建包含这个后端。

### 3.2 安装依赖的顺序

`pyproject.toml` 的 Python 范围是 `>=3.10,<3.13`，可选择 3.11。已经有合适环境时跳过创建步骤。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install packaging ninja
```

先依据机器驱动和操作系统，在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择支持 CUDA 的安装命令，再执行上面的 PyTorch 检查。这里不预设一个 CUDA wheel 版本，因为尚未知道你的 GPU 和驱动条件。

随后，在匹配的环境中安装项目需要的其他依赖：

```bash
python -m pip install "transformers>=4.51.0" "triton>=3.0.0" xxhash numpy safetensors tqdm huggingface_hub
python -m pip install flash-attn --no-build-isolation
python -m pip install -e .
python -m pip check
```

这些范围对应项目要求，不是一个经过所有硬件验证的版本锁定文件。成功后保存实际版本。若 FlashAttention 需要从源码编译，通常还需要匹配的 CUDA toolkit、编译器和足够构建资源；可用预编译包的环境不一定需要本地 `nvcc`。遇到安装问题先查看 [FlashAttention 项目说明](https://github.com/Dao-AILab/flash-attention) 的兼容条件。

`pip install -e .` 的含义是使用当前工作区的源码，后续阅读和小改动会对应到正在运行的代码。

检查依赖和导入位置：

```bash
python - <<'PY'
import importlib.metadata as metadata
for package in ("torch", "transformers", "triton", "flash-attn", "xxhash", "numpy", "safetensors", "tqdm"):
    print(package, metadata.version(package))
import nanovllm
print("nanovllm loaded from:", nanovllm.__file__)
PY
```

确认最后的路径属于你正在学习的仓库。

### 3.3 下载并检查模型

使用当前 Hugging Face CLI 的示例：

```bash
hf download Qwen/Qwen3-0.6B --local-dir "$HOME/huggingface/Qwen3-0.6B"
```

模型已存在时直接使用实际目录。需要严格复现时，记录模型仓库的真实 revision，并在下载时指定它。

```bash
python - <<'PY'
from pathlib import Path
import json
path = Path.home() / "huggingface" / "Qwen3-0.6B"
config = json.loads((path / "config.json").read_text())
print("Path:", path)
print("model_type:", config.get("model_type"))
print("dtype fields:", config.get("dtype"), config.get("torch_dtype"))
print("Weight files:", [p.name for p in path.glob("*.safetensors")])
print("Tokenizer files:", [p.name for p in path.glob("*tokenizer*")])
PY
```

应该有模型配置、实际权重和 tokenizer 资源。下载了一个配置文件不等于下载了整个模型；文件存在检查也不能代替权重完整性验证。

### 3.4 固定本周起步配置

配套脚本使用如下配置：

| 配置 | 本周值 | 原因 |
| --- | --- | --- |
| 模型 | Qwen3-0.6B | 降低模型和显存规模，保持流程完整 |
| `tensor_parallel_size` | 1 | 先理解单卡路径 |
| `enforce_eager` | `True` | 跳过 CUDA Graph 捕获与重放，部分算子仍会编译 |
| `max_model_len` | 1024，且不超过模型自身上限 | 使用短上下文 |
| `max_num_batched_tokens` | 512 | 缩小本轮 Prefill 预算和初始化预热规模 |
| `max_num_seqs` | 8 | 限制本轮请求数量 |
| `gpu_memory_utilization` | 0.75 | 为 KV 池计算提供起步预算，实际可用空间仍需检查 |
| 正常生成 `max_tokens` | 32 | 观察短输出，减少等待 |
| 轨迹生成 `max_tokens` | 分别 3、5 | 明确观察两个请求在不同轮次完成 |

这些是教学配置，不是最优性能配置，也不保证所有环境都不会 OOM。

## 4. 七天执行安排

每天遵循“先预测 → 执行或推演 → 对照源码 → 写下结论”的顺序。

| 日期 | 时间 | 主要任务 | 当天产物 |
| --- | --- | --- | --- |
| 第 1 天 | 1 小时 | 读术语和项目入口，完成纸上生成例子 | 术语表、初版调用图 |
| 第 2 天 | 1.5 小时 | 检查环境，安装/确认依赖，准备模型 | 环境记录、问题清单 |
| 第 3 天 | 1 小时 | 做实验 A 和 B，运行单请求与双请求 | 两份运行日志 |
| 第 4 天 | 1 小时 | 精读 `generate / add_request / step` | 三个函数的输入、修改、输出表 |
| 第 5 天 | 1 小时 | 阅读 `Sequence`，做单请求手工状态表 | 字段解释与完整状态表 |
| 第 6 天 | 2 小时 | 做实验 C，核对双请求逐轮轨迹 | 带解释的轨迹和最终调用图 |
| 第 7 天 | 1 小时 | 完成自测和复盘，检查验收项 | 一页周报告与待解决问题 |

### 第 1 天：建立整体认识

阅读本文第 1～2 节，再读源码分析文档的“整体结构”和“从 `example.py` 进入系统”。打开真实的 `example.py`，找出模型目录、Tokenizer、LLM、采样配置和输出打印分别在哪里。

完成三个问题：Tokenizer 是否属于大模型权重？`max_tokens` 限制的是输入还是新增输出？模型为什么不能在第一次普通前向中直接得到所有后续采样 token？

当天先画粗图：文本 → token IDs → 请求 → 调度 → 模型 → 下一个 token → 请求更新 → 文本。第 6 天再修正细节。

### 第 2 天：准备环境

执行第 3 节适用的步骤，填写第 7 节环境表。每条报错都记录具体命令和最早出现的有效异常。

如果 1.5 小时内没有解决依赖问题，先把问题缩小到驱动/PyTorch/扩展编译/模型文件中的哪一层，再继续第 4、5 天的无 GPU 学习任务。运行任务仍保留为待完成。

### 第 3 天：第一次看见真实生成

完成实验 A 和 B。先检查请求数量、实际输出 token 数和正常结束，再观察文本。Qwen3 在短输出限制下可能只生成部分句子，不能仅靠“回答是否完整”判断引擎是否跑通。

写下初始化时间与生成时间的区别；本周不做“比另一个推理框架快多少”的结论。

### 第 4 天：读懂生成循环

按源码分析文档的函数分析，完成下表：

| 函数 | 接收什么 | 修改什么 | 返回什么 |
| --- | --- | --- | --- |
| `add_request` | 自己填写 | 自己填写 | 自己填写 |
| `step` | 自己填写 | 自己填写 | 自己填写 |
| `generate` | 自己填写 | 自己填写 | 自己填写 |

特别寻找 `while not self.is_finished()`、`scheduler.schedule()`、`model_runner.call("run", ...)` 和 `scheduler.postprocess(...)`。先用一句话解释每一处，再讨论内部算法。

### 第 5 天：理解请求携带的状态

重点阅读 `Sequence.__init__`、`append_token` 和三个 token 列表/数量属性。根据源码分析中的 4-token 输入示例，独立填写每一轮的总 token 数、输出 token 数、已缓存 token 数和状态。

当你能解释“生成了一个 token，但它的 KV 尚未计算”时，就抓住了本周最重要的状态关系。

### 第 6 天：用真实轨迹验证理解

运行实验 C。先预计两个请求分别在第几轮结束，再查日志。读不懂某个字段时，先查源码分析的字段表。

可在调试器设置断点：`LLMEngine.step` 入口、`schedule` 返回后、`ModelRunner.run` 入口、`postprocess` 返回后。GPU 环境下断点会扰动执行时间，调试结果只用于理解。

### 第 7 天：验收与复盘

不看文档，用 5～10 分钟讲解一次单请求和一次双请求运行。然后完成第 8 节自测题，对照参考答案修正遗漏。最后提交第 7 节周报告。

## 5. 三个配套实验

先创建本周结果目录：

```bash
mkdir -p learning-results/week01
python -m pip freeze > learning-results/week01/environment.txt
git rev-parse HEAD > learning-results/week01/commit.txt
```

### 实验 A：单请求跑通

从仓库根目录运行：

```bash
python docs/week01/observe.py --model "$HOME/huggingface/Qwen3-0.6B" --mode single
```

脚本依次打印 JSON 事件：`inputs` → `initialized` → `result` → `finished`。第三方库可能另外输出初始化日志。

- `inputs`：实际 tokenized prompt 长度、输出上限和环境信息。
- `initialized`：`LLM(...)` 构造耗时和初始化后的显存；外部预先加载 tokenizer 等步骤不计入这个时间。
- `result`：模型实际生成的 token IDs 和文本。
- `finished`：生成阶段耗时、总输出数量，以及结束时 waiting/running 队列长度。

预期结构：1 个 `result`，输出 token 数在 1～32 之间，最终两个队列长度都为 0。具体文本和耗时不固定，EOS 可能让请求提前结束。

需要保存完整运行输出时：

```bash
python docs/week01/observe.py --model "$HOME/huggingface/Qwen3-0.6B" --mode single > learning-results/week01/single.log 2>&1
```

重跑会覆盖同名日志；保留多次记录时使用不同文件名。

**实验记录：** 初始化多少秒？生成多少秒？实际 prompt/output 各多少 token？生成后显存是否仍然占用？根据源码解释最后一个现象。

### 实验 B：两个不同长度的请求

```bash
python docs/week01/observe.py --model "$HOME/huggingface/Qwen3-0.6B" --mode batch
```

该模式一次提交两个不同长度的 prompt，分别允许最多生成 32 个 token。预期得到两个 `result`，按提交顺序输出。它不是用两个独立引擎分别执行两次。

**先预测：** 输入更长的请求一定更晚结束吗？两个输出一定都是 32 个 token 吗？

**分析方向：** 完成时间取决于实际生成长度、EOS 和调度等因素。不同 prompt 的文本、输出长度都可能不同；本实验主要验证多请求流程，不能直接用它定量证明 batching 加速。

### 实验 C：逐轮追踪请求

```bash
python docs/week01/observe.py --model "$HOME/huggingface/Qwen3-0.6B" --mode trace
```

该模式调用真实的 `add_request` 和 `step`，每一步前后保存请求状态。两个请求输出上限分别为 3、5，并设置 `ignore_eos=True`，方便固定轮数；短输出可能是不完整句子，甚至包含 EOS 对应的特殊 token。

脚本检查两个输入可放进一轮 Prefill。在显存足够、没有抢占的正常起步场景中，预计轨迹如下：

| 轮次 | 阶段 | A 已输出 token 数 | B 已输出 token 数 | 本轮完成的请求 |
| --- | --- | --- | --- | --- |
| 1 | Prefill | 1 | 1 | 无 |
| 2 | Decode | 2 | 2 | 无 |
| 3 | Decode | 3，完成 | 3 | A |
| 4 | Decode | 已完成 | 4 | 无 |
| 5 | Decode | 已完成 | 5，完成 | B |

这里用 A/B 表示提交顺序；实际 `seq_id` 不保证从 0 开始，因为初始化预热也创建 `Sequence`。

阅读日志时注意：

1. 第一轮 `signed_num_tokens` 是本轮 Prefill 实际调度的 token 数；后续通常是 `-2, -2, -1, -1`。负号是引擎给 Decode 统计使用的标记。
2. `completed_seq_ids` 只包含本轮完成的请求；为空不代表没有生成 token。
3. 完成的请求在 `after` 里仍可见，是脚本提前保留了对象引用；它已从真实运行队列移除。
4. 完成后 `num_cached_tokens` 归零、`block_table` 清空，是释放资源引用的行为；生成 token 列表仍保留。
5. `num_scheduled_tokens` 在 step 前后常为 0，中间的调度阶段才会被设置。想看中间值，在 `schedule()` 返回后断点查看。
6. 日志中 `is_prefill_flag` 对应对象字段，首轮之后它可能仍为真；本轮阶段以 `step` 使用的局部 `is_prefill` 为准。详细原因见源码分析。

**验收：** 找出 A 完成的那一轮，并解释为什么 B 继续运行；找出 Prefill 产生第一个 token 的证据。

### 5.1 观察脚本与原始示例的区别

原始 `example.py` 使用 `apply_chat_template(..., tokenize=False)` 先生成字符串，再交给引擎编码。配套脚本使用 `tokenize=True` 直接取得 token IDs，随后传入引擎，便于准确记录输入长度，并避免重复添加特殊 token 的歧义。

脚本为 Qwen3 设置 `enable_thinking=False`，让短输出更容易阅读；这属于该模型聊天模板的行为，不能推广为所有模型的通用开关。脚本没有修改引擎和模型的执行逻辑。

`trace` 模式额外打印状态，因此它的耗时包含观察成本。三个模式都是教学实验，尚不是用于横向性能比较的基准。

## 6. 常见问题与定位方法

先定位失败发生在哪一阶段：依赖导入 → 模型配置/分词器 → 引擎初始化 → 真实请求执行 → 输出整理。

| 现象 | 可能原因 | 优先检查 |
| --- | --- | --- |
| `ModuleNotFoundError: nanovllm` | 没有安装当前仓库，或解释器环境不一致 | 当前目录、`python -m pip --version`、`pip install -e .` |
| 缺少 Triton / FlashAttention | 依赖没有安装或安装失败 | 完整安装日志、GPU/系统支持范围 |
| `undefined symbol` 等扩展导入失败 | PyTorch、CUDA 和扩展二进制不匹配 | 三者版本和包来源，按兼容组合重装扩展 |
| `torch.cuda.is_available()` 为假 | CPU 版 PyTorch、GPU 不可见或驱动问题 | `nvidia-smi`、`torch.version.cuda`、GPU 可见性配置 |
| NCCL 初始化失败 | 后端、环境或本机通信初始化问题 | 即使 TP=1 也初始化 NCCL；检查最早的 NCCL 错误 |
| 端口 2333 已占用 | 当前实现使用固定本机初始化端口，另一个引擎进程可能正在使用 | 检查自己的运行实例；本周顺序运行实验 |
| 模型路径断言或文件错误 | 路径不存在、下载不完整、模型目录不正确 | `config.json`、权重、tokenizer 文件是否齐全 |
| `hf_config.dtype` 相关错误 | Transformers 接口/模型配置与当前代码不兼容 | 实际 Transformers 版本和模型配置；别只根据依赖下限判断兼容 |
| CUDA OOM | 权重、初始化预热、中间张量或缓存池超出空间 | 先查 traceback 所在阶段和可用显存，再调整相应预算 |
| `num_kvcache_blocks > 0` 断言失败 | 预算扣除现有占用与预热峰值后，无法容纳缓存块 | 空闲显存、其他占用、模型规模、预热预算；盲目降低利用率可能更糟 |
| `greedy sampling is not permitted` | 温度设置为 0 或过小 | 本版本要求正温度；使用示例中的 0.6 |
| 输出比上限短 | 遇到 EOS，或只是文本显示与 token 数不一致 | 实际 `token_ids` 数量、`ignore_eos`、特殊 token |
| 首次运行很慢 | 权重加载、CUDA 初始化、算子编译、预热 | 区分初始化和生成时间；`enforce_eager=True` 仍保留算子编译 |
| 生成结束显存没明显下降 | 模型权重和 KV 池仍由引擎持有 | 区分请求释放块引用和引擎释放 GPU 张量 |

OOM 的处理应对应发生位置：预热期间可先缩小 `max_num_batched_tokens` 等；KV 池分配期间核对预算与可用空间；权重加载期间需要更小模型或更多可用显存。不要同时随机修改所有参数。

引擎构造后已通过 `atexit` 注册清理。配套脚本让进程正常结束，不手动重复调用 `llm.exit()`。异常后若另有清理报错，优先保留并分析最早的根因。

## 7. 可直接复制的学习记录模板

### 7.1 环境记录

| 项目 | 实际值 |
| --- | --- |
| 日期 / 机器说明 | 待填写 |
| 操作系统 / Python | 待填写 |
| 仓库提交 | 待填写 |
| GPU 型号 / 总显存 / 开始前空闲显存 | 待填写 |
| 驱动 / PyTorch / PyTorch CUDA 构建 | 待填写 |
| Transformers / Triton / FlashAttention | 待填写 |
| 模型目录 / 模型 revision | 待填写；不确定的 revision 明确标注未记录 |
| 本次配置变更 | 待填写 |

### 7.2 实验记录

```text
实验名称：
运行命令：
运行条件与配置：
执行前的预测：
实际输入 token 数：
实际输出 token 数：
初始化时间 / 生成时间：
结束时 waiting / running：
日志文件位置：
预测与结果不一致的地方：
对应的源码证据：
仍需验证的问题：
```

### 7.3 第一周分析报告

```text
1. 用 5 句话介绍本项目如何处理一次生成。
2. 附调用图，标出 CPU 逻辑与 GPU 执行。
3. 解释 LLMEngine、Sequence、Scheduler、ModelRunner 的职责。
4. 附单请求手工状态表和双请求实际轨迹。
5. 分别说明请求结束、全部请求结束、进程结束时发生什么。
6. 记录一个误解，以及用哪段代码修正了它。
7. 列出已完成项、未完成项和下一步验证动作。
```

## 8. 自测题与参考答案

先独立回答，再向下查看答案。

1. `LLM` 类代码很少，核心生成逻辑在哪里？
2. `max_tokens=32` 是否表示最多输入 32 个 token？
3. 为什么首个输出 token 通常在 Prefill 轮产生？
4. 为什么后续 Decode 不需要再次输入全部历史 token？
5. `step()` 返回空的完成列表，是否意味着模型没做任何工作？
6. 一个请求生成结束的两个主要条件是什么？全部请求结束如何判断？
7. `generate()` 的实际返回值是什么？
8. 两条输入顺序与完成顺序不一致时，正常 `generate()` 最后按什么顺序返回？
9. 输入有 4 个 token、输出有 3 个 token，忽略 EOS、无分块无抢占，一共几轮 Prefill/Decode？
10. 为什么第一个真实请求的 `seq_id` 可能不是 0？
11. 为什么结束后 KV 块归还了，GPU 显存仍可能保持高占用？
12. `enforce_eager=True` 是否同时关闭 CUDA Graph 和所有 `torch.compile`？

参考答案：

1. `LLM` 继承 `LLMEngine`，核心逻辑在 `nanovllm/engine/llm_engine.py`。
2. 它是最多新增的输出 token 数；输入由 prompt 的分词长度决定。
3. 模型在输入上下文最后位置得到预测下一 token 的分数，采样后就是首个输出。
4. 已处理上下文的 K/V 保存在缓存中；新 token 的 Query 仍会通过注意力读取历史 KV。
5. 不是。请求可能生成了新 token，但还未达到结束条件；分块 Prefill 也可能只推进输入计算。
6. 未忽略 EOS 时采样到 EOS，或新增 token 数达到 `max_tokens`；全部结束要求 waiting 和 running 都为空。
7. 字典列表，每项有 `text` 和 `token_ids`。当前源码里的 `list[str]` 返回注解与实际实现不一致。
8. 普通独立 `generate` 调用按 `seq_id` 排序，该编号按提交时的创建顺序递增，因此恢复输入顺序。
9. 1 轮 Prefill、2 轮 Decode；首个输出来自 Prefill。
10. 初始化预热也会创建 `Sequence` 并消耗编号。
11. 请求释放的是块的使用关系，模型权重和预分配 KV 张量仍由引擎持有。
12. 只跳过 CUDA Graph；若干层和采样器仍有 `torch.compile`。

## 9. 第一周验收清单

- [ ] 有完整环境记录，能定位实际运行的仓库版本。
- [ ] 成功完成一个单请求和一个双请求实验，保存日志。
- [ ] 能画出初始化链路与生成链路，并解释两者区别。
- [ ] 能解释 `Sequence` 的输入、输出、总 token 数与已缓存 token 数。
- [ ] 能独立推演“4 个输入 token、3 个输出 token”的每轮状态。
- [ ] 能从真实轨迹说明一个请求先结束，另一个继续运行。
- [ ] 能解释 `step` 完成列表、EOS、输出上限和空队列结束条件。
- [ ] 已提交一页复盘，并明确尚未验证的内容。

如果暂时没有 GPU，先完成可以静态验证的项目，保留运行相关待办。进入第二周时继续复用这些请求例子，进一步追踪 `input_ids`、`positions` 和 Q/K/V 的张量形状。
