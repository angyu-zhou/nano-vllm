import os
from dataclasses import dataclass
from transformers import AutoConfig


# dataclass 自动生成初始化等方法；slots=True 限定实例属性，减少内存开销。
@dataclass(slots=True)
class Config:
    # 本地模型目录，包含模型配置、权重和分词器文件。
    model: str
    # 单次 prefill（处理提示词）调度的 token 总预算，超过预算的长提示词可分块处理。
    max_num_batched_tokens: int = 16384
    # 单次调度最多处理的序列数，prefill 和 decode（逐步生成）阶段均受此限制。
    max_num_seqs: int = 512
    # 配置的最大上下文长度（以 token 为单位），用于预热和 CUDA Graph 缓冲区规划。
    max_model_len: int = 4096
    # GPU 总显存的使用预算比例；扣除已有占用和预热峰值开销后，计算 KV Cache 容量。
    gpu_memory_utilization: float = 0.9
    # 张量并行的 GPU 数量；1 表示单卡，多卡时将模型中的张量计算拆分到各卡。
    tensor_parallel_size: int = 1
    # 为 True 时强制即时执行，禁用 CUDA Graph 的捕获与重放。
    enforce_eager: bool = False
    # Hugging Face 模型配置，在 __post_init__ 中从模型目录加载并覆盖此字段。
    hf_config: AutoConfig | None = None
    # 结束标记的 token ID；-1 是占位值，由 LLMEngine 加载分词器后填入。
    eos: int = -1
    # 每个 KV Cache 块容纳的 token 数，用于分页管理注意力的 Key/Value 缓存。
    kvcache_block_size: int = 256
    # KV Cache 块数；-1 是占位值，由 ModelRunner 根据显存预算计算并回填。
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # dataclass 生成的 __init__ 在字段赋值后自动调用此方法。
        # 当前实现要求传入已存在的本地目录，不直接接受 Hugging Face 仓库名称。
        assert os.path.isdir(self.model)
        # 当前实现要求 KV Cache 块大小为 256 的整数倍，张量并行规模为 1～8 张卡。
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 读取模型结构参数，例如层数、注意力头数和最大位置编码长度。
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 配置的上下文长度不能超过模型声明的最大位置编码长度。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
