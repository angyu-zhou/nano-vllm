"""Week 1 single-GPU teaching demo for Qwen3; see learning-guide.md.

This records request states, not benchmark-quality timings. GPU dependencies
are imported after argument parsing so --help also works without CUDA.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def snapshot(seq):
    return {
        "seq_id": seq.seq_id,
        "status": seq.status.name,
        "num_prompt_tokens": seq.num_prompt_tokens,
        "num_tokens": seq.num_tokens,
        "num_completion_tokens": seq.num_completion_tokens,
        "num_cached_tokens": seq.num_cached_tokens,
        "num_scheduled_tokens": seq.num_scheduled_tokens,
        "last_token": seq.last_token,
        "block_table": list(seq.block_table),
        "is_prefill_flag": seq.is_prefill,
    }


def trace_requests(llm, prompts, params):
    for prompt, param in zip(prompts, params, strict=True):
        llm.add_request(prompt, param)

    completed = {}
    step_number = 0
    while not llm.is_finished():
        step_number += 1
        # Keep references so a sequence removed on completion is still visible.
        active = list(llm.scheduler.waiting) + list(llm.scheduler.running)
        before = [snapshot(seq) for seq in active]
        outputs, signed_tokens = llm.step()
        after = [snapshot(seq) for seq in active]
        emit(
            "step",
            step=step_number,
            phase="prefill" if signed_tokens > 0 else "decode",
            signed_num_tokens=signed_tokens,
            before=before,
            after=after,
            completed_seq_ids=[seq_id for seq_id, _ in outputs],
        )
        for seq_id, token_ids in outputs:
            completed[seq_id] = list(token_ids)

    return [
        {
            "seq_id": seq_id,
            "token_ids": completed[seq_id],
            "text": llm.tokenizer.decode(completed[seq_id]),
        }
        for seq_id in sorted(completed)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3 model directory")
    parser.add_argument(
        "--mode", choices=("single", "batch", "trace"), default="single"
    )
    args = parser.parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        parser.error(f"Model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        parser.error("The model directory must contain config.json")
    if not any(model_path.glob("*.safetensors")):
        parser.error("The model directory must contain downloaded .safetensors weights")

    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("This demo requires a supported NVIDIA CUDA GPU on Linux.")

    from nanovllm import LLM, SamplingParams

    torch.manual_seed(0)
    hf_config = AutoConfig.from_pretrained(str(model_path))
    if hf_config.model_type != "qwen3":
        parser.error("This teaching demo is prepared for Qwen3, preferably Qwen3-0.6B")
    max_model_len = min(1024, hf_config.max_position_embeddings)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    texts = [
        "Say hello in one short sentence.",
        "Explain in two short sentences why a language model generates text one token at a time.",
    ]
    if args.mode == "single":
        texts = texts[:1]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in texts
    ]
    params = (
        [SamplingParams(temperature=0.6, max_tokens=n, ignore_eos=True) for n in (3, 5)]
        if args.mode == "trace"
        else [SamplingParams(temperature=0.6, max_tokens=32) for _ in prompts]
    )
    for token_ids, param in zip(prompts, params, strict=True):
        if not token_ids or len(token_ids) + param.max_tokens > max_model_len:
            parser.error("Each prompt must be nonempty and fit the configured context with its output")
    if sum(map(len, prompts)) > 512:
        parser.error("This week's trace assumes all prompts fit in one 512-token prefill batch")

    emit(
        "inputs",
        mode=args.mode,
        model=str(model_path),
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda,
        prompt_lengths=[len(prompt) for prompt in prompts],
        max_tokens=[param.max_tokens for param in params],
        ignore_eos=[param.ignore_eos for param in params],
    )
    torch.cuda.synchronize()
    started = perf_counter()
    llm = LLM(
        str(model_path),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=512,
        max_num_seqs=8,
        gpu_memory_utilization=0.75,
    )
    torch.cuda.synchronize()
    emit(
        "initialized",
        initialization_seconds=perf_counter() - started,
        num_kvcache_blocks=llm.model_runner.config.num_kvcache_blocks,
        allocated_mib=round(torch.cuda.memory_allocated() / 2**20, 2),
        reserved_mib=round(torch.cuda.memory_reserved() / 2**20, 2),
    )

    torch.cuda.synchronize()
    started = perf_counter()
    if args.mode == "trace":
        outputs = trace_requests(llm, prompts, params)
    else:
        outputs = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    if len(outputs) != len(prompts):
        raise RuntimeError("The number of outputs does not match the number of requests")
    for index, (text, output) in enumerate(zip(texts, outputs, strict=True)):
        emit(
            "result",
            request_index=index,
            prompt=text,
            output_tokens=len(output["token_ids"]),
            **output,
        )
    emit(
        "finished",
        generation_seconds=elapsed,
        actual_output_tokens=sum(len(output["token_ids"]) for output in outputs),
        waiting=len(llm.scheduler.waiting),
        running=len(llm.scheduler.running),
        allocated_mib=round(torch.cuda.memory_allocated() / 2**20, 2),
        reserved_mib=round(torch.cuda.memory_reserved() / 2**20, 2),
    )
    # LLMEngine already registered exit() with atexit. Do not call it twice.


if __name__ == "__main__":
    main()
