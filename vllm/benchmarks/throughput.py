# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark offline inference throughput."""

from datetime import datetime

from pathlib import Path

from dataclasses import replace

import argparse
import json
import random
import time
from typing import Any

import torch

from vllm.benchmarks.datasets import (
    RandomDataset,
    SampleRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.outputs import RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser


def run_vllm(
    requests: list[SampleRequest],
    n: int,
    llm,
    do_profile: bool,
    disable_detokenize: bool = False,
    warmup_requests: list[SampleRequest] | None = None,
    prequeue_requests: bool = False,
) -> tuple[float, list[RequestOutput] | None]:

    if warmup_requests:
        print(f"Warming up with {len(warmup_requests)} requests...")
        _run_vllm_requests(
            llm,
            warmup_requests,
            n,
            disable_detokenize,
            do_profile=False,
            prequeue_requests=prequeue_requests,
        )

    return _run_vllm_requests(
        llm,
        requests,
        n,
        disable_detokenize,
        do_profile=do_profile,
        prequeue_requests=prequeue_requests,
    )


def _run_vllm_requests(
    llm: Any,
    requests: list[SampleRequest],
    n: int,
    disable_detokenize: bool,
    do_profile: bool,
    prequeue_requests: bool,
) -> tuple[float, list[RequestOutput] | None]:
    from vllm import SamplingParams

    prompts: list[TextPrompt | TokensPrompt] = []
    sampling_params: list[SamplingParams] = []
    for request in requests:
        if isinstance(request.prompt, dict) and "prompt_token_ids" in request.prompt:
            prompt_token_ids = request.prompt["prompt_token_ids"]
            assert isinstance(prompt_token_ids, list)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        else:
            assert isinstance(request.prompt, str)
            prompt = TextPrompt(prompt=request.prompt)
        prompts.append(prompt)

        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )

    start = time.perf_counter()
    if do_profile:
        llm.start_profile()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    if do_profile:
        llm.stop_profile()
    end = time.perf_counter()
    return end - start, outputs


def get_requests(args, tokenizer, num_prompts) -> list[SampleRequest]:
    requests = RandomDataset(random_seed=args.seed).sample(
        num_requests=num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        tokenizer=tokenizer,
    )
    return requests


def validate_args(args):
    """
    Validate command-line arguments.
    """

    num_devices = torch.cuda.device_count()
    if num_devices < args.tp_size:
        raise ValueError(f"Not enough devices {num_devices} for tp size {args.tp_size}")

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model


def add_cli_args(parser: FlexibleArgumentParser):
    parser.add_argument("--tp-size", type=int, default=None)

    parser.add_argument("--node-size", type=int, default=None)

    parser.add_argument(
        "--max-eff-batch-size",
        type=int,
        default=None,
        help="Sweep batch size from 1,2,4..max-batch-size",
    )

    parser.add_argument(
        "--max-input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )

    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize the response (i.e. do not include "
            "detokenization time in the measurement)"
        ),
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Use vLLM Profiling. --profiler-config must be provided on the server.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)


def init_llm(engine_args, tp_size: int):

    engine_args.tensor_parallel_size = tp_size
    engine_args.pipeline_parallel_size = 1
    engine_args.data_parallel_size = 1

    engine_args.device_ids = list(range(tp_size))

    from vllm import LLM

    path = Path("/tmp/vllm_kv_cache.json")
    path.unlink(True)

    llm = LLM.from_engine_args(engine_args)

    while not path.exists():
        time.sleep(1)

    kv_cache_tokens = int(path.read_text())

    return llm, kv_cache_tokens


def main(args: argparse.Namespace):
    validate_args(args)
    if args.seed is None:
        args.seed = 0
    random.seed(args.seed)

    tokenizer = get_tokenizer(
        args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    eff_max_bs = args.max_eff_batch_size
    tp_size = args.tp_size

    requests = get_requests(args, tokenizer, eff_max_bs)

    results = []

    engine_args = EngineArgs.from_cli_args(args)

    llm, max_tokens = init_llm(engine_args, tp_size)

    CONTEXT_LENS = [512, 2048, 8192, 32768]
    OUTPUT_LENS = [128, 512, 1024]

    for input_len in CONTEXT_LENS:
        if input_len > args.max_input_len:
            break
        for output_len in OUTPUT_LENS:
            if output_len > args.max_output_len:
                break
            bs = 1
            while bs * (args.node_size // tp_size) <= eff_max_bs:
                common_args = {
                    "input_len": input_len,
                    "output_len": output_len,
                    "tp_size": tp_size,
                    "batch_size": bs,
                }

                if bs * (input_len + output_len) > max_tokens:
                    results.append(
                        {
                            "elapsed_time": None,
                            "reason": "batch too large",
                            **common_args,
                        }
                    )
                    continue

                warmup_reqs = [replace(req) for req in requests]
                for req in warmup_reqs:
                    req.expected_output_len = 1

                elapsed_time, request_outputs = run_vllm(
                    requests[:bs],
                    1,
                    llm,
                    disable_detokenize=args.disable_detokenize,
                    do_profile=args.profile,
                    warmup_requests=warmup_reqs[:bs],
                    prequeue_requests=args.prequeue_requests,
                )

                total_prompt_tokens = 0
                total_output_tokens = 0
                for ro in request_outputs:
                    if not isinstance(ro, RequestOutput):
                        continue
                    total_prompt_tokens += (
                        len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
                    )
                    total_output_tokens += sum(
                        len(o.token_ids)
                        for o in ro.outputs
                        if o is not None and o.token_ids is not None
                    )
                total_num_tokens = total_prompt_tokens + total_output_tokens

                print(f"Batch Size: {bs}, TP Size: {tp_size}")
                print(
                    f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
                    f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
                    f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
                )
                print(f"Total num prompt tokens:  {total_prompt_tokens}")
                print(f"Total num output tokens:  {total_output_tokens}")

                results.append({"elapsed_time": elapsed_time, **common_args})

                bs *= 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = f"results/{args.model}_ns{args.node_size}_tp{tp_size}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
