import argparse
import logging
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator, AutoTokenizer

# for checkpointing
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import matplotlib.pyplot as plt
import random

import sys
sys.path.append("/Users/lilylassiter/Desktop/kronfluence-main/examples/tinystories2")

from pipeline import construct_gpt2, get_tinystories_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on WikiText dataset.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path that is storing the final checkpoint of the model.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=False,
        help="Whether to use torch compile for computing factors and scores.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--compute_per_token_scores",
        action="store_true",
        default=False,
        help="Boolean flag to compute per token scores.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


class LanguageModelingTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # We could also compute the log-likelihood or averaged margin.
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    # 100 samples for training and 2 samples for evaluation.
    train_dataset = get_tinystories_dataset(split="eval_train", num_samples=100)
    eval_dataset = get_tinystories_dataset(split="valid", num_samples=2)
    
    # Prepare the trained model.
    model = construct_gpt2()
    
    # TinyStories HF repo 
    # 'https://huggingface.co/rock-z/tiny_gpt2_tiny_stories/resolve/main/checkpoint-49683/model.safetensors'
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint-49683/model.safetensors")
    if not os.path.isfile(checkpoint_path):
        hf_hub_download(repo_id='rock-z/tiny_gpt2_tiny_stories', filename='checkpoint-49683/model.safetensors', local_dir='./checkpoints')

    weights = load_file(checkpoint_path)
    for k, v in weights.items():
        # keep transformer.wpe.weight and transformer.wte.weight as they are
        if 'transformer.wpe.weight' in k or 'transformer.wte.weight' in k:
            continue
        weights[k] = v.T

    model.load_state_dict(weights, strict=False)

    # Define task and prepare model.
    task = LanguageModelingTask()
    model = prepare_model(model, task)

    if args.use_compile:
        model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name="tinystories",
        model=model,
        task=task,
        profile=args.profile,
    )
        
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    print("Computing influence factors...")
    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    if args.use_compile:
        factors_name += "_compile"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        initial_per_device_batch_size_attempt=64,
        overwrite_output_dir=False,
    )
    
    print("Computing pairwise scores...")
    
    # Compute pairwise scores.
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    if args.use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"
    if args.use_compile:
        scores_name += "_compile"
    if args.compute_per_token_scores:
        score_args.compute_per_token_scores = True
        scores_name += "_per_token"
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    if rank is not None:
        score_args.query_gradient_low_rank = rank
        score_args.query_gradient_accumulation_steps = 10
        scores_name += f"_qlr{rank}"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    
    print(scores)
    logging.info(f"Scores shape: {scores.shape}")
    
    
    # print("Visualizing pairwise scores...")
    # score_args = ScoreArguments(compute_per_module_scores=True)
    # analyzer.compute_pairwise_scores(
    #     score_args=score_args,
    #     scores_name=scores_name,
    #     factors_name=factors_name,
    #     query_dataset=eval_dataset,
    #     train_dataset=train_dataset,
    #     per_device_query_batch_size=args.query_batch_size,
    #     overwrite_output_dir=False,
    # )
    # per_module_scores = analyzer.load_pairwise_scores(scores_name=scores_name)
    # per_module_scores.keys()    
    # plt.matshow(per_module_scores['all_modules'])
    # plt.colorbar()
    # plt.savefig("output2.png")  
    
    # order the scores per test example, and print corresponding tokenized top 5 training documents
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # top 5 highest scores in descending order
    top5_indices = scores.argsort(axis=1)[:, -5:]
    # top 5 training examples for each test example
    top5_input_ids = [train_dataset['input_ids'][i] for i in top5_indices.flatten()]
    top5_tokenized = [tokenizer.decode(ids, skip_special_tokens=True) for ids in top5_input_ids]

    with open('top5.txt', 'w') as f:
        for i in range(len(top5_tokenized)):
            if i % 5 == 0:
                f.write("<<<<<<<<<<Test example %d:\n" % (i // 5 + 1))
                f.write("%s\n" % tokenizer.decode(eval_dataset['input_ids'][i // 5], skip_special_tokens=True))
                f.write("\n<<<<<<<<<<Top 5 training examples for test example %d:\n" % (i // 5 + 1))
            f.write("\n-----------------\n")
            f.write("Index: %d\n" % top5_indices[i // 5, i % 5])
            f.write("%s\n" % top5_tokenized[i])
   
    # top 5 lowest scores
    bottom5_indices = scores.argsort(axis=1)[:, :5]
    bottom5_input_ids = [train_dataset['input_ids'][i] for i in bottom5_indices.flatten()]
    bottom5_tokenized = [tokenizer.decode(ids, skip_special_tokens=True) for ids in bottom5_input_ids]
    with open('bottom5.txt', 'w') as f: 
        for i in range(len(bottom5_tokenized)):
            if i % 5 == 0:
                f.write("<<<<<<<<<<Test example %d:\n" % (i // 5 + 1))
                f.write("%s\n" % tokenizer.decode(eval_dataset['input_ids'][i // 5], skip_special_tokens=True))
                f.write("\n<<<<<<<<<<Bottom 5 training examples for test example %d:\n" % (i // 5 + 1))
            f.write("\n-----------------\n")
            f.write("Index: %d\n" % bottom5_indices[i // 5, i % 5])
            f.write("%s\n" % bottom5_tokenized[i])

if __name__ == "__main__":
    main()
