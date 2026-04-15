#!/usr/bin/env python
# coding=utf-8

import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from arguments import DataTrainingArguments, GLMTrainingArguments, ModelArguments
from trainer import build_prediction_preview, cal_auc_map, log_prediction_preview


logger = logging.getLogger(__name__)

PROMPT_CONTEXT_TEMPLATE = (
    "Identify whether the candidate transaction matches the account's normal behavior pattern.\n"
    " Here is a collection of historical transactions of the account:\n"
    " ### {}"
)
PROMPT_QUERY_TEMPLATE = (
    "Determine whether the following statement is true and answer 'Yes' or 'No'. "
    '"{}" belongs to the account\'s normal behavior pattern.'
)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def format_pub_text(pub: Dict, feature: str) -> str:
    title = (pub.get("title") or "").strip()
    if feature == "title":
        return title

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")

    venue = (pub.get("venue") or "").strip()
    if venue:
        parts.append(f"Venue: {venue}")

    authors = pub.get("authors") or []
    if authors:
        author_text = "; ".join(
            f"{(item.get('name') or '').strip()} @ {(item.get('org') or '').strip()}".strip()
            for item in authors[:10]
            if (item.get("name") or "").strip() or (item.get("org") or "").strip()
        )
        if author_text:
            parts.append(f"Authors: {author_text}")

    abstract = (pub.get("abstract") or "").strip()
    if abstract:
        parts.append(f"Abstract: {abstract}")

    keywords = pub.get("keywords") or []
    if keywords:
        keyword_text = ", ".join(str(keyword).strip() for keyword in keywords[:20] if str(keyword).strip())
        if keyword_text:
            parts.append(f"Keywords: {keyword_text}")

    return " ; ".join(parts) if parts else title


def build_prompt_pair(
    profile_pub_ids: List[str],
    candidate_pub_id: str,
    pub_data: Dict[str, Dict],
    feature: str,
) -> Dict[str, str]:
    history_texts = [format_pub_text(pub_data[pub_id], feature) for pub_id in profile_pub_ids if pub_id in pub_data]
    candidate_text = format_pub_text(pub_data[candidate_pub_id], feature)
    context = PROMPT_CONTEXT_TEMPLATE.format(" # ".join(history_texts))
    query = PROMPT_QUERY_TEMPLATE.format(candidate_text)
    return {"context": context, "query": query}


@dataclass
class ExampleRecord:
    author: str
    pub: str
    label: Optional[int]


class RobertaPromptDataset(Dataset):
    def __init__(
        self,
        author_data: Dict[str, Dict],
        pub_data: Dict[str, Dict],
        ground_truth: Optional[Dict[str, Dict]],
        tokenizer,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        mode: str,
    ):
        self.author_data = author_data
        self.pub_data = pub_data
        self.ground_truth = ground_truth or {}
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.mode = mode
        self.feature = model_args.text_feature if model_args.text_feature != "title" else model_args.feature
        self.examples = self._build_examples()

    def _lookup_eval_label(self, author_id: str, pub_id: str) -> Optional[int]:
        author_labels = self.ground_truth.get(author_id)
        if not author_labels:
            return None
        if pub_id in set(author_labels.get("normal_data", [])):
            return 1
        if pub_id in set(author_labels.get("outliers", [])):
            return 0
        return None

    def _build_examples(self) -> List[ExampleRecord]:
        examples: List[ExampleRecord] = []
        normal_to_fraud_ratio = 4
        for author_id, author_info in self.author_data.items():
            if self.mode == "train":
                normal_data = list(author_info.get("normal_data", []))
                outliers = list(author_info.get("outliers", []))
                if self.data_args.sample and outliers:
                    if len(normal_data) >= len(outliers) * normal_to_fraud_ratio:
                        sampled_normal = random.sample(normal_data, len(outliers) * normal_to_fraud_ratio)
                    else:
                        sampled_normal = normal_data
                    sampled_outliers = outliers
                    selected = [(pub_id, 1) for pub_id in sampled_normal] + [(pub_id, 0) for pub_id in sampled_outliers]
                else:
                    selected = [(pub_id, 1) for pub_id in normal_data] + [(pub_id, 0) for pub_id in outliers]
            else:
                papers = author_info.get("papers")
                if papers is None:
                    papers = list(author_info.get("normal_data", [])) + list(author_info.get("outliers", []))
                selected = [(pub_id, self._lookup_eval_label(author_id, pub_id)) for pub_id in papers]

            random.shuffle(selected)
            for pub_id, label in selected:
                if self.mode != "train" and self.ground_truth and label is None:
                    continue
                if pub_id in self.pub_data:
                    examples.append(ExampleRecord(author=author_id, pub=pub_id, label=label))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _get_profile_pub_ids(self, author_id: str, candidate_pub_id: str) -> List[str]:
        author_info = self.author_data[author_id]
        if self.mode == "train":
            profile_pub_ids = list(author_info.get("normal_data", [])) + list(author_info.get("outliers", []))
            random.shuffle(profile_pub_ids)
        else:
            papers = author_info.get("papers")
            if papers is None:
                papers = list(author_info.get("normal_data", [])) + list(author_info.get("outliers", []))
            profile_pub_ids = list(papers)
            if self.data_args.shuffle_profile:
                random.shuffle(profile_pub_ids)

        if len(profile_pub_ids) > 1:
            profile_pub_ids = [pub_id for pub_id in profile_pub_ids if pub_id != candidate_pub_id]

        truncation_limit = self.data_args.profile_truncation or len(profile_pub_ids)
        return profile_pub_ids[:truncation_limit]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        profile_pub_ids = self._get_profile_pub_ids(example.author, example.pub)
        prompt = build_prompt_pair(profile_pub_ids, example.pub, self.pub_data, self.feature)
        encoded = self.tokenizer(
            prompt["context"],
            prompt["query"],
            max_length=self.data_args.max_source_length,
            truncation="only_first",
            padding=False,
        )

        if example.label is not None:
            encoded["labels"] = example.label
        return encoded


def save_prediction_outputs(
    output_dir: str,
    dataset: RobertaPromptDataset,
    predictions,
    ground_truth: Optional[Dict],
) -> Dict[str, float]:
    logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
    overall_result = build_overall_result(dataset, logits)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predict_res.json"), "w", encoding="utf-8") as file_obj:
        json.dump(overall_result, file_obj)

    preview = build_prediction_preview(overall_result, ground_truth)
    with open(os.path.join(output_dir, "predict_preview.json"), "w", encoding="utf-8") as file_obj:
        json.dump(preview, file_obj, ensure_ascii=False, indent=2)
    log_prediction_preview(preview, "roberta baseline")

    metrics: Dict[str, float] = {}
    if ground_truth is not None:
        auc, mean_ap = cal_auc_map(overall_result, ground_truth)
        metrics = {"AUC": auc, "MAP": mean_ap}
        with open(os.path.join(output_dir, "result.txt"), "a", encoding="utf-8") as file_obj:
            file_obj.write(f"AUC:{auc},MAP:{mean_ap}\n")
        logger.info("RoBERTa baseline metrics: AUC=%s MAP=%s", auc, mean_ap)
    return metrics


def build_overall_result(dataset: RobertaPromptDataset, logits) -> Dict[str, Dict[str, float]]:
    scores = torch.softmax(torch.as_tensor(logits), dim=-1)[:, 1].tolist()
    overall_result: Dict[str, Dict[str, float]] = {}
    for example, score in zip(dataset.examples, scores):
        overall_result.setdefault(example.author, {})[example.pub] = float(score)
    return overall_result


def build_compute_metrics(dataset: Optional[RobertaPromptDataset], ground_truth: Optional[Dict]):
    if dataset is None or ground_truth is None:
        return None

    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
        overall_result = build_overall_result(dataset, logits)
        auc, mean_ap = cal_auc_map(overall_result, ground_truth)
        return {"AUC": auc, "MAP": mean_ap}

    return compute_metrics


def main():
    cfgs = [item for item in sys.argv if item.endswith(".json")]
    if not cfgs:
        raise ValueError("A json config path is required.")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GLMTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(cfgs[0]))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    model_path = model_args.ptm_model_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=model_args.use_fast_tokenizer, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        trust_remote_code=True,
    )

    author_data = load_json(data_args.author_data)
    pub_data = load_json(data_args.pub_data)
    eval_source = data_args.test_data or data_args.eval_data
    eval_data = load_json(eval_source) if eval_source else None
    ground_truth = load_json(training_args.eval_ground_truth) if training_args.eval_ground_truth and os.path.exists(training_args.eval_ground_truth) else None

    train_dataset = None
    if training_args.do_train:
        train_dataset = RobertaPromptDataset(author_data, pub_data, None, tokenizer, model_args, data_args, mode="train")
        logger.info("Loaded %s training examples", len(train_dataset))

    eval_dataset = None
    if eval_data is not None:
        eval_dataset = RobertaPromptDataset(eval_data, pub_data, ground_truth, tokenizer, model_args, data_args, mode="eval")
        logger.info("Loaded %s evaluation examples", len(eval_dataset))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(eval_dataset, ground_truth),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(training_args.output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if eval_dataset is not None and (training_args.do_eval or not training_args.do_train):
        predictions = trainer.predict(eval_dataset, metric_key_prefix="predict")
        if trainer.is_world_process_zero():
            metrics = save_prediction_outputs(training_args.output_dir, eval_dataset, predictions, ground_truth)
            if metrics:
                trainer.log({f"predict_{key}": value for key, value in metrics.items()})


if __name__ == "__main__":
    main()