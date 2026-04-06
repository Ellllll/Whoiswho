# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
# coding=utf-8
from transformers.trainer import *
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import torch
from transformers import Trainer
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.utils import logging
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from sklearn.metrics import average_precision_score,roc_auc_score
import torch.nn.functional as F
import json
from tqdm import tqdm
logger = logging.get_logger(__name__)
from utils import LABEL_TOKEN

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class LoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model_to_save = unwrap_model(self.model)

        # 这个判断会导致模型保存全部参数，因此强制仅保存lora参数
        # if isinstance(model_to_save, PreTrainedModel):
        #     state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if v.requires_grad}
        #     # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
        #     model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save,safe_serialization=False)
        # else:
        #     logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        #     torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        state_dict = {}
        for k,v in model_to_save.named_parameters():
            for module_to_save in self.modules_to_save:
                if module_to_save in k:
                    state_dict[k] = v.to("cpu")

        # state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if (v.requires_grad or 'lora' in k)}
        # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
        model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save,safe_serialization=False)
        
        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))


class GLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        # if output_dir is None:
        #     output_dir = self.args.output_dir
        # self.model.save_pretrained(output_dir)
        # if self.tokenizer is not None:
        #     self.tokenizer.save_pretrained(output_dir)
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = unwrap_model(self.model)
        
        #for debug
        state_dict = {}
        for k,v in model_to_save.named_parameters():
            for module_to_save in self.modules_to_save:
                if module_to_save in k:
                    state_dict[k] = v.to("cpu")
        # state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if (v.requires_grad or 'lora' in k)}
        # Using torch.save instead of huggingface transformers or PEFT save_pretrained, so as to invoid other trainable parameter which is not defined by peft.modules_to_save be ingnored by PEFTmodel.save_pretrained method
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
        # model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save,safe_serialization=False)
        
        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))
        # state_dict = self.accelerator.get_state_dict(self.deepspeed)
        # if self.args.should_save:
        #     self._save(output_dir, state_dict=state_dict)
    def predict_without_train(        
        self,
        test_dataset=None,
        ground_truth = None,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        eval_result = None

        """
            borrow from trainer.train, to initialize model 
            ONLY FOR INFERENCE WITHOUT TRAINING
        """
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model
        
        """
            borrow from customized evaluate func
        """
        
        self._memory_tracker.start()
        args = self.args


        dataloader = self.get_eval_dataloader(test_dataset)

        # if self.is_deepspeed_enabled and self.deepspeed is None:
        #     _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            # model = (
            #     self.accelerator.prepare(model)
            #     if self.is_deepspeed_enabled
            #     else self.accelerator.prepare_model(model, evaluation_mode=True)
            # )
            model = self.accelerator.prepare_model(model, evaluation_mode=True)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        eval_result = []
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
            #     observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            # length.append(self.test_data(inputs)) # only for debug
            
            # continue
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None)
            logits = self.get_logits(logits,inputs)
            res = [{"author":inputs['author'],
                    "pub":inputs['pubs'][i],
                    "logit":logits[i].item()}
                    for i in range(len(inputs['pubs']))]
            raw = self.accelerator.gather_for_metrics(res)
            if self.accelerator.is_main_process:
                eval_result.extend(raw)
        if self.accelerator.is_main_process:
            overall_result = {}
            for i in eval_result:
                author = i['author']
                pub = i['pub']
                logit = i['logit']
                if author not in overall_result.keys():
                    overall_result[author] = {}
                overall_result[author][pub]= logit
            if self.args.predict_saved_path is not None:
                json.dump(overall_result, f)
            else:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir, exist_ok=True)
                output_dir = args.output_dir if output_dir is None else output_dir
                with open(os.path.join(output_dir,f"predict_res.json"), 'w') as f:
                    json.dump(overall_result, f)
            if ground_truth is not None:
                try:
                    AUCs,MAPs = cal_auc_map(overall_result,ground_truth)
                    print(f" AUC and MAP before training is {AUCs}")
                    with open(os.path.join(args.output_dir,f"result.txt"), 'a') as f:
                        f.write(f"step:0,epoch:0,AUC:{AUCs},MAP:{MAPs}\n") 
                except :
                    raise Exception("calculating AUC error:  unmatched predict dataset and ground truth dataset")     
            # with open(os.path.join(output_dir,f"predict_res.json"), 'w') as f:
            #     json.dump(overall_result, f)
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        

        if not hasattr(self, "eval_ground_truth"):
            with open(self.args.eval_ground_truth, "r", encoding="utf-8") as f:
                self.eval_ground_truth = json.load(f)
        self._memory_tracker.start()
        args = self.args
        dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        eval_result = []
        # length = []
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
            #     observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            # length.append(self.test_data(inputs)) # only for debug
            
            # continue
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys)
            logits = self.get_logits(logits,inputs)
            res = [{"author":inputs['author'],
                    "pub":inputs['pubs'][i],
                    "logit":logits[i].item()}
                    for i in range(len(inputs['pubs']))]
            raw = self.accelerator.gather_for_metrics(res)
            if self.accelerator.is_main_process:
                eval_result.extend(raw)
        if self.accelerator.is_main_process:
            overall_result = {}
            for i in eval_result:
                author = i['author']
                pub = i['pub']
                logit = i['logit']
                
                if author not in overall_result.keys():
                    overall_result[author] = {}
                overall_result[author][pub]= logit
                
            os.makedirs(os.path.join(args.output_dir,f"result"),exist_ok=True)
            with open(os.path.join(args.output_dir,f"result/step-{self.state.global_step}.json"), 'w') as f:
                json.dump(overall_result, f)   
             
            AUCs,MAPs = cal_auc_map(overall_result,self.eval_ground_truth)


            #update best metric
            if self.state.best_metric is None:
                self.state.best_metric = AUCs
                self.state.best_model_checkpoint = os.path.join(self.args.output_dir,f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            elif AUCs > self.state.best_metric:
                self.state.best_metric = AUCs
                self.state.best_model_checkpoint = os.path.join(self.args.output_dir,f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            
            output = {'AUC':AUCs,'MAP':MAPs,'step':self.state.global_step,'epoch':self.state.epoch}
            self.log(output)
            with open(os.path.join(args.output_dir,f"result.txt"), 'a') as f:
                f.write(f"step:{self.state.global_step},epoch:{self.state.epoch},AUC:{AUCs},MAP:{MAPs}\n")       
        

    def get_logits(self, logits, inputs): #only for IND task
        # # ==========================
        # # 1. 打印 logits 的完整形状（最重要）
        # # ==========================
        # print("\n===== 🔍 DEBUG logits 结构 =====")
        # print(f"logits 形状: {logits.shape}")
        # print(f"logits 设备: {logits.device}")
        # print(f"logits 数据类型: {logits.dtype}")
        
        # # ==========================
        # # 2. 打印局部/全部内容（防止输出太多刷屏，控制个数）
        # # ==========================
        # # 如果形状太大，只打印前几个数值
        # print("\n===== 🔍 logits 局部数值预览（前5个 token，前10个词） =====")
        # # 取 batch=0，seq_len=前5个位置，vocab=前10个词
        # if logits.numel() > 100: # 如果元素太多
        #     print(logits[0, :5, :10].cpu().detach().numpy())
        # else:
        #     print(logits.cpu().detach().numpy())
    
        # # ==========================
        # # 3. 精准打印：你要找的 <label_token> 位置附近的值
        # # ==========================
        # print("\n===== 🔍 重点：寻找 label_token 位置 =====")
        # label_id = self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN)
        # mask = (inputs['input_ids'] == label_id)[0].cpu().numpy() # 假设batch=1
        # positions = np.where(mask)[0]
        
        # print(f"词表中 <label_token> 的ID: {label_id}")
        # print(f"输入序列中 <label_token> 出现在位置: {positions}")
    
        # if len(positions) > 0:
        #     pos = positions[0] # 取第一个出现的位置
        #     print(f"\n📌 位置 {pos} 的真实 logits 值:")
        #     # 打印这个位置的前10个词的分数
        #     print(logits[0, pos, :10].cpu().detach().numpy()) 
        # else:
        #     print("❌ 没找到 <label_token>！")


        # # ========== 你要加的打印 ==========
        # print("\n===== yfx DEBUG get_logits =====")
        # print("logits 有没有 NaN:", torch.isnan(logits).any().item())
        # print("logits 有没有 Inf:", torch.isinf(logits).any().item())
        # print("labels_pos 要找的 LABEL_TOKEN:", LABEL_TOKEN)
        # print("input_ids 里有没有 LABEL_TOKEN:", (inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN)).any().item())
        # print("输入作者ID:", inputs.get('author', 'N/A')[:10])
        # # =================================

        labels_pos = torch.masked_select(torch.arange(inputs['input_ids'].shape[-1], device = self.model.device), inputs['input_ids'] == self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        labels_pos -= 1
        
        if "glm4" in self.tokenizer.name_or_path:# for glm4 tokenizer
            YES_TOKEN_IDS, NO_TOKEN_IDS = self.tokenizer.encode('Yes',add_special_tokens = False)[0],self.tokenizer.encode('No',add_special_tokens = False)[0]
        else:# for llama tokenizer
            YES_TOKEN_IDS, NO_TOKEN_IDS = self.tokenizer.convert_tokens_to_ids(['Yes','No'])
        
        yes_logit,no_logit= logits[:,labels_pos,YES_TOKEN_IDS],logits[:,labels_pos,NO_TOKEN_IDS]
        # # ========== 你要加的打印 ==========
        # print("\n===== yfx DEBUG =====")
        # print("yes_logit 形状:", yes_logit.shape)
        # print("yes_logit 有没有 NaN:", torch.isnan(yes_logit).any().item())
        # print("no_logit 有没有 NaN:", torch.isnan(no_logit).any().item())
        # # =================================l
        logit = F.softmax(torch.concat([yes_logit,no_logit],dim=0),dim=0)[0]
        return logit
    
        # labels_pos = torch.masked_select(torch.arange(inputs['llm_inputs']['input_ids'].shape[-1], device = self.model.device), inputs['llm_inputs']['input_ids'] == self.tokenizer.convert_tokens_to_ids(LABEL_TOKEN))
        # if not labels_pos.shape[-1] == inputs['labels']:
        #     breakpoint()
        # [n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)]
    
def cal_auc_map(pred, ground_truth):
    # logger.info(f"===== yfx cal_auc_map =====")
    data_dict = pred
    labels_dict = ground_truth

    # 新增：记录缺失样本的统计信息
    missing_authors = []       # 缺失的作者ID列表
    missing_papers = []        # 缺失的(作者ID, 论文ID)列表
    valid_sample_count = 0     # 有效样本数
    total_sample_count = 0     # 原始总样本数

    total_w = 0
    total_auc = 0
    total_ap = 0
    for aid in labels_dict:
        # 1. 跳过无预测结果的作者
        if aid not in data_dict:
            missing_authors.append(aid)
            continue

        cur_normal_data = labels_dict[aid]["normal_data"]
        cur_outliers = labels_dict[aid]["outliers"]

        # 统计该作者的原始总样本数
        total_sample_count += len(cur_normal_data) + len(cur_outliers)

        cur_labels = []
        cur_preds = []
        cur_w = len(cur_outliers)

        for item in cur_normal_data:
            # 2. 处理正样本，跳过无预测的论文
            if item in data_dict[aid]:
                cur_labels.append(1)
                cur_preds.append(data_dict[aid][item])
                # # ========== 你要加的打印 ==========
                # print(f"yfx 1 作者 {aid} 预测值: {data_dict[aid][item]}, 是否NaN: {np.isnan(pred_val)}")
                # # =================================
                # cur_preds.append(1)
                valid_sample_count += 1
            else:
                missing_papers.append((aid, item))

            # cur_labels.append(1)
            # cur_preds.append(data_dict[aid][item])
            # # cur_preds.append(1)

        for item in cur_outliers:
            # 3. 处理负样本，跳过无预测的论文
            if item in data_dict[aid]:
                cur_labels.append(0)
                cur_preds.append(data_dict[aid][item])
                # # ========== 你要加的打印 ==========
                # print(f"yfx 0 作者 {aid} 预测值: {data_dict[aid][item]}, 是否NaN: {np.isnan(data_dict[aid][item])}")
                # # =================================
                valid_sample_count += 1
            else:
                missing_papers.append((aid, item))
                
            # cur_labels.append(0)
            # cur_preds.append(data_dict[aid][item])
            # # cur_preds.append(0)

        # 4. 跳过无有效样本的作者
        if len(cur_preds) < 2 or len(set(cur_labels)) < 2:
            continue
        
        # 5. 计算指标
        try:
            cur_auc = roc_auc_score(cur_labels, cur_preds)
            cur_map = average_precision_score(cur_labels, cur_preds)
            total_ap += cur_w * cur_map
            total_auc += cur_w * cur_auc
            total_w += cur_w
        except Exception as e:
            logger.error(f"作者 {aid} 指标计算失败：{e}")
            continue

        # cur_auc = roc_auc_score(cur_labels, cur_preds)
        # cur_map = average_precision_score(cur_labels, cur_preds)
        # total_ap += cur_w * cur_map
        # total_auc += cur_w * cur_auc
        # total_w += cur_w

    # 打印缺失样本统计（关键：知道舍弃了多少）
    logger.info(f"===== yfx 样本统计 =====")
    logger.info(f"原始总样本数: {total_sample_count}")
    logger.info(f"有效样本数: {valid_sample_count}")
    logger.info(f"缺失作者数: {len(missing_authors)} (前5个: {missing_authors[:5]})")
    logger.info(f"缺失论文数: {len(missing_papers)} (前5个: {missing_papers[:5]})")
    logger.info(f"样本有效率: {valid_sample_count/total_sample_count*100:.2f}%" if total_sample_count>0 else "无样本")

    # 避免除以0
    if total_w == 0:
        logger.error("无有效样本参与计算，返回默认值")
        return 0.0, 0.0

    mAP = total_ap / total_w
    avg_auc = total_auc / total_w
    return avg_auc,mAP 
    
def compute_metrics(preds,inputs,ground_truth):
    res = {}
    for logits,author,pubs in zip(preds,inputs['author'],inputs['pubs']):
        if author not in res:
            res[author] = {}
        for i in range(len(pubs)):
            res[author][pubs[i]] = logits[i].item()
    AUC,MAP = cal_auc_map(res,ground_truth)
    return {"AUC":AUC,"MAP":MAP}

class INDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        # if output_dir is None:
        #     output_dir = self.args.output_dir
        # self.model.save_pretrained(output_dir)
        # if self.tokenizer is not None:
        #     self.tokenizer.save_pretrained(output_dir)
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = unwrap_model(self.model)
        
        #for debug
        state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if (v.requires_grad or 'lora' in k)}
        # Using torch.save instead of huggingface transformers or PEFT save_pretrained, so as to invoid other trainable parameter which is not defined by peft.modules_to_save be ingnored by PEFTmodel.save_pretrained method
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
        # model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save,safe_serialization=False)
        
        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))
        # state_dict = self.accelerator.get_state_dict(self.deepspeed)
        # if self.args.should_save:
        #     self._save(output_dir, state_dict=state_dict)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        all_inputs = EvalInputContainer(keys_for_eval=['author','pubs'])

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if is_torch_xla_available():
                xm.mark_step()
            # Update containers
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                all_losses.add(losses)
            if inputs is not None:
                inputs = self.gather_function([[inputs]])
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            
        # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
        if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
            all_losses.to_cpu_and_numpy()
            all_preds.to_cpu_and_numpy()
            all_labels.to_cpu_and_numpy()
            all_inputs.to_cpu_and_numpy()

            del losses, logits, labels, inputs
            torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and not self.args.batch_eval_metrics
        ):

            if not hasattr(self, "eval_ground_truth"):
                with open(self.args.eval_ground_truth, "r") as f:
                    eval_ground_truth = json.load(f)
            metrics = self.compute_metrics(preds=all_preds, inputs=all_inputs, ground_truth = eval_ground_truth)
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    

class EvalInputContainer:
    def __init__(self, keys_for_eval):
        self.keys_for_eval = keys_for_eval
        self.inputs = {}
        for key in self.keys_for_eval:
            self.inputs[key] = []

    def add(self, inputs) -> None:
        """Add tensors to the stored objects. If `do_nested_concat=True`, the tensors will be concatenated recursively."""
        for cur_input in inputs:
            if isinstance(cur_input,list) or isinstance(cur_input,tuple):
                cur_input = cur_input[0]
            
            for key in self.keys_for_eval:
                
                self.inputs[key].append(cur_input[key])
    def to_cpu_and_numpy(self) -> None:
        pass

    def get_arrays(self):
        return self.inputs
    
#TODO
class INDTrainingCallback(TrainerCallback):
    def __init__(self):
        pass
        
    def on_train_begin(self, args, state, control):
        # to eval before training, to ensure the model load parameters correctly
        pass
        
    def on_epoch_end(self, args, state, control):
        # to resample the balancing positive and negitive data in of dataset
        
        pass