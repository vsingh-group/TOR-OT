import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,6"

import json
import time
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from scipy import sparse
import numpy as np
import logging

from TF4OT import utils

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Please install transformers and accelerate")
    raise

logger = logging.getLogger(__name__)

OT_INPUT_TOTAL         = "input_total"
OT_INPUT_COST_ONEHOT   = "input_cost_type"
OT_TARGET_PLAN_MATRIX_SPARSE  = "target_otplan_solve_sparse"

def format_example(source, target, cost_metric, plan_sparse=None):
    src_str = "[" + ", ".join([f"{x:.4f}" for x in source]) + "]"
    tgt_str = "[" + ", ".join([f"{x:.4f}" for x in target]) + "]"
    
    s = f"Source: {src_str}\n"
    s += f"Target: {tgt_str}\n"
    s += f"Cost Metric: {cost_metric}\n"
    if plan_sparse is not None:
        nonzeros = []
        for r, c, v in zip(plan_sparse.row, plan_sparse.col, plan_sparse.data):
            if v > 1e-6:
                nonzeros.append([int(r), int(c), float(f"{v:.4f}")])
        s += f"Plan: {json.dumps(nonzeros)}\n"
    else:
        s += "Plan:"
    return s

def parse_qwen_plan(text, n_bins):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[-2].split("```")[0]
    else:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end+1]
            
    text = text.strip()
    try:
        data = json.loads(text)
        rows, cols, vals = [], [], []
        for item in data:
            if len(item) >= 3:
                rows.append(int(item[0]))
                cols.append(int(item[1]))
                vals.append(float(item[2]))
        
        pred_sparse = sparse.coo_matrix((vals, (rows, cols)), shape=(n_bins, n_bins))
        return pred_sparse
    except Exception as e:
        logger.error(f"Error parsing Qwen output:\n{text}\nException: {e}")
        return sparse.coo_matrix((n_bins, n_bins))

def inference_routine(cfg):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        force=True)

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    logger.info("Loading dataset: " + cfg.data.hf_dataset)
    dataset = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.hf_cache_dir)
    test_data = dataset["test"]
    
    cost_type = cfg.data.dataset.input_cost_type
    n_bins = cfg.data.dataset.support_size
    
    num_icl = 5
    num_test = 20 # small experiment limits API/compute cost
    total_needed = num_icl + num_test
    
    testset_size = total_needed * cfg.data.dataset.support_size
    test_data_adjusted = test_data.select(range(min(testset_size, len(test_data))))
        
    batch_size_adjusted = cfg.data.dataset.support_size
    test_loader = DataLoader(test_data_adjusted,
                            batch_size=batch_size_adjusted,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=utils.collate_from_string)
                            
    logger.info(f"Loaded {len(test_loader)} examples for ICL and evaluation.")
    
    device = torch.device("cpu")
    current_metric_name = cfg.data.dataset.input_cost_metric
    x_norm = torch.linspace(0, 1, n_bins, device=device).view(-1, 1)
    if current_metric_name == 'sqeuclidean':
        C = torch.cdist(x_norm, x_norm, p=2)**2
    C = C.numpy()

    icl_examples = []
    test_examples = []
    
    for k, batch in enumerate(test_loader):
        input_total_vec = batch[OT_INPUT_TOTAL][0]
        source = input_total_vec[:n_bins].numpy()
        target = input_total_vec[n_bins:2*n_bins].numpy()
        
        ot_target_sparse_rows = batch[OT_TARGET_PLAN_MATRIX_SPARSE]
        gt_rows, gt_cols, gt_data = [], [], []
        for i, row_data in enumerate(ot_target_sparse_rows):
            if isinstance(row_data, torch.Tensor): row_data = row_data.tolist()
            for idx in range(0, len(row_data), 2):
                 gt_rows.append(i)
                 gt_cols.append(int(row_data[idx]))
                 gt_data.append(float(row_data[idx+1]))
        ot_target_sparse = sparse.coo_matrix((gt_data, (gt_rows, gt_cols)), shape=(n_bins, n_bins))
        
        ex = {
            'source': source,
            'target': target,
            'metric': current_metric_name,
            'sparse': ot_target_sparse,
            'dense': ot_target_sparse.toarray()
        }
        
        if k < num_icl:
            icl_examples.append(ex)
        else:
            test_examples.append(ex)
            if len(test_examples) >= num_test:
                break
                
    system_prompt = "You are an expert at computing 1D discrete optimal transport plans.\nGiven a source distribution (sum=1) and a target distribution (sum=1) over N bins, and a cost metric, output the corresponding optimal transport plan matrix.\nThe optimal transport plan must satisfy the marginal constraints (rows sum to source, columns sum to target) and minimize the total transport cost.\nOutput ONLY a JSON-formatted list of lists `[[row_idx, col_idx, value], ...]`, representing the NON-ZERO entries of the transport plan.\nDo not output any markdown formatting, explanations, or extra text. Just the JSON array."

    messages_template = [{"role": "system", "content": system_prompt}]
    for ex in icl_examples:
        messages_template.append({"role": "user", "content": format_example(ex['source'], ex['target'], ex['metric'])})
        ans = json.dumps([[int(r), int(c), float(f"{v:.4f}")] for r,c,v in zip(ex['sparse'].row, ex['sparse'].col, ex['sparse'].data) if v > 1e-6])
        messages_template.append({"role": "assistant", "content": ans})
        
    all_mae, all_mape, all_hellinger, all_tv, all_otcost_diff, all_row_viol, all_col_viol, all_max_viol = [], [], [], [], [], [], [], []
    
    for i, ex in enumerate(test_examples):
        prompt_text = format_example(ex['source'], ex['target'], ex['metric'])
        messages = messages_template + [{"role": "user", "content": prompt_text}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        logger.info(f"Querying Qwen for eval example {i+1}/{len(test_examples)}...")
        
        try:
            start_time = time.time()
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            ans = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"  Inference time: {time.time()-start_time:.2f}s")
            
            ot_pred_sparse = parse_qwen_plan(ans, n_bins)
            ot_pred_dense = ot_pred_sparse.toarray()
            
            mae_sparse  = utils.sparse_mean_abs_error(ex['sparse'], ot_pred_sparse)
            mape_sparse = utils.sparse_weighted_mean_abs_percentage_error(ex['sparse'], ot_pred_sparse)
            hellinger_sparse = utils.sparse_hellinger_distance(ex['sparse'], ot_pred_sparse)
            tv_sparse = utils.sparse_total_variation(ex['sparse'], ot_pred_sparse)
            
            otcost_gt   = (ex['dense'] * C).sum()
            otcost_pred = (ot_pred_dense * C).sum()
            otcost_diff = abs(otcost_pred - otcost_gt)
            
            metrics_viol = utils.compute_constraint_violations(ot_pred_sparse, ex['source'], ex['target'])
            row_violation = metrics_viol['row_violation']
            col_violation = metrics_viol['col_violation']
            max_violation = metrics_viol['max_violation']
            
            all_mae.append(mae_sparse)
            all_mape.append(mape_sparse)
            all_hellinger.append(hellinger_sparse)
            all_tv.append(tv_sparse)
            all_otcost_diff.append(otcost_diff)
            all_row_viol.append(row_violation)
            all_col_viol.append(col_violation)
            all_max_viol.append(max_violation)
            
            logger.info(f"  MAE: {mae_sparse:.4e} | MAPE: {mape_sparse:.4e} | MaxViol: {max_violation:.4e}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate: {e}")
            continue
            
    logger.info("="*80)
    logger.info(f"Summary for {len(all_mae)} examples using Qwen 7B")
    logger.info("="*80)
    logger.info(f"MAE (sparse): Mean={np.mean(all_mae):.6e}, Std={np.std(all_mae):.6e}")
    logger.info(f"MAPE (sparse): Mean={np.mean(all_mape):.6e}, Std={np.std(all_mape):.6e}")
    logger.info(f"Hellinger: Mean={np.mean(all_hellinger):.6e}, Std={np.std(all_hellinger):.6e}")
    logger.info(f"TV: Mean={np.mean(all_tv):.6e}, Std={np.std(all_tv):.6e}")
    logger.info(f"OT Cost Diff: Mean={np.mean(all_otcost_diff):.6e}, Std={np.std(all_otcost_diff):.6e}")
    logger.info(f"Row Violation: Mean={np.mean(all_row_viol):.6e}")
    logger.info(f"Col Violation: Mean={np.mean(all_col_viol):.6e}")
    logger.info(f"Max Violation: Mean={np.mean(all_max_viol):.6e}")
    
@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    inference_routine(cfg)

if __name__ == "__main__":
    main()
