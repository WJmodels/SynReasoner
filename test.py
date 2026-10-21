"""SynReasoner inference, chemical scoring, and self-consistency voting.

All inference and evaluation implementations are contained in this file.
Run ``python test.py --help`` for inference options, or use the inspect,
evaluate, and vote subcommands. Training is implemented in train.py.
"""
import argparse
import concurrent.futures
import glob
import importlib
import importlib.util
import inspect
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from functools import partial
from itertools import combinations, permutations
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import rdkit
import torch
import torch.multiprocessing as mp
from datasets import Dataset, concatenate_datasets, load_dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# METRICS

def get_subdirectories(path):
    """
    获取指定路径下的一级子文件夹路径列表

    参数:
    path (str): 需要获取子文件夹的路径

    返回:
    list: 包含一级子文件夹路径的列表

    使用方法:
    subdirs = get_subdirectories('your_path_here')
    print(subdirs)
    """
    if not os.path.exists(path):
        raise ValueError(f"路径 '{path}' 不存在")
    subdirectories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return subdirectories

def get_files_with_suffix(folder_path, suffix, traverse_sub_folder=False):
    """
    获取指定文件夹中指定后缀的所有文件路径
    
    参数：
        folder_path (str): 文件夹路径
        suffix (str): 文件的后缀，如 '.jpg' 或 '.png'
        traverse_sub_folder (bool): 是否遍历子文件夹，默认为 False
    
    返回：
        list: 包含所有符合条件的文件路径的列表
    """
    list_target_file_path = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                list_target_file_path.append(os.path.join(root, file))
        if not traverse_sub_folder:
            break
    return list_target_file_path

def extract_dataset_name_from_txt_file_name_with_suffix(txt_file_name_with_suffix):
    """
    从文件名中提取数据集名称。
    兼容两种模式：
    1. 带step前缀: model_step_datasetname.txt -> datasetname
    2. 纯文件名: datasetname.txt -> datasetname
    """
    match = re.search('step_([^.]+)\\.txt', txt_file_name_with_suffix)
    if match:
        return match.group(1)
    if txt_file_name_with_suffix.endswith('.txt'):
        if txt_file_name_with_suffix.endswith('.txt.txt'):
            return txt_file_name_with_suffix[:-8]
        return txt_file_name_with_suffix[:-4]
    return None

def get_key_index_by_value(dict, target_value):
    values = list(dict.values())
    try:
        index = values.index(target_value)
        return index
    except ValueError:
        return None

def import_functions(list_tuple_py_file_abs_path_function_name):
    """
    动态导入指定列表中的每个 Python 文件的指定函数，
    同时将其添加到调用模块和__main__模块的命名空间。
    
    参数:
    list_tuple_py_file_abs_path_function_name: 一个列表，列表中的每个元素是一个元组，元组包含：
        - py_file_abs_path: Python 文件的绝对路径
        - function_name: 要导入的函数名称
    """
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    caller_module_name = caller_globals['__name__']
    if caller_module_name == '__main__':
        import __main__
        caller_module = __main__
    else:
        caller_module = sys.modules[caller_module_name]
    import __main__
    main_globals = __main__.__dict__
    imported_functions = {}
    for py_file_abs_path, function_name in list_tuple_py_file_abs_path_function_name:
        try:
            module_name = f'dynamic_module_{function_name}_{hash(py_file_abs_path)}'
            spec = importlib.util.spec_from_file_location(module_name, py_file_abs_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                setattr(caller_module, function_name, func)
                caller_globals[function_name] = func
                if caller_module_name != '__main__':
                    main_globals[function_name] = func
                imported_functions[function_name] = func
                print(f'成功导入 {function_name} 函数，来自文件：{py_file_abs_path}')
            else:
                print(f'警告：{py_file_abs_path} 中未找到函数 {function_name}')
        except Exception as e:
            print(f'错误：无法加载 {function_name} 函数，来自文件：{py_file_abs_path}，错误信息：{e}')
    return imported_functions

def replace_keys_with_values(one_result_folder_dict_stat, dict_key_test_arrow_folder_path_value_dataset_name):
    updated_result = {}
    for key, value in one_result_folder_dict_stat.items():
        if key in dict_key_test_arrow_folder_path_value_dataset_name:
            new_key = dict_key_test_arrow_folder_path_value_dataset_name[key]
            updated_result[new_key] = value
        else:
            updated_result[key] = value
    return updated_result

def save_evaluation_results_as_json_and_csv(dict_results, save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_filename_prefix='evaluation_result', decimal_places=4):
    """
    将评估结果保存为JSON和CSV文件。
    
    更新内容:
    1. 在JSON中注入 "_task_type_debug" 字段。
    2. 根据任务类型动态生成 CSV。
    3. [新增] 强制指定 NMR Forward 指标在 CSV 中的显示顺序 (MAE优先)。
    4. [新增] 修复 Pandas Pivot 导致的字母自动排序问题。
    """
    import json
    import csv
    import pandas as pd
    import os
    all_dataset_names = list(dict_key_test_arrow_folder_path_value_dataset_name.values())
    try:
        sorted_train_steps = sorted([int(step) for step in dict_results.keys()])
    except:
        sorted_train_steps = sorted(dict_results.keys())
    if not sorted_train_steps:
        return []
    datasets_general = []
    datasets_nmr_forward = []
    for step in sorted_train_steps:
        step_str = str(step)
        step_data = dict_results.get(step_str, {})
        for ds_name in all_dataset_names:
            if ds_name not in step_data:
                continue
            metrics_dict = step_data[ds_name]
            if 'average_mae_zero_padding_1H_nmr_shift' in metrics_dict:
                task_type = 'nmr_forward'
                if ds_name not in datasets_nmr_forward and ds_name not in datasets_general:
                    datasets_nmr_forward.append(ds_name)
            else:
                task_type = 'general'
                if ds_name not in datasets_general and ds_name not in datasets_nmr_forward:
                    datasets_general.append(ds_name)
            metrics_dict['_task_type_debug'] = task_type
    print(f'检测到任务分类 (基于指标内容):')
    print(f'  - General Task Datasets ({len(datasets_general)}): {datasets_general}')
    print(f'  - NMR Forward Datasets  ({len(datasets_nmr_forward)}): {datasets_nmr_forward}')
    json_file_path = os.path.join(save_folder_path, f'{save_result_filename_prefix}.json')
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dict_results, json_file, indent=4)
    print(f'已保存完整 JSON: {json_file_path}')
    generated_files = [json_file_path]
    metrics_general = ['accuracy', 'prediction_validity_rate', 'average_tanimoto', 'valid_smiles', 'accuracy_040', 'accuracy_060', 'accuracy_080', 'accuracy_090']
    metrics_nmr_forward = ['average_mae_zero_padding_13C_nmr_shift', 'average_mae_zero_padding_1H_nmr_shift', 'average_deta_13C_nmr_shift_count', 'average_deta_1H_nmr_shift_count', 'average_rmsd_zero_padding_13C_nmr_shift', 'average_rmsd_zero_padding_1H_nmr_shift', 'valid_samples_for_error_metrics', 'average_cot_13c_accuracy', 'average_cot_1h_accuracy', 'average_cot_brics_accuracy', 'average_cot_13c_tanimoto', 'average_cot_1h_tanimoto', 'average_cot_brics_tanimoto', 'total_samples', 'valid_samples_for_deta_13c_count', 'valid_samples_for_deta_1h_count']

    def _write_csv(filename_suffix, target_datasets, target_metrics):
        if not target_datasets:
            return None
        csv_data = []
        for train_step in sorted_train_steps:
            train_step_str = str(train_step)
            step_data = dict_results.get(train_step_str, {})
            for metric in target_metrics:
                row = [train_step_str, metric]
                for ds_name in target_datasets:
                    val = step_data.get(ds_name, {}).get(metric, '')
                    if isinstance(val, (int, float)):
                        formatted_val = f'{val:.{decimal_places}f}'
                    else:
                        formatted_val = str(val)
                    row.append(formatted_val)
                csv_data.append(row)
        full_path = os.path.join(save_folder_path, f'{save_result_filename_prefix}{filename_suffix}.csv')
        header = ['train_step', 'metric'] + target_datasets
        try:
            with open(full_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(csv_data)
            print(f'已保存 CSV: {full_path}')
            return full_path
        except Exception as e:
            print(f'保存 CSV 失败: {e}')
            return None
    path_gen = _write_csv('_general', datasets_general, metrics_general)
    if path_gen:
        generated_files.append(path_gen)
    path_nmr = _write_csv('_nmr_forward', datasets_nmr_forward, metrics_nmr_forward)
    if path_nmr:
        generated_files.append(path_nmr)
    if datasets_general:
        acc_path = os.path.join(save_folder_path, f'{save_result_filename_prefix}_general_accuracy.csv')
        acc_rows = []
        for step in sorted_train_steps:
            row = [str(step)]
            for ds in datasets_general:
                val = dict_results.get(str(step), {}).get(ds, {}).get('accuracy', '')
                row.append(f'{val:.{decimal_places}f}' if isinstance(val, (int, float)) else str(val))
            acc_rows.append(row)
        with open(acc_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['train_step'] + datasets_general)
            csv.writer(f).writerows(acc_rows)
        generated_files.append(acc_path)
    if datasets_nmr_forward:
        mae_path = os.path.join(save_folder_path, f'{save_result_filename_prefix}_nmr_forward_mae_1h.csv')
        mae_rows = []
        for step in sorted_train_steps:
            row = [str(step)]
            for ds in datasets_nmr_forward:
                val = dict_results.get(str(step), {}).get(ds, {}).get('average_mae_zero_padding_1H_nmr_shift', '')
                row.append(f'{val:.{decimal_places}f}' if isinstance(val, (int, float)) else str(val))
            mae_rows.append(row)
        with open(mae_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['train_step'] + datasets_nmr_forward)
            csv.writer(f).writerows(mae_rows)
        generated_files.append(mae_path)
    try:
        if path_gen:
            df = pd.read_csv(path_gen)
            for c in df.columns[2:]:
                df[c] = pd.to_numeric(df[c], errors='ignore')
            valid_metrics = [m for m in metrics_general if m in df['metric'].unique()]
            df['metric'] = pd.Categorical(df['metric'], categories=valid_metrics, ordered=True)
            fmt_path = path_gen.replace('.csv', '_formatted.csv')
            df.pivot(index='train_step', columns='metric', values=datasets_general).to_csv(fmt_path)
            generated_files.append(fmt_path)
        if path_nmr:
            df = pd.read_csv(path_nmr)
            for c in df.columns[2:]:
                df[c] = pd.to_numeric(df[c], errors='ignore')
            valid_metrics = [m for m in metrics_nmr_forward if m in df['metric'].unique()]
            df['metric'] = pd.Categorical(df['metric'], categories=valid_metrics, ordered=True)
            fmt_path = path_nmr.replace('.csv', '_formatted.csv')
            df.pivot(index='train_step', columns='metric', values=datasets_nmr_forward).to_csv(fmt_path)
            generated_files.append(fmt_path)
    except Exception as e:
        print(f'生成 formatted csv 时出错: {e}')
        import traceback
        traceback.print_exc()
    return generated_files

def extract_formula_from_generate_result(str_generate_result_without_prompt):
    """
    从生成结果中提取分子式。
    
    参数:
        str_generate_result_without_prompt (str): 去除prompt后的生成结果字符串
        
    返回值:
        str: 提取的分子式字符串，如果未找到则返回None
    """
    import re
    pattern = 'Formula:([^;]+);'
    match = re.search(pattern, str_generate_result_without_prompt)
    if match:
        formula = match.group(1).strip()
        return formula
    else:
        return None

def normalize_molecular_formula(formula):
    """
    标准化分子式格式，处理不同顺序写法的分子式。
    将分子式转换为标准格式：C优先，然后H，再按字母顺序排列其他元素。
    
    参数:
        formula (str): 原始分子式字符串
        
    返回值:
        str: 标准化后的分子式字符串
    """
    import re
    from collections import defaultdict
    if not formula:
        return ''
    element_count = defaultdict(int)
    pattern = '([A-Z][a-z]?)(\\d*)'
    matches = re.findall(pattern, formula)
    for element, count_str in matches:
        count = int(count_str) if count_str else 1
        element_count[element] += count
    result_parts = []
    if 'C' in element_count:
        count = element_count['C']
        result_parts.append(f"C{(count if count > 1 else '')}")
        del element_count['C']
    if 'H' in element_count:
        count = element_count['H']
        result_parts.append(f"H{(count if count > 1 else '')}")
        del element_count['H']
    for element in sorted(element_count.keys()):
        count = element_count[element]
        result_parts.append(f"{element}{(count if count > 1 else '')}")
    return ''.join(result_parts)

def is_valid_smiles_with_formula_check(smiles, target_formula):
    """
    检查SMILES是否合格并且分子式匹配。
    
    参数:
        smiles (str): 待检查的SMILES字符串
        target_formula (str): 目标分子式
        
    返回值:
        tuple: (bool, str) - (是否合格, 标准SMILES)，如果不合格则标准SMILES为None
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    if not smiles or smiles == 'F' or len(smiles) > 5000:
        return (False, None)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (False, None)
        mol_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        normalized_mol_formula = normalize_molecular_formula(mol_formula)
        normalized_target_formula = normalize_molecular_formula(target_formula)
        if normalized_mol_formula != normalized_target_formula:
            return (False, None)
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return (True, canonical_smiles)
    except Exception as e:
        return (False, None)

def extract_str_generate_result_without_prompt(str_generate_result):
    return str_generate_result.split('###Response:')[-1]

def remove_disordered_tags(s, tuple_first_second):
    """
    移除字符串中不符合顺序的标签（如果 second 在 first 之前，则删除该 second）。
    
    参数:
        s (str): 输入字符串
        tuple_first_second (tuple): 包含两个标签的元组 (first, second)
        
    返回:
        str: 处理后的字符串
    """
    first, second = tuple_first_second
    first_positions = [match.start() for match in re.finditer(re.escape(first), s)]
    second_positions = [match.start() for match in re.finditer(re.escape(second), s)]
    filtered_s = s
    for pos in second_positions:
        if not any((f_pos < pos for f_pos in first_positions)):
            filtered_s = filtered_s[:pos] + filtered_s[pos + len(second):]
    return filtered_s

def extract_str_answer(str_generate_result_without_prompt):
    str_generate_result_without_prompt = remove_disordered_tags(str_generate_result_without_prompt, ('<answer>', '</answer>'))
    bool_answer_start = '<answer>' in str_generate_result_without_prompt
    bool_answer_end = '</answer>' in str_generate_result_without_prompt
    bool_specific_str = '])' in str_generate_result_without_prompt
    bool_nat_question_str = 'infer the molecular SMILES:' in str_generate_result_without_prompt
    if bool_answer_start and bool_answer_end:
        str_answer = str_generate_result_without_prompt.split('</answer>')[-2].split('<answer>')[-1]
    elif bool_answer_start and (not bool_answer_end):
        str_answer = str_generate_result_without_prompt.split('<answer>')[1]
    elif not bool_answer_start and bool_answer_end:
        str_answer = 'F'
    elif bool_nat_question_str:
        str_answer = str_generate_result_without_prompt.split('infer the molecular SMILES:')[-1].split('"')[0]
    elif bool_specific_str:
        str_answer = str_generate_result_without_prompt.split('])')[-1].split('"')[0]
    else:
        str_answer = 'F'
    return str_answer

def extract_str_cot(str_generate_result_without_prompt):
    str_generate_result_without_prompt = remove_disordered_tags(str_generate_result_without_prompt, ('<think>', '</think>'))
    bool_think_start = '<think>' in str_generate_result_without_prompt
    bool_think_end = '</think>' in str_generate_result_without_prompt
    if bool_think_start and bool_think_end:
        str_cot = str_generate_result_without_prompt.split('</think>')[-2].split('<think>')[-1]
    elif bool_think_start and (not bool_think_end):
        str_cot = str_generate_result_without_prompt.split('<think>')[1]
    elif not bool_think_start and bool_think_end:
        str_cot = str_generate_result_without_prompt.split('</think>')[0]
    else:
        str_cot = str_generate_result_without_prompt
    if '<answer>' in str_cot:
        str_cot = str_cot.split('<answer>')[0]
    if '</answer>' in str_cot:
        str_cot = str_cot.split('</answer>')[0]
    return str_cot

def get_df_sample_info_from_arrow_file(arrow_file_path):
    dataset = Dataset.from_file(arrow_file_path)
    df = dataset.to_pandas()
    return df

def load_question_cot_answer_list_from_arrow_folder(arrow_folder_path):
    """
    从arrow文件夹中加载所有arrow文件的问题、CoT和答案列表。
    
    参数:
        arrow_folder_path (str): 包含arrow文件的文件夹路径
        
    返回值:
        tuple: (list_question, list_cot, list_answer) - 合并所有arrow文件的数据列表
    """
    list_arrow_file_path = get_files_with_suffix(arrow_folder_path, '.arrow', traverse_sub_folder=False)
    if not list_arrow_file_path:
        raise ValueError(f"在文件夹 '{arrow_folder_path}' 中未找到任何.arrow文件")
    list_arrow_file_path.sort()
    combined_questions = []
    combined_cots = []
    combined_answers = []
    for arrow_file_path in list_arrow_file_path:
        print(f'正在加载arrow文件: {arrow_file_path}')
        df = get_df_sample_info_from_arrow_file(arrow_file_path)
        combined_questions.extend(df['question'].tolist())
        combined_cots.extend(df['cot'].tolist())
        combined_answers.extend(df['answer'].tolist())
    print(f'已成功合并 {len(list_arrow_file_path)} 个arrow文件，总样本数: {len(combined_questions)}')
    return (combined_questions, combined_cots, combined_answers)

def load_question_cot_answer_list_from_arrow_file(arrow_file_path):
    df = get_df_sample_info_from_arrow_file(arrow_file_path)
    list_question = df['question'].tolist()
    list_cot = df['cot'].tolist()
    list_answer = df['answer'].tolist()
    return (list_question, list_cot, list_answer)

def read_model_predictions(model_predict_result_txt_file_path):
    """
    生成器函数：逐行读取模型预测结果文件，并返回每行的内容（字符串形式）。

    参数:
        model_predict_result_txt_file_path (str): 模型预测结果的 .txt 文件路径。

    生成:
        str: 每次返回文件中的一行内容（去除换行符）。
    """
    try:
        with open(model_predict_result_txt_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.strip()
    except FileNotFoundError:
        print(f"错误：文件 '{model_predict_result_txt_file_path}' 未找到。")
    except Exception as e:
        print(f'发生错误：{e}')

def get_certain_line_content_from_model_predict_result_txt_file(model_predict_result_txt_file_path, line_count):
    """
    从模型预测结果文件中获取指定行的内容。
    
    参数:
        model_predict_result_txt_file_path (str): 模型预测结果的 .txt 文件路径
        line_count (int): 要返回的目标行号（从1开始计数）
        
    返回值:
        str: 指定行的内容；如果行号超出文件范围，则返回None
    """
    if line_count < 1:
        print(f'错误：行号 {line_count} 无效，行号应从1开始。')
        return None
    current_line = 1
    for line in read_model_predictions(model_predict_result_txt_file_path):
        if current_line == line_count:
            return line
        current_line += 1
    print(f'错误：文件只有 {current_line - 1} 行，无法获取第 {line_count} 行。')
    return None

def get_one_line_cot_and_answer(model_predict_result_txt_file_path, line_count):
    content = get_certain_line_content_from_model_predict_result_txt_file(model_predict_result_txt_file_path, line_count)
    str_generate_result_without_prompt = extract_str_generate_result_without_prompt(content)
    str_cot = extract_str_cot(str_generate_result_without_prompt)
    str_answer = extract_str_answer(str_generate_result_without_prompt)
    return (str_cot, str_answer)

def read_list_generate_result_from_txt(model_predict_result_txt_file_path):
    """
    从模型预测结果txt文件中读取所有预测答案，并返回答案列表。
    
    参数:
        model_predict_result_txt_file_path (str): 模型预测结果的 .txt 文件路径
        
    返回值:
        list: 包含所有预测答案的列表
    
    说明:
        该函数会逐行读取txt文件，对每一行：
        1. 提取"###Response:"后的生成结果
        2. 从生成结果中提取<answer>标签内的答案
        3. 将答案添加到列表中并返回
    """
    list_generate_answer = []
    for line in read_model_predictions(model_predict_result_txt_file_path):
        str_generate_result_without_prompt = extract_str_generate_result_without_prompt(line)
        str_answer = extract_str_answer(str_generate_result_without_prompt)
        list_generate_answer.append(str_answer)
    return list_generate_answer

def calculate_smiles_validity_accuracy_from_ref_list_and_pred_list(list_answer, list_generate_answer, str_task='nmr', dechirality=False, check_length=5000, list_false_prefix=None, retrosyn_tanimoto_method='concatenated'):
    """
    计算SMILES有效性和准确率（已修复 syn/retrosyn 任务下的 Validity 和 Tanimoto 计算问题）。
    """
    if list_false_prefix is None:
        list_false_prefix = ['F_']
    if len(list_answer) != len(list_generate_answer):
        raise ValueError(f'答案列表长度不匹配: {len(list_answer)} vs {len(list_generate_answer)}')
    if str_task.lower() == 'nmr':
        judge_func = judge_single_nmr_answer
    elif str_task.lower() == 'retrosyn':
        judge_func = judge_single_retrosyn_answer
    elif str_task.lower() == 'syn':
        judge_func = judge_single_syn_answer
    else:
        raise ValueError(f'不支持的任务类型: {str_task}')
    total_count = len(list_answer)
    valid_smiles_count = 0
    correct_count = 0
    tanimoto_sum = 0.0
    tanimoto_count = 0
    correct_040_count = 0
    correct_060_count = 0
    correct_080_count = 0
    correct_090_count = 0
    for gt_raw, pred_raw in zip(list_answer, list_generate_answer):
        pred_clean = pred_raw
        gt_clean = gt_raw
        if str_task.lower() in ['retrosyn', 'syn']:
            pred_clean = remove_reaction_tags(pred_raw)
            gt_clean = remove_reaction_tags(gt_raw)
            if str_task.lower() == 'retrosyn':
                pred_clean = _replace_reactant_placeholders(pred_clean)
                gt_clean = _replace_reactant_placeholders(gt_clean)
                if '>>' in gt_clean:
                    gt_clean = gt_clean.split('>>')[0].strip()
            if str_task.lower() == 'syn':
                pred_clean = _replace_product_placeholders(pred_clean)
                gt_clean = _replace_product_placeholders(gt_clean)
                if '>>' in gt_clean:
                    gt_clean = gt_clean.split('>>')[-1].strip()
        try:
            if not pred_clean or any((pred_clean.startswith(p) for p in list_false_prefix)):
                is_valid = False
            elif str_task.lower() == 'nmr':
                pred_mol = Chem.MolFromSmiles(pred_clean)
                is_valid = pred_mol is not None and len(pred_clean) < check_length
            else:
                parts = pred_clean.split('.')
                is_valid = True
                for part in parts:
                    if not part.strip():
                        continue
                    mol = Chem.MolFromSmiles(part.strip())
                    if mol is None or len(part) >= check_length:
                        is_valid = False
                        break
        except:
            is_valid = False
        if is_valid:
            valid_smiles_count += 1
        try:
            is_correct = judge_func(pred_raw, gt_raw, dechirality=dechirality, check_length=check_length, list_false_prefix=list_false_prefix)
            if is_correct:
                correct_count += 1
        except:
            pass
        if is_valid:
            try:
                if str_task.lower() == 'nmr':
                    gt_mol = Chem.MolFromSmiles(gt_clean)
                    pred_mol = Chem.MolFromSmiles(pred_clean)
                    if gt_mol is not None and pred_mol is not None:
                        if dechirality:
                            Chem.RemoveStereochemistry(gt_mol)
                            Chem.RemoveStereochemistry(pred_mol)
                        gt_fp = AllChem.GetMorganFingerprintAsBitVect(gt_mol, 2, nBits=1024)
                        pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=1024)
                        tanimoto = DataStructs.TanimotoSimilarity(gt_fp, pred_fp)
                        tanimoto_sum += tanimoto
                        tanimoto_count += 1
                else:
                    tanimoto = calculate_retrosyn_tanimoto_similarity(gt_clean, pred_clean, method=retrosyn_tanimoto_method, dechirality=dechirality)
                    if tanimoto > 0:
                        tanimoto_sum += tanimoto
                        tanimoto_count += 1
                if 'tanimoto' in locals() or 'tanimoto' in globals():
                    if tanimoto >= 0.4:
                        correct_040_count += 1
                    if tanimoto >= 0.6:
                        correct_060_count += 1
                    if tanimoto >= 0.8:
                        correct_080_count += 1
                    if tanimoto >= 0.9:
                        correct_090_count += 1
            except:
                pass
    stats = {'prediction_validity_rate': valid_smiles_count / total_count if total_count > 0 else 0.0, 'accuracy': correct_count / total_count if total_count > 0 else 0.0, 'average_tanimoto': tanimoto_sum / tanimoto_count if tanimoto_count > 0 else 0.0, 'valid_smiles': valid_smiles_count / total_count if total_count > 0 else 0.0, 'accuracy_040': correct_040_count / total_count if total_count > 0 else 0.0, 'accuracy_060': correct_060_count / total_count if total_count > 0 else 0.0, 'accuracy_080': correct_080_count / total_count if total_count > 0 else 0.0, 'accuracy_090': correct_090_count / total_count if total_count > 0 else 0.0}
    return stats

def remove_extra_txt_extension(file_path):
    file_name = os.path.basename(file_path)
    if file_name.endswith('.txt.txt'):
        new_file_name = file_name[:-4]
        return os.path.join(os.path.dirname(file_path), new_file_name)
    return file_path

def vote_for_best_answer(list_candidate_info, list_priority_index):
    """
    对候选答案进行投票，选择最佳答案。
    
    参数:
        list_candidate_info (list): 候选信息列表，每个元素为(canonical_smiles, original_index)
        list_priority_index (list): 优先级索引列表，对应list_model_predict_result_txt_file_path的索引
        
    返回值:
        tuple: (best_canonical_smiles, best_original_index) - 最佳的标准SMILES和对应的原始索引
    """
    from collections import Counter, defaultdict
    if not list_candidate_info:
        return (None, None)
    smiles_votes = Counter((info[0] for info in list_candidate_info))
    max_votes = max(smiles_votes.values())
    top_smiles = [smiles for smiles, votes in smiles_votes.items() if votes == max_votes]
    if len(top_smiles) == 1:
        best_smiles = top_smiles[0]
        candidates_with_best_smiles = [info for info in list_candidate_info if info[0] == best_smiles]
        best_original_index = min((info[1] for info in candidates_with_best_smiles))
        return (best_smiles, best_original_index)
    else:
        best_smiles = None
        best_priority = float('inf')
        best_original_index = None
        for smiles in top_smiles:
            candidates_with_this_smiles = [info for info in list_candidate_info if info[0] == smiles]
            min_original_index = min((info[1] for info in candidates_with_this_smiles))
            priority = list_priority_index[min_original_index]
            if priority < best_priority:
                best_priority = priority
                best_smiles = smiles
                best_original_index = min_original_index
        return (best_smiles, best_original_index)

def process_single_sample_cot_sc(args):
    """
    处理单个样本的CoT Self-Consistency，用于并行处理。
    
    参数:
        args (tuple): 包含(sample_idx, list_model_predict_result_txt_file_path, list_priority_index)
        
    返回值:
        str: 该样本的最终预测答案
    """
    sample_idx, list_model_predict_result_txt_file_path, list_priority_index = args
    list_candidate_info = []
    list_all_raw_answers = []
    for file_idx, txt_file_path in enumerate(list_model_predict_result_txt_file_path):
        try:
            line_content = get_certain_line_content_from_model_predict_result_txt_file(txt_file_path, sample_idx + 1)
            if line_content is None:
                continue
            str_generate_result_without_prompt = extract_str_generate_result_without_prompt(line_content)
            str_answer = extract_str_answer(str_generate_result_without_prompt)
            str_formula = extract_formula_from_generate_result(str_generate_result_without_prompt)
            list_all_raw_answers.append((str_answer, file_idx))
            if str_formula:
                is_valid, canonical_smiles = is_valid_smiles_with_formula_check(str_answer, str_formula)
                if is_valid:
                    list_candidate_info.append((canonical_smiles, file_idx))
        except Exception as e:
            continue
    if list_candidate_info:
        best_canonical_smiles, best_file_index = vote_for_best_answer(list_candidate_info, list_priority_index)
        return best_canonical_smiles
    elif list_all_raw_answers:
        list_all_raw_answers.sort(key=lambda x: x[1])
        fallback_answer = list_all_raw_answers[0][0]
        return fallback_answer
    else:
        return 'F'

def test_model_predict_result_cot_sc(arrow_folder_path, list_model_predict_result_txt_file_path, use_parallel=True, max_workers=64, save_json_path=None):
    """
    测试模型预测结果，支持CoT Self-Consistency功能。
    对同一个问题的多个预测结果进行投票，选择最佳答案。
    
    参数:
        arrow_folder_path (str): 包含arrow文件的文件夹路径
        list_model_predict_result_txt_file_path (list): 模型预测结果txt文件路径列表，按优先级排序
        use_parallel (bool): 是否使用并行处理，默认True
        max_workers (int): 最大并行进程数，默认None（自动选择）
        save_json_path (str): 保存统计结果成.json文件的路径，None则不保存
    返回值:
        dict: 统计结果字典
    """
    import multiprocessing as mp
    from functools import partial
    list_question, list_cot, list_answer = load_question_cot_answer_list_from_arrow_folder(arrow_folder_path)
    num_samples = len(list_question)
    print(f'开始处理 {num_samples} 个样本，使用 {len(list_model_predict_result_txt_file_path)} 个预测文件')
    print(f"并行处理: {('启用' if use_parallel else '禁用')}")
    list_priority_index = list(range(len(list_model_predict_result_txt_file_path)))
    if use_parallel and num_samples > 1:
        if max_workers is None:
            max_workers = min(mp.cpu_count(), num_samples)
        print(f'使用 {max_workers} 个进程进行并行处理')
        args_list = [(sample_idx, list_model_predict_result_txt_file_path, list_priority_index) for sample_idx in range(num_samples)]
        try:
            with mp.Pool(processes=max_workers) as pool:
                results_dict = {}
                from tqdm import tqdm
                for i, result in enumerate(tqdm(pool.imap(process_single_sample_cot_sc, args_list), total=num_samples, desc='处理样本')):
                    results_dict[args_list[i][0]] = result
                list_final_generate_answer = [results_dict[sample_idx] for sample_idx in range(num_samples)]
        except Exception as e:
            print(f'并行处理出现错误: {e}')
            print('回退到串行处理...')
            use_parallel = False
    if not use_parallel or num_samples <= 1:
        print('使用串行处理')
        list_final_generate_answer = []
        for sample_idx in range(num_samples):
            if (sample_idx + 1) % 100 == 0:
                print(f'正在处理第 {sample_idx + 1}/{num_samples} 个样本')
            args = (sample_idx, list_model_predict_result_txt_file_path, list_priority_index)
            result = process_single_sample_cot_sc(args)
            list_final_generate_answer.append(result)
    stats = calculate_smiles_validity_accuracy_from_ref_list_and_pred_list(list_answer, list_final_generate_answer)
    print('\nCoT Self-Consistency 评估结果:')
    for key, value in stats.items():
        print(f'{key}: {value}')
    if save_json_path is not None:
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f'统计结果已保存到 {save_json_path}')
    return stats

def _build_cot_sc_structure_with_accuracy_sorting(list_model_infer_save_folder_path, list_dataset_name, save_folder_path, save_result_filename_prefix):
    """
    按准确率排序构建CoT SC数据结构（需要读取CSV）
    """
    import pandas as pd
    import os
    list_model_csv_data = []
    for model_idx, model_folder_path in enumerate(list_model_infer_save_folder_path):
        print(f'处理第 {model_idx + 1} 个模型文件夹: {model_folder_path}')
        list_accuracy_csv_files = get_files_with_suffix(model_folder_path, '_accuracy.csv', traverse_sub_folder=False)
        if len(list_accuracy_csv_files) == 0:
            raise FileNotFoundError(f'在文件夹 {model_folder_path} 中未找到 _accuracy.csv 文件')
        elif len(list_accuracy_csv_files) > 1:
            print(f'警告：在文件夹 {model_folder_path} 中找到多个 _accuracy.csv 文件，使用第一个: {list_accuracy_csv_files[0]}')
        accuracy_csv_path = list_accuracy_csv_files[0]
        print(f'读取文件: {accuracy_csv_path}')
        try:
            df_accuracy = pd.read_csv(accuracy_csv_path)
            if 'train_step' not in df_accuracy.columns:
                raise ValueError(f"CSV文件 {accuracy_csv_path} 中缺少 'train_step' 列")
            model_data = {'model_folder_path': model_folder_path, 'csv_path': accuracy_csv_path, 'df': df_accuracy, 'model_index': model_idx}
            list_model_csv_data.append(model_data)
            print(f'成功读取CSV，共 {len(df_accuracy)} 行数据')
        except Exception as e:
            raise Exception(f'读取CSV文件 {accuracy_csv_path} 时出错: {e}')
    num_rows = len(list_model_csv_data[0]['df'])
    for model_data in list_model_csv_data:
        if len(model_data['df']) != num_rows:
            raise ValueError(f"所有模型的CSV文件行数必须一致，发现不一致的文件: {model_data['csv_path']}")
    print(f'所有CSV文件行数验证通过，共 {num_rows} 个训练阶段')
    list_dict_result = []
    for row_idx in range(num_rows):
        print(f'处理第 {row_idx + 1}/{num_rows} 个训练阶段')
        dict_current_stage = {}
        for dataset_name in list_dataset_name:
            print(f'  处理数据集: {dataset_name}')
            list_model_file_info = []
            for model_data in list_model_csv_data:
                model_folder_path = model_data['model_folder_path']
                df = model_data['df']
                model_idx = model_data['model_index']
                train_step = str(int(df.iloc[row_idx]['train_step']))
                step_folder_path = os.path.join(model_folder_path, train_step)
                if not os.path.exists(step_folder_path):
                    print(f'    警告：文件夹不存在 {step_folder_path}')
                    continue
                pattern_txt_files = get_files_with_suffix(step_folder_path, '.txt', traverse_sub_folder=False)
                matching_txt_files = [txt_file for txt_file in pattern_txt_files if txt_file.endswith(f'_{dataset_name}.txt')]
                if len(matching_txt_files) == 0:
                    print(f'    警告：在 {step_folder_path} 中未找到匹配 {dataset_name} 的txt文件')
                    continue
                elif len(matching_txt_files) > 1:
                    print(f'    警告：在 {step_folder_path} 中找到多个匹配 {dataset_name} 的txt文件，使用第一个')
                txt_file_path = matching_txt_files[0]
                if dataset_name in df.columns:
                    accuracy = float(df.iloc[row_idx][dataset_name])
                else:
                    print(f'    警告：CSV中缺少数据集 {dataset_name} 的列，使用默认准确率0.0')
                    accuracy = 0.0
                list_model_file_info.append({'txt_file_path': txt_file_path, 'accuracy': accuracy, 'model_folder': model_folder_path, 'model_index': model_idx})
                print(f'    找到文件: {txt_file_path}, 准确率: {accuracy}')
            list_model_file_info.sort(key=lambda x: x['accuracy'], reverse=True)
            print(f'    数据集 {dataset_name} 按准确率排序')
            list_sorted_txt_files = [info['txt_file_path'] for info in list_model_file_info]
            dict_current_stage[dataset_name] = list_sorted_txt_files
            print(f'    数据集 {dataset_name} 处理完成，共 {len(list_sorted_txt_files)} 个文件')
        list_dict_result.append(dict_current_stage)
    return list_dict_result

def _save_summary_info(list_dict_result, list_model_infer_save_folder_path, list_dataset_name, save_folder_path, save_result_filename_prefix, sort_by_accuracy):
    """保存处理摘要信息"""
    import json
    import os
    summary_info = {'num_training_stages': len(list_dict_result), 'num_models': len(list_model_infer_save_folder_path), 'dataset_names': list_dataset_name, 'model_folders': list_model_infer_save_folder_path, 'sort_by_accuracy': sort_by_accuracy}
    print('\n处理完成摘要:')
    print(f"训练阶段数: {summary_info['num_training_stages']}")
    print(f"模型数量: {summary_info['num_models']}")
    print(f"数据集数量: {len(summary_info['dataset_names'])}")
    print(f"排序方式: {('按准确率排序' if sort_by_accuracy else '按训练步数排序')}")
    if save_folder_path and os.path.exists(save_folder_path):
        summary_file_path = os.path.join(save_folder_path, f'{save_result_filename_prefix}_cot_sc_summary.json')
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, indent=4, ensure_ascii=False)
        print(f'摘要信息已保存至: {summary_file_path}')

def _build_cot_sc_structure_with_step_sorting(list_model_infer_save_folder_path, list_dataset_name, save_folder_path, save_result_filename_prefix):
    """
    按训练步数排序构建CoT SC数据结构（不读取CSV，直接基于文件夹结构）
    """
    import os
    import re
    print('使用基于文件夹结构的处理方式，不读取CSV文件')
    dict_model_step_folders = {}
    for model_idx, model_folder_path in enumerate(list_model_infer_save_folder_path):
        print(f'扫描第 {model_idx + 1} 个模型文件夹: {model_folder_path}')
        subdirs = get_subdirectories(model_folder_path)
        step_folders = {}
        for subdir in subdirs:
            folder_name = os.path.basename(subdir)
            if re.match('^\\d+$', folder_name):
                step_num = int(folder_name)
                step_folders[step_num] = subdir
        if not step_folders:
            print(f'    警告：在 {model_folder_path} 中未找到数字命名的文件夹')
            continue
        dict_model_step_folders[model_idx] = step_folders
        print(f'    找到 {len(step_folders)} 个训练步数文件夹: {sorted(step_folders.keys())}')
    if not dict_model_step_folders:
        raise ValueError('所有模型文件夹中都没有找到数字命名的子文件夹')
    dict_model_sorted_steps = {}
    for model_idx, step_folders in dict_model_step_folders.items():
        sorted_steps = sorted(step_folders.keys())
        dict_model_sorted_steps[model_idx] = sorted_steps
        print(f'    模型 {model_idx} 排序后的训练步数: {sorted_steps}')
    if dict_model_sorted_steps:
        step_counts = [len(steps) for steps in dict_model_sorted_steps.values()]
        min_stages = min(step_counts)
        max_stages = max(step_counts)
        if min_stages != max_stages:
            print(f'警告：模型间训练阶段数不一致，最少 {min_stages} 个，最多 {max_stages} 个')
            print(f'将使用前 {min_stages} 个训练阶段进行对齐')
        print(f'共处理 {min_stages} 个训练阶段（按排序位置对应）')
    else:
        raise ValueError('没有找到任何有效的训练步数')
    list_dict_result = []
    for stage_idx in range(min_stages):
        print(f'处理第 {stage_idx + 1}/{min_stages} 个训练阶段（位置索引 {stage_idx}）')
        dict_current_stage = {}
        stage_steps_info = []
        for model_idx in sorted(dict_model_sorted_steps.keys()):
            actual_step = dict_model_sorted_steps[model_idx][stage_idx]
            stage_steps_info.append(f'模型{model_idx}:{actual_step}')
        print(f"  当前阶段对应步数: {', '.join(stage_steps_info)}")
        for dataset_name in list_dataset_name:
            print(f'  处理数据集: {dataset_name}')
            list_txt_files = []
            for model_idx in sorted(dict_model_sorted_steps.keys()):
                actual_step_num = dict_model_sorted_steps[model_idx][stage_idx]
                step_folder_path = dict_model_step_folders[model_idx][actual_step_num]
                pattern_txt_files = get_files_with_suffix(step_folder_path, '.txt', traverse_sub_folder=False)
                matching_txt_files = [txt_file for txt_file in pattern_txt_files if txt_file.endswith(f'_{dataset_name}.txt')]
                if len(matching_txt_files) == 0:
                    print(f'    警告：在 {step_folder_path} 中未找到匹配 {dataset_name} 的txt文件')
                    continue
                elif len(matching_txt_files) > 1:
                    print(f'    警告：在 {step_folder_path} 中找到多个匹配 {dataset_name} 的txt文件，使用第一个')
                txt_file_path = matching_txt_files[0]
                list_txt_files.append(txt_file_path)
                print(f'    模型{model_idx}(步数{actual_step_num}): {os.path.basename(txt_file_path)}')
            dict_current_stage[dataset_name] = list_txt_files
            print(f'    数据集 {dataset_name} 按模型顺序排序，共 {len(list_txt_files)} 个文件')
        list_dict_result.append(dict_current_stage)
    return list_dict_result

def get_list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path(list_model_infer_save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_filename_prefix, save_folder_path, sort_by_accuracy=False):
    """
    获取用于CoT Self-Consistency的数据结构，按准确率排序多个模型的预测结果文件。
    
    参数:
        list_model_infer_save_folder_path (list): 多个模型推理结果文件夹路径列表
        dict_key_test_arrow_folder_path_value_dataset_name (dict): 测试文件夹路径到数据集名称的映射
        save_result_filename_prefix (str): 结果文件名前缀
        save_folder_path (str): 输出结果的文件夹路径
        sort_by_accuracy (bool): 排序方式，True为按准确率降序排序（默认），False为按模型文件夹顺序排序
        
    返回值:
        list: list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path
              列表中每个元素是一个字典，键为数据集名称，值为排序后的预测文件路径列表
    """
    print(f'开始处理 {len(list_model_infer_save_folder_path)} 个模型文件夹')
    print(f"排序方式: {('按准确率排序' if sort_by_accuracy else '按训练步数排序')}")
    list_dataset_name = list(dict_key_test_arrow_folder_path_value_dataset_name.values())
    print(f'数据集列表: {list_dataset_name}')
    if sort_by_accuracy:
        result = _build_cot_sc_structure_with_accuracy_sorting(list_model_infer_save_folder_path, list_dataset_name, save_folder_path, save_result_filename_prefix)
    else:
        result = _build_cot_sc_structure_with_step_sorting(list_model_infer_save_folder_path, list_dataset_name, save_folder_path, save_result_filename_prefix)
    _save_summary_info(result, list_model_infer_save_folder_path, list_dataset_name, save_folder_path, save_result_filename_prefix, sort_by_accuracy)
    return result

def batch_evaluate_cot_sc_and_save_results(list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path, dict_key_test_arrow_folder_path_value_dataset_name, save_folder_path, save_result_filename_prefix='cot_sc_evaluation_result', use_parallel=True, max_workers=None, decimal_places=3):
    """
    批量执行CoT Self-Consistency评估并保存结果为JSON和CSV格式。
    
    参数:
        list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path (list): 
            CoT SC数据结构，每个元素对应一个训练阶段的数据集-文件路径映射
        dict_key_test_arrow_folder_path_value_dataset_name (dict): 
            测试文件夹路径到数据集名称的映射字典
        save_folder_path (str): 保存结果文件的文件夹路径
        save_result_filename_prefix (str): 保存结果文件的前缀名称，默认"cot_sc_evaluation_result"
        use_parallel (bool): 是否使用并行处理，默认True
        max_workers (int): 最大并行进程数，默认None（自动选择）
        decimal_places (int): CSV文件中小数值保留的小数位数，默认3
        
    返回值:
        dict: 格式化后的评估结果字典，键为训练阶段（字符串），值为该阶段各数据集的评估统计
    """
    import os
    from tqdm import tqdm
    print('开始批量CoT Self-Consistency评估...')
    if not list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path:
        raise ValueError('输入的CoT SC数据结构为空')
    dict_key_dataset_name_value_arrow_folder_path = {dataset_name: arrow_folder_path for arrow_folder_path, dataset_name in dict_key_test_arrow_folder_path_value_dataset_name.items()}
    list_dataset_name = list(dict_key_dataset_name_value_arrow_folder_path.keys())
    print(f'将评估 {len(list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path)} 个训练阶段')
    print(f'涉及数据集: {list_dataset_name}')
    print(f"并行处理: {('启用' if use_parallel else '禁用')}")
    dict_results = {}
    for stage_idx, dict_current_stage in enumerate(tqdm(list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path, desc='评估训练阶段')):
        print(f'\n处理第 {stage_idx + 1} 个训练阶段')
        stage_results = {}
        for dataset_name in list_dataset_name:
            if dataset_name not in dict_current_stage:
                print(f'  警告：当前阶段缺少数据集 {dataset_name}，跳过')
                continue
            list_model_predict_result_txt_file_path = dict_current_stage[dataset_name]
            if not list_model_predict_result_txt_file_path:
                print(f'  警告：数据集 {dataset_name} 没有可用的预测文件，跳过')
                continue
            print(f'  评估数据集 {dataset_name}，使用 {len(list_model_predict_result_txt_file_path)} 个预测文件')
            arrow_folder_path = dict_key_dataset_name_value_arrow_folder_path[dataset_name]
            try:
                stats = test_model_predict_result_cot_sc(arrow_folder_path=arrow_folder_path, list_model_predict_result_txt_file_path=list_model_predict_result_txt_file_path, use_parallel=use_parallel, max_workers=max_workers)
                stage_results[dataset_name] = stats
                print(f'  数据集 {dataset_name} 评估完成')
            except Exception as e:
                print(f'  错误：评估数据集 {dataset_name} 时出现异常: {e}')
                stage_results[dataset_name] = {'total_samples': 0, 'valid_references': 0, 'reference_validity_rate': 0.0, 'valid_predictions': 0, 'prediction_validity_rate': 0.0, 'correct_predictions': 0, 'accuracy': 0.0, 'average_tanimoto': 0.0, 'tanimoto_pairs_count': 0, 'correct_indices_start_0': []}
                continue
        dict_results[str(stage_idx)] = stage_results
    print(f'\n所有训练阶段评估完成，共处理了 {len(dict_results)} 个阶段')
    print('开始保存评估结果...')
    try:
        json_file_path, csv_file_path, formatted_csv_path, accuracy_csv_path = save_evaluation_results_as_json_and_csv(dict_results=dict_results, save_folder_path=save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name=dict_key_test_arrow_folder_path_value_dataset_name, save_result_filename_prefix=save_result_filename_prefix, decimal_places=decimal_places)
        print('CoT Self-Consistency评估结果已成功保存:')
        print(f'  JSON文件: {json_file_path}')
        print(f'  CSV文件: {csv_file_path}')
        if formatted_csv_path:
            print(f'  格式化CSV: {formatted_csv_path}')
        print(f'  准确率CSV: {accuracy_csv_path}')
    except Exception as e:
        print(f'保存结果时出现错误: {e}')
        raise
    print('\n评估摘要:')
    print(f'训练阶段数: {len(dict_results)}')
    dataset_counts = [len(stage_data) for stage_data in dict_results.values()]
    if dataset_counts:
        print(f'平均每阶段数据集数: {sum(dataset_counts) / len(dataset_counts):.1f}')
        print(f'数据集数量范围: {min(dataset_counts)} - {max(dataset_counts)}')
    return dict_results

def get_cot_sc_pipeline_results(list_model_infer_save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_filename_prefix, save_folder_path, use_parallel=True, max_workers=None, decimal_places=3):
    """
    完整的CoT Self-Consistency评估管道：从模型结果文件夹到最终保存的评估结果。
    
    这是一个便捷函数，将整个CoT SC评估流程封装在一起：
    1. 从模型文件夹构建CoT SC数据结构
    2. 执行批量CoT SC评估
    3. 保存结果为JSON和CSV格式
    
    参数:
        list_model_infer_save_folder_path (list): 多个模型推理结果文件夹路径列表
        dict_key_test_arrow_folder_path_value_dataset_name (dict): 测试文件夹路径到数据集名称的映射
        save_result_filename_prefix (str): 结果文件名前缀
        save_folder_path (str): 输出结果的文件夹路径
        use_parallel (bool): 是否使用并行处理，默认True
        max_workers (int): 最大并行进程数，默认None（自动选择）
        decimal_places (int): CSV文件中小数值保留的小数位数，默认3
        
    返回值:
        tuple: (list_dict_cot_sc_data, dict_evaluation_results)
               - list_dict_cot_sc_data: CoT SC数据结构
               - dict_evaluation_results: 评估结果字典
    """
    print('=' * 60)
    print('开始完整的CoT Self-Consistency评估管道')
    print('=' * 60)
    print('步骤1：构建CoT Self-Consistency数据结构...')
    list_dict_cot_sc_data = get_list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path(list_model_infer_save_folder_path=list_model_infer_save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name=dict_key_test_arrow_folder_path_value_dataset_name, save_result_filename_prefix=save_result_filename_prefix, save_folder_path=save_folder_path)
    print(f'CoT SC数据结构构建完成，共 {len(list_dict_cot_sc_data)} 个训练阶段')
    print('\n步骤2：执行批量CoT Self-Consistency评估...')
    dict_evaluation_results = batch_evaluate_cot_sc_and_save_results(list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path=list_dict_cot_sc_data, dict_key_test_arrow_folder_path_value_dataset_name=dict_key_test_arrow_folder_path_value_dataset_name, save_folder_path=save_folder_path, save_result_filename_prefix=save_result_filename_prefix, use_parallel=use_parallel, max_workers=max_workers, decimal_places=decimal_places)
    print('=' * 60)
    print('CoT Self-Consistency评估管道完成！')
    print('=' * 60)
    return (list_dict_cot_sc_data, dict_evaluation_results)

def _is_content_hallucinated(text: str) -> bool:
    """
    检测字符串是否包含 NMR 数据中不该出现的“幻觉/崩坏”特征。
    用于在提取数值前拦截严重崩坏的生成结果（如死循环、SMILES泄漏、异常关键词）。
    如果不拦截，这些数据可能会被错误地解析出数值，导致统计指标异常。
    
    注意：不应过滤 [MASK]，因为它是合法的占位符。
    """
    if not text:
        return False
    if re.search('([\\]\\.\\>\\)])\\1{4,}', text):
        return True
    if text.count('):):') > 3:
        return True
    forbidden_markers = ['=C', 'c1', 'c2', '*):', '>>', '(:*)']
    for marker in forbidden_markers:
        if marker in text:
            return True
    hallucination_keywords = ['corrupted', 'irrelevant', 'twisted', 'terminating', 'approached', 'letting', 'expressing', 'Finally']
    for kw in hallucination_keywords:
        if kw in text:
            return True
    return False

def _extract_tag_smiles(text: str, tag_prefix: str) -> dict:
    """提取 [C1]...[/C1] 格式的标签内容"""
    pattern = '\\[' + re.escape(tag_prefix) + '(\\d+)\\](.*?)\\[/' + re.escape(tag_prefix) + '\\1\\]'
    regex = re.compile(pattern, flags=re.DOTALL)
    result = {}
    for idx_str, content in regex.findall(text):
        idx = int(idx_str)
        smiles = content.strip()
        if smiles:
            result[idx] = smiles
    return result

def str_cot_to_dict(str_cot: str) -> dict:
    """将 CoT 字符串转换为结构化字典"""
    dict_cot = {'depth': {'13c': {}, '1h': {}}, 'brics': {}}
    if not isinstance(str_cot, str):
        return dict_cot
    c_dict = _extract_tag_smiles(str_cot, 'C')
    h_dict = _extract_tag_smiles(str_cot, 'H')
    brics_dict = _extract_tag_smiles(str_cot, 'FS')
    dict_cot['depth']['13c'] = {idx: c_dict[idx] for idx in sorted(c_dict.keys())} if c_dict else {}
    dict_cot['depth']['1h'] = {idx: h_dict[idx] for idx in sorted(h_dict.keys())} if h_dict else {}
    dict_cot['brics'] = {idx: brics_dict[idx] for idx in sorted(brics_dict.keys())} if brics_dict else {}
    return dict_cot

def _smiles_to_mol_safe(smiles: str):
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except:
        return None

def _stat_single_section(section_gt: dict, section_generate: dict, dechirality: bool) -> dict:
    """统计单个 CoT 部分（如 13C 片段）的准确率和相似度"""
    count = len(section_generate)
    if count == 0:
        return {'count': 0, 'correct_count': 0, 'average_tanimoto': 0.0}
    correct_count = 0
    tanimoto_sum = 0.0
    tanimoto_pair_count = 0
    for idx, smiles_gen in section_generate.items():
        mol_gen = _smiles_to_mol_safe(smiles_gen)
        smiles_gt = section_gt.get(idx, None)
        if smiles_gt is None:
            continue
        mol_gt = _smiles_to_mol_safe(smiles_gt)
        if mol_gen is None or mol_gt is None:
            continue
        if dechirality:
            mol_gen = Chem.Mol(mol_gen)
            Chem.RemoveStereochemistry(mol_gen)
            mol_gt = Chem.Mol(mol_gt)
            Chem.RemoveStereochemistry(mol_gt)
        try:
            smi_gen_std = Chem.MolToSmiles(mol_gen, canonical=True)
            smi_gt_std = Chem.MolToSmiles(mol_gt, canonical=True)
            if smi_gen_std == smi_gt_std:
                correct_count += 1
        except:
            pass
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_gen, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_gt, 2, nBits=2048)
            tanimoto_sum += DataStructs.TanimotoSimilarity(fp1, fp2)
            tanimoto_pair_count += 1
        except:
            pass
    average_tanimoto = tanimoto_sum / tanimoto_pair_count if tanimoto_pair_count > 0 else 0.0
    return {'count': count, 'correct_count': correct_count, 'average_tanimoto': float(average_tanimoto)}

def stat_dict_cot_generate_info(dict_cot_gt: dict, dict_cot_generate: dict, dechirality: bool=True) -> dict:
    """统计整个 CoT 的结构信息"""
    depth_gt = dict_cot_gt.get('depth', {})
    depth_gen = dict_cot_generate.get('depth', {})
    gt_13c = depth_gt.get('13c', {}) or {}
    gen_13c = depth_gen.get('13c', {}) or {}
    gt_1h = depth_gt.get('1h', {}) or {}
    gen_1h = depth_gen.get('1h', {}) or {}
    gt_brics = dict_cot_gt.get('brics', {}) or {}
    gen_brics = dict_cot_generate.get('brics', {}) or {}
    return {'depth': {'13c': _stat_single_section(gt_13c, gen_13c, dechirality), '1h': _stat_single_section(gt_1h, gen_1h, dechirality)}, 'brics': _stat_single_section(gt_brics, gen_brics, dechirality)}

def _extract_nmr_data_from_answer(answer_str: str) -> dict:
    """
    从 Answer 字符串提取 NMR 数值 (13C 和 1H)。
    
    修改说明:
    1. 增加了脏数据预检 (_is_content_hallucinated)，遇到崩坏数据直接返回空结果。
    2. 使用分号 ';' 精准切分 13C 和 1H 区域，避免正则跨区域误匹配。
    3. 在局部解析时也进行了二次脏数据检查。
    
    参数:
        answer_str (str): 模型生成的答案字符串。
        
    返回:
        dict: 包含 '13c_shifts', '1h_shifts' (列表) 和对应的 'shift_count' (整数)。
              如果解析失败或数据崩坏，列表为空，计数为 0。
    """
    result = {'13c_shifts': [], '1h_shifts': [], '13c_shift_count': 0, '1h_shift_count': 0}
    if not answer_str or not isinstance(answer_str, str):
        return result
    if answer_str == 'F_' or answer_str == 'F':
        return result
    if _is_content_hallucinated(answer_str):
        return result
    parts = answer_str.split(';')
    str_13c_content = ''
    str_1h_content = ''
    for part in parts:
        part = part.strip()
        if '13C NMR:' in part:
            str_13c_content = part.split('13C NMR:')[-1].strip()
        elif '1H NMR:' in part:
            str_1h_content = part.split('1H NMR:')[-1].strip()
    if not str_13c_content and (not str_1h_content):
        match_13c = re.search('13C NMR:(.*?)(?:1H NMR:|$)', answer_str)
        if match_13c:
            str_13c_content = match_13c.group(1)
        match_1h = re.search('1H NMR:(.*)', answer_str)
        if match_1h:
            str_1h_content = match_1h.group(1)
    if str_13c_content and (not _is_content_hallucinated(str_13c_content)):
        result['13c_nmr'] = str_13c_content
        idx_bracket_close = str_13c_content.find(']')
        if idx_bracket_close != -1:
            raw_nums = str_13c_content[idx_bracket_close + 1:]
        else:
            raw_nums = str_13c_content
        shifts = re.findall('[-+]?\\d*\\.\\d+|[-+]?\\d+', raw_nums)
        if shifts:
            result['13c_shifts'] = sorted([float(s) for s in shifts if s])
            result['13c_shift_count'] = len(result['13c_shifts'])
    if str_1h_content and (not _is_content_hallucinated(str_1h_content)):
        result['1h_nmr'] = str_1h_content
        idx_bracket_close = str_1h_content.find(']')
        raw_tuples = str_1h_content[idx_bracket_close + 1:] if idx_bracket_close != -1 else str_1h_content
        shifts = re.findall('\\(\\s*([-+]?\\d*\\.?\\d+)', raw_tuples)
        if shifts:
            result['1h_shifts'] = sorted([float(s) for s in shifts if s])
            result['1h_shift_count'] = len(result['1h_shifts'])
    return result

def _calculate_error_metrics(shifts_gen: list, shifts_gt: list):
    """计算 MSE, MAE, RMSD (Zero-padding)"""
    if not shifts_gen and (not shifts_gt):
        return (0.0, 0.0, 0.0)
    max_len = max(len(shifts_gen), len(shifts_gt))
    gen_padded = shifts_gen + [0.0] * (max_len - len(shifts_gen))
    gt_padded = shifts_gt + [0.0] * (max_len - len(shifts_gt))
    abs_errors = [abs(g - t) for g, t in zip(gen_padded, gt_padded)]
    squared_errors = [e ** 2 for e in abs_errors]
    mae = sum(abs_errors) / max_len if max_len > 0 else 0.0
    mse = sum(squared_errors) / max_len if max_len > 0 else 0.0
    rmsd = math.sqrt(mse)
    return (mse, mae, rmsd)

def stat_nmr_forward_answer_info(gt_answer: str, generate_answer: str, with_answer_for_deta_count: bool=False, max_length_for_deta_count: int=5000 * 6) -> dict:
    """
    [修复版] 统计 NMR Forward 的 Answer 部分。
    修复了 13C 和 1H 过滤逻辑耦合的问题，现在它们是独立判断的。
    """
    gt_data = _extract_nmr_data_from_answer(gt_answer)
    gen_data = _extract_nmr_data_from_answer(generate_answer)
    is_tag_broken = generate_answer in ['F', 'F_', '']
    answer_length = len(generate_answer) if generate_answer else 0
    answer_too_long = answer_length > max_length_for_deta_count
    filter_13c = False
    if is_tag_broken or answer_too_long:
        filter_13c = True
    elif with_answer_for_deta_count and gen_data['13c_shift_count'] == 0:
        filter_13c = True
    if filter_13c:
        deta_13c = 'FILTERED'
    else:
        deta_13c = gen_data['13c_shift_count'] - gt_data['13c_shift_count']
    filter_1h = False
    if is_tag_broken or answer_too_long:
        filter_1h = True
    elif with_answer_for_deta_count and gen_data['1h_shift_count'] == 0:
        filter_1h = True
    if filter_1h:
        deta_1h = 'FILTERED'
    else:
        deta_1h = gen_data['1h_shift_count'] - gt_data['1h_shift_count']
    mse_13c, mae_13c, rmsd_13c = _calculate_error_metrics(gen_data['13c_shifts'], gt_data['13c_shifts'])
    mse_1h, mae_1h, rmsd_1h = _calculate_error_metrics(gen_data['1h_shifts'], gt_data['1h_shifts'])
    return {'deta_13c': deta_13c, 'deta_1h': deta_1h, 'mse_13c': mse_13c, 'mae_13c': mae_13c, 'rmsd_13c': rmsd_13c, 'mse_1h': mse_1h, 'mae_1h': mae_1h, 'rmsd_1h': rmsd_1h}

def calc_global_stats_from_memory(list_dict_main: list, delta_1h_count: int=0, delta_13c_count: int=0) -> dict:
    """
    [修复版] NMR Forward 全局指标统计函数
    实现了严格的 Joint Filter：只有 13C 和 1H 均有效的样本才计入 Delta 平均值。
    """
    if not list_dict_main:
        return {}
    total_count = len(list_dict_main)
    acc = {'deta_13c_cnt': 0.0, 'deta_1h_cnt': 0.0, 'cot_13c_tani': 0.0, 'cot_1h_tani': 0.0, 'cot_brics_tani': 0.0, 'cot_13c_acc_sum': 0.0, 'cot_1h_acc_sum': 0.0, 'cot_brics_acc_sum': 0.0, 'mse_13c': 0.0, 'mae_13c': 0.0, 'rmsd_13c': 0.0, 'mse_1h': 0.0, 'mae_1h': 0.0, 'rmsd_1h': 0.0}
    valid_joint_sample_count = 0
    filtered_count_for_error_metrics = 0
    for item in list_dict_main:
        ans = item.get('answer', {})
        cot = item.get('cot', {})
        deta_13c = ans.get('deta_13c')
        deta_1h = ans.get('deta_1h')
        is_13c_valid = isinstance(deta_13c, (int, float))
        is_1h_valid = isinstance(deta_1h, (int, float))
        if is_13c_valid and is_1h_valid:
            acc['deta_13c_cnt'] += deta_13c
            acc['deta_1h_cnt'] += deta_1h
            valid_joint_sample_count += 1
            cond_1h = delta_1h_count is None or abs(deta_1h) <= delta_1h_count
            cond_13c = delta_13c_count is None or abs(deta_13c) <= delta_13c_count
            if cond_1h and cond_13c:
                filtered_count_for_error_metrics += 1
                acc['mse_13c'] += ans.get('mse_13c', 0.0)
                acc['mae_13c'] += ans.get('mae_13c', 0.0)
                acc['rmsd_13c'] += ans.get('rmsd_13c', 0.0)
                acc['mse_1h'] += ans.get('mse_1h', 0.0)
                acc['mae_1h'] += ans.get('mae_1h', 0.0)
                acc['rmsd_1h'] += ans.get('rmsd_1h', 0.0)
        d13c = cot.get('depth', {}).get('13c', {})
        d1h = cot.get('depth', {}).get('1h', {})
        brics = cot.get('brics', {})
        acc['cot_13c_tani'] += d13c.get('average_tanimoto', 0.0)
        acc['cot_1h_tani'] += d1h.get('average_tanimoto', 0.0)
        acc['cot_brics_tani'] += brics.get('average_tanimoto', 0.0)
        if d13c.get('count', 0) > 0:
            acc['cot_13c_acc_sum'] += d13c.get('correct_count', 0) / d13c.get('count')
        if d1h.get('count', 0) > 0:
            acc['cot_1h_acc_sum'] += d1h.get('correct_count', 0) / d1h.get('count')
        if brics.get('count', 0) > 0:
            acc['cot_brics_acc_sum'] += brics.get('correct_count', 0) / brics.get('count')
    results = {}
    results['total_samples'] = total_count
    results['valid_samples_for_deta_13c_count'] = valid_joint_sample_count
    results['valid_samples_for_deta_1h_count'] = valid_joint_sample_count
    results['valid_samples_for_error_metrics'] = filtered_count_for_error_metrics
    if valid_joint_sample_count > 0:
        results['average_deta_13C_nmr_shift_count'] = acc['deta_13c_cnt'] / valid_joint_sample_count
        results['average_deta_1H_nmr_shift_count'] = acc['deta_1h_cnt'] / valid_joint_sample_count
    else:
        results['average_deta_13C_nmr_shift_count'] = 0.0
        results['average_deta_1H_nmr_shift_count'] = 0.0
    results['average_cot_13c_tanimoto'] = acc['cot_13c_tani'] / total_count
    results['average_cot_1h_tanimoto'] = acc['cot_1h_tani'] / total_count
    results['average_cot_brics_tanimoto'] = acc['cot_brics_tani'] / total_count
    results['average_cot_13c_accuracy'] = acc['cot_13c_acc_sum'] / total_count
    results['average_cot_1h_accuracy'] = acc['cot_1h_acc_sum'] / total_count
    results['average_cot_brics_accuracy'] = acc['cot_brics_acc_sum'] / total_count
    if filtered_count_for_error_metrics > 0:
        for m in ['mae', 'mse', 'rmsd']:
            for iso in ['13c', '1h']:
                key = f'{m}_{iso}'
                metric_name = f'average_{m}_zero_padding_{iso.upper()}_nmr_shift'
                results[metric_name] = acc[key] / filtered_count_for_error_metrics
    else:
        for m in ['mae', 'mse', 'rmsd']:
            for iso in ['13C', '1H']:
                results[f'average_{m}_zero_padding_{iso}_nmr_shift'] = 0.0
    return results

def judge_task_from_answer_by_keyword(list_answer, dict_key_task_value_list_task_keyword=None):
    """
    根据list_answer[0]判断任务类型。
    修改规则: 
    1. 使用 nmr_forward=["Formula"], retrosyn=["Reactant", "F1"], syn=["Product"] 进行判断
    2. 如果都不匹配，兜底返回 nmr (通常是纯SMILES输出的结构解析任务)
    """
    if dict_key_task_value_list_task_keyword is None:
        dict_key_task_value_list_task_keyword = {'nmr_forward': ['Formula'], 'nmr': [], 'retrosyn': ['Reactant', 'F1'], 'syn': ['Product']}
    if not list_answer:
        raise ValueError('list_answer不能为空')
    answer_text = str(list_answer[0])
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        if not keywords:
            continue
        for kw in keywords:
            if kw in answer_text:
                return task
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        if not keywords:
            return task
    raise ValueError('未匹配到任何任务，且未定义默认兜底任务（关键词为空列表的任务）')

def _replace_reactant_placeholders(content: str) -> str:
    """ 
    将文本中的 [ReactantN] 占位符替换为 SMILES 的点分隔表示。
    """

    def _reactant_repl(m: re.Match) -> str:
        idx = m.group(1)
        return '' if idx == '1' else '.'
    content = re.sub('\\[Reactant(\\d+)\\]', _reactant_repl, content)
    return content

def _replace_product_placeholders(content: str) -> str:
    """ 
    将文本中的 [ProductN] 占位符替换为 SMILES 的点分隔表示。
    
    参数:
        content: 包含占位符的文本
        
    返回值:
        str: 替换后的文本
        
    示例:
        输入: "[Product1].[Product2]"
        输出: ".[Product2]"  (第一个替换为空，第二个开始替换为.)
    """
    import re
    pattern = '\\[Product\\d*\\]'
    matches = list(re.finditer(pattern, content))
    if not matches:
        return content
    result = content
    for i, match in enumerate(reversed(matches)):
        pos = match.start()
        end = match.end()
        replacement = '' if i == len(matches) - 1 else '.'
        result = result[:pos] + replacement + result[end:]
    return result

def _to_morgan_fp(smiles: str, radius: int=2, n_bits: int=1024, use_chirality: bool=False):
    """
    将 SMILES 转为 RDKit 分子并计算 Morgan 指纹。
    参数:
        smiles (str): SMILES 字符串
        radius (int): Morgan 指纹半径，默认 2
        n_bits (int): 指纹位数，默认 1024
        use_chirality (bool): 是否考虑手性信息
    返回:
        (mol, fp): 若解析失败返回 (None, None)
    """
    if smiles is None:
        return (None, None)
    s = smiles.strip()
    if not s:
        return (None, None)
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return (None, None)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
        return (mol, fp)
    except Exception:
        return (None, None)

def judge_single_nmr_answer(infer_answer, gt_answer, dechirality=True, check_length=5000, list_false_prefix=None):
    """
    判断单个NMR答案是否正确。
    """
    if list_false_prefix is None:
        list_false_prefix = ['F_']
    if check_length is not None and len(infer_answer) > check_length:
        return False
    for prefix in list_false_prefix:
        if infer_answer.startswith(prefix):
            return False
    gt_mol = Chem.MolFromSmiles(gt_answer) if gt_answer else None
    infer_mol = Chem.MolFromSmiles(infer_answer) if infer_answer else None
    if gt_mol is None or infer_mol is None:
        return False
    if dechirality:
        gt_mol_copy = Chem.Mol(gt_mol)
        infer_mol_copy = Chem.Mol(infer_mol)
        Chem.RemoveStereochemistry(gt_mol_copy)
        Chem.RemoveStereochemistry(infer_mol_copy)
        gt_canonical = Chem.MolToSmiles(gt_mol_copy, canonical=True)
        infer_canonical = Chem.MolToSmiles(infer_mol_copy, canonical=True)
    else:
        gt_canonical = Chem.MolToSmiles(gt_mol, isomericSmiles=True, canonical=True)
        infer_canonical = Chem.MolToSmiles(infer_mol, isomericSmiles=True, canonical=True)
    return gt_canonical == infer_canonical

def remove_reaction_tags(smiles_with_tags):
    """
    移除SMILES字符串中的反应物/产物标签。
    
    处理规则:
        - 第一个标签(如[Reactant1]或[Product1])删除
        - 后续标签(如[Reactant2], [Reactant3]或[Product2], [Product3])替换为"."以连接多个分子
    
    参数:
        smiles_with_tags (str): 带标签的SMILES字符串
        
    返回值:
        str: 清理后的SMILES字符串
        
    示例:
        输入: "[Reactant1]CCO[Reactant2]CC(C)C"
        输出: "CCO.CC(C)C"
        输入: "[Product1]CCO[Product2]CC(C)C"
        输出: "CCO.CC(C)C"
    """
    if not isinstance(smiles_with_tags, str):
        return ''
    result = smiles_with_tags.strip()
    import re
    pattern = '\\[(Reactant\\d*|Product\\d*)\\]'
    matches = list(re.finditer(pattern, result))
    if not matches:
        return result
    for i in range(len(matches) - 1, -1, -1):
        match = matches[i]
        start, end = match.span()
        if i == 0:
            result = result[:start] + result[end:]
        else:
            result = result[:start] + '.' + result[end:]
    return result

def judge_single_retrosyn_answer(infer_answer, gt_answer, dechirality=True, check_length=5000, list_false_prefix=None):
    """
    判断单个逆合成答案是否正确（基于 Morgan 指纹 Tanimoto==1.0 的等价判定）。

    参数:
        infer_answer (str): 推理得到的答案（SMILES 字符串，可能带标签，如 [Reactant1]SMILES1[Reactant2]SMILES2）
        gt_answer (str):    正确答案（可能带标签的SMILES字符串）
        dechirality (bool): 是否忽略手性信息。True=忽略（默认），False=考虑手性
        check_length (int or None): 长度阈值，超过此长度的答案视为错误，默认 5000；None 表示不检查
        list_false_prefix (list[str] or None): 以列表中任一前缀开头的答案视为错误，默认 ["F_"]

    返回值:
        bool: 推理答案是否正确（Tanimoto 相似度==1.0）
    """
    if list_false_prefix is None:
        list_false_prefix = ['F_']
    if check_length is not None and isinstance(infer_answer, str) and (len(infer_answer) > check_length):
        return False
    if isinstance(infer_answer, str):
        infer_clean = remove_reaction_tags(infer_answer)
        for prefix in list_false_prefix:
            if infer_clean.startswith(prefix):
                return False
    else:
        return False
    try:
        if not isinstance(gt_answer, str):
            return False
        gt_clean = remove_reaction_tags(gt_answer)
        infer_clean = remove_reaction_tags(infer_answer)
        if '>>' in gt_clean:
            gt_product = gt_clean.split('>>', 1)[0].strip()
        else:
            gt_product = gt_clean.strip()
        if not gt_product or not infer_clean:
            return False
    except Exception:
        return False
    use_chirality = not dechirality
    _, gt_fp = _to_morgan_fp(gt_product, radius=2, n_bits=1024, use_chirality=use_chirality)
    _, infer_fp = _to_morgan_fp(infer_clean, radius=2, n_bits=1024, use_chirality=use_chirality)
    if gt_fp is None or infer_fp is None:
        return False
    try:
        tanimoto = DataStructs.FingerprintSimilarity(gt_fp, infer_fp)
        return tanimoto == 1.0
    except Exception:
        return False

def judge_single_syn_answer(infer_answer, gt_answer, dechirality=True, check_length=5000, list_false_prefix=None):
    """
    判断单个合成答案是否正确（基于 Morgan 指纹 Tanimoto==1.0 的等价判定）。

    参数:
        infer_answer (str): 推理得到的答案（SMILES 字符串，可能带标签，如 [Product1]SMILES1[Product2]SMILES2）
        gt_answer (str):    正确答案（可能带标签的SMILES字符串）
        dechirality (bool): 是否忽略手性信息。True=忽略（默认），False=考虑手性
        check_length (int or None): 长度阈值，超过此长度的答案视为错误，默认 5000；None 表示不检查
        list_false_prefix (list[str] or None): 以列表中任一前缀开头的答案视为错误，默认 ["F_"]

    返回值:
        bool: 推理答案是否正确（Tanimoto 相似度==1.0）
    """
    if list_false_prefix is None:
        list_false_prefix = ['F_']
    if check_length is not None and isinstance(infer_answer, str) and (len(infer_answer) > check_length):
        return False
    if isinstance(infer_answer, str):
        infer_clean = remove_reaction_tags(infer_answer)
        for prefix in list_false_prefix:
            if infer_clean.startswith(prefix):
                return False
    else:
        return False
    try:
        if not isinstance(gt_answer, str):
            return False
        gt_clean = remove_reaction_tags(gt_answer)
        infer_clean = remove_reaction_tags(infer_answer)
        if '>>' in gt_clean:
            gt_product = gt_clean.split('>>', 1)[1].strip()
        else:
            gt_product = gt_clean.strip()
        if not gt_product or not infer_clean:
            return False
    except Exception:
        return False
    use_chirality = not dechirality
    _, gt_fp = _to_morgan_fp(gt_product, radius=2, n_bits=1024, use_chirality=use_chirality)
    _, infer_fp = _to_morgan_fp(infer_clean, radius=2, n_bits=1024, use_chirality=use_chirality)
    if gt_fp is None or infer_fp is None:
        return False
    try:
        tanimoto = DataStructs.FingerprintSimilarity(gt_fp, infer_fp)
        return tanimoto == 1.0
    except Exception:
        return False

def calculate_concatenated_fingerprint_similarity(smiles_list1: List[str], smiles_list2: List[str], dechirality: bool=True, fp_radius: int=2, fp_nbits: int=1024) -> float:
    """
    方法1：串联指纹法 - 计算多分子集合的Tanimoto相似度（推荐）
    
    原理：
        将每个集合中所有分子的指纹串联起来，形成一个大指纹
        然后计算两个大指纹的Tanimoto相似度
    
    参数:
        smiles_list1: 第一个分子列表（ground truth）
        smiles_list2: 第二个分子列表（预测结果）
        dechirality: 是否去除手性
        fp_radius: Morgan指纹的半径，默认2
        fp_nbits: 每个分子指纹的位数，默认1024
    
    返回值:
        float: Tanimoto相似度 [0, 1]，失败返回0.0
    
    示例:
        gt = ["CCO", "CC(C)C"]
        pred = ["CCO", "CCC"]
        sim = calculate_concatenated_fingerprint_similarity(gt, pred)
    """
    try:
        fp1_concat = None
        for smiles in smiles_list1:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            if fp1_concat is None:
                fp1_concat = fp
            else:
                fp1_concat = fp1_concat | fp
        fp2_concat = None
        for smiles in smiles_list2:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            if fp2_concat is None:
                fp2_concat = fp
            else:
                fp2_concat = fp2_concat | fp
        if fp1_concat is not None and fp2_concat is not None:
            return DataStructs.TanimotoSimilarity(fp1_concat, fp2_concat)
        else:
            return 0.0
    except Exception:
        return 0.0

def calculate_average_pairwise_similarity(smiles_list1: List[str], smiles_list2: List[str], dechirality: bool=True, fp_radius: int=2, fp_nbits: int=1024) -> float:
    """
    方法2：平均相似度法 - 最优配对后计算平均Tanimoto相似度
    
    原理：
        1. 计算所有可能的分子对之间的相似度矩阵
        2. 使用贪心算法找到最优配对
        3. 计算配对相似度的平均值
    
    参数:
        smiles_list1: 第一个分子列表
        smiles_list2: 第二个分子列表
        dechirality: 是否去除手性
        fp_radius: Morgan指纹的半径
        fp_nbits: 指纹位数
    
    返回值:
        float: 平均Tanimoto相似度 [0, 1]
    
    注意:
        如果两个列表长度不同，会对较短的列表进行惩罚
    """
    try:
        if len(smiles_list1) != len(smiles_list2):
            min_len = min(len(smiles_list1), len(smiles_list2))
            if min_len == 0:
                return 0.0
        fps1 = []
        for smiles in smiles_list1:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            fps1.append(fp)
        fps2 = []
        for smiles in smiles_list2:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            fps2.append(fp)
        n1, n2 = (len(fps1), len(fps2))
        sim_matrix = np.zeros((n1, n2))
        for i, fp1 in enumerate(fps1):
            for j, fp2 in enumerate(fps2):
                sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fp1, fp2)
        used1 = set()
        used2 = set()
        total_sim = 0.0
        pairs = min(n1, n2)
        for _ in range(pairs):
            max_sim = -1
            best_i, best_j = (-1, -1)
            for i in range(n1):
                if i in used1:
                    continue
                for j in range(n2):
                    if j in used2:
                        continue
                    if sim_matrix[i, j] > max_sim:
                        max_sim = sim_matrix[i, j]
                        best_i, best_j = (i, j)
            if best_i != -1:
                used1.add(best_i)
                used2.add(best_j)
                total_sim += max_sim
        avg_sim = total_sim / max(n1, n2)
        return avg_sim
    except Exception:
        return 0.0

def calculate_retrosyn_tanimoto_similarity(gt_answer: str, pred_answer: str, method: str='concatenated', dechirality: bool=True) -> float:
    """
    计算逆合成任务的Tanimoto相似度（统一接口）
    
    参数:
        gt_answer: ground truth答案（格式：smiles1.smiles2）
        pred_answer: 预测答案（格式：smiles3.smiles4）
        method: 计算方法
            - "concatenated": 串联指纹法（默认，推荐）
            - "average": 平均相似度法（更精细）
        dechirality: 是否去除手性
    
    返回值:
        float: Tanimoto相似度 [0, 1]
    
    使用示例:
        sim = calculate_retrosyn_tanimoto_similarity(
            "CCO.CC(C)C",
            "CCO.CCC",
            method="concatenated"
        )
    """
    try:
        gt_parts = [s.strip() for s in gt_answer.split('.') if s.strip()]
        pred_parts = [s.strip() for s in pred_answer.split('.') if s.strip()]
        if not gt_parts or not pred_parts:
            return 0.0
        if method == 'concatenated':
            return calculate_concatenated_fingerprint_similarity(gt_parts, pred_parts, dechirality=dechirality)
        elif method == 'average':
            return calculate_average_pairwise_similarity(gt_parts, pred_parts, dechirality=dechirality)
        else:
            raise ValueError(f'不支持的方法: {method}')
    except Exception:
        return 0.0

def test_model_predict_result(arrow_folder_path, model_predict_result_txt_file_path, dict_key_task_value_list_task_keyword=None, dechirality=True, check_length=5000, list_false_prefix=None, with_answer_for_deta_count=True, max_length_for_deta_count=5000 * 6):
    """
    测试模型预测结果（集成 NMR Forward 逻辑）。
    
    参数:
        arrow_folder_path (str): Arrow 数据集文件夹路径
        model_predict_result_txt_file_path (str): 模型预测结果 txt 文件路径
        dict_key_task_value_list_task_keyword (dict): 任务关键词映射
        dechirality (bool): 是否忽略手性
        check_length (int): 长度检查阈值
        list_false_prefix (list): 错误前缀列表
        with_answer_for_deta_count (bool): [NMR Forward] 是否仅对有答案的样本计算delta count
        max_length_for_deta_count (int): [NMR Forward] 答案最大长度阈值，超过则过滤delta count
        
    返回:
        dict: 评估统计结果
    """
    if list_false_prefix is None:
        list_false_prefix = ['F_']
    print(f'正在加载数据: {arrow_folder_path}')
    list_gt_question, list_gt_cot, list_gt_answer = load_question_cot_answer_list_from_arrow_folder(arrow_folder_path)
    str_task = judge_task_from_answer_by_keyword(list_gt_answer, dict_key_task_value_list_task_keyword=dict_key_task_value_list_task_keyword)
    print(f'检测到任务类型: {str_task}')
    if str_task == 'nmr_forward':
        print(f'正在执行 NMR Forward 评估 (使用 calc_global_stats_from_memory)')
        print(f'正在加载预测文件: {model_predict_result_txt_file_path}')
        if with_answer_for_deta_count:
            print('  - 启用过滤: 仅统计有答案的样本的 delta count')
        print(f'  - 答案长度阈值: {max_length_for_deta_count}')
        list_dict_main = []
        gen = read_model_predictions(model_predict_result_txt_file_path)
        for idx, (gt_q, gt_c, gt_a) in enumerate(zip(list_gt_question, list_gt_cot, list_gt_answer)):
            try:
                line = next(gen)
                str_gen_raw = extract_str_generate_result_without_prompt(line)
                gen_c = extract_str_cot(str_gen_raw)
                gen_a = extract_str_answer(str_gen_raw)
                dict_cot_gt = str_cot_to_dict(gt_c) if isinstance(gt_c, str) else {'depth': {'13c': {}, '1h': {}}, 'brics': {}}
                dict_cot_gen = str_cot_to_dict(gen_c)
                dict_answer_info = stat_nmr_forward_answer_info(gt_a, gen_a, with_answer_for_deta_count=with_answer_for_deta_count, max_length_for_deta_count=max_length_for_deta_count)
                dict_cot_info = stat_dict_cot_generate_info(dict_cot_gt, dict_cot_gen, dechirality=dechirality)
                list_dict_main.append({'answer': dict_answer_info, 'cot': dict_cot_info})
            except StopIteration:
                print(f'警告: 预测文件行数少于 GT 样本数，在第 {idx} 行停止。')
                break
            except Exception as e:
                print(f'警告: 处理第 {idx} 行时出错: {e}')
                list_dict_main.append({'answer': {}, 'cot': {}})
        stats = calc_global_stats_from_memory(list_dict_main, delta_1h_count=0, delta_13c_count=0)
        print(f'\nNMR Forward 评估摘要:')
        print(f"Avg 13C MAE (Filter |Δ|=0): {stats.get('average_mae_zero_padding_13C_nmr_shift', 0):.4f}")
        print(f"Avg 1H MAE (Filter |Δ|=0):  {stats.get('average_mae_zero_padding_1H_nmr_shift', 0):.4f}")
        print(f"Valid Samples for Error:    {stats.get('valid_samples_for_error_metrics', 0)}")
        return stats
    else:
        if str_task == 'retrosyn':
            print('逆合成任务：正在预处理ground truth答案...')
            list_gt_answer = [_replace_reactant_placeholders(ans) for ans in list_gt_answer]
        print(f'正在加载预测结果: {model_predict_result_txt_file_path}')
        list_generate_answer = read_list_generate_result_from_txt(model_predict_result_txt_file_path)
        if str_task == 'retrosyn':
            print('逆合成任务：正在预处理预测答案...')
            list_generate_answer = [_replace_reactant_placeholders(ans) for ans in list_generate_answer]
        if len(list_gt_answer) != len(list_generate_answer):
            print(f'警告: 答案数量不匹配 (GT: {len(list_gt_answer)}, Pred: {len(list_generate_answer)})')
        stats = calculate_smiles_validity_accuracy_from_ref_list_and_pred_list(list_gt_answer, list_generate_answer, str_task=str_task, dechirality=dechirality, check_length=check_length, list_false_prefix=list_false_prefix)
        print(f"\n{'=' * 50}")
        print(f'任务类型: {str_task.upper()}')
        print(f"{'=' * 50}")
        for key, value in stats.items():
            if value is not None:
                print(f'{key}: {value:.4f}')
        print(f"{'=' * 50}\n")
        return stats

def evaluate_multiple_test_files(save_folder_path, list_save_file_name, dict_key_test_arrow_folder_path_value_dataset_name, with_answer_for_deta_count=True, max_length_for_deta_count=5000 * 6):
    """
    批量评估多个测试文件和预测结果文件。
    
    参数:
        save_folder_path: 存储txt结果文件的文件夹路径
        list_save_file_name: 结果文件的文件名列表（不包含.txt后缀）
        dict_key_test_arrow_folder_path_value_dataset_name: 字典，键是测试文件夹路径，值是数据集名称
        
        
    返回值:
        all_stats: 包含所有评估统计结果的字典，键为测试文件夹路径
    """
    list_test_arrow_folder_path = list(dict_key_test_arrow_folder_path_value_dataset_name.keys())
    if len(list_test_arrow_folder_path) != len(list_save_file_name):
        raise ValueError(f'测试文件夹列表和保存文件名列表的长度不匹配, len(list_test_arrow_folder_path):{len(list_test_arrow_folder_path)}, len(list_save_file_name):{len(list_save_file_name)}')
    list_model_predict_result_txt_file_path = [os.path.join(save_folder_path, file_name + '.txt') for file_name in list_save_file_name]
    all_stats = {}
    for i, (arrow_folder_path, txt_file_path) in enumerate(zip(list_test_arrow_folder_path, list_model_predict_result_txt_file_path)):
        txt_file_path = remove_extra_txt_extension(txt_file_path)
        print(f'\n评估文件 {i + 1}/{len(list_test_arrow_folder_path)}:')
        print(f'Arrow文件夹: {arrow_folder_path}')
        print(f'预测结果文件: {txt_file_path}')
        stats = test_model_predict_result(arrow_folder_path, txt_file_path, with_answer_for_deta_count=with_answer_for_deta_count, max_length_for_deta_count=max_length_for_deta_count)
        all_stats[arrow_folder_path] = stats
    print('\n所有文件的评估已完成')
    return all_stats

def batch_evaluate_multiple_test_files(save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_as_json_and_csv=True, save_result_filename_prefix=None, with_answer_for_deta_count=True, max_length_for_deta_count=5000 * 6):
    """
    批量评估多个权重模型对多个测试文件的预测结果。
    
    更新：支持单 Checkpoint 模式（直接在目录下寻找 txt 文件）。
    """
    try:
        list_result_folder_path = get_subdirectories(save_folder_path)
    except ValueError as e:
        list_result_folder_path = []
    if not list_result_folder_path:
        files_in_root = get_files_with_suffix(save_folder_path, '.txt', traverse_sub_folder=False)
        if files_in_root:
            print(f'检测到单 Checkpoint 模式：在 {save_folder_path} 下直接发现 {len(files_in_root)} 个 txt 文件')
            list_result_folder_path = [save_folder_path]
        else:
            print(f'警告：在 {save_folder_path} 中既没有子文件夹，也没有直接的 .txt 结果文件。')
            return {}
    train_info = os.path.basename(os.path.normpath(save_folder_path))
    dict_key_int_train_step_value_one_result_folder_dict_stat = {}
    for result_folder_path in list_result_folder_path:
        if result_folder_path == save_folder_path:
            train_step = 'final'
        else:
            train_step = os.path.basename(os.path.normpath(result_folder_path))
        print(f'\n正在处理 Checkpoint/Step: {train_step} (路径: {result_folder_path})')
        list_txt_file_path_unsorted = get_files_with_suffix(result_folder_path, '.txt', traverse_sub_folder=False)
        if not list_txt_file_path_unsorted:
            print(f'  警告: {result_folder_path} 中没有 .txt 文件，跳过')
            continue
        list_txt_file_name_sorted_index = []
        valid_files = True
        valid_txt_paths = []
        valid_indices = []
        for txt_file_path_unsorted in list_txt_file_path_unsorted:
            txt_file_name_with_suffix = os.path.basename(txt_file_path_unsorted)
            dataset_name = extract_dataset_name_from_txt_file_name_with_suffix(txt_file_name_with_suffix)
            if dataset_name is None:
                continue
            txt_file_name_index = get_key_index_by_value(dict_key_test_arrow_folder_path_value_dataset_name, dataset_name)
            if txt_file_name_index is not None:
                valid_indices.append(txt_file_name_index)
                valid_txt_paths.append(txt_file_path_unsorted)
            else:
                pass
        if not valid_txt_paths:
            print(f'  警告: 在 {result_folder_path} 中未找到任何匹配目标数据集的 txt 文件')
            continue
        list_txt_file_path_sorted = [x for _, x in sorted(zip(valid_indices, valid_txt_paths))]
        one_result_folder_dict_stat = evaluate_multiple_test_files(result_folder_path, list_txt_file_path_sorted, dict_key_test_arrow_folder_path_value_dataset_name, with_answer_for_deta_count=with_answer_for_deta_count, max_length_for_deta_count=max_length_for_deta_count)
        dict_key_int_train_step_value_one_result_folder_dict_stat[train_step] = replace_keys_with_values(one_result_folder_dict_stat, dict_key_test_arrow_folder_path_value_dataset_name)
    if save_result_filename_prefix is None and save_result_as_json_and_csv:
        save_result_filename_prefix = train_info
    if save_result_as_json_and_csv and dict_key_int_train_step_value_one_result_folder_dict_stat:
        print('\n正在保存评估汇总结果...')
        try:
            generated_files_list = save_evaluation_results_as_json_and_csv(dict_key_int_train_step_value_one_result_folder_dict_stat, save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_filename_prefix)
            print(f'成功生成以下文件:')
            for f in generated_files_list:
                print(f'  - {f}')
        except Exception as e:
            print(f'保存结果时出错: {e}')
            import traceback
            traceback.print_exc()
    return dict_key_int_train_step_value_one_result_folder_dict_stat

# GENERATION

def get_immediate_subfolders(target_folder_path):
    """
    Get all immediate subfolder paths under the specified folder.
    
    Args:
        target_folder_path (str): The target parent folder path.
        
    Returns:
        list: A list containing full paths of all immediate subfolders. 
              Returns an empty list if the target path does not exist.
    """
    if not os.path.exists(target_folder_path):
        print(f"Error: Folder '{target_folder_path}' does not exist.")
        return []
    subfolders = []
    try:
        with os.scandir(target_folder_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    subfolders.append(entry.path)
    except Exception as e:
        print(f'Error reading folder: {e}')
        return []
    subfolders.sort()
    return subfolders

def get_subdirectories_with_certain_prefix(path, str_prefix):
    """
    Get a list of immediate subfolder paths under the specified path that start with a specific prefix.
    
    Args:
        path (str): The path to search for subfolders.
        str_prefix (str): The prefix of the subfolder names.
    
    Returns:
        list: A list of immediate subfolder paths matching the prefix condition.
    
    Usage:
        subdirs = get_subdirectories_with_certain_prefix('your_path_here', 'data_')
        print(subdirs)
    """
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
    subdirectories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name.startswith(str_prefix)]
    return subdirectories

def extract_checkpoint_iters(list_model_checkpoint_folder_path):
    """
    Extract the numeric part from checkpoint folder paths and convert them to a list of strings.

    Args:
        list_model_checkpoint_folder_path: List of checkpoint paths, where the last part should look like 'checkpoint-1000'.

    Returns:
        list_checkpoint_iter: A list of extracted iteration numbers (as strings).
    """
    list_checkpoint_iter = []
    for folder_path in list_model_checkpoint_folder_path:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        match = re.match('checkpoint-(\\d+)', folder_name)
        if match:
            list_checkpoint_iter.append(match.group(1))
    return list_checkpoint_iter

def get_architecture_from_config(model_checkpoint_folder_path: str) -> str:
    config_path = os.path.join(model_checkpoint_folder_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.json does not exist at path: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    if 'architectures' not in config_data:
        raise KeyError('"architectures" field not found in config.json')
    architectures = config_data['architectures']
    if not isinstance(architectures, list) or len(architectures) == 0:
        raise ValueError('"architectures" field should be a non-empty list')
    return architectures[0]

def detect_qwen_model_type(model_checkpoint_folder_path):
    """
    Determine the model type based on the value of the "architectures" key in the config.json file
    under model_checkpoint_folder_path, to facilitate subsequent question mapping.
    
    Args:
    model_checkpoint_folder_path: The folder path of the model files. Note: this should be the full model path, not the LoRA folder path.

    Returns:
    bool_qwen3_model: True indicates a Qwen3 model, False indicates a Qwen2 model (Deepseek R1 distilled small model). Raises an error if neither.
    """
    str_version = get_architecture_from_config(model_checkpoint_folder_path)
    if 'Qwen3' in str_version:
        bool_qwen3_model = True
    elif 'Qwen2' in str_version:
        bool_qwen3_model = False
    else:
        raise ValueError(f'The "architectures" field in config.json at {model_checkpoint_folder_path} indicates this model is not a supported Qwen2 or Qwen3 model.')
    return bool_qwen3_model

def build_save_folder_paths(parent_parent_save_folder_path, model_train_info, list_checkpoint_iter):
    """
    Construct a list of save path strings with the structure:
        parent_parent_save_folder_path / model_train_info / checkpoint_iter

    Args:
        parent_parent_save_folder_path: Top-level save directory (string or Path).
        model_train_info: Name of the intermediate folder (string).
        list_checkpoint_iter: List of checkpoint numbers (list of strings).

    Returns:
        list_save_folder_path: A list where each path is a string.
    """
    parent_parent_save_folder_path = Path(parent_parent_save_folder_path)
    list_save_folder_path = [str(parent_parent_save_folder_path / model_train_info / checkpoint_iter) for checkpoint_iter in list_checkpoint_iter]
    return list_save_folder_path

def build_save_file_names(model_train_info, list_checkpoint_iter, dict_arrow_folder_path_save_file_name):
    """
    Construct a list of save file names with the format:
        {model_train_info}_{checkpoint_iter}_step_{save_file_name}

    Args:
        model_train_info: Model test info string.
        list_checkpoint_iter: List of checkpoint numbers (list of strings).
        dict_arrow_folder_path_save_file_name: Dictionary mapping arrow paths to save file names.

    Returns:
        list_save_file_name: A list of constructed save file name strings.
    """
    list_save_file_name_raw = list(dict_arrow_folder_path_save_file_name.values())
    if len(list_checkpoint_iter) != len(list_save_file_name_raw):
        raise ValueError('Number of checkpoints does not match number of save file names')
    list_save_file_name = [f'{model_train_info}_{list_checkpoint_iter[i]}_step_{list_save_file_name_raw[i]}' for i in range(len(list_checkpoint_iter))]
    return list_save_file_name

def get_folder_name_by_level(folder_path: str, level: int) -> str:
    """
    Get the folder name based on the specified level.
    
    Args:
        folder_path (str): Full folder path.
        level (int): Level to go up. 0 means current folder name, 1 means parent folder name, and so on.
    
    Returns:
        str: Folder name at the specified level.
    """
    path_obj = Path(folder_path)
    if level < 0:
        raise ValueError(f'Level must be a non-negative integer, current value: {level}')
    parts = path_obj.parts
    if level >= len(parts):
        raise ValueError(f"level={level} exceeds path depth, path '{folder_path}' only has {len(parts)} levels")
    return parts[-(level + 1)]

def transform_chem_question_to_nat_question(question, language='en'):
    """
    Transform a chemical question into natural language format.
    
    Args:
        question: String containing the chemical question.
        language: Output language, "zh" for Chinese, "en" for English. Defaults to "zh" (logic maintained, though typically called with "en").
        
    Returns:
        nat_question: Transformed natural language question string.
    """
    formula_match = re.search('Formula:(.*?);', question)
    formula = formula_match.group(1) if formula_match else None
    fw_match = re.search('FW:(.*?);', question)
    fw = fw_match.group(1) if fw_match else None
    hsqc_match = re.search('HSQC:(.*?);', question)
    hsqc = hsqc_match.group(1) if hsqc_match else None
    hmbc_match = re.search('HMBC:(.*?);', question)
    hmbc = hmbc_match.group(1) if hmbc_match else None
    cosy_match = re.search('COSY:(.*?);', question)
    cosy = cosy_match.group(1) if cosy_match else None
    cnmr_match = re.search('13C NMR:\\[NCP:(\\d+)\\](.*?);', question)
    c_nmr_count = cnmr_match.group(1) if cnmr_match else None
    c_nmr_shifts = cnmr_match.group(2).split(',') if cnmr_match and cnmr_match.group(2) else []
    hnmr_match = re.search('1H NMR:\\[NHP:(\\d+)\\](.*?)($|;)', question)
    h_nmr_count = hnmr_match.group(1) if hnmr_match else None
    h_nmr_signals_str = hnmr_match.group(2) if hnmr_match else ''
    has_mask_in_hnmr = '[MASK]' in h_nmr_signals_str
    result_parts = []
    if language == 'zh':
        if formula and fw and (formula != '[MASK]') and (fw != '[MASK]'):
            result_parts.append(f'该分子的分子式为{formula}，分子量为{fw}')
        else:
            if formula == '[MASK]' or formula is None:
                result_parts.append('分子式信息未给出')
            elif formula:
                result_parts.append(f'该分子的分子式为{formula}')
            if fw == '[MASK]' or fw is None:
                result_parts.append('分子量信息未给出')
            elif fw:
                result_parts.append(f'分子量为{fw}')
        if hsqc == '[MASK]' and hmbc == '[MASK]' and (cosy == '[MASK]'):
            result_parts.append('HSQC、HMBC、COSY信息未给出')
        else:
            if hsqc == '[MASK]':
                result_parts.append('HSQC信息未给出')
            elif hsqc:
                result_parts.append(f'HSQC信息为{hsqc}')
            if hmbc == '[MASK]':
                result_parts.append('HMBC信息未给出')
            elif hmbc:
                result_parts.append(f'HMBC信息为{hmbc}')
            if cosy == '[MASK]':
                result_parts.append('COSY信息未给出')
            elif cosy:
                result_parts.append(f'COSY信息为{cosy}')
        if c_nmr_count == '[MASK]' or (c_nmr_count and len(c_nmr_shifts) == 1 and (c_nmr_shifts[0] == '[MASK]')):
            result_parts.append('13C NMR信息未给出')
        elif c_nmr_count and c_nmr_shifts:
            result_parts.append(f"13C NMR一共有{c_nmr_count}个化学位移信号，分别为：{','.join(c_nmr_shifts)}")
        if h_nmr_count == '[MASK]' or (h_nmr_count and h_nmr_signals_str == '[MASK]'):
            result_parts.append('1H NMR信息未给出')
        elif h_nmr_count:
            if h_nmr_signals_str:
                result_parts.append(f'1H NMR一共有{h_nmr_count}个信号，以(chemical shift,integration,(multiplicity,coupling patterns))的形式分别表示为{h_nmr_signals_str}')
            else:
                result_parts.append(f'1H NMR一共有{h_nmr_count}个信号')
            if has_mask_in_hnmr:
                result_parts.append('其中[MASK]表示该信息未给出')
        result_parts.append('根据以上信息推理分子的SMILES:')
    else:
        if formula and fw and (formula != '[MASK]') and (fw != '[MASK]'):
            result_parts.append(f'The molecular formula is {formula} with molecular weight of {fw}')
        else:
            if formula == '[MASK]' or formula is None:
                result_parts.append('The molecular formula is not available')
            elif formula:
                result_parts.append(f'The molecular formula is {formula}')
            if fw == '[MASK]' or fw is None:
                result_parts.append('The molecular weight is not available')
            elif fw:
                result_parts.append(f'The molecular weight is {fw}')
        if hsqc == '[MASK]' and hmbc == '[MASK]' and (cosy == '[MASK]'):
            result_parts.append('HSQC, HMBC, and COSY data are not available')
        else:
            if hsqc == '[MASK]':
                result_parts.append('HSQC data is not available')
            elif hsqc:
                result_parts.append(f'HSQC data is {hsqc}')
            if hmbc == '[MASK]':
                result_parts.append('HMBC data is not available')
            elif hmbc:
                result_parts.append(f'HMBC data is {hmbc}')
            if cosy == '[MASK]':
                result_parts.append('COSY data is not available')
            elif cosy:
                result_parts.append(f'COSY data is {cosy}')
        if c_nmr_count == '[MASK]' or (c_nmr_count and len(c_nmr_shifts) == 1 and (c_nmr_shifts[0] == '[MASK]')):
            result_parts.append('13C NMR data is not available')
        elif c_nmr_count and c_nmr_shifts:
            result_parts.append(f"The 13C NMR spectrum shows {c_nmr_count} chemical shift signals: {','.join(c_nmr_shifts)}")
        if h_nmr_count == '[MASK]' or (h_nmr_count and h_nmr_signals_str == '[MASK]'):
            result_parts.append('1H NMR data is not available')
        elif h_nmr_count:
            if h_nmr_signals_str:
                result_parts.append(f'The 1H NMR spectrum exhibits {h_nmr_count} signals in the format (chemical shift, integration, (multiplicity, coupling patterns)): {h_nmr_signals_str}')
            else:
                result_parts.append(f'The 1H NMR spectrum exhibits {h_nmr_count} signals')
            if has_mask_in_hnmr:
                result_parts.append('Where [MASK] indicates that the information is not provided')
        result_parts.append('Based on the above information, infer the molecular SMILES:')
    nat_question = '。'.join(result_parts) if language == 'zh' else '. '.join(result_parts)
    return nat_question

def standardize_shift_decimal_place(question, adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2):
    """
    Standardize the decimal places of NMR chemical shift data.
    Args:
        question: String containing NMR data.
        c_shift_decimal_place: Decimal places for 13C NMR chemical shift, default 1.
        h_shift_decimal_place: Decimal places for 1H NMR chemical shift, default 2.
    Returns:
        processed_question: The question string after decimal place standardization.
    """
    if not adjust_decimal_places:
        return question
    processed_question = question
    c_nmr_pattern = '(13C NMR:\\[NCP:\\d+\\])([\\d.,]+)(;)'
    c_nmr_match = re.search(c_nmr_pattern, processed_question)
    if c_nmr_match:
        prefix = c_nmr_match.group(1)
        shift_values = c_nmr_match.group(2)
        suffix = c_nmr_match.group(3)
        values = shift_values.split(',')
        formatted_values = []
        for value in values:
            try:
                float_value = float(value.strip())
                formatted_value = f'{float_value:.{c_shift_decimal_place}f}'
                formatted_values.append(formatted_value)
            except ValueError:
                formatted_values.append(value.strip())
        new_c_nmr = prefix + ','.join(formatted_values) + suffix
        processed_question = processed_question.replace(c_nmr_match.group(0), new_c_nmr)
    h_nmr_pattern = '(1H NMR:\\[NHP:\\d+\\])(.*?)(?=;|$)'
    h_nmr_match = re.search(h_nmr_pattern, processed_question)
    if h_nmr_match:
        prefix = h_nmr_match.group(1)
        tuples_content = h_nmr_match.group(2)
        tuple_pattern = '\\((\\d+\\.?\\d*),(\\d+),(\\[MASK\\])\\)'

        def replace_tuple(match):
            first_value = float(match.group(1))
            formatted_first = f'{first_value:.{h_shift_decimal_place}f}'
            return f'({formatted_first},{match.group(2)},{match.group(3)})'
        new_tuples_content = re.sub(tuple_pattern, replace_tuple, tuples_content)
        new_h_nmr = prefix + new_tuples_content
        processed_question = processed_question.replace(h_nmr_match.group(0), new_h_nmr)
    return processed_question

def mapping_list_qusetion_to_certain_form(list_question, tokenizer, bool_qwen3_model, bool_beam_search):
    """
    Prepare questions for inference (apply templates)
    1. Args:
    (1) list_question: List of questions, where each element is a question (string).
    (2) tokenizer: The tokenizer corresponding to the model.
    (3) bool_qwen3_model: Whether the model used for inference is a Qwen3 model. True for Qwen3, False for Qwen2 (Deepseek R1 distilled small model).
    (4) bool_beam_search: Whether to use beam search.

    2. Returns:
    prompt_texts: Mapped list_question.
    """
    prompt_texts = []
    for question in list_question:
        if bool_qwen3_model:
            full_prompt = f'{question}<think>'
            messages = [{'role': 'user', 'content': full_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(text)
        else:
            prompt_texts.append(f'{question}<think>')
    if bool_beam_search:
        prompt_texts = [{'prompt': p} for p in prompt_texts]
    return prompt_texts

def load_multiple_arrow_files(directory_path, split='train'):
    """
    Load multiple .arrow files from a directory and concatenate them into a single dataset.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f'Directory {directory_path} does not exist')
    arrow_files = glob.glob(os.path.join(directory_path, '*.arrow'))
    if not arrow_files:
        raise FileNotFoundError(f'No .arrow files found in directory {directory_path}')
    print(f'Found {len(arrow_files)} .arrow files in {directory_path}')
    datasets_list = []
    for file_path in arrow_files:
        print(f'Loading file: {os.path.basename(file_path)}')
        try:
            dataset = load_dataset('arrow', data_files=file_path, split=split)
            datasets_list.append(dataset)
            print(f'Successfully loaded, this file contains {len(dataset)} samples')
        except Exception as e:
            print(f'Error loading file {file_path}: {e}')
    if not datasets_list:
        raise ValueError('No datasets were successfully loaded')
    if len(datasets_list) == 1:
        return datasets_list[0]
    else:
        return concatenate_datasets(datasets_list)

def get_list_question_from_arrow_folder_path(arrow_folder_path, bool_nat_question, adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2):
    arrow_folder_datasets = load_multiple_arrow_files(arrow_folder_path)
    list_question = arrow_folder_datasets['question']
    list_question = [standardize_shift_decimal_place(q, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place) for q in list_question]
    if bool_nat_question:
        list_nat_question = []
        for chem_question in list_question:
            nat_question = transform_chem_question_to_nat_question(chem_question, language='en')
            list_nat_question.append(nat_question)
        list_question = list_nat_question
    return list_question

def _load_vllm():
    global LLM, SamplingParams, BeamSearchParams
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams

def generation_import_functions(list_tuple_py_file_abs_path_function_name):
    """
    Dynamically import specific functions from each Python file in the provided list,
    and add them to the namespace of both the calling module and the __main__ module.
    
    Args:
    list_tuple_py_file_abs_path_function_name: A list where each element is a tuple containing:
        - py_file_abs_path: Absolute path to the Python file.
        - function_name: Name of the function to import.
    """
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    caller_module_name = caller_globals['__name__']
    if caller_module_name == '__main__':
        import __main__
        caller_module = __main__
    else:
        caller_module = sys.modules[caller_module_name]
    import __main__
    main_globals = __main__.__dict__
    imported_functions = {}
    for py_file_abs_path, function_name in list_tuple_py_file_abs_path_function_name:
        try:
            module_name = f'dynamic_module_{function_name}_{hash(py_file_abs_path)}'
            spec = importlib.util.spec_from_file_location(module_name, py_file_abs_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                setattr(caller_module, function_name, func)
                caller_globals[function_name] = func
                if caller_module_name != '__main__':
                    main_globals[function_name] = func
                imported_functions[function_name] = func
                print(f"Successfully imported function '{function_name}' from file: {py_file_abs_path}")
            else:
                print(f"Warning: Function '{function_name}' not found in {py_file_abs_path}")
        except Exception as e:
            print(f"Error: Unable to load function '{function_name}' from file: {py_file_abs_path}. Error info: {e}")
    return imported_functions

def load_model_from_model_checkpoint_folder_path(model_checkpoint_folder_path):
    _load_vllm()
    llm = LLM(model=model_checkpoint_folder_path, tensor_parallel_size=len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')))
    return llm

def save_vllm_infer_result_to_jsonl_file(vllm_infer_result, save_folder_path, save_file_name):
    """
    Save vLLM inference results to a JSONL file.
    This function automatically detects whether the results were generated by the generate method or beam_search.

    Args:
        vllm_infer_result: List of vLLM inference results (generated by generate or beam_search method).
        save_folder_path: Path of the folder to save to.
        save_file_name: Name of the file to save (without extension).
    
    Returns:
        None: Function returns nothing, results are saved directly to file.
    """
    os.makedirs(save_folder_path, exist_ok=True)
    file_path = os.path.join(save_folder_path, f'{save_file_name}.jsonl')
    jsonl_data = []
    if not vllm_infer_result:
        return
    first_item = vllm_infer_result[0]
    if hasattr(first_item, 'sequences'):
        for item in vllm_infer_result:
            beam_results = []
            for sequence in item.sequences:
                beam_results.append(sequence.text)
            jsonl_data.append({'text': beam_results})
    else:
        for item in vllm_infer_result:
            generat_result = []
            for generate_output in item.outputs:
                text = item.prompt + generate_output.text
                generat_result.append(text)
            jsonl_data.append({'text': generat_result})
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_jsonl_to_txt_file(model_infer_result_folder_path):
    """
    Convert all .jsonl files in the specified path to .txt files.
    
    Args:
        model_infer_result_folder_path: Folder path containing model training info.
    
    Returns:
        None: Function returns nothing, but creates .txt files in the file system.
    """
    print(f'ctime: {time.ctime()}')
    model_train_info = os.path.basename(model_infer_result_folder_path)
    for root, dirs, files in os.walk(model_infer_result_folder_path):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, model_infer_result_folder_path)
                if rel_path == '.' or not rel_path:
                    continue
                train_iter = rel_path.split(os.sep)[0]
                dataset_name = os.path.splitext(file)[0]
                txt_file_name = f'{model_train_info}_{train_iter}_step_{dataset_name}.txt'
                txt_file_path = os.path.join(root, txt_file_name)
                print(f'Processing {jsonl_file_path} -> {txt_file_path}')
                with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_in, open(txt_file_path, 'w', encoding='utf-8') as txt_out:
                    for line_num, line in enumerate(jsonl_in, 1):
                        try:
                            data = json.loads(line.strip())
                            test_field = data.get('text', [])
                            if not isinstance(test_field, list):
                                print(f"Warning: 'text' field in line {line_num} of file {jsonl_file_path} is not a list")
                                txt_out.write(str(test_field).replace('\n', '\\n') + '\n')
                                continue
                            if not test_field:
                                print(f"Warning: 'text' list in line {line_num} of file {jsonl_file_path} is empty")
                                txt_out.write('\n')
                                continue
                            first_string = test_field[0]
                            first_string_normalized = first_string.replace('\n', '\\n').replace('\r', '\\r')
                            txt_out.write(first_string_normalized + '\n')
                        except json.JSONDecodeError:
                            print(f'Error: JSON parse error occurred at line {line_num} of file {jsonl_file_path}: {line.strip()}')
                            txt_out.write('\n')
                print(f'Converted {jsonl_file_path} to {txt_file_path}')

def evaluate_infer_result_by_convert_to_txt(parent_checkpoint_folder_path, parent_parent_save_folder_path, list_tuple_py_file_abs_path_function_name, dict_key_test_arrow_folder_path_value_dataset_name, level=0):
    print(f'time.ctime{time.ctime}')
    generation_import_functions(list_tuple_py_file_abs_path_function_name)
    checkpoint_folder_name = get_folder_name_by_level(parent_checkpoint_folder_path, level)
    save_folder_path = str(Path(parent_parent_save_folder_path) / checkpoint_folder_name)
    convert_jsonl_to_txt_file(save_folder_path)
    batch_evaluate_multiple_test_files(save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_as_json_and_csv=True)

def one_model_vllm_infer_one_dataset(model_checkpoint_folder_path, arrow_folder_path, bool_nat_question, save_folder_path='.', save_file_name='vllm_infer_result', beam_width=1, max_tokens=5000, temperature=0, top_p=0.95, top_k=20, c_shift_decimal_place=1, h_shift_decimal_place=2):
    _load_vllm()
    visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    tensor_parallel_size = len(visible_gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_folder_path)
    llm = LLM(model=model_checkpoint_folder_path, tensor_parallel_size=tensor_parallel_size)
    list_question = get_list_question_from_arrow_folder_path(arrow_folder_path, bool_nat_question=bool_nat_question, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place)
    bool_qwen3_model = detect_qwen_model_type(model_checkpoint_folder_path)
    bool_beam_search = True if beam_width > 1 else False
    list_question = mapping_list_qusetion_to_certain_form(list_question, tokenizer, bool_qwen3_model, bool_beam_search)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)
    beam_params = BeamSearchParams(beam_width=beam_width, temperature=temperature, max_tokens=max_tokens, ignore_eos=False, length_penalty=1.0, stop_token_ids=[tokenizer.eos_token_id])
    if bool_beam_search:
        outputs = llm.beam_search(list_question, beam_params)
    else:
        outputs = llm.generate(list_question, sampling_params)
    save_vllm_infer_result_to_jsonl_file(outputs, save_folder_path, save_file_name)
    return outputs

def one_model_vllm_infer_multi_dataset(model_checkpoint_folder_path, list_arrow_folder_path, list_save_file_name, bool_nat_question, save_folder_path='.', beam_width=1, max_tokens=5000, temperature=0, top_p=0.95, top_k=20, adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2):
    """
    Use the same model to infer on multiple datasets to avoid reloading the model.
    
    Args:
        model_checkpoint_folder_path: Path to model checkpoint folder
        list_arrow_folder_path: List of arrow dataset folder paths
        list_save_file_name: List of save file names corresponding to each dataset
        bool_nat_question: Whether it is a natural language question
        save_folder_path: Folder path to save results, defaults to current directory
        beam_width: Width for beam search, defaults to 1
        max_tokens: Maximum generated tokens, defaults to 5000
        temperature: Sampling temperature, defaults to 0
        top_p: top-p sampling parameter, defaults to 0.95
        top_k: top-k sampling parameter, defaults to 20
    
    Returns:
        list_infer_results: List containing inference results for multiple datasets
    """
    _load_vllm()
    if len(list_arrow_folder_path) != len(list_save_file_name):
        raise ValueError(f'Number of test files ({len(list_arrow_folder_path)}) does not match number of save file names ({len(list_save_file_name)})')
    print(f"\n{'=' * 80}")
    print(f'Starting batch inference task')
    print(f'Model path: {model_checkpoint_folder_path}')
    print(f'Number of datasets: {len(list_arrow_folder_path)}')
    print(f'Beam width: {beam_width}')
    print(f"{'=' * 80}\n")
    print(f'[Init] Loading model and tokenizer...')
    visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    tensor_parallel_size = len(visible_gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_folder_path)
    llm = LLM(model=model_checkpoint_folder_path, tensor_parallel_size=tensor_parallel_size)
    print(f'[Init] Model loading complete (Using {tensor_parallel_size} GPUs)\n')
    bool_qwen3_model = detect_qwen_model_type(model_checkpoint_folder_path)
    print(f"[Init] Model type: {('Qwen3' if bool_qwen3_model else 'Qwen2')}")
    bool_beam_search = True if beam_width > 1 else False
    print(f"[Init] Inference mode: {('Beam Search' if bool_beam_search else 'Greedy Decoding')}\n")
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)
    beam_params = BeamSearchParams(beam_width=beam_width, temperature=temperature, max_tokens=max_tokens, ignore_eos=False, length_penalty=1.0)
    list_infer_results = []
    for i in range(len(list_arrow_folder_path)):
        arrow_folder_path = list_arrow_folder_path[i]
        save_file_name = list_save_file_name[i]
        print(f"\n{'=' * 80}")
        print(f'Processing dataset [{i + 1}/{len(list_arrow_folder_path)}]: {save_file_name}')
        print(f'Data path: {arrow_folder_path}')
        print(f"{'=' * 80}\n")
        print(f'[Step 1/4] Loading data...')
        list_question = get_list_question_from_arrow_folder_path(arrow_folder_path, bool_nat_question=bool_nat_question, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place)
        print(f'[Step 1/4] ✓ Data loading complete, total {len(list_question)} samples\n')
        print(f'[Step 2/4] Mapping question format...')
        list_question = mapping_list_qusetion_to_certain_form(list_question, tokenizer, bool_qwen3_model, bool_beam_search)
        print(f'[Step 2/4] ✓ Question format mapping complete\n')
        print(f'[Step 3/4] Starting model inference...')
        if bool_beam_search:
            print(f'            Using Beam Search (beam_width={beam_width})')
            print(f'            ⚠️  First inference may take 1-3 minutes for compilation optimization, please wait...')
            print(f'            Progress bar will be displayed during inference\n')
            outputs = llm.beam_search(list_question, beam_params)
        else:
            print(f'            Using Greedy Decoding')
            print(f'            Progress bar will be displayed during inference\n')
            outputs = llm.generate(list_question, sampling_params)
        print(f'\n[Step 3/4] ✓ Inference complete\n')
        print(f'[Step 4/4] Saving results...')
        save_vllm_infer_result_to_jsonl_file(outputs, save_folder_path, save_file_name)
        save_path = os.path.join(save_folder_path, f'{save_file_name}.jsonl')
        print(f'[Step 4/4] ✓ Results saved to: {save_path}\n')
        list_infer_results.append(outputs)
        print(f'Dataset [{i + 1}/{len(list_arrow_folder_path)}] processing complete\n')
    print(f"\n{'=' * 80}")
    print(f'All {len(list_arrow_folder_path)} datasets inference complete!')
    print(f"{'=' * 80}\n")
    return list_infer_results

def multi_model_vllm_infer_multi_dataset(list_model_checkpoint_folder_path, list_arrow_folder_path, list_save_file_name, bool_nat_question, list_save_folder_path, beam_width=1, max_tokens=5000, temperature=0, top_p=0.95, top_k=20, adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2):
    if len(list_model_checkpoint_folder_path) != len(list_save_folder_path):
        raise ValueError(f'Number of model weights ({len(list_model_checkpoint_folder_path)}) does not match number of save folders ({len(list_save_folder_path)})')
    list_list_one_model_vllm_infer_multi_dataset_result = []
    for i in range(len(list_model_checkpoint_folder_path)):
        list_one_model_vllm_infer_multi_dataset_result = one_model_vllm_infer_multi_dataset(list_model_checkpoint_folder_path[i], list_arrow_folder_path, list_save_file_name, bool_nat_question, save_folder_path=list_save_folder_path[i], beam_width=beam_width, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place)
        list_list_one_model_vllm_infer_multi_dataset_result.append(list_one_model_vllm_infer_multi_dataset_result)
    return list_list_one_model_vllm_infer_multi_dataset_result

def quick_multi_model_vllm_infer_multi_dataset(parent_checkpoint_folder_path, dict_arrow_folder_path_save_file_name, parent_parent_save_folder_path, bool_nat_question, max_tokens=5000, beam_width=1, temperature=0, top_p=0.95, top_k=20, adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2, level=0):
    """
    Further encapsulation based on multi_model_vllm_infer_multi_dataset for quick calls.

    Args:
    (1) parent_checkpoint_folder_path: The parent folder of the checkpoint folders, i.e., this folder contains multiple checkpoint-xxxx folders.
    (2) dict_arrow_folder_path_save_file_name: A dictionary.
        Key: Arrow dataset folder path.
        Value: File name to use when saving test results for this dataset.
    (3) parent_parent_save_folder_path: Test results will be saved under this path as follows:
        - parent_parent_save_folder_path
        -- Folder matching parent_checkpoint_folder_path name
        --- Folder named after checkpoint iteration
        ---- .jsonl files named after values in dict_arrow_folder_path_save_file_name
    (4) bool_nat_question: Whether to convert questions to natural language.

    Returns:
    list_list_one_model_vllm_infer_multi_dataset_result: Each result in the list is a list_one_model_vllm_infer_multi_dataset_result object, i.e., a list of results for one model inferring on multiple datasets.
    """
    list_model_checkpoint_folder_path = get_subdirectories_with_certain_prefix(parent_checkpoint_folder_path, str_prefix='checkpoint')
    list_arrow_folder_path = list(dict_arrow_folder_path_save_file_name.keys())
    model_train_info = get_folder_name_by_level(parent_checkpoint_folder_path, level)
    list_checkpoint_iter = extract_checkpoint_iters(list_model_checkpoint_folder_path)
    list_save_folder_path = build_save_folder_paths(parent_parent_save_folder_path, model_train_info, list_checkpoint_iter)
    list_save_file_name = list(dict_arrow_folder_path_save_file_name.values())
    list_list_one_model_vllm_infer_multi_dataset_result = multi_model_vllm_infer_multi_dataset(list_model_checkpoint_folder_path, list_arrow_folder_path, list_save_file_name, bool_nat_question, list_save_folder_path=list_save_folder_path, beam_width=beam_width, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place)
    return list_list_one_model_vllm_infer_multi_dataset_result

def vllm_infer_multi_folder_weight(list_parent_checkpoint_folder_path: Union[str, List[str]], dict_arrow_folder_path_save_file_name: dict, parent_parent_save_folder_path: str, list_tuple_py_file_abs_path_function_name: list, bool_nat_question: bool=False, beam_width: int=1, max_tokens: int=5000, temperature: float=0, top_p: float=0.95, top_k: int=-1, adjust_decimal_places: bool=True, c_shift_decimal_place: int=1, h_shift_decimal_place: int=2, level: Union[int, str]='auto', list_need_level_1: List[str]=None):
    """
    Complete workflow for inference and evaluation of single or multiple folder weights using VLLM.
    
    Args:
        list_parent_checkpoint_folder_path (Union[str, List[str]]): Single checkpoint folder path (string) or list of multiple paths.
        dict_arrow_folder_path_save_file_name (dict): Mapping dictionary of dataset paths and save file names.
        parent_parent_save_folder_path (str): Parent directory path for saving results.
        list_tuple_py_file_abs_path_function_name (list): List of tuples containing file paths and function names for validation functions.
        bool_nat_question (bool): Whether it is a natural language question, default False.
        beam_width (int): Beam search width, default 1.
        max_tokens (int): Maximum generated tokens, default 5000.
        temperature (float): Sampling temperature, default 0 (greedy search).
        top_p (float): nucleus sampling parameter, default 0.95.
        top_k (int): top-k sampling parameter, default -1 (disabled).
        adjust_decimal_places (bool): Whether to adjust decimal places, default True.
        c_shift_decimal_place (int): Decimal places for Carbon chemical shift, default 1.
        h_shift_decimal_place (int): Decimal places for Hydrogen chemical shift, default 2.
        level (Union[int, str]): Hierarchy level for result folder names. Integer indicates levels up (0 for current, 1 for parent), 
                                 "auto" means automatic determination based on list_need_level_1, default "auto".
        list_need_level_1 (List[str]): List of folder names that trigger level=1 when level="auto". 
                                       Otherwise level=0 is used. Default is ["outputs", "output", "ouput"].
        
    Returns:
        None: Function performs complete inference and evaluation workflow for each model folder, saving results to specified directory.
    """
    if list_need_level_1 is None:
        list_need_level_1 = ['outputs', 'output', 'ouput']
    if isinstance(list_parent_checkpoint_folder_path, str):
        checkpoint_folder_paths = [list_parent_checkpoint_folder_path]
        print(f'Processing single model weight folder: {list_parent_checkpoint_folder_path}')
    else:
        checkpoint_folder_paths = list_parent_checkpoint_folder_path
        print(f'Batch processing {len(checkpoint_folder_paths)} model weight folders')
    for i, current_checkpoint_folder_path in enumerate(checkpoint_folder_paths):
        if len(checkpoint_folder_paths) > 1:
            print(f'Processing {i + 1}/{len(checkpoint_folder_paths)} model weight folder: {current_checkpoint_folder_path}')
        if level == 'auto':
            folder_name_level_0 = Path(current_checkpoint_folder_path).name
            if folder_name_level_0 in list_need_level_1:
                actual_level = 1
            else:
                actual_level = 0
        else:
            actual_level = level
        list_list_one_model_vllm_infer_multi_dataset_result = quick_multi_model_vllm_infer_multi_dataset(current_checkpoint_folder_path, dict_arrow_folder_path_save_file_name, parent_parent_save_folder_path, bool_nat_question, max_tokens=max_tokens, beam_width=beam_width, temperature=temperature, top_p=top_p, top_k=top_k, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place, level=actual_level)
        result_folder_name = get_folder_name_by_level(current_checkpoint_folder_path, actual_level)
        test_result_folder_path = Path(parent_parent_save_folder_path) / result_folder_name
        convert_jsonl_to_txt_file(test_result_folder_path)
        evaluate_infer_result_by_convert_to_txt(current_checkpoint_folder_path, parent_parent_save_folder_path, list_tuple_py_file_abs_path_function_name, dict_arrow_folder_path_save_file_name, level=actual_level)
        if len(checkpoint_folder_paths) > 1:
            print(f'Model weight folder {i + 1} processing complete: {current_checkpoint_folder_path}')
    if len(checkpoint_folder_paths) > 1:
        print(f'All {len(checkpoint_folder_paths)} model weight folders processing complete!')
    else:
        print(f'Model weight folder processing complete: {checkpoint_folder_paths[0]}')

def single_checkpoint_infer_pipeline(checkpoint_folder_path: str, ground_truth_arrow_folder_path: str, dataset_name: str, output_folder_path: str, result_name: str, stat_metrics_scription_file_path: str=str(Path(__file__).resolve()), beam_width: int=1, max_tokens: int=5000, temperature: float=0, top_p: float=0.95, top_k: int=20, adjust_decimal_places: bool=False, c_shift_decimal_place: int=1, h_shift_decimal_place: int=2):
    """
    Complete pipeline for performing inference on a single checkpoint and a single dataset, 
    converting the format, and executing evaluation.

    The generated files (.jsonl and .txt) will be named in the format: {result_name}_{dataset_name}.

    Args:
        checkpoint_folder_path (str): The path to the model checkpoint folder.
        ground_truth_arrow_folder_path (str): The path to the folder containing the Ground Truth dataset.
        dataset_name (str): The name of the dataset. This will be used as a suffix for the output filenames.
        output_folder_path (str): The root directory where output results will be stored.
        result_name (str): The specific result identifier. A subfolder with this name will be created inside `output_folder_path`, 
                           and it will also be used as a prefix for the output filenames.
        stat_metrics_scription_file_path (str, optional): The path to the Python script containing the evaluation function 
                                                          `batch_evaluate_multiple_test_files`. Defaults to "./stat_metrics_scription.py".
        beam_width (int, optional): The beam width for beam search decoding. Defaults to 1 (Greedy Search).
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 5000.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
        top_k (int, optional): Top-k sampling parameter. Defaults to 20.
        adjust_decimal_places (bool, optional): Whether to standardize decimal places in NMR data. Defaults to False.
        c_shift_decimal_place (int, optional): The number of decimal places for 13C NMR shifts. Defaults to 1.
        h_shift_decimal_place (int, optional): The number of decimal places for 1H NMR shifts. Defaults to 2.

    Returns:
        None: The function saves the inference results and evaluation metrics (CSV/JSON) to the disk.
    """
    save_folder_path = os.path.join(output_folder_path, result_name)
    os.makedirs(save_folder_path, exist_ok=True)
    full_save_name = f'{result_name}_{dataset_name}'
    print(f'Start processing single checkpoint: {checkpoint_folder_path}')
    print(f'Output directory: {save_folder_path}')
    print(f'Target filename prefix: {full_save_name}')
    one_model_vllm_infer_multi_dataset(model_checkpoint_folder_path=checkpoint_folder_path, list_arrow_folder_path=[ground_truth_arrow_folder_path], list_save_file_name=[full_save_name], bool_nat_question=False, save_folder_path=save_folder_path, beam_width=beam_width, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place)
    jsonl_file_path = os.path.join(save_folder_path, f'{full_save_name}.jsonl')
    txt_file_path = os.path.join(save_folder_path, f'{full_save_name}.txt')
    print(f'Converting JSONL to TXT: {jsonl_file_path} -> {txt_file_path}')
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_in, open(txt_file_path, 'w', encoding='utf-8') as txt_out:
            for line in jsonl_in:
                data = json.loads(line.strip())
                test_field = data.get('text', [])
                if isinstance(test_field, list) and test_field:
                    content = test_field[0].replace('\n', '\\n').replace('\r', '\\r')
                    txt_out.write(content + '\n')
                else:
                    txt_out.write('\n')
    except Exception as e:
        print(f'Error during TXT conversion: {e}')
    eval_func_name = 'batch_evaluate_multiple_test_files'
    if not os.path.exists(stat_metrics_scription_file_path):
        print(f'Warning: Evaluation script not found at {stat_metrics_scription_file_path}. Skipping evaluation.')
        return
    imported_funcs = generation_import_functions([(stat_metrics_scription_file_path, eval_func_name)])
    eval_func = imported_funcs.get(eval_func_name, globals().get(eval_func_name))
    if eval_func:
        print(f'Starting evaluation using {eval_func_name}...')
        dict_arrow_mapping = {ground_truth_arrow_folder_path: full_save_name}
        try:
            eval_func(save_folder_path, dict_arrow_mapping, save_result_as_json_and_csv=True)
            print('Evaluation finished. CSV files should be in the output folder.')
        except Exception as e:
            print(f'Error during evaluation execution: {e}')
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: Function '{eval_func_name}' could not be loaded.")

# VOTING

def voting_get_files_with_suffix(folder_path: str, suffix: str, traverse_sub_folder: bool=False) -> List[str]:
    """
    Scans a directory for files with a specific extension.

    Args:
        folder_path: The root directory to scan.
        suffix: The file extension to look for (e.g., '.arrow').
        traverse_sub_folder: If True, searches recursively through subdirectories.

    Returns:
        A list of absolute or relative paths to the matching files.
    """
    list_target_file_path = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                list_target_file_path.append(os.path.join(root, file))
        if not traverse_sub_folder:
            break
    return list_target_file_path

def voting_get_df_sample_info_from_arrow_file(arrow_file_path: str) -> pd.DataFrame:
    """
    Loads a HuggingFace Dataset arrow file and converts it into a DataFrame.

    Args:
        arrow_file_path: Path to the .arrow file.

    Returns:
        A pandas DataFrame containing the dataset samples.
    """
    dataset = Dataset.from_file(arrow_file_path)
    df = dataset.to_pandas()
    return df

def voting_judge_task_from_answer_by_keyword(list_answer: List[Any], dict_key_task_value_list_task_keyword: Dict[str, List[str]]={'nmr': [], 'retrosyn': ['Reactant', 'F1'], 'syn': ['Product']}) -> str:
    """
    Identifies the chemistry task type based on keywords found in the ground truth answer.

    Args:
        list_answer: A list of ground truth answers.
        dict_key_task_value_list_task_keyword: Mapping of task names to list of identifying keywords.

    Returns:
        The task name (e.g., 'nmr', 'retrosyn', or 'syn').

    Raises:
        ValueError: If keywords overlap between tasks or if no task can be matched.
    """
    if not list_answer:
        raise ValueError('list_answer cannot be empty')
    answer_text = str(list_answer[0])
    seen_keywords = set()
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        for kw in keywords:
            if kw in seen_keywords:
                raise ValueError(f"Keyword '{kw}' is duplicated across tasks.")
            seen_keywords.add(kw)
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        for kw in keywords:
            if kw in answer_text:
                return task
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        if not keywords:
            return task
    raise ValueError('No task matched and no default task defined.')

def load_question_cot_answer_list_and_str_task_from_arrow_folder(arrow_folder_path: str, dict_key_task_value_list_task_keyword: Dict[str, List[str]]={'nmr': [], 'retrosyn': ['Reactant', 'F1'], 'syn': ['Product']}) -> Tuple[List[str], List[str], List[str], str]:
    """
    Aggregates all arrow files in a folder and determines the task type.

    Args:
        arrow_folder_path: Folder containing .arrow files.
        dict_key_task_value_list_task_keyword: Keywords for task identification.

    Returns:
        A tuple containing (list of questions, list of CoTs, list of answers, task_type).
    """
    list_arrow_file_path = voting_get_files_with_suffix(arrow_folder_path, '.arrow', traverse_sub_folder=False)
    if not list_arrow_file_path:
        raise ValueError(f"No .arrow files found in '{arrow_folder_path}'")
    list_arrow_file_path.sort()
    combined_questions = []
    combined_cots = []
    combined_answers = []
    for arrow_file_path in list_arrow_file_path:
        print(f'Loading arrow file: {arrow_file_path}')
        df = voting_get_df_sample_info_from_arrow_file(arrow_file_path)
        combined_questions.extend(df['question'].tolist())
        combined_cots.extend(df['cot'].tolist())
        combined_answers.extend(df['answer'].tolist())
    str_task = voting_judge_task_from_answer_by_keyword(combined_answers, dict_key_task_value_list_task_keyword)
    if str_task == 'retrosyn':
        combined_answers = [voting__replace_reactant_placeholders(ans) for ans in combined_answers]
    elif str_task == 'syn':
        combined_answers = [voting__replace_product_placeholders(ans) for ans in combined_answers]
    print(f'Successfully merged {len(list_arrow_file_path)} files. Total samples: {len(combined_questions)}, Task: {str_task}')
    return (combined_questions, combined_cots, combined_answers, str_task)

def read_multi_jsonl_to_list_list_list_infer_cotent(list_jsonl_file_path: List[str], extract_key: str='text', fix_escaped_backslash: bool=True, beamsearch_top_n_involve: Optional[int]=None) -> Tuple[List[List[List[str]]], Dict[int, str]]:
    """
    Reads multiple JSONL files (each from a different model) into a nested structure.

    Args:
        list_jsonl_file_path: List of paths to inference result files.
        extract_key: JSON key containing the list of Top-K strings.
        fix_escaped_backslash: If True, converts '\\\\' to '\\' (common in SMILES JSON exports).
        beamsearch_top_n_involve: Truncate Top-K results to this number.

    Returns:
        A tuple of (3D list [model_idx][question_idx][candidate_idx], index_to_filename_map).
    """
    if not isinstance(list_jsonl_file_path, list) or len(list_jsonl_file_path) == 0:
        raise ValueError('list_jsonl_file_path must be a non-empty list.')
    list_list_list_infer_cotent: List[List[List[str]]] = []
    dict_key_model_index_value_model_result_jsonl_file_path: Dict[int, str] = {}
    for model_idx, file_path in enumerate(list_jsonl_file_path):
        dict_key_model_index_value_model_result_jsonl_file_path[model_idx] = file_path
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'File not found for model index {model_idx}: {file_path}')
        model_questions_topk: List[List[str]] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    raise ValueError(f'Failed to parse JSON: {file_path} Line {line_no}. Error: {e}')
                topk_list = obj[extract_key]
                topk_str_list: List[str] = []
                for elem in topk_list:
                    if elem is None:
                        topk_str_list.append('')
                    else:
                        elem_str = str(elem)
                        if fix_escaped_backslash and '\\\\' in elem_str:
                            elem_str = elem_str.replace('\\\\', '\\')
                        topk_str_list.append(elem_str)
                if beamsearch_top_n_involve is not None and beamsearch_top_n_involve > 0:
                    topk_str_list = topk_str_list[:beamsearch_top_n_involve]
                model_questions_topk.append(topk_str_list)
        list_list_list_infer_cotent.append(model_questions_topk)
    return (list_list_list_infer_cotent, dict_key_model_index_value_model_result_jsonl_file_path)

def voting_remove_disordered_tags(s: str, tuple_first_second: Tuple[str, str]) -> str:
    """
    Removes closing tags that appear before an opening tag in a string.

    Args:
        s: Input string.
        tuple_first_second: A tuple of (opening_tag, closing_tag).

    Returns:
        The string with disordered tags removed.
    """
    first, second = tuple_first_second
    first_positions = [match.start() for match in re.finditer(re.escape(first), s)]
    second_positions = [match.start() for match in re.finditer(re.escape(second), s)]
    filtered_s = s
    for pos in second_positions:
        if not any((f_pos < pos for f_pos in first_positions)):
            filtered_s = filtered_s[:pos] + filtered_s[pos + len(second):]
    return filtered_s

def voting_extract_str_answer(str_generate_result_without_prompt: str) -> str:
    """
    Parses the raw generation to extract the content between <answer> tags.
    If tags are missing, it attempts to use common chemistry pattern fallbacks.

    Args:
        str_generate_result_without_prompt: Raw text generated by the model.

    Returns:
        The extracted answer string or 'F_' if extraction fails.
    """
    str_generate_result_without_prompt = voting_remove_disordered_tags(str_generate_result_without_prompt, ('<answer>', '</answer>'))
    bool_answer_start = '<answer>' in str_generate_result_without_prompt
    bool_answer_end = '</answer>' in str_generate_result_without_prompt
    bool_specific_str = '])' in str_generate_result_without_prompt
    bool_nat_question_str = 'infer the molecular SMILES:' in str_generate_result_without_prompt
    if bool_answer_start and bool_answer_end:
        str_answer = str_generate_result_without_prompt.split('</answer>')[-2].split('<answer>')[-1]
    elif bool_answer_start and (not bool_answer_end):
        str_answer = str_generate_result_without_prompt.split('<answer>')[1]
    elif not bool_answer_start and bool_answer_end:
        str_answer = 'F_'
    elif bool_nat_question_str:
        str_answer = str_generate_result_without_prompt.split('infer the molecular SMILES:')[-1].split('"')[0]
    elif bool_specific_str:
        str_answer = str_generate_result_without_prompt.split('])')[-1].split('"')[0]
    else:
        str_answer = 'F_'
    return str_answer.strip()

def list_list_infer_cotent_to_list_list_infer_answer(list_list_infer_cotent: List[List[str]]) -> List[List[str]]:
    """
    Converts a 2D list of full generated texts into a 2D list of extracted answers.

    Args:
        list_list_infer_cotent: Nested list [question][candidate_text].

    Returns:
        Nested list [question][extracted_answer].
    """
    list_list_infer_answer = []
    for row in list_list_infer_cotent:
        list_list_infer_answer.append([voting_extract_str_answer(str(content)) for content in row])
    return list_list_infer_answer

def voting_extract_formula_from_generate_result(str_generate_result_without_prompt: str) -> Optional[str]:
    """
    Extracts molecular formula (e.g., C6H12O6) from the reasoning trace.

    Args:
        str_generate_result_without_prompt: The model's reasoning text.

    Returns:
        Formula string if found, otherwise None.
    """
    pattern = 'Formula:([^;]+);'
    match = re.search(pattern, str_generate_result_without_prompt)
    return match.group(1).strip() if match else None

def voting_normalize_molecular_formula(formula: str) -> str:
    """
    Parses and normalizes a molecular formula string to a standard Hill system order (C, then H, then alphabetical).

    Args:
        formula: A raw formula string (e.g., 'H2O', '(NH4)2SO4').

    Returns:
        Normalized formula string.
    """
    if not formula:
        return ''
    import re
    from collections import defaultdict

    def parse_segment_counts(segment: str) -> dict[str, int]:
        s, n, i = (segment, len(segment), 0)

        def parse_int(idx: int):
            if idx >= n or not s[idx].isdigit():
                return (None, idx)
            j = idx
            while j < n and s[j].isdigit():
                j += 1
            return (int(s[idx:j]), j)

        def parse_element(idx: int):
            if idx >= n or not s[idx].isupper():
                return (None, idx)
            j = idx + 1
            if j < n and s[j].islower():
                j += 1
            return (s[idx:j], j)

        def merge(dst, src, mul=1):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v * mul

        def parse_block(idx: int):
            out = defaultdict(int)
            while idx < n:
                ch = s[idx]
                if ch in '([{':
                    idx += 1
                    inner, idx = parse_block(idx)
                    if idx < n and s[idx] in ')]}':
                        idx += 1
                    mul, idx = parse_int(idx)
                    merge(out, inner, mul if mul is not None else 1)
                    continue
                if ch in ')]}':
                    return (out, idx)
                el, j = parse_element(idx)
                if el:
                    cnt, j2 = parse_int(j)
                    out[el] += cnt if cnt is not None else 1
                    idx = j2
                    continue
                idx += 1
            return (out, idx)
        counts, _ = parse_block(0)
        return dict(counts)
    s = re.sub('\\s+', '', formula).replace('−', '-')
    s = s.replace('+', '').replace('-', '').replace('^', '')
    parts = re.split('[·⋅∙\\.•]', s)
    total_counts = defaultdict(int)
    for part in parts:
        if not part:
            continue
        m = re.match('^(\\d+)(.+)$', part)
        if m:
            part = f'({m.group(2)}){m.group(1)}'
        for el, c in parse_segment_counts(part).items():
            total_counts[el] += c
    if not total_counts:
        return ''
    res = []
    if 'C' in total_counts:
        c = total_counts.pop('C')
        res.append(f"C{(c if c > 1 else '')}")
    if 'H' in total_counts:
        h = total_counts.pop('H')
        res.append(f"H{(h if h > 1 else '')}")
    for el in sorted(total_counts.keys()):
        n = total_counts[el]
        res.append(f"{el}{(n if n > 1 else '')}")
    return ''.join(res)

def voting_is_valid_smiles_with_formula_check(smiles: str, target_formula: str, return_canonical: bool=False, check_length: int=5000, return_special_str_when_smiles_false: str | None='F_', check_nmr_formula: bool=True) -> Tuple[bool, str]:
    """
    Validates a SMILES string and optionally checks if its formula matches the target.

    Args:
        smiles: The candidate SMILES string.
        target_formula: The formula extracted from the model trace.
        return_canonical: If True, returns the RDKit canonical SMILES.
        check_length: Rejects strings exceeding this length.
        return_special_str_when_smiles_false: String to return on failure.
        check_nmr_formula: If True, compares SMILES formula to target_formula.

    Returns:
        Tuple of (is_valid, final_smiles).
    """
    if not isinstance(smiles, str) or not isinstance(target_formula, str):
        return (False, return_special_str_when_smiles_false)
    normalized_target_formula = voting_normalize_molecular_formula(target_formula)
    if check_length is not None and len(smiles) > check_length:
        return (False, return_special_str_when_smiles_false)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (False, return_special_str_when_smiles_false)
        if check_nmr_formula:
            mol_formula = rdMolDescriptors.CalcMolFormula(mol)
            normalized_mol_formula = voting_normalize_molecular_formula(mol_formula)
            if normalized_mol_formula != normalized_target_formula:
                return (False, return_special_str_when_smiles_false)
        return (True, Chem.MolToSmiles(mol, canonical=True) if return_canonical else smiles)
    except:
        return (False, return_special_str_when_smiles_false)

def extract_nmr_list_list_question_formula(list_list_infer_cotent: List[List[str]]) -> List[List[Optional[str]]]:
    """
    Extracts formulas from reasoning traces for a whole batch.

    Args:
        list_list_infer_cotent: Nested list [question][candidate_trace].

    Returns:
        Nested list [question][extracted_formula].
    """
    nmr_list_list_question_formula = []
    for row in list_list_infer_cotent:
        nmr_list_list_question_formula.append([voting_extract_formula_from_generate_result(str(c)) for c in row])
    return nmr_list_list_question_formula

def extract_filter_nmr_list_infer_answer(list_list_infer_cotent: List[List[str]], check_nmr_formula: bool=True) -> List[List[str]]:
    """
    Extracts answers and filters them based on SMILES validity and formula consistency (Specific to NMR).

    Args:
        list_list_infer_cotent: Full traces.
        check_nmr_formula: Whether to enforce formula matching.

    Returns:
        Filtered extracted answers.
    """
    list_list_infer_answer = list_list_infer_cotent_to_list_list_infer_answer(list_list_infer_cotent)
    list_list_question_formula = extract_nmr_list_list_question_formula(list_list_infer_cotent)
    filtered_answers = []
    for ans_row, fml_row in zip(list_list_infer_answer, list_list_question_formula):
        new_row = []
        for smiles_candidate, formula in zip(ans_row, fml_row):
            ok, out_smiles = voting_is_valid_smiles_with_formula_check(smiles_candidate, formula or '', check_nmr_formula=check_nmr_formula)
            new_row.append(out_smiles)
        filtered_answers.append(new_row)
    return filtered_answers

def voting__replace_product_placeholders(content: str) -> str:
    """
    Converts '[Product]' or '[Product1][Product2]' to dot-separated SMILES format.

    Args:
        content: String containing product placeholders.

    Returns:
        String with placeholders replaced by dots where appropriate.
    """

    def _product_repl(m):
        idx = m.group(1)
        return '' if not idx or idx == '1' else '.'
    return re.sub('\\[Product(\\d*)\\]', _product_repl, content)

def voting__replace_reactant_placeholders(content: str) -> str:
    """
    Converts '[Reactant]' placeholders to dot-separated SMILES format.

    Args:
        content: String containing reactant placeholders.

    Returns:
        Cleaned string.
    """

    def _reactant_repl(m):
        idx = m.group(1)
        return '' if not idx or idx == '1' else '.'
    return re.sub('\\[Reactant(\\d*)\\]', _reactant_repl, content)

def is_smilesdotsmiles_valid_smiles(smilesdotsmiles: str, check_length: int=5000, return_special_str_when_smilesdotsmiles_false: str='F_') -> Tuple[bool, str]:
    """
    Validates a dot-separated SMILES string (e.g., mixtures or reactants).

    Args:
        smilesdotsmiles: Candidate string.
        check_length: Threshold for length check.
        return_special_str_when_smilesdotsmiles_false: Error string.

    Returns:
        Tuple of (is_valid, processed_string).
    """
    s = str(smilesdotsmiles).strip()
    if not s:
        return (False, return_special_str_when_smilesdotsmiles_false)
    parts = s.split('.')
    if any((not p.strip() for p in parts)):
        return (False, return_special_str_when_smilesdotsmiles_false)
    for p in parts:
        sp = p.strip()
        if len(sp) >= check_length or Chem.MolFromSmiles(sp) is None:
            return (False, return_special_str_when_smilesdotsmiles_false)
    return (True, s)

def extract_filter_list_list_infer_answer_from_list_list_infer_cotent(list_list_infer_cotent: List[List[str]], str_task: str, check_length: int=5000, return_special_str_when_check_result_false: str='F_', check_nmr_formula: bool=True) -> List[List[str]]:
    """
    The unified entry point for extracting and filtering candidate answers based on the task type.

    Args:
        list_list_infer_cotent: The model generated traces.
        str_task: 'nmr', 'retrosyn', or 'syn'.
        check_length: Length threshold for validity.
        return_special_str_when_check_result_false: Placeholder for invalid candidates.
        check_nmr_formula: Whether to use trace formulas for NMR filtering.

    Returns:
        A 2D list of cleaned and filtered SMILES strings.
    """
    task = str_task.strip().lower()
    if task == 'nmr':
        return extract_filter_nmr_list_infer_answer(list_list_infer_cotent, check_nmr_formula=check_nmr_formula)
    if task in ['retrosyn', 'syn']:
        raw_answers = list_list_infer_cotent_to_list_list_infer_answer(list_list_infer_cotent)
        final_result = []
        for row in raw_answers:
            filtered_inner = []
            for ans in row:
                ans = str(ans or '').strip()
                ans = voting__replace_reactant_placeholders(ans) if task == 'retrosyn' else voting__replace_product_placeholders(ans)
                ans = re.sub('\\s+', '', ans)
                ans = re.sub('\\.{2,}', '.', ans).strip('.')
                ok, valid_or_placeholder = is_smilesdotsmiles_valid_smiles(ans, check_length, return_special_str_when_check_result_false)
                filtered_inner.append(valid_or_placeholder)
            final_result.append(filtered_inner)
        return final_result
    raise ValueError(f'Unsupported task: {str_task}')

def _canonicalize_smiles(smi: str, dechirality: bool=True, check_length: int=5000) -> Optional[str]:
    """
    Produces a canonical SMILES string using RDKit for voting comparison.

    Args:
        smi: Raw SMILES string.
        dechirality: If True, removes stereochemical information for more relaxed voting.
        check_length: Maximum allowed length.

    Returns:
        Canonicalized SMILES or None if invalid.
    """
    if not smi or (check_length and len(smi) > check_length):
        return None
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            return None
        if dechirality:
            Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not dechirality)
    except:
        return None

def multi_model_one_question_smiles_answer_to_index_answer(list_dict_key_model_name_value_topk_list_one_question: List[Dict[str, List[str]]], list_dict_key_model_name_value_full_content_list_one_question: List[Dict[str, List[str]]], dechirality: bool=True, remove_smiles: str='F_', check_length: int=5000, ignore_order: bool=True) -> Tuple[List[List[List[int]]], Dict[int, Dict[str, List[str]]], Dict[Tuple[int, ...], str]]:
    """
    Converts raw SMILES candidates from multiple models into index-based sequences to handle mixtures.
    Also builds mapping dictionaries to reconstruct original strings later.

    Args:
        list_dict_key_model_name_value_topk_list_one_question: List of candidate answers per model.
        list_dict_key_model_name_value_full_content_list_one_question: List of full generated content per model.
        dechirality: Remove chirality for indexing.
        remove_smiles: Invalid answer placeholder to ignore.
        ignore_order: If True, 'A.B' is indexed same as 'B.A'.

    Returns:
        - Nested list of indices [model][candidate_idx][sequence_of_ids].
        - Mapping from ID to original model-specific string.
        - Mapping from ID sequence (tuple) to full reasoning trace content.
    """
    global_smi2idx: Dict[str, int] = {}
    idx2model2originals = defaultdict(lambda: defaultdict(list))
    tuple_indices_to_full_content: Dict[Tuple[int, ...], str] = {}
    list_multi_model_result_indices = []
    for m_idx, (model_dict, full_dict) in enumerate(zip(list_dict_key_model_name_value_topk_list_one_question, list_dict_key_model_name_value_full_content_list_one_question)):
        for model_name, answers in model_dict.items():
            full_contents = full_dict[model_name]
            model_indexed_answers = []
            for ans_idx, ans_str in enumerate(answers):
                if not ans_str:
                    model_indexed_answers.append([])
                    continue
                parts = [p.strip() for p in str(ans_str).split('.')]
                parts = [p for p in parts if p and p != remove_smiles]
                one_answer_indices = []
                for raw_piece in parts:
                    std_smi = _canonicalize_smiles(raw_piece, dechirality, check_length)
                    if std_smi is None:
                        continue
                    if std_smi not in global_smi2idx:
                        global_smi2idx[std_smi] = len(global_smi2idx)
                    idx = global_smi2idx[std_smi]
                    one_answer_indices.append(idx)
                    idx2model2originals[idx][model_name].append(raw_piece)
                model_indexed_answers.append(one_answer_indices)
                if one_answer_indices:
                    k = tuple(sorted(one_answer_indices)) if ignore_order else tuple(one_answer_indices)
                    if k not in tuple_indices_to_full_content:
                        tuple_indices_to_full_content[k] = full_contents[ans_idx]
            list_multi_model_result_indices.append(model_indexed_answers)
    return (list_multi_model_result_indices, {k: dict(v) for k, v in idx2model2originals.items()}, tuple_indices_to_full_content)

def vote_list_multi_model_result_list_one_model_result_list_smilesdotsmiles_smiles_index_to_topk_list_list_smiles_index(list_multi_model_result_indices: List[List[List[int]]], ignore_order: bool, force_to_topk: Optional[int]=None, add_special_str_when_lack_result_to_fit_topk: str='F_') -> Tuple[List[Union[List[int], str]], Dict[Tuple[int, ...], List[int]]]:
    """
    The core voting engine. Counts occurrences of chemical identities across models.

    Args:
        list_multi_model_result_indices: Nested list of IDs.
        ignore_order: Whether mixture order matters.
        force_to_topk: If set, pad the results to this length.
        add_special_str_when_lack_result_to_fit_topk: Padding string.

    Returns:
        - Sorted list of ID sequences (Top candidates).
        - Mapping from ID sequence to list of model indices that voted for it.
    """

    def key_of(ans):
        return tuple(sorted(ans)) if ignore_order else tuple(ans)
    counts = defaultdict(int)
    first_pos = {}
    voters_in_order = defaultdict(list)
    voters_seen = defaultdict(set)
    for m_idx, model_answers in enumerate(list_multi_model_result_indices):
        for a_idx, ans in enumerate(model_answers):
            if not ans:
                continue
            k = key_of(ans)
            counts[k] += 1
            if k not in first_pos or (m_idx, a_idx) < first_pos[k]:
                first_pos[k] = (m_idx, a_idx)
            if m_idx not in voters_seen[k]:
                voters_seen[k].add(m_idx)
                voters_in_order[k].append(m_idx)
    if not counts:
        res = []
        if force_to_topk:
            res.extend([add_special_str_when_lack_result_to_fit_topk] * force_to_topk)
        return (res, OrderedDict())
    uniq_keys = sorted(counts.keys(), key=lambda k: (-counts[k], first_pos[k][0], first_pos[k][1]))
    topk_list = [list(k) for k in uniq_keys]
    if force_to_topk:
        if len(topk_list) > force_to_topk:
            topk_list = topk_list[:force_to_topk]
        else:
            topk_list.extend([add_special_str_when_lack_result_to_fit_topk] * (force_to_topk - len(topk_list)))
    mapping = OrderedDict(((k, voters_in_order[k]) for k in uniq_keys))
    return (topk_list, mapping)

def convert_dict_key_tuple_smiles_index_value_vote_this_answer_model_index_to_dict_smiles_value_vote_this_answer_model_index(list_multi_model_result_indices: List[List[List[int]]], idx2model2originals: Dict[int, Dict[str, List[str]]], ordered_mapping: Dict[Tuple[int, ...], List[int]], tuple_indices_to_full_content: Dict[Tuple[int, ...], str], ignore_order: bool=True) -> OrderedDict:
    """
    Reconstructs original reasoning traces or SMILES strings from the integer-based voting results.

    Args:
        list_multi_model_result_indices: Trace ID sequences.
        idx2model2originals: ID to raw string map.
        ordered_mapping: Winning sequences and their voters.
        tuple_indices_to_full_content: Winning sequences and their Reasoning Traces.
        ignore_order: Mixture order flag.

    Returns:
        Ordered dictionary: {ReconstructedContent: {voters: [...], reconstructed_smiles: "..."}}
    """
    all_names = sorted({n for idx_d in idx2model2originals.values() for n in idx_d.keys()})
    if not all_names:
        all_names = ['model0']
    model_index_to_name = {i: n for i, n in enumerate(all_names)}

    def _first_seq(m_idx, key_tuple):
        model_answers = list_multi_model_result_indices[m_idx]
        sorted_target = tuple(sorted(key_tuple)) if ignore_order else key_tuple
        for ans in model_answers:
            current = tuple(sorted(ans)) if ignore_order else tuple(ans)
            if current == sorted_target:
                return list(ans)
        return list(key_tuple)
    usage_ptr = defaultdict(int)
    out = OrderedDict()
    for key_tuple, voter_indices in ordered_mapping.items():
        search_key = tuple(sorted(key_tuple)) if ignore_order else key_tuple
        full_content = tuple_indices_to_full_content.get(search_key)
        lead_m_idx = voter_indices[0]
        lead_m_name = model_index_to_name.get(lead_m_idx, all_names[0])
        seq_indices = _first_seq(lead_m_idx, key_tuple)
        raw_pieces = []
        for idx in seq_indices:
            name2list = idx2model2originals.get(idx, {})
            lst = name2list.get(lead_m_name, [])
            if lst:
                pos = usage_ptr[idx, lead_m_name]
                raw_pieces.append(lst[pos] if pos < len(lst) else lst[0])
                usage_ptr[idx, lead_m_name] += 1
            else:
                raw_pieces.append(next(iter(name2list.values()))[0] if name2list else '')
        reconstructed = '.'.join(raw_pieces)
        out[full_content or reconstructed] = {'votes': voter_indices, 'reconstructed': reconstructed}
    return out

def _process_one_question_for_sc(question_idx: int, per_model_ans_strict: List[List[str]], per_model_ans_relaxed: List[List[str]], per_model_full: List[List[str]], ignore_order: bool, dechirality: bool, force_to_topk: Optional[int], add_special_str: str, check_length: int) -> OrderedDict:
    """
    Orchestrates the Self-Consistency logic for a single question.
    It first attempts a 'strict' vote (e.g., formula match); 
    if no valid candidate survives, it falls back to 'relaxed' candidates.

    Args:
        question_idx: Question ID.
        per_model_ans_strict: Candidate answers that passed strict filters.
        per_model_ans_relaxed: Candidate answers that passed basic filters.
        per_model_full: Full model generation traces.
        ignore_order: Mixture order flag.
        dechirality: Stereochemical flag for voting.
        force_to_topk: Results padding count.
        add_special_str: Padding value.
        check_length: Max SMILES length.

    Returns:
        Ordered dictionary of results.
    """

    def _run_voting(candidate_answers):
        list_dict_ans = [{f'model{i}': topk} for i, topk in enumerate(candidate_answers)]
        list_dict_full = [{f'model{i}': full} for i, full in enumerate(per_model_full)]
        list_indices, idx2originals, tuple2full = multi_model_one_question_smiles_answer_to_index_answer(list_dict_ans, list_dict_full, dechirality, add_special_str, check_length, ignore_order)
        _, ordered_mapping = vote_list_multi_model_result_list_one_model_result_list_smilesdotsmiles_smiles_index_to_topk_list_list_smiles_index(list_indices, ignore_order, force_to_topk, add_special_str)
        return convert_dict_key_tuple_smiles_index_value_vote_this_answer_model_index_to_dict_smiles_value_vote_this_answer_model_index(list_indices, idx2originals, ordered_mapping, tuple2full, ignore_order)
    strict_res = _run_voting(per_model_ans_strict)
    top1 = next(iter(strict_res)) if strict_res else add_special_str
    if top1 == add_special_str or top1.startswith(add_special_str):
        return _run_voting(per_model_ans_relaxed)
    return strict_res

def sc_multi_topk_answer_to_sc_topk_answer(list_list_list_infer_cotent: List[List[List[str]]], task: str, save_folder_path: str, save_file_name: str, ignore_order: bool=True, dechirality: bool=True, force_to_topk: Optional[int]=None, check_length: int=5000, add_special_str_when_lack_result_to_fit_topk: str='F_', num_process: int=16, check_nmr_formula: bool=True) -> Tuple[str, List[OrderedDict]]:
    """
    Runs the multi-model Self-Consistency pipeline in parallel for all questions.

    Args:
        list_list_list_infer_cotent: Input traces [model][question][candidate].
        task: Task type.
        save_folder_path: Output dir.
        save_file_name: Output filename.
        ignore_order: Mixture order flag.
        dechirality: Stereochemical flag.
        force_to_topk: Top candidates to keep.
        check_length: Validity threshold.
        add_special_str_when_lack_result_to_fit_topk: Placeholder.
        num_process: Parallel workers.
        check_nmr_formula: NMR specific filtering.

    Returns:
        - Path to saved results.
        - List of voting result dictionaries per question.
    """
    num_models, num_questions = (len(list_list_list_infer_cotent), len(list_list_list_infer_cotent[0]))
    list_list_list_strict = []
    list_list_list_relaxed = []
    print('Extracting and filtering answers (Strict & Relaxed)...')
    for m_idx in range(num_models):
        strict = extract_filter_list_list_infer_answer_from_list_list_infer_cotent(list_list_list_infer_cotent[m_idx], task, check_length, add_special_str_when_lack_result_to_fit_topk, check_nmr_formula)
        list_list_list_strict.append(strict)
        relaxed = extract_filter_list_list_infer_answer_from_list_list_infer_cotent(list_list_list_infer_cotent[m_idx], task, check_length, add_special_str_when_lack_result_to_fit_topk, False) if check_nmr_formula else strict
        list_list_list_relaxed.append(relaxed)
    per_q_inputs = []
    for q_idx in range(num_questions):
        per_q_inputs.append((q_idx, [m[q_idx] for m in list_list_list_strict], [m[q_idx] for m in list_list_list_relaxed], [m[q_idx] for m in list_list_list_infer_cotent], ignore_order, dechirality, force_to_topk, add_special_str_when_lack_result_to_fit_topk, check_length))
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_process) as ex:
        futures = [ex.submit(_process_one_question_for_sc, *inp) for inp in per_q_inputs]
        for fut in tqdm(futures, desc='Self-Consistency Parallel Execution'):
            results.append(fut.result())
    os.makedirs(save_folder_path, exist_ok=True)
    out_path = os.path.join(save_folder_path, f'{save_file_name}.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for od in results:
            f.write(json.dumps({k: v['votes'] for k, v in od.items()}, ensure_ascii=False) + '\n')
    return (out_path, results)

def _get_target_smiles_from_gt(gt_text: str, task: str) -> str:
    """
    Parses Ground Truth strings to extract only the target SMILES.

    Args:
        gt_text: Raw GT answer.
        task: 'nmr', 'retrosyn', or 'syn'.

    Returns:
        Target SMILES string.
    """
    gt_text = str(gt_text).strip()
    if task == 'nmr':
        return gt_text
    if '>>' in gt_text:
        parts = gt_text.split('>>')
        return parts[0].strip() if task == 'retrosyn' else parts[-1].strip()
    return gt_text

def _compute_metrics_single_row(args: Tuple) -> Dict[str, Any]:
    """
    Calculates Accuracy and Tanimoto Similarity for a single question.

    Args:
        args: (list_of_predictions, gt_text, task, check_length, max_k).

    Returns:
        Dictionary containing metric results for the row.
    """
    list_preds, gt_text, task, check_length, max_k = args
    res = {'valid': False, 'tanimoto': 0.0, 'exact_match': False, 'top_hits': [False] * max_k}
    if not list_preds:
        return res
    try:
        gt_target = _get_target_smiles_from_gt(gt_text, task)
        mol_gt = Chem.MolFromSmiles(gt_target)
        if not mol_gt:
            return res
        can_gt = Chem.MolToSmiles(mol_gt, canonical=True, isomericSmiles=False)
        fp_gt = AllChem.GetMorganFingerprintAsBitVect(mol_gt, 2, nBits=1024)
        found_match = False
        for i, pred in enumerate(list_preds[:max_k]):
            if not pred or pred.startswith('F_'):
                continue
            m = Chem.MolFromSmiles(pred)
            if not m:
                continue
            if i == 0:
                res['valid'] = True
                fp_p = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
                res['tanimoto'] = DataStructs.FingerprintSimilarity(fp_p, fp_gt)
            can_p = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
            if can_p == can_gt:
                found_match = True
                if i == 0:
                    res['exact_match'] = True
            if found_match:
                for k in range(i, max_k):
                    res['top_hits'][k] = True
                break
    except:
        pass
    return res

def calculate_metrics_standard(list_list_pred: List[List[str]], list_gt: List[str], task: str, check_length: int=5000, num_process: int=16, calc_topk: int=10) -> Dict[str, float]:
    """
    Calculates batch-level metrics including Validity Rate, Accuracy, and Tanimoto.

    Args:
        list_list_pred: Predictions per question.
        list_gt: Ground truths.
        task: Task type.
        check_length: Threshold for metrics validation.
        num_process: Parallel workers.
        calc_topk: K value for Top-K metrics.

    Returns:
        Final averaged metrics.
    """
    args = [(p, g, task, check_length, calc_topk) for p, g in zip(list_list_pred, list_gt)]
    with Pool(num_process) as pool:
        results = pool.map(_compute_metrics_single_row, args)
    total = len(results)
    if not total:
        return {}
    metrics = {'prediction_validity_rate': sum((1 for r in results if r['valid'])) / total, 'accuracy': sum((1 for r in results if r['exact_match'])) / total, 'average_tanimoto': sum((r['tanimoto'] for r in results)) / total}
    for k in range(calc_topk):
        metrics[f'top{k + 1}'] = sum((1 for r in results if r['top_hits'][k])) / total
    return metrics

def save_results_aligned(dict_results: Dict[str, Any], save_folder_path: str, save_filename_prefix: str):
    """
    Saves experimental results into a structured JSON and various CSV formats.

    Args:
        dict_results: The nested result dictionary.
        save_folder_path: Dir to save.
        save_filename_prefix: Filename prefix.
    """
    import csv
    json_path = os.path.join(save_folder_path, f'{save_filename_prefix}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dict_results, f, indent=4)
    steps = sorted(dict_results.keys(), key=lambda x: int(x) if x.isdigit() else x)
    all_datasets = sorted({ds for s in steps for ds in dict_results[s].keys()})
    core_metrics = ['accuracy', 'average_tanimoto', 'prediction_validity_rate']
    fmt_path = os.path.join(save_folder_path, f'{save_filename_prefix}_formatted.csv')
    with open(fmt_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + [ds for ds in all_datasets for _ in core_metrics])
        writer.writerow(['metric'] + core_metrics * len(all_datasets))
        writer.writerow(['train_step'] + [''] * (len(all_datasets) * len(core_metrics)))
        for s in steps:
            row = [s]
            for ds in all_datasets:
                data = dict_results[s].get(ds, {})
                for m in core_metrics:
                    val = data.get(m, 0)
                    row.append(f'{val:.4f}' if isinstance(val, (int, float)) else str(val))
            writer.writerow(row)
    acc_path = os.path.join(save_folder_path, f'{save_filename_prefix}_accuracy.csv')
    with open(acc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train_step'] + all_datasets)
        for s in steps:
            row = [s]
            for ds in all_datasets:
                acc = dict_results[s].get(ds, {}).get('accuracy', 0)
                row.append(f'{acc:.4f}')
            writer.writerow(row)
    print(f'Results saved to {save_folder_path} with prefix {save_filename_prefix}')

def cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc(arrow_folder_path: str, list_jsonl_file_path: List[str], save_folder_path: str, save_folder_name: str, topk: int, ignore_order: bool=True, dechirality: bool=True, force_to_topk: Optional[int]=None, check_length: int=5000, num_process: int=8, extract_key: str='text', fix_escaped_backslash: bool=True, add_special_str_when_lack_result_to_fit_topk: str='F_', check_nmr_formula: bool=True) -> Dict[str, float]:
    """
    The main high-level API to execute multi-model Self-Consistency voting and calculate Top-K accuracy.

    Args:
        arrow_folder_path: Ground truth path.
        list_jsonl_file_path: Inference result paths from multiple models.
        save_folder_path: Root output directory.
        save_folder_name: Sub-folder for this experiment.
        topk: K for Top-K accuracy.
        ignore_order: Mixture order flag for voting.
        dechirality: Stereochemical flag for voting.
        force_to_topk: Candidate count per question.
        check_length: Max SMILES length.
        num_process: Worker count.
        extract_key: JSON key for candidates.
        fix_escaped_backslash: JSON escape fix.
        add_special_str_when_lack_result_to_fit_topk: Fail placeholder.
        check_nmr_formula: NMR formula trace check.

    Returns:
        Summary metrics dictionary.
    """
    target_dir = os.path.join(save_folder_path, save_folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print('=' * 80 + '\nStarting Multi-Model Self-Consistency Voting Pipeline\n' + '=' * 80)
    _, _, list_gt, str_task = load_question_cot_answer_list_and_str_task_from_arrow_folder(arrow_folder_path)
    list_content, _ = read_multi_jsonl_to_list_list_list_infer_cotent(list_jsonl_file_path, extract_key, fix_escaped_backslash)
    _, voting_results = sc_multi_topk_answer_to_sc_topk_answer(list_content, str_task, target_dir, 'sc_results', ignore_order, dechirality, force_to_topk, check_length, add_special_str_when_lack_result_to_fit_topk, num_process, check_nmr_formula)
    sc_preds = []
    for q_res in voting_results:
        row = [v['reconstructed'] for k, v in q_res.items() if not (k.startswith('F_') and len(k) < 5)]
        sc_preds.append(row if row else ['F_'])
    metrics = calculate_metrics_standard(sc_preds, list_gt, str_task, check_length, num_process, topk)
    print('\nFinal Averaged Metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    with open(os.path.join(target_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics

# COMMAND-LINE WORKFLOWS

PROJECT_ROOT = Path(__file__).resolve().parent

def load_data(path):
    from datasets import Dataset, concatenate_datasets
    files = sorted(Path(path).glob('*.arrow'))
    if not files:
        raise FileNotFoundError(f'No Arrow shards: {path}')
    data = concatenate_datasets([Dataset.from_file(str(p)) for p in files])
    missing = {'question', 'cot', 'answer'} - set(data.column_names)
    if missing:
        raise ValueError(f'Missing dataset columns: {missing}')
    if not len(data):
        raise ValueError('Empty dataset')
    return data

def checkpoint(path):
    root = Path(path).resolve()
    candidates = [root] if (root / 'config.json').is_file() else sorted((p.parent for p in root.rglob('config.json')))
    candidates = [p for p in candidates if list(p.glob('*.safetensors')) or list(p.glob('pytorch_model*.bin'))]
    if len(candidates) != 1:
        raise ValueError(f'Expected one checkpoint under {root}; found {candidates}. Specify the exact checkpoint.')
    if not (candidates[0] / 'tokenizer_config.json').is_file():
        raise FileNotFoundError(f'Missing tokenizer in {candidates[0]}')
    return candidates[0]

def verify_task(data):
    task = judge_task_from_answer_by_keyword(list(data['answer']))
    if task not in {'syn', 'retrosyn'}:
        raise ValueError(f'Task {task!r} is outside this project scope')
    return task

def infer(config):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    started = time.time()
    data = load_data(config['data'])
    total = len(data)
    task = verify_task(data)
    limit = config.get('limit', 3)
    if limit < 0:
        raise ValueError('limit must be >= 0 (0 means full dataset)')
    if limit:
        data = data.select(range(min(limit, len(data))))
    model_dir = checkpoint(config['model'])
    out = Path(config.get('output', 'runs/inference')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(out / 'ground_truth'))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    questions = [standardize_shift_decimal_place(q, config.get('adjust_decimal_places', False), config.get('c_shift_decimal_place', 1), config.get('h_shift_decimal_place', 2)) for q in data['question']]
    prompts = mapping_list_qusetion_to_certain_form(questions, tokenizer, detect_qwen_model_type(str(model_dir)), False)
    backend = config.get('backend', 'transformers')
    beam = config.get('beam_width', 1)
    temperature = config.get('temperature', 0)
    max_tokens = config.get('max_tokens', 5000)
    details = []
    texts = []
    if backend == 'transformers':
        device = config.get('device', 'cuda:0')
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32).to(device).eval()
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        batch_size = config.get('batch_size', 1)
        with (out / 'predictions.partial.jsonl').open('w') as partial:
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start:start + batch_size]
                inputs = tokenizer(batch, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
                options = dict(max_new_tokens=max_tokens, do_sample=temperature > 0, num_beams=beam, num_return_sequences=beam, pad_token_id=tokenizer.pad_token_id)
                if temperature > 0:
                    options.update(temperature=temperature, top_p=config.get('top_p', .95), top_k=config.get('top_k', 20))
                with torch.inference_mode():
                    generated = model.generate(**inputs, **options)
                continuations = generated[:, inputs['input_ids'].shape[1]:]
                eos_ids = model.generation_config.eos_token_id
                eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
                for offset, prompt in enumerate(batch):
                    candidates = []
                    lengths = []
                    for tokens in continuations[offset * beam:(offset + 1) * beam]:
                        ids = tokens.tolist()
                        length = next((i + 1 for i, value in enumerate(ids) if value in eos_ids), len(ids))
                        lengths.append(length)
                        candidates.append(prompt + tokenizer.decode(ids[:length], skip_special_tokens=True))
                    texts.append(candidates)
                    partial.write(json.dumps({'text': candidates}, ensure_ascii=False) + '\n')
                    details.append({'index': start + offset, 'input_tokens': int(inputs['attention_mask'][offset].sum()),
                                    'generated_tokens': lengths[0], 'has_answer': bool(extract_str_answer(candidates[0])),
                                    'reached_token_limit': lengths[0] >= max_tokens})
                partial.flush()
                (out / 'progress.json').write_text(json.dumps({'completed': len(texts), 'total': len(prompts),
                                                               'elapsed_seconds': time.time() - started}, indent=2))
                print(f'Generated {len(texts)}/{len(prompts)}', flush=True)
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    elif backend == 'vllm':
        from vllm import LLM, SamplingParams
        engine = LLM(model=str(model_dir), tensor_parallel_size=config.get('tensor_parallel_size', 1), gpu_memory_utilization=config.get('gpu_memory_utilization', 0.8), max_model_len=config.get('context_length', 8192), max_num_seqs=config.get('batch_size', 64), seed=config.get('seed', 3407))
        if beam > 1:
            from vllm.sampling_params import BeamSearchParams
            generated = engine.beam_search([{'prompt': p} for p in prompts], BeamSearchParams(beam_width=beam, max_tokens=max_tokens, temperature=temperature))
            texts = [[s.text for s in item.sequences] for item in generated]
        else:
            token_prompts = [{'prompt_token_ids': tokenizer.encode(prompt, add_special_tokens=False)} for prompt in prompts]
            request_batch_size = config.get('request_batch_size', 512)
            with (out / 'predictions.partial.jsonl').open('w') as partial:
                for start in range(0, len(prompts), request_batch_size):
                    generated = engine.generate(token_prompts[start:start + request_batch_size], SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=config.get('top_p', .95), top_k=config.get('top_k', 20)))
                    for offset, (prompt, item) in enumerate(zip(prompts[start:start + request_batch_size], generated)):
                        candidates = [prompt + s.text for s in item.outputs]
                        texts.append(candidates)
                        partial.write(json.dumps({'text': candidates}, ensure_ascii=False) + '\n')
                        details.append({'index': start + offset, 'input_tokens': len(item.prompt_token_ids), 'generated_tokens': len(item.outputs[0].token_ids), 'has_answer': bool(extract_str_answer(candidates[0])), 'reached_token_limit': item.outputs[0].finish_reason == 'length', 'finish_reason': item.outputs[0].finish_reason})
                    partial.flush()
                    (out / 'progress.json').write_text(json.dumps({'completed': len(texts), 'total': len(prompts), 'elapsed_seconds': time.time() - started}, indent=2))
                    print(f'Generated {len(texts)}/{len(prompts)}', flush=True)
    else:
        raise ValueError(f'Unknown backend {backend}')
    with (out / 'predictions.jsonl').open('w') as handle:
        for candidates in texts:
            handle.write(json.dumps({'text': candidates}, ensure_ascii=False) + '\n')
    with (out / 'predictions.txt').open('w') as handle:
        for candidates in texts:
            handle.write(candidates[0].replace('\n', '\\n').replace('\r', '\\r') + '\n')
    result = test_model_predict_result(str(out / 'ground_truth'), str(out / 'predictions.txt'))
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    report = {'project': 'SynReasoner', 'task': task, 'model': str(model_dir), 'data': str(Path(config['data']).resolve()), 'dataset_rows': total, 'evaluated_rows': len(data), 'backend': backend, 'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'config': config, 'elapsed_seconds': time.time() - started, 'generation': details, 'metrics': result}
    (out / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'Results: {out}', flush=True)

def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {'infer', 'inspect', 'evaluate', 'vote'}:
        arguments.insert(0, 'infer')
    parser = argparse.ArgumentParser(description='SynReasoner')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['inspect', 'infer']:
        p = commands.add_parser(command)
        p.add_argument('--config', default=str(PROJECT_ROOT / 'configs/local.json'))
        for name in ['data', 'model', 'output', 'backend', 'device']:
            p.add_argument('--' + name)
        for name in ['limit', 'max-tokens', 'beam-width', 'batch-size']:
            p.add_argument('--' + name, type=int)
    p = commands.add_parser('evaluate')
    p.add_argument('--data', required=True)
    p.add_argument('--predictions', required=True, help='One prediction per line, TXT format')
    p.add_argument('--output', required=True)
    p = commands.add_parser('vote')
    p.add_argument('--data', required=True)
    p.add_argument('--predictions', nargs='+', required=True, help='JSONL paths, highest priority first')
    p.add_argument('--output', required=True)
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(arguments)
    if args.command in ['inspect', 'infer']:
        config = json.loads(Path(args.config).read_text())
        for key, value in vars(args).items():
            if key not in ['command', 'config'] and value is not None:
                config[key] = value
        if args.command == 'infer':
            infer(config)
        else:
            data = load_data(config['data'])
            print(json.dumps({'rows': len(data), 'columns': data.column_names, 'task': verify_task(data), 'checkpoint': str(checkpoint(config['model'])), 'first_question': data[0]['question']}, indent=2, ensure_ascii=False))
    elif args.command == 'evaluate':
        data = load_data(args.data)
        verify_task(data)
        lines = Path(args.predictions).read_text().splitlines()
        if len(lines) != len(data):
            raise ValueError(f'Prediction count {len(lines)} != dataset count {len(data)}')
        result = test_model_predict_result(args.data, args.predictions)
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    elif args.command == 'vote':
        verify_task(load_data(args.data))
        cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc(args.data, args.predictions, args.output, 'consensus', topk=args.topk, num_process=args.workers, check_nmr_formula=False)
if __name__ == '__main__':
    main()
