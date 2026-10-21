"""SynReasoner supervised fine-tuning, including data preparation and logging.

All training implementation is contained in this file. Run python train.py --help.
"""
import argparse
import os
from pathlib import Path


def normalize_export_config(model_directory):
    """Keep model/tokenizer export metadata readable by Transformers 4 and 5."""
    import json
    directory = Path(model_directory)
    changes = {}
    path = directory / "config.json"
    config = json.loads(path.read_text())
    rope = config.get("rope_parameters") or {}
    if config.get("model_type") == "qwen2" and rope.get("rope_type", "default") == "default" and "rope_theta" in rope:
        if config.get("rope_theta") != rope["rope_theta"] or "rope_scaling" not in config:
            changes["model_config"] = {"before": {name: config.get(name) for name in ["rope_theta", "rope_scaling"]}, "rope_theta": rope["rope_theta"], "rope_scaling": None}
            config["rope_theta"] = rope["rope_theta"]
            config["rope_scaling"] = None
            temporary = path.with_suffix(".compat.tmp")
            temporary.write_text(json.dumps(config, indent=2) + "\n")
            temporary.replace(path)
    path = directory / "tokenizer_config.json"
    if path.is_file():
        tokenizer = json.loads(path.read_text())
        if tokenizer.get("tokenizer_class") == "TokenizersBackend":
            tokenizer["tokenizer_class"] = "PreTrainedTokenizerFast"
            temporary = path.with_suffix(".compat.tmp")
            temporary.write_text(json.dumps(tokenizer, indent=2) + "\n")
            temporary.replace(path)
            changes["tokenizer_config"] = {"before": "TokenizersBackend", "after": "PreTrainedTokenizerFast"}
    return changes or None


def train():
    # ======= IMPORTANT: Set environment variables before importing any libraries =======
    import os
    import tempfile
    import sys
    from datetime import datetime
    
    # ===== Add terminal output logging functionality =====
    class TerminalLogger:
        """Simple class to log all terminal output to a log file"""
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, 'a', encoding='utf-8')
            start_msg = f"\n=== Training Session Started: {datetime.now()} ===\n"
            self.log.write(start_msg)
            self.log.flush()
        
        def write(self, message):
            self.terminal.write(message)  # Display in terminal
            self.log.write(message)       # Save to file
            self.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Set up terminal output logging
    os.makedirs("./training_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    terminal_log_file = f"./training_logs/terminal_output_{timestamp}.log"
    print(f"Start recording terminal output to: {terminal_log_file}")
    sys.stdout = TerminalLogger(terminal_log_file)
    
    # 1. Solve Issue 2: Set GPU environment variables at the very beginning
    # Control which GPU to use
    GPU_IDS = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]  
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # 2. Solve Issue 1: Set HF cache directory
    # Set Hugging Face cache directory
    cache_dir = "./hf_cache"   # Can be changed to any path
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Set new temporary directory
    new_temp_dir = tempfile.mkdtemp(prefix="synreasoner_", dir="/tmp")
    os.makedirs(new_temp_dir, exist_ok=True)
    tempfile.tempdir = new_temp_dir
    os.environ["TMPDIR"] = new_temp_dir
    print(f"Temporary directory set to: {new_temp_dir}")
    print(f"HF cache directory set to: {cache_dir}")
    
    import torch
    import numpy as np
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    # Monkey-patch torch.load to force weights_only=False
    _real_torch_load = torch.load
    def _unsafe_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _real_torch_load(*args, **kwargs)
    torch.load = _unsafe_torch_load
    
    # Now import other libraries
    from unsloth import FastLanguageModel
    import glob
    from datasets import load_dataset, concatenate_datasets
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments, Trainer, TrainerCallback
    from unsloth import is_bfloat16_supported
    try:
        import wandb
    except ImportError:
        wandb = None
    import time
    import shutil
    import copy
    from typing import Dict, List, Any, Optional, Union
    from pathlib import Path
    
    # ==============================================================================
    # >>>>>>>>>>>>>>>>>>>>>>>>>  SETTINGS START HERE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================
    
    # Training data directory
    TRAIN_DATA_DIR = os.environ["REASONER_TRAIN_DATA"]
    
    # Training epochs 
    NUM_EPOCHS = int(os.environ.get("REASONER_EPOCHS", "20"))
    # Number of processes for data loading
    dataset_num_proc = 1
    
    # Model parameters
    MAX_SEQ_LENGTH = int(os.environ.get("REASONER_MAX_SEQ_LENGTH", "5000"))
    DTYPE = torch.bfloat16 
    LOAD_IN_4BIT = False
    # Pretrained weights 
    MODEL_NAME = os.environ["REASONER_BASE_MODEL"]
    
    # Training parameters
    BATCH_SIZE = int(os.environ.get("REASONER_BATCH_SIZE", "64")) 
    GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("REASONER_GRAD_ACCUM", "4")) 
    max_grad_norm = 0.15
    WARMUP_STEPS = 5
    LEARNING_RATE = float(os.environ.get("REASONER_LEARNING_RATE", "2e-4"))
    WEIGHT_DECAY = 0.01
    # How many steps to save once: Training Data Size / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * How many epochs per save
    SAVE_STEPS = 8000 # Currently saving every 1 epoch
    SAVE_TOTAL_LIMIT = 300
    MY_WANDB_KEY = None
    
    # LoRA parameters
    full_finetuning = True
    load_in_4bit = False
    load_in_8bit = False
    LORA_R = 512
    LORA_ALPHA = 2048
    LORA_DROPOUT = 0
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    # ==============================================================================
    # <<<<<<<<<<<<<<<<<<<<<<<<<   SETTINGS END HERE   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ==============================================================================
    
    
    # Prompt template
    TRAIN_PROMPT_STYLE = '{}<think>{}</think><answer>{}</answer>'
    
    # =====================
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # ============= WandB Connection Fix Section - Minimal Changes =============
    def setup_wandb_with_fallback(wandb_key=None):
        """
        Configures WandB logging with multiple fallback levels.
        1. Online: If wandb_key is provided, attempts to connect to the cloud.
        2. Offline: If key is missing or connection fails, records data locally in WandB format.
        3. Local: If both fail, switches to a simple local .txt log file.
        
        Args:
            wandb_key (str, optional): The WandB API Key. Defaults to None.
            
        Returns:
            tuple: (wandb_available: bool, report_to: list)
        """
        # 1. Attempt Online Mode (Only if key is provided)
        if wandb_key:
            try:
                print("WandB key detected. Attempting online connection...")
                os.environ['WANDB_INIT_TIMEOUT'] = '30'
                
                # Login with the provided key
                wandb.login(key=wandb_key, timeout=30)
                
                # Connection test
                test_run = wandb.init(project="medical-reasoning-sft", mode="online", reinit=True)
                if test_run:
                    test_run.finish()
                    print("✓ WandB online connection successful")
                    return True, ["wandb"]
            except Exception as e:
                print(f"WandB online login failed or timed out: {e}")
                print("Switching to fallback modes...")
        else:
            print("No WandB key provided. Skipping online mode.")
    
        # 2. Attempt Offline Mode (Records metrics locally without needing a key)
        try:
            print("Attempting WandB offline mode...")
            offline_dir = "./wandb_offline"
            os.makedirs(offline_dir, exist_ok=True)
            
            os.environ['WANDB_MODE'] = 'offline'
            os.environ['WANDB_DIR'] = offline_dir
            
            # Offline init does not require a key
            test_run = wandb.init(project="medical-reasoning-sft", mode="offline", reinit=True)
            if test_run:
                test_run.finish()
                print(f"✓ WandB offline mode enabled. Data saved to: {offline_dir}")
                print("Note: Run 'wandb sync ./wandb_offline' later to upload data.")
                return True, ["wandb"]
                
        except Exception as e2:
            print(f"WandB offline mode also failed: {e2}")
    
        # 3. Disable WandB and enable simple local text logging
        print("✓ WandB unavailable. Enabling local text logging mode.")
        os.environ['WANDB_DISABLED'] = 'true'
        
        log_dir = "./training_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        
        # Write initial session info
        try:
            with open(log_file, "w", encoding='utf-8') as f:
                f.write("===== Training Log (Local Mode) =====\n")
                f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                # Uses .get() to avoid crashing if variables are not defined globally
                f.write(f"Model: {globals().get('MODEL_NAME', 'Unknown')}\n")
                f.write("=" * 50 + "\n\n")
            
            global LOCAL_LOG_FILE
            LOCAL_LOG_FILE = log_file
            print(f"Local log file created at: {log_file}")
        except Exception as e3:
            print(f"Failed to create local log file: {e3}")
    
        return False, ["none"]
    
    # Local logging function
    def log_to_file(message, metrics=None):
        """Log message to local log file"""
        if 'LOCAL_LOG_FILE' in globals():
            with open(LOCAL_LOG_FILE, "a", encoding='utf-8') as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
                if metrics:
                    f.write(f"Metrics: {metrics}\n")
                f.write("\n")
    
    # Execute WandB setup
    WANDB_AVAILABLE, REPORT_TO_SETTING = setup_wandb_with_fallback(wandb_key=MY_WANDB_KEY)
    # ============= WandB Connection Fix Section End =============
    
    # Data construction function - Training set
    def formatting_prompts_func(examples):
        """
        Prepare input data for training set, including thinking process and final answer.
        """
        inputs = examples["question"]
        cots = examples["cot"]
        outputs = examples["answer"]
        texts = []
        for input, cot, output in zip(inputs, cots, outputs):
            text = TRAIN_PROMPT_STYLE.format(input, cot, output) + EOS_TOKEN
            # text = TRAIN_PROMPT_STYLE.format(input, "", output) + EOS_TOKEN   # Use this line for no CoT 
            texts.append(text)
        return {
            "text": texts,
        }
    
    # Load multiple arrow files and merge into one dataset
    def load_multiple_arrow_files(directory_path, split="train"):
        """
        Load multiple .arrow files from a directory and merge them into one dataset.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist")
        
        # Get all .arrow files in the directory
        arrow_files = glob.glob(os.path.join(directory_path, "*.arrow"))
        
        if not arrow_files:
            raise FileNotFoundError(f"No .arrow files found in directory {directory_path}")
        
        print(f"Found {len(arrow_files)} .arrow files in {directory_path}")
        
        # Load each file and create a list of datasets
        datasets_list = []
        for file_path in arrow_files:
            print(f"Loading file: {os.path.basename(file_path)}")
            try:
                dataset = load_dataset("arrow", data_files=file_path, split=split)
                datasets_list.append(dataset)
                print(f"Successfully loaded, this file contains {len(dataset)} samples")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        
        if not datasets_list:
            raise ValueError("No datasets successfully loaded")
        
        # Merge all datasets
        if len(datasets_list) == 1:
            return datasets_list[0]
        else:
            return concatenate_datasets(datasets_list)
    
    # 1. Instantiate Model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit = load_in_4bit,
        load_in_8bit = load_in_8bit,
        full_finetuning=full_finetuning,
        trust_remote_code=False,
        local_files_only=True, # 强制只查找本地
    )
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    print("Model loaded")
    
    # 2. Load Training Data
    print("Loading training data...")
    try:
        train_dataset = load_multiple_arrow_files(TRAIN_DATA_DIR)
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        print(f"Training data loaded, total {len(train_dataset)} samples")
        log_to_file(f"Training data loaded", {"Total Samples": len(train_dataset)})
    except Exception as e:
        print(f"Error loading training data: {e}")
        log_to_file(f"Failed to load training data", {"Error": str(e)})
        sys.exit(1)
    
    # 3. Create Fine-tuning Model
    print("Configuring LoRA fine-tuning model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("LoRA model configuration complete")
    log_to_file("LoRA model configuration complete", {"r": LORA_R, "alpha": LORA_ALPHA})
    
    # 4. Create Training Arguments
    print("Configuring training arguments...")
    training_args = SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=dataset_num_proc,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=False,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        max_grad_norm = max_grad_norm, 
        # Save regular checkpoints
        save_strategy="epoch",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        # Use fixed reporting settings
        report_to=REPORT_TO_SETTING,
        run_name="medical-reasoning-sft" if WANDB_AVAILABLE else None,
    )
    
    log_to_file("Training arguments configuration complete", {
        "Batch Size": BATCH_SIZE,
        "Gradient Accumulation Steps": GRADIENT_ACCUMULATION_STEPS,
        "Epochs": NUM_EPOCHS,
        "Learning Rate": LEARNING_RATE,
        "WandB Available": WANDB_AVAILABLE
    })
    
    #############################   Data Check Specific   ################################
    # 2. Save training dataset content before passing to model
    import json
    import time
    from tqdm import tqdm
    
    def save_complete_dataset_to_json(dataset, output_file="complete_dataset.json"):
        """
        Save complete dataset to JSON file without omitting any content.
        
        Args:
            dataset: The dataset to save
            output_file: Output JSON file path
        """
        start_time = time.time()
        total_samples = len(dataset)
        
        print(f"Starting to save complete dataset, total {total_samples} samples...")
        
        # Create data structure
        dataset_info = {
            "total_samples": total_samples,
            "samples": []
        }
        
        # Use tqdm to show progress bar
        for i in tqdm(range(total_samples), desc="Saving dataset"):
            sample_data = {
                "index": i,
                "data": {}
            }
            
            # Add all fields in the sample, do not truncate
            for key in dataset[i]:
                sample_data["data"][key] = dataset[i][key]
            
            dataset_info["samples"].append(sample_data)
        
        # Save to JSON file
        print(f"Writing to JSON file: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        # Calculate and show file size
        import os
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Dataset save complete!")
        print(f"File size: {file_size:.2f} MB")
        print(f"Processing time: {duration:.2f} seconds")
        
        return output_file
    
    # save_complete_dataset_to_json(train_dataset)  # Save all samples
    
    ##############################################################################
    # 6. Create Trainer
    print("Configuring Trainer...")
    class ExportCompatibilityCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            normalize_export_config(Path(args.output_dir) / f"checkpoint-{state.global_step}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        callbacks=[ExportCompatibilityCallback()],
        
    )
    print("Trainer configuration complete")
    log_to_file("Trainer configuration complete")
    
    # 7. Training
    print("Starting training...")
    log_to_file("Starting training")
    try:
        trainer_stats = trainer.train(resume_from_checkpoint=os.environ.get("REASONER_RESUME") or None)
        # trainer_stats = trainer.train(resume_from_checkpoint = True)
        print("Training complete")
        log_to_file("Training complete", {"Training Stats": str(trainer_stats)})
        
        # 8. Save Final Model
        print("Saving final model...")
        trainer.save_model("./final_model")
        tokenizer.save_pretrained("./final_model")
        normalize_export_config("./final_model")
        trainer.save_state()
        trainer.save_metrics("train", trainer_stats.metrics)
        print("Final model saved to ./final_model")
        log_to_file("Final model saved", {"Path": "./final_model"})
    except Exception as e:
        print(f"Error during training: {e}")
        log_to_file("Error during training", {"Error": str(e)})
        raise
    
    # If using offline mode, remind user to sync data
    if not WANDB_AVAILABLE or 'WANDB_MODE' in os.environ and os.environ['WANDB_MODE'] == 'offline':
        print("\n===== WandB Data Sync Reminder =====")
        if os.path.exists("./wandb_offline"):
            print("Training complete! Since WandB offline mode was used, please run the following command to sync data:")
            print("cd ./wandb_offline && wandb sync .")
        print(f"Local log file location: {LOCAL_LOG_FILE if 'LOCAL_LOG_FILE' in globals() else 'None'}")
    
    log_to_file("Training fully completed!")
    
    # ===== Restore terminal output and show log info =====
    # Write end marker to log
    end_msg = f"=== Training Session Ended: {datetime.now()} ===\n"
    sys.stdout.log.write(end_msg)
    sys.stdout.log.close()
    
    # Restore original stdout (so the final summary only shows in terminal)
    sys.stdout = sys.stdout.terminal
    
    print("\n===== Training fully completed! =====")
    print(f"Full terminal output saved to: {terminal_log_file}")
    
    # Show log file size
    if os.path.exists(terminal_log_file):
        file_size = os.path.getsize(terminal_log_file) / (1024 * 1024)  # MB
        print(f"Terminal log file size: {file_size:.2f} MB")
    
    # Show all generated log files
    print(f"\nGenerated log files:")
    print(f"1. Full Terminal Output: {terminal_log_file}")
    if 'LOCAL_LOG_FILE' in globals():
        print(f"2. Training Key Milestones: {LOCAL_LOG_FILE}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train SynReasoner with Unsloth")
    parser.add_argument("--data", required=True, help="Arrow training dataset directory")
    parser.add_argument("--model", required=True, help="Local base model directory")
    parser.add_argument("--output", required=True, help="Training output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=5000)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args(argv)
    from datasets import Dataset
    files = sorted(Path(args.data).glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No Arrow shards: {args.data}")
    for file in files:
        data = Dataset.from_file(str(file))
        if not {"question", "cot", "answer"}.issubset(data.column_names):
            raise ValueError(f"Missing question/cot/answer columns: {file}")
        if not len(data):
            raise ValueError(f"Empty training shard: {file}")
        first_answer = str(data[0]["answer"])
        if "Formula" in first_answer or not any(tag in first_answer for tag in ["Reactant", "F1", "Product"]):
            raise ValueError(f"Expected a synthesis or retrosynthesis training dataset: {file}")
    model_dir = Path(args.model).resolve()
    candidates = [model_dir] if (model_dir / "config.json").is_file() else sorted(path.parent for path in model_dir.rglob("config.json"))
    candidates = [path for path in candidates if list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))]
    if len(candidates) != 1:
        raise ValueError(f"Expected one base checkpoint; found {candidates}")
    os.environ.update(
        REASONER_TRAIN_DATA=str(Path(args.data).resolve()),
        REASONER_BASE_MODEL=str(candidates[0]),
        REASONER_EPOCHS=str(args.epochs),
        REASONER_BATCH_SIZE=str(args.batch_size),
        REASONER_GRAD_ACCUM=str(args.gradient_accumulation_steps),
        REASONER_LEARNING_RATE=str(args.learning_rate),
        REASONER_MAX_SEQ_LENGTH=str(args.max_seq_length),
        REASONER_RESUME=str(Path(args.resume_from_checkpoint).resolve()) if args.resume_from_checkpoint else "",
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    train()


if __name__ == "__main__":
    main()
