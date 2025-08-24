import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re
from dataclasses import dataclass, field
import gc
import sys

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_id: str
    adapter_path: str = None
    use_quantization: bool = True
    torch_dtype: torch.dtype = field(default=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

MODEL_CONFIGS = [
    # Base Models (commented out for now)
    ModelConfig(
        name="Qwen2.5-7B-Instruct",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="Mistral-8B-Instruct-2410",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        use_quantization=False
    ),

    # ToW Trained Models
    ModelConfig(
        name="Qwen2.5-7B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Qwen2.5-7B-Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Qwen2.5-7B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Mistral-8B-Instruct-2410-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Mistral-8B-Instruct-2410",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Mistral-8B-Instruct-2410-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="Llama-3.1-8B-Instruct-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/Llama3.1_8B_Instruct",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/Llama-3.1-8B-Instruct-ToW",
        use_quantization=False
    ),
    ModelConfig(
        name="DeepSeek-R1-0528-Qwen3-8B-ToW",
        model_id="/scratch/jsong132/Increase_MLLM_Ability/Base_Models/DeepSeek-R1-0528-Qwen3-8B",
        adapter_path="/scratch/jsong132/Increase_MLLM_Ability/5_training/ToW_Models/DeepSeek-R1-0528-Qwen3-8B-ToW",
        use_quantization=False
    ),
]

# --- General Configuration ---
ARC_DATASET_PATH = "../../2_datasets/ARC-C_Ko-ARC/ARC.json"
KO_ARC_DATASET_PATH = "../../2_datasets/ARC-C_Ko-ARC/Ko-ARC.json"
BASE_OUTPUT_DIR = "arc_tow_model1_5shot_maxtoken_256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache" if not os.path.exists("/scratch/jsong132/.cache/huggingface") else "/scratch/jsong132/.cache/huggingface"
BATCH_SIZE = 32

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions for 5-shot ARC Evaluation ---
def create_5shot_prompt(item, examples, dataset_type="arc"):
    """
    Creates a 5-shot ARC prompt with examples from the same dataset type.
    """
    if dataset_type == "arc":
        prompt_parts = ["The following are multiple choice questions about science and reasoning."]
    else:  # ko-arc
        prompt_parts = ["다음은 과학과 추론에 관한 객관식 문제들입니다."]
    
    prompt_parts.append("")
    
    # Add 5 examples
    for i, example in enumerate(examples[:5], 1):
        question = example.get("question", "")
        choices = [
            example.get("A", ""),
            example.get("B", ""),
            example.get("C", ""),
            example.get("D", "")
        ]
        answer = example.get("answer", "")
        
        prompt_parts.append(question)
        prompt_parts.append(f"A. {choices[0]}")
        prompt_parts.append(f"B. {choices[1]}")
        prompt_parts.append(f"C. {choices[2]}")
        prompt_parts.append(f"D. {choices[3]}")
        if dataset_type == "arc":
            prompt_parts.append(f"Response: Let's think step by step. [thinking process] So the answer is {answer}.")
        else:  # ko-arc
            prompt_parts.append(f"응답: 단계적으로 생각해봅시다. [사고 과정] 따라서 답은 {answer}입니다.")
        prompt_parts.append("")
    
    # Add test question
    test_question = item.get("question", "")
    test_choices = [
        item.get("A", ""),
        item.get("B", ""),
        item.get("C", ""),
        item.get("D", "")
    ]
    
    prompt_parts.append(test_question)
    prompt_parts.append(f"A. {test_choices[0]}")
    prompt_parts.append(f"B. {test_choices[1]}")
    prompt_parts.append(f"C. {test_choices[2]}")
    prompt_parts.append(f"D. {test_choices[3]}")
    prompt_parts.append("")
    
    if dataset_type == "arc":
        prompt_parts.append("You should ONLY choose the letters from the options as your final answer.")
        prompt_parts.append("Response: Let's think step by step.")
    else:  # ko-arc
        prompt_parts.append("선택지에서 문자만을 최종 답변으로 선택해야 합니다.")
        prompt_parts.append("응답: 단계적으로 생각해봅시다.")
    
    return "\n".join(prompt_parts)

def extract_answer_first_token(model_output):
    """
    Extract answer from model output using first token approach.
    """
    cleaned_output = model_output.strip().upper()
    for char in cleaned_output:
        if char in ['A', 'B', 'C', 'D']:
            return char
    return None

def load_arc_data(filepath):
    """Loads ARC data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        return None

def get_ground_truth(item):
    """Returns the ground truth answer letter."""
    answer = item.get("answer", "")
    if answer in ['A', 'B', 'C', 'D']:
        return answer
    return None

def select_examples(data, test_item, num_examples=5):
    """
    Select examples for few-shot prompting, excluding the test item.
    """
    examples = []
    test_id = test_item.get("id", "")
    
    for item in data:
        if item.get("id", "") != test_id and len(examples) < num_examples:
            examples.append(item)
    
    return examples

# --- Single Model Evaluation Function with 5-shot Prompting ---
def evaluate_single_model(config: ModelConfig, arc_data: list, ko_arc_data: list, model_specific_output_dir: str):
    """
    Performs 5-shot ARC evaluation for a single model on both datasets.
    """
    results_filepath = os.path.join(model_specific_output_dir, f"results_{config.name}_5shot.json")
    log_filepath = os.path.join(model_specific_output_dir, f"eval_{config.name}_5shot.log")
    raw_gen_filepath = os.path.join(model_specific_output_dir, f"raw_generations_{config.name}_5shot.json")

    # --- Setup Logging for this specific model ---
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    
    # Remove previous file handlers to avoid duplicate logging
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"--- Starting 5-shot Evaluation for Model: {config.name} ({config.model_id}) ---")
    logger.info(f"Results will be saved to: {results_filepath}")

    model = None
    tokenizer = None
    raw_generations_list = []

    try:
        # --- Load Model and Tokenizer ---
        tokenizer_load_path = config.adapter_path if config.adapter_path else config.model_id
        logger.info(f"Loading tokenizer from: {os.path.abspath(tokenizer_load_path)}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, cache_dir=CACHE_DIR)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model {config.model_id}...")
        quantization_config_bnb = None
        if config.use_quantization:
            quantization_config_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=config.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            quantization_config=quantization_config_bnb,
            device_map=DEVICE,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

        if config.adapter_path:
            logger.info(f"Loading adapter from: {config.adapter_path}")
            model = PeftModel.from_pretrained(model, config.adapter_path)
            logger.info("Successfully loaded LoRA adapter.")

        model.eval()
        logger.info("Model and tokenizer loaded successfully.")

        # --- Evaluate on both datasets ---
        all_results = {}
        
        datasets = [
            ("ARC", arc_data, "arc"),
            ("Ko-ARC", ko_arc_data, "ko-arc")
        ]
        
        for dataset_name, dataset, dataset_type in datasets:
            logger.info(f"Starting evaluation on {dataset_name} dataset...")
            
            correct_predictions = 0
            total_predictions = 0
            errors_or_skipped = 0
            results_details = []
            
            # Batch processing loop
            num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
            pbar = tqdm(range(num_batches), desc=f"Evaluating {config.name} on {dataset_name} (5-shot, errors: 0)")
            for i in pbar:
                batch_start = i * BATCH_SIZE
                batch_end = batch_start + BATCH_SIZE
                batch = dataset[batch_start:batch_end]
                
                prompts = []
                ground_truths = []
                valid_items_in_batch = []

                for item in batch:
                    ground_truth = get_ground_truth(item)
                    if ground_truth is None:
                        errors_or_skipped += 1
                        # Log skipped item if needed
                        continue

                    examples = select_examples(dataset, item, num_examples=5)
                    prompt = create_5shot_prompt(item, examples, dataset_type)
                    prompts.append(prompt)
                    ground_truths.append(ground_truth)
                    valid_items_in_batch.append(item)

                if not prompts:
                    continue

                try:
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                        )
                    
                    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    input_lengths = inputs['input_ids'].shape[1]
                    # The generated text includes the prompt, so we need to remove it.
                    # Be careful with batch_decode, it might handle prompts differently.
                    # A safer way is to decode only the generated part.
                    output_only_tokens = outputs[:, input_lengths:]
                    decoded_outputs = tokenizer.batch_decode(output_only_tokens, skip_special_tokens=True)


                    for j, (item, ground_truth, gen_text) in enumerate(zip(valid_items_in_batch, ground_truths, decoded_outputs)):
                        generated_text_log = gen_text.strip()
                        model_answer_log = extract_answer_first_token(generated_text_log)
                        is_correct_log = False

                        if model_answer_log:
                            total_predictions += 1
                            if model_answer_log == ground_truth:
                                correct_predictions += 1
                                is_correct_log = True
                        else:
                            errors_or_skipped += 1
                            generated_text_log = f"EXTRACTION_FAILED: {generated_text_log}"

                        current_item_index = batch_start + j # or find a better way to get original index
                        results_details.append({
                            "index": current_item_index, 
                            "id": item.get("id", ""),
                            "ground_truth": ground_truth, 
                            "model_raw_output": generated_text_log,
                            "predicted_answer": model_answer_log, 
                            "is_correct": is_correct_log
                        })
                        
                        raw_generations_list.append({
                            "dataset": dataset_name,
                            "index": current_item_index, 
                            "id": item.get("id", ""),
                            "ground_truth": ground_truth,
                            "raw_output": generated_text_log, 
                            "extracted_answer": model_answer_log
                        })

                except Exception as e:
                    logger.error(f"Batch {i}: Inference error: {e}", exc_info=False)
                    errors_or_skipped += len(prompts)
                
                # Update progress bar with current error count
                pbar.set_description(f"Evaluating {config.name} on {dataset_name} (5-shot, errors: {errors_or_skipped})")

            
            # Calculate accuracy
            accuracy_standard = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            accuracy_strict = (correct_predictions / len(dataset) * 100) if len(dataset) > 0 else 0

            logger.info(f"--- 5-shot {dataset_name} Results for {config.name} ---")
            logger.info(f"Test Items: {len(dataset)}")
            logger.info(f"Valid Predictions: {total_predictions}")
            logger.info(f"Correct Predictions: {correct_predictions}")
            logger.info(f"Accuracy Standard: {accuracy_standard:.2f}%")
            logger.info(f"Accuracy Strict: {accuracy_strict:.2f}%")
            
            all_results[dataset_name] = {
                "test_items": len(dataset),
                "valid_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "accuracy_standard": accuracy_standard,
                "accuracy_strict": accuracy_strict,
                "details": results_details
            }

        # --- Save Results ---
        final_summary = {
            "model_config": {k: str(v) for k, v in config.__dict__.items()},
            "evaluation_type": "5-shot ARC Challenge",
            "datasets": all_results
        }
        
        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        with open(raw_gen_filepath, 'w', encoding='utf-8') as f:
            json.dump(raw_generations_list, f, indent=2, ensure_ascii=False)

        return all_results

    except Exception as e:
        logger.exception(f"A critical error occurred during evaluation for {config.name}: {e}")
        return None
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'file_handler' in locals() and file_handler in root_logger.handlers:
             root_logger.removeHandler(file_handler)
             file_handler.close()

def main():
    # Load datasets
    arc_data = load_arc_data(ARC_DATASET_PATH)
    ko_arc_data = load_arc_data(KO_ARC_DATASET_PATH)
    
    if not arc_data or not ko_arc_data:
        logger.error("Failed to load one or both datasets")
        return

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Store summary results for all models
    summary_results = {}

    for config in MODEL_CONFIGS:
        model_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        
        results = evaluate_single_model(config, arc_data, ko_arc_data, model_specific_output_dir)
        
        if results:
            summary_results[config.name] = {
                "model_id": config.model_id,
                "adapter_path": config.adapter_path,
                "ARC_accuracy_standard": results["ARC"]["accuracy_standard"],
                "ARC_accuracy_strict": results["ARC"]["accuracy_strict"],
                "Ko-ARC_accuracy_standard": results["Ko-ARC"]["accuracy_standard"],
                "Ko-ARC_accuracy_strict": results["Ko-ARC"]["accuracy_strict"]
            }

    # Save summary results
    summary_filepath = os.path.join(BASE_OUTPUT_DIR, "summary.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary results saved to: {summary_filepath}")
    
    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model Name':<30} {'ARC Acc (%)':<15} {'Ko-ARC Acc (%)':<15}")
    print("-"*80)
    
    for model_name, results in summary_results.items():
        arc_acc = results["ARC_accuracy_standard"]
        ko_arc_acc = results["Ko-ARC_accuracy_standard"]
        print(f"{model_name:<30} {arc_acc:<15.2f} {ko_arc_acc:<15.2f}")
    
    print("="*80)

if __name__ == "__main__":
    main()