import json
import os
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM
from accelerate import Accelerator
from tqdm import tqdm
import time
import re

# Model configurations
MODELS = {
    "code-llama": ("codellama/CodeLlama-7b-hf", 2048),
    "mistral-7B": ("mistralai/Mistral-7B-v0.1", 2048),
    "gpt-j-6b": ("EleutherAI/gpt-j-6B", 2048)
}

# Generation parameters
BATCH_SIZE = 2  # Reduced to prevent OOM
TEMPERATURE = 0.4  # Lower temperature for more focused output
TOP_P = 0.9  # Slightly more restrictive
TOP_K = 50
MAX_NEW_TOKENS = 250  # Reduced to prevent OOM

def get_optimal_batch_size(num_gpus: int, model_name: str) -> int:
    """Determine optimal batch size based on available GPUs and model."""
    if num_gpus >= 2:
        if model_name == "llama-3-8B":
            return 4  # Can handle larger batches with 2 GPUs
        elif model_name == "mistral-7B":
            return 3  # Medium batch size for Mistral
        elif model_name == "gpt-j-6b":
            return 2  # Conservative for GPT-J
    else:
        return 1  # Single GPU or CPU
    
    return 1  # Default fallback

def monitor_gpu_memory():
    """Monitor and print GPU memory usage."""
    if torch.cuda.is_available():
        print("\nGPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - allocated
            print(f"GPU {i}: {allocated:.2f}GB used, {free:.2f}GB free, {total:.2f}GB total")
        print()

def load_huggingface_token():
    """Load HuggingFace API token from file."""
    try:
        with open("huggingface_api.txt", "r") as f:
            token = f.read().strip()
        return token
    except FileNotFoundError:
        print("Warning: huggingface_api.txt not found. Using public models only.")
        return None

def load_ast_test_data(dataset_name: str, max_queries: int | None = None) -> List[Dict]:
    """Load AST test data from the specified dataset."""
    file_path = f"ast-datasets/{dataset_name}-ast.test.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if max_queries is not None:
            data = data[:max_queries]
        print(f"Loaded {len(data)} test queries from {dataset_name}")
        return data
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return []

def load_similar_queries(dataset_name: str, sql_query: str) -> List[Dict]:
    """Load top 5 similar queries by matching the exact SQL query."""
    dataset_dir = Path(f"similar_queries_results/{dataset_name}")
    if not dataset_dir.exists():
        print(f"Warning: {dataset_dir} does not exist")
        return []
    
    # Search through all files to find the exact query match
    for file_path in dataset_dir.glob('*.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this file contains our query
            if data.get('dev_query', '').strip() == sql_query.strip():
                similar_queries = data.get('similar_train_queries', [])[:5]  # Ensure we get exactly 5
                print(f"Found {len(similar_queries)} similar queries for: {sql_query[:50]}...")
                return similar_queries
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Warning: No matching query found for: {sql_query[:100]}...")
    return []

def create_few_shot_prompt(sql_query: str, similar_queries: List[Dict]) -> str:
    """Create a few-shot prompt with examples."""
    # Advanced Chain of Thought prompt for high-quality SQL-to-NL translation
    BASE_PROMPT_TEMPLATE = """You are an expert SQL analyst who translates SQL queries into natural language questions. Follow this detailed Chain of Thought process:

**Step 1: Analyze SQL Structure**
- Identify SELECT columns and their aliases
- Identify FROM/JOIN tables and relationships  
- Identify WHERE conditions and filters
- Identify GROUP BY, ORDER BY, LIMIT clauses
- Understand the query's logical flow

**Step 2: Understand Business Intent**
- What business question does this query answer?
- What entities are being queried?
- What specific information is being requested?
- What relationships are being explored?

**Step 3: Generate Natural Language Question**
- Use conversational, human-like language
- Include ALL important details from the SQL
- Mention specific columns being selected
- Include table/entity names when relevant
- Use appropriate verbs (show, list, find, count, etc.)
- Make it specific and detailed, not generic

**Step 4: Quality Check**
- Does the question capture ALL the SQL logic?
- Is it natural and conversational?
- Does it include the right level of detail?
- Would a human ask this question this way?

**Guidelines for High-Quality Translation:**
- Be SPECIFIC: Include column names, table names, conditions
- Be NATURAL: Use conversational language
- Be COMPLETE: Don't omit important details
- Be PRECISE: Match the exact intent of the SQL

**CRITICAL: Output ONLY the natural language question. Do NOT generate additional examples or explanations. Stop after the first translation.**

Now translate this SQL query to natural language:"""
    
    prompt = f"{BASE_PROMPT_TEMPLATE}\n\n"
    
    # Add few-shot examples (ensure we use exactly 5)
    examples_to_use = similar_queries[:5]  # Take exactly 5 examples
    print(f"Using {len(examples_to_use)} examples for few-shot learning")
    
    for i, example in enumerate(examples_to_use, 1):
        train_query = example.get('train_query', '')
        train_translation = example.get('train_translation', '')
        prompt += f"SQL: {train_query}\n"
        prompt += f"Natural Language: {train_translation}\n\n"
    
    # Add the target query
    prompt += f"SQL: {sql_query}\n"
    prompt += "Natural Language:"
    
    return prompt

def prepare_prompts(prompts: List[str], tokenizer, batch_size: int, accelerator) -> List[Dict]:
    """Prepare prompts for batch processing with advanced GPU optimization."""
    batches = []
    # Set a reasonable max length to avoid "int too big to convert" error
    max_length = min(tokenizer.model_max_length, 8192) - MAX_NEW_TOKENS
    
    # Toggle padding side for better performance
    original_padding_side = tokenizer.padding_side
    original_trunc_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left" 
    
    try:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Pad to multiples of 8 for better GPU performance
            tokenized = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Ensure padding to multiples of 8
            seq_len = tokenized["input_ids"].shape[1]
            if seq_len % 8 != 0:
                pad_len = 8 - (seq_len % 8)
                if tokenizer.padding_side == "left":
                    # Pad on the left
                    pad_tensor = torch.full((tokenized["input_ids"].shape[0], pad_len), 
                                          tokenizer.pad_token_id, dtype=tokenized["input_ids"].dtype)
                    tokenized["input_ids"] = torch.cat([pad_tensor, tokenized["input_ids"]], dim=1)
                    tokenized["attention_mask"] = torch.cat([torch.zeros_like(pad_tensor), tokenized["attention_mask"]], dim=1)
                else:
                    # Pad on the right
                    pad_tensor = torch.full((tokenized["input_ids"].shape[0], pad_len), 
                                          tokenizer.pad_token_id, dtype=tokenized["input_ids"].dtype)
                    tokenized["input_ids"] = torch.cat([tokenized["input_ids"], pad_tensor], dim=1)
                    tokenized["attention_mask"] = torch.cat([tokenized["attention_mask"], torch.zeros_like(pad_tensor)], dim=1)
            
            batches.append(tokenized)
    finally:
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_trunc_side
    
    return batches

def clean_translation_text(text: str) -> str:
    """Clean and post-process translation text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common artifacts and prefixes
    text = re.sub(r'^Natural Language:\s*', '', text)
    text = re.sub(r'^Translation:\s*', '', text)
    text = re.sub(r'^Response:\s*', '', text)
    text = re.sub(r'^Answer:\s*', '', text)
    text = re.sub(r'^Output:\s*', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove quotes if they wrap the entire text
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    # CRITICAL: Extract only the first translation by stopping at the first "SQL:" or similar pattern
    # Split by common patterns that indicate the start of additional examples
    patterns_to_split = [
        r'\s+SQL:\s*',
        r'\s+Natural Language:\s*',
        r'\s+Translation:\s*',
        r'\s+Example\s+\d+:',
        r'\s+Query\s+\d+:',
        r'\s+Results:\s*',
        r'\s+Output:\s*',
        r'\s+Answer:\s*',
        r'\s+Here\s+is\s+the\s+result',
        r'\s+The\s+result\s+is',
        r'\s+Here\s+are\s+the\s+results',
        r'\s+The\s+results\s+are',
        r'\s+This\s+will\s+return',
        r'\s+The\s+output\s+will\s+be'
    ]
    
    for pattern in patterns_to_split:
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[0]  # Take only the first part
            break
    
    # Remove any remaining SQL or technical artifacts
    text = re.sub(r'SQL Query \d+:\s*.*?Natural Language:', '', text, flags=re.DOTALL)
    text = re.sub(r'### System:.*?### User:', '', text, flags=re.DOTALL)
    text = re.sub(r'###.*?###', '', text, flags=re.DOTALL)
    
    # Remove table structure artifacts (very aggressive)
    text = re.sub(r'CREATE TABLE.*?;', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Let me know.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'I will assist.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'feel free to ask.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'FOREIGN KEY.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'PRIMARY KEY.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any text after the first sentence that looks like a question
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('CREATE') and not line.startswith('Let me') and not line.startswith('I will') and not line.startswith('Here is'):
            clean_lines.append(line)
    
    text = ' '.join(clean_lines)
    
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Ensure proper sentence ending
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    # Final cleanup - remove any remaining artifacts
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text.strip()

def generate_translations(model, tokenizer, prompts: List[str], accelerator) -> List[str]:
    """Generate natural language translations for SQL queries with advanced GPU optimization."""
    translations = []
    
    # Check if model is on GPU or CPU
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Check if model is distributed across multiple GPUs
    is_distributed = False
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        is_distributed = len(set(model.hf_device_map.values())) > 1
        print(f"Model is distributed across {len(set(model.hf_device_map.values()))} devices")
    
    # Prepare prompts in batches
    prompt_batches = prepare_prompts(prompts, tokenizer, batch_size=BATCH_SIZE, accelerator=accelerator)
    
    for batch_idx, prompts_tokenized in enumerate(tqdm(prompt_batches, desc="Generating translations")):
        try:
            # Move to same device as model
            if device.type == 'cuda':
                prompts_tokenized = {k: v.cuda() for k, v in prompts_tokenized.items()}
            else:
                prompts_tokenized = {k: v.cpu() for k, v in prompts_tokenized.items()}
            
            # Use gradient-free generation for better memory efficiency
            with torch.no_grad():
                outputs_tokenized = model.generate(
                    **prompts_tokenized,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    num_return_sequences=1,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    use_cache=True
                )
            
            # Remove prompt from generated tokens
            outputs_tokenized = [
                tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)
            ]
            
            # Decode generated tokens
            batch_translations = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            
            # Clean and post-process translations
            cleaned_translations = [clean_translation_text(text) for text in batch_translations]
            translations.extend(cleaned_translations)
            
            # Clear intermediate tensors to free memory
            del prompts_tokenized, outputs_tokenized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM during generation for batch {batch_idx}: {e}")
            # Try with smaller batch or CPU fallback
            if BATCH_SIZE > 1:
                print("Trying with batch size 1...")
                # This would require restructuring, but for now just return error
                translations.append(f"ERROR: CUDA out of memory during generation")
            else:
                translations.append(f"ERROR: CUDA out of memory - batch size already 1")
        except Exception as e:
            print(f"Error during generation for batch {batch_idx}: {e}")
            translations.append(f"ERROR: {str(e)}")
    
    return translations

def process_dataset(model, tokenizer, dataset_name: str, model_name: str, accelerator, max_queries: int | None = None) -> List[Dict]:
    """Process a single dataset and generate translations."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset with {model_name}")
    print(f"{'='*60}")
    
    # Load test data
    test_data = load_ast_test_data(dataset_name, max_queries)
    if not test_data:
        print(f"ERROR: No test data loaded for {dataset_name}")
        return []
    
    print(f"Total test queries: {len(test_data)}")
    
    # Check existing results to skip already processed queries
    output_dir = Path("ast_result2")
    output_file = output_dir / f"{dataset_name}_{model_name}_translations.json"
    existing_results = []
    
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"Found {len(existing_results)} existing translations for {dataset_name}_{model_name}")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            existing_results = []
    
    # Create a map of existing translations and identify failed ones
    existing_map = {result.get('query_index', -1): result for result in existing_results}
    failed_indices = set()
    missing_indices = set()
    
    # Check all queries in the dataset
    for i in range(len(test_data)):
        if i in existing_map:
            # Check if existing translation is failed
            translation = existing_map[i].get('natural_language_translation', '')
            is_failed = (
                translation.startswith('ERROR:') or 
                'CUDA out of memory' in translation or
                'out of memory' in translation.lower() or
                translation.strip() == '' or
                len(translation.strip()) < 10  # Too short to be a valid translation
            )
            if is_failed:
                failed_indices.add(i)
                print(f"Found failed translation for query {i}: {translation[:50]}...")
        else:
            # Query is missing entirely
            missing_indices.add(i)
            if i < 10:  # Highlight early missing queries
                print(f"Missing early query {i}")
    
    print(f"Analysis: {len(existing_map)} existing, {len(failed_indices)} failed, {len(missing_indices)} missing")
    print(f"Total queries to process: {len(failed_indices) + len(missing_indices)}")
    
    # Create a set of all queries that need processing
    queries_to_process = failed_indices.union(missing_indices)
    print(f"Processing {len(queries_to_process)} queries: {len(failed_indices)} failed + {len(missing_indices)} missing")
    
    if not queries_to_process:
        print("No queries to process!")
        return []
    
    results = []
    skipped_count = 0
    fixed_count = 0
    processed_count = 0
    
    # Process only the queries that need attention
    for i in tqdm(queries_to_process, desc=f"Processing {dataset_name}"):
        try:
            test_item = test_data[i]
            sql_query = test_item.get('query', '')
            if not sql_query:
                print(f"Warning: No SQL query found for index {i}")
                continue
            
            print(f"\n--- Processing Query {i+1}/{len(test_data)} (Index {i}) ---")
            print(f"SQL: {sql_query[:100]}...")
            
            # Load similar queries for few-shot learning
            similar_queries = load_similar_queries(dataset_name, sql_query)
            
            # Create prompt with few-shot examples
            prompt = create_few_shot_prompt(sql_query, similar_queries)
            
            # Generate translation
            try:
                translation = generate_translations(model, tokenizer, [prompt], accelerator)[0]
                
                result = {
                    'query_index': i,
                    'sql_query': sql_query,
                    'natural_language_translation': translation.strip(),
                    'few_shot_examples': similar_queries
                }
                results.append(result)
                
                # Save result immediately after each translation
                save_single_result(result, dataset_name, model_name)
                
                # Print translation and progress
                print(f"Translation: {translation.strip()}")
                print(f"Progress: {processed_count + 1}/{len(queries_to_process)} queries processed")
                print("-" * 50)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error generating translation for query {i}: {e}")
                import traceback
                traceback.print_exc()
                
                result = {
                    'query_index': i,
                    'sql_query': sql_query,
                    'natural_language_translation': f"ERROR: {str(e)}",
                    'few_shot_examples': similar_queries
                }
                results.append(result)
                
                # Save error result immediately
                save_single_result(result, dataset_name, model_name)
                
                # Continue with next query instead of stopping
                continue
            
            # Add a small delay to prevent overwhelming the system
            import time
            time.sleep(0.1)
            
        except Exception as e:
            print(f"ERROR processing query {i}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error result and continue
            error_result = {
                'query_index': i,
                'sql_query': test_data[i].get('query', '') if i < len(test_data) else '',
                'natural_language_translation': f"ERROR: {str(e)}",
                'few_shot_examples': []
            }
            save_single_result(error_result, dataset_name, model_name)
            
            # Continue with next query instead of stopping
            continue
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY FOR {dataset_name.upper()} - {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Total queries in dataset: {len(test_data)}")
    print(f"Queries processed in this run: {processed_count}")
    print(f"Total results saved: {len(results)}")
    print(f"Total successful translations: {len([r for r in results if not r.get('natural_language_translation', '').startswith('ERROR:')])}")
    print(f"{'='*60}")
    
    return results

def save_results(results: List[Dict], dataset_name: str, model_name: str):
    """Save results to the ast_result2 directory."""
    output_dir = Path("ast_result2")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{dataset_name}_{model_name}_translations.json"
    
    # Debug: Print first few results
    print(f"\nSaving {len(results)} results to {output_file}")
    if results:
        print(f"First result sample:")
        print(f"  Query: {results[0].get('sql_query', 'N/A')[:100]}...")
        print(f"  Translation: {results[0].get('natural_language_translation', 'N/A')[:100]}...")
        print(f"  Examples used: {len(results[0].get('few_shot_examples', []))}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Total translations generated: {len(results)}")

def save_single_result(result: Dict, dataset_name: str, model_name: str):
    """Save a single result to the ast_result2 directory."""
    output_dir = Path("ast_result2")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{dataset_name}_{model_name}_translations.json"
    
    # Load existing results if file exists
    existing_results = []
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Error reading existing file: {e}")
            existing_results = []
    
    # Check if we're replacing an existing result or adding new one
    query_index = result.get('query_index', -1)
    replaced = False
    
    for i, existing_result in enumerate(existing_results):
        if existing_result.get('query_index', -1) == query_index:
            # Replace existing result (especially failed ones)
            existing_results[i] = result
            replaced = True
            print(f"Replaced result for query {query_index}")
            break
    
    if not replaced:
        # Add new result
        existing_results.append(result)
        print(f"Added new result for query {query_index}")
    
    # Save updated results
    with open(output_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"Saved {len(existing_results)} total results to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate natural language translations of SQL queries using LLMs")
    parser.add_argument("--model_name", type=str, choices=list(MODELS.keys()) + ["all"], 
                       default="all", help="Model to use for generation (use 'all' for all models)")
    parser.add_argument("--datasets", nargs="+", default=["spider", "sparc", "cosql"],
                       help="Datasets to process")
    parser.add_argument("--max_queries", type=int, default=None,
                       help="Maximum number of queries to process per dataset (None for all)")
    parser.add_argument("--test_mode", action="store_true",
                       help="Run in test mode with limited queries")
    
    args = parser.parse_args()
    
    # Set max_queries based on test_mode
    if args.test_mode:
        max_queries = 10
    else:
        max_queries = args.max_queries if args.max_queries is not None and args.max_queries > 0 else None
    
    # Determine which models to run
    if args.model_name == "all":
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = [args.model_name]
    
    # Load HuggingFace token
    token = load_huggingface_token()
    
    # Initialize accelerator with simple configuration
    accelerator = Accelerator()
    
    # Track models processed to ensure proper cleanup
    models_processed = 0
    
    # Process each model
    for model_name in models_to_run:
        print(f"\n{'='*80}")
        print(f"PROCESSING MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        # Load model and tokenizer
        model_path = MODELS[model_name][0]
        print(f"Loading model: {model_name}")
        
        # Check number of available GPUs and memory
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(num_gpus):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")
        else:
            print("WARNING: CUDA is not available!")
        
        try:
            # COMPLETE GPU MEMORY CLEANUP BEFORE LOADING NEW MODEL
            print(f"\n{'='*60}")
            print(f"PREPARING GPU MEMORY FOR {model_name.upper()}")
            print(f"{'='*60}")
            
            # AGGRESSIVE GPU MEMORY CLEANUP BEFORE LOADING NEW MODEL
            if torch.cuda.is_available():
                print("Performing aggressive GPU memory cleanup before loading...")
                
                # Force memory release on all GPUs
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Force memory release by allocating and freeing a small tensor
                    try:
                        temp_tensor = torch.zeros(1, device=f'cuda:{i}')
                        del temp_tensor
                        torch.cuda.empty_cache()
                    except:
                        pass
                
                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                print(f"Aggressive GPU memory cleanup completed before loading {model_name}")
                
                # Monitor memory before loading
                monitor_gpu_memory()
            
            # Force garbage collection before loading
            import gc
            for _ in range(5):  # Increased iterations
                gc.collect()
            
            # Force GPU loading for all models with 2 GPUs available
            if num_gpus >= 2:
                print(f"Loading {model_name} with FORCED GPU distribution across {num_gpus} GPUs")
                
                if model_name == "gpt-j-6b":
                    # GPT-J-6B: Distributed loading across both GPUs
                    print(f"Loading GPT-J-6B with distributed device map across {num_gpus} GPUs")
                    # Create balanced device map for GPT-J-6B (28 layers)
                    device_map = {}
                    layers_per_gpu = 28 // num_gpus
                    for i in range(28):
                        gpu_id = i // layers_per_gpu
                        if gpu_id >= num_gpus:
                            gpu_id = num_gpus - 1
                        device_map[f"transformer.h.{i}"] = gpu_id
                    
                    # Put embedding and output layers on GPU 0
                    device_map["transformer.wte"] = 0
                    device_map["transformer.ln_f"] = 0
                    device_map["lm_head"] = 0
                    
                    model = GPTJForCausalLM.from_pretrained(
                        "EleutherAI/gpt-j-6B",
                        revision="float16",
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        cache_dir=f"/mnt/data2/ast-icl/huggingface_cache"
                    )
                    print(f"GPT-J-6B loaded successfully across {num_gpus} GPUs")
                else:
                    # Llama-3-8B and Mistral-7B: Use advanced device mapping
                    print(f"Loading {model_name} with optimized device map across {num_gpus} GPUs")
                    
                    # Get model config to determine number of layers
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_path, token=token)
                    num_layers = config.num_hidden_layers
                    print(f"Model has {num_layers} layers")
                    
                    # Create balanced device map
                    device_map = {}
                    layers_per_gpu = num_layers // num_gpus
                    
                    # Distribute layers evenly across GPUs
                    for i in range(num_layers):
                        gpu_id = i // layers_per_gpu
                        if gpu_id >= num_gpus:
                            gpu_id = num_gpus - 1
                        device_map[f"model.layers.{i}"] = gpu_id
                    
                    # Put embedding, norm, and output layers on GPU 0
                    device_map["model.embed_tokens"] = 0
                    device_map["model.norm"] = 0
                    device_map["lm_head"] = 0
                    
                    print(f"Device map for {model_name}: {len([k for k, v in device_map.items() if v == 0])} components on GPU 0, {len([k for k, v in device_map.items() if v == 1])} components on GPU 1")
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=device_map,
                        torch_dtype=torch.bfloat16,
                        cache_dir=f"/mnt/data2/ast-icl/huggingface_cache",
                        token=token,
                        low_cpu_mem_usage=True
                    )
            else:
                # Single GPU or CPU fallback
                if model_name == "gpt-j-6b":
                    # Force GPU loading even for single GPU
                    print(f"Loading GPT-J-6B on single GPU with device map")
                    device_map = {"": 0}  # Force all to GPU 0
                    model = GPTJForCausalLM.from_pretrained(
                        "EleutherAI/gpt-j-6B",
                        revision="float16", 
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        cache_dir=f"/mnt/data2/ast-icl/huggingface_cache"
                    )
                    print(f"GPT-J-6B loaded on single GPU")
                else:
                    # Single GPU loading
                    device_map = {"": 0}
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=device_map,
                        torch_dtype=torch.bfloat16,
                        cache_dir=f"/mnt/data2/ast-icl/huggingface_cache",
                        token=token,
                        low_cpu_mem_usage=True
                    )
        except Exception as e:
            print(f"Error loading model {model_name} on GPU: {e}")
            print("Trying with alternative GPU loading methods...")
            try:
                # Try with more aggressive memory optimization
                if model_name == "gpt-j-6b":
                    # For GPT-J, try multiple GPU loading strategies
                    print(f"Attempting alternative GPU loading for GPT-J...")
                    
                    # Strategy 1: Try with auto device map
                    try:
                        print(f"Strategy 1: Auto device map for GPT-J")
                        model = GPTJForCausalLM.from_pretrained(
                            "EleutherAI/gpt-j-6B",
                            revision="float16",
                            device_map="auto",  # Let accelerate handle device mapping
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            cache_dir=f"/mnt/data/llm/{model_name}"
                        )
                        print(f"Successfully loaded GPT-J with auto device map")
                    except Exception as auto_error:
                        print(f"Auto device map failed: {auto_error}")
                        
                                            # Strategy 2: Try with auto device map (exact model)
                        try:
                            print(f"Strategy 2: Auto device map for exact GPT-J model")
                            model = GPTJForCausalLM.from_pretrained(
                                "EleutherAI/gpt-j-6B",
                                revision="float16",
                                device_map="auto",
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True,
                                cache_dir=f"/mnt/data2/ast-icl/huggingface_cache"
                            )
                            print(f"Successfully loaded exact GPT-J with auto device map")
                        except Exception as auto2_error:
                            print(f"Auto device map failed: {auto2_error}")
                            
                            # Strategy 3: Force single GPU with exact model
                            try:
                                print(f"Strategy 3: Single GPU with exact GPT-J model")
                                model = GPTJForCausalLM.from_pretrained(
                                    "EleutherAI/gpt-j-6B",
                                    revision="float16",
                                    device_map={"": 0},  # Force to GPU 0
                                    torch_dtype=torch.float16,
                                    low_cpu_mem_usage=True,
                                    cache_dir=f"/mnt/data2/ast-icl/huggingface_cache"
                                )
                                print(f"Successfully loaded exact GPT-J on single GPU")
                            except Exception as single_error:
                                print(f"Single GPU loading failed: {single_error}")
                                
                                # Final fallback to CPU
                                print(f"All GPU strategies failed, falling back to CPU for GPT-J")
                                model = GPTJForCausalLM.from_pretrained(
                                    "EleutherAI/gpt-j-6B",
                                    revision="float16", 
                                    low_cpu_mem_usage=True,
                                    torch_dtype=torch.float32,
                                    device_map="cpu"
                                )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        cache_dir=f"/mnt/data2/ast-icl/huggingface_cache",
                        token=token,
                        low_cpu_mem_usage=True
                    )
                print(f"Successfully loaded {model_name} with fallback method")
            except Exception as cpu_error:
                print(f"CRITICAL ERROR: Failed to load {model_name} on both GPU and CPU: {cpu_error}")
                print("Skipping this model and continuing...")
                continue
        
        # Prepare model with accelerator (skip if model is on CPU)
        try:
            if next(model.parameters()).device.type != 'cpu':
                model = accelerator.prepare(model)
            else:
                print("Model is on CPU, skipping accelerator preparation")
        except Exception as e:
            print(f"Error preparing model with accelerator: {e}")
            print("Continuing without accelerator preparation")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, cache_dir=f"/mnt/data2/ast-icl/huggingface_cache")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Fix for "int too big to convert" error
        if tokenizer.model_max_length > 1000000:  # If tokenizer has very large max length
            tokenizer.model_max_length = 8192  # Set reasonable limit
        
        # Check where the model is actually placed
        model_device = next(model.parameters()).device
        print(f"Model loaded successfully: {model_name}")
        print(f"Model is placed on device: {model_device}")
        
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            print(f"Model device map: {model.hf_device_map}")
            unique_devices = set(model.hf_device_map.values())
            print(f"Model distributed across devices: {unique_devices}")
        else:
            print(f"Model is not distributed (single device)")
        
        # Check GPU memory after loading
        if torch.cuda.is_available():
            print("GPU memory after model loading:")
            for i in range(num_gpus):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Set optimal batch size for this model and GPU setup
        optimal_batch_size = get_optimal_batch_size(num_gpus, model_name)
        global BATCH_SIZE
        BATCH_SIZE = optimal_batch_size
        print(f"Using optimal batch size: {BATCH_SIZE} for {model_name} with {num_gpus} GPUs")
        
        # Monitor GPU memory before processing
        monitor_gpu_memory()
        
        # Process each dataset for this model
        for dataset_name in args.datasets:
            try:
                print(f"\n{'='*60}")
                print(f"Starting dataset: {dataset_name}")
                print(f"{'='*60}")
                
                results = process_dataset(model, tokenizer, dataset_name, model_name, accelerator, max_queries)
                
                if results:
                    print(f"Successfully processed {len(results)} queries for {dataset_name}")
                    save_results(results, dataset_name, model_name)
                else:
                    print(f"No results generated for {dataset_name}")
                
                # Monitor GPU memory after each dataset
                monitor_gpu_memory()
                
                print(f"Completed dataset: {dataset_name}")
                
            except Exception as e:
                print(f"ERROR processing dataset {dataset_name} with model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                print(f"Continuing with next dataset...")
                continue
        
        # COMPLETE GPU MEMORY CLEANUP AFTER EACH MODEL
        print(f"\n{'='*60}")
        print(f"COMPLETING GPU MEMORY CLEANUP FOR {model_name.upper()}")
        print(f"{'='*60}")
        
        # Delete model and tokenizer to free memory
        del model
        del tokenizer
        
        # AGGRESSIVE GPU MEMORY CLEANUP
        if torch.cuda.is_available():
            print("Performing aggressive GPU memory cleanup...")
            
            # Clear cache and synchronize on all GPUs
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force memory release by allocating and freeing a small tensor
                try:
                    temp_tensor = torch.zeros(1, device=f'cuda:{i}')
                    del temp_tensor
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print(f"Aggressive GPU memory cleanup completed for {model_name}")
            
            # Monitor memory after cleanup
            monitor_gpu_memory()
        
        # Force garbage collection multiple times
        import gc
        for _ in range(5):  # Increased iterations
            gc.collect()
        
        print(f"Memory cleanup completed for {model_name}")
        print(f"{'='*60}")
        
        # Increment models processed counter
        models_processed += 1
        
        # Reset accelerator for next model (if any)
        if models_processed < len(models_to_run):
            print(f"\nPreparing for next model ({models_processed + 1}/{len(models_to_run)})...")
            # Longer delay to ensure complete cleanup
            import time
            print("Waiting 5 seconds for complete memory cleanup...")
            time.sleep(5)
            
            # Final memory check before next model
            if torch.cuda.is_available():
                print("Final GPU memory status before next model:")
                monitor_gpu_memory()
    
    print(f"\n{'='*60}")
    print("TRANSLATION GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Models processed: {models_processed}/{len(models_to_run)}")
    print(f"Datasets processed: {len(args.datasets)}")
    print(f"Results saved in 'ast_result2' directory")
    
    # Final GPU memory status
    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Status:")
        monitor_gpu_memory()
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
