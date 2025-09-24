#!/usr/bin/env python3
"""
Correct Method Comparison: Use the actual different data sources for Graph-AST ICL vs Weighted-AST
Based on the original LaTeX table results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import re
from typing import Dict, List, Optional

class CorrectMethodComparison:
    """Compare Graph-AST ICL and Weighted-AST methods using their actual different data sources"""
    
    def __init__(self):
        self.output_dir = Path("EX_EM_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Based on the original LaTeX table, these are the actual method mappings:
        self.method_data_sources = {
            'Graph-AST ICL': {
                'COSQL': {
                    'GPT-J-6B': 'benchmark/merged_sql_full_dataset/cosql_gpt-j-6b_translations_with_sql.csv',
                    'CODE-LLAMA': 'benchmark/merged_sql_full_dataset/cosql_code-llama_translations_with_sql.csv',
                    'MISTRAL-7B': 'benchmark/merged_sql_full_dataset/cosql_mistral-7B_translations_with_sql.csv'
                },
                'SPARC': {
                    'GPT-J-6B': 'benchmark/merged_sql_full_dataset/sparc_gpt-j-6b_translations_with_sql.csv',
                    'CODE-LLAMA': 'benchmark/merged_sql_full_dataset/sparc_code-llama_translations_with_sql.csv',
                    'MISTRAL-7B': 'benchmark/merged_sql_full_dataset/sparc_mistral-7B_translations_with_sql.csv'
                },
                'SPIDER': {
                    'GPT-J-6B': 'benchmark/merged_sql_full_dataset/spider_gpt-j-6b_translations_with_sql.csv',
                    'CODE-LLAMA': 'benchmark/merged_sql_full_dataset/spider_code-llama_translations_with_sql.csv',
                    'MISTRAL-7B': 'benchmark/merged_sql_full_dataset/spider_mistral-7B_translations_with_sql.csv'
                }
            },
            'Weighted-AST': {
                'COSQL': {
                    'GPT-J-6B': 'extracted_sql_results/cosql_gpt_j_6b_extracted_sql.csv',
                    'CODE-LLAMA': 'extracted_sql_results/cosql_code_llama_extracted_sql.csv',
                    'MISTRAL-7B': 'extracted_sql_results/cosql_mistral_7b_extracted_sql.csv'
                },
                'SPARC': {
                    'GPT-J-6B': 'extracted_sql_results/sparc_gpt_j_6b_extracted_sql.csv',
                    'CODE-LLAMA': 'extracted_sql_results/sparc_code_llama_extracted_sql.csv',
                    'MISTRAL-7B': 'extracted_sql_results/sparc_mistral_7b_extracted_sql.csv'
                },
                'SPIDER': {
                    'GPT-J-6B': 'extracted_sql_results/spider_gpt_j_6b_extracted_sql.csv',
                    'CODE-LLAMA': 'extracted_sql_results/spider_code_llama_extracted_sql.csv',
                    'MISTRAL-7B': 'extracted_sql_results/spider_mistral_7b_extracted_sql.csv'
                }
            }
        }
    
    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL for consistent comparison"""
        if not sql or pd.isna(sql):
            return ""
        
        sql_str = str(sql).strip()
        sql_str = re.sub(r'\s+', ' ', sql_str)
        sql_str = sql_str.lower()
        sql_str = sql_str.rstrip(';')
        sql_str = re.sub(r'\s*([=<>!+\-*/])\s*', r'\1', sql_str)
        sql_str = re.sub(r'\(\s*', '(', sql_str)
        sql_str = re.sub(r'\s*\)', ')', sql_str)
        sql_str = re.sub(r'\s*,\s*', ',', sql_str)
        
        return sql_str.strip()
    
    def calculate_exact_match(self, generated_sql: str, gold_sql: str) -> bool:
        """Calculate exact match between generated and gold SQL"""
        gen_normalized = self.normalize_sql(generated_sql)
        gold_normalized = self.normalize_sql(gold_sql)
        return gen_normalized == gold_normalized
    
    def find_database_file(self, dataset_name: str, db_id: str) -> Optional[str]:
        """Find the database file for a given dataset and database ID"""
        possible_paths = [
            f"datasets/{dataset_name}/database/{db_id}/{db_id}.sqlite",
            f"datasets/{dataset_name}/database/{db_id}.sqlite",
            f"datasets/{dataset_name}/{db_id}/{db_id}.sqlite",
            f"datasets/{dataset_name}/{db_id}.sqlite"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        return None
    
    def execute_sql_query(self, sql_query: str, db_path: str) -> Dict:
        """Execute a SQL query against a database and return execution results"""
        if not sql_query or not sql_query.strip():
            return {
                'success': False,
                'error': 'Empty or invalid SQL query',
                'result_count': 0,
                'results': []
            }
        
        if not db_path or not Path(db_path).exists():
            return {
                'success': False,
                'error': 'Database file not found',
                'result_count': 0,
                'results': []
            }
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            formatted_results = []
            for row in results:
                row_dict = {}
                for i, value in enumerate(row):
                    col_name = column_names[i] if i < len(column_names) else f"col_{i}"
                    row_dict[col_name] = value
                formatted_results.append(row_dict)
            
            conn.close()
            
            return {
                'success': True,
                'error': None,
                'result_count': len(formatted_results),
                'results': formatted_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'SQL Error: {str(e)}',
                'result_count': 0,
                'results': []
            }
    
    def normalize_sql_results(self, results: List[Dict]) -> List[tuple]:
        """Normalize SQL results for comparison"""
        normalized = []
        for result in results:
            normalized_row = []
            for key, value in result.items():
                if value is None:
                    normalized_row.append(('NULL', key))
                elif isinstance(value, (int, float)):
                    normalized_row.append((str(value), key))
                else:
                    normalized_row.append((str(value).strip().lower(), key))
            normalized.append(tuple(sorted(normalized_row)))
        return sorted(normalized)
    
    def compare_sql_results(self, generated_results: Dict, gold_results: Dict) -> bool:
        """Compare results from generated SQL and gold SQL"""
        if not generated_results['success'] or not gold_results['success']:
            return False
        
        gen_results = generated_results['results']
        gold_results_list = gold_results['results']
        
        if len(gen_results) == 0 and len(gold_results_list) == 0:
            return True
        
        if len(gen_results) == 0 or len(gold_results_list) == 0:
            return False
        
        gen_normalized = self.normalize_sql_results(gen_results)
        gold_normalized = self.normalize_sql_results(gold_results_list)
        
        return gen_normalized == gold_normalized
    
    def analyze_method_dataset_model(self, method: str, dataset: str, model: str) -> Dict:
        """Analyze a specific method-dataset-model combination"""
        filepath = Path(self.method_data_sources[method][dataset][model])
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return None
        
        print(f"🔍 Analyzing {method} - {dataset} - {model}")
        print(f"   Data source: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            print(f"   Loaded {len(df)} queries")
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            return None
        
        results = []
        exact_matches = 0
        execution_matches = 0
        successful_executions = 0
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   Processing query {idx+1}/{len(df)}...")
            
            # Extract data based on the file structure
            if 'generated_sql' in df.columns:
                # extracted_sql_results format
                generated_sql = row.get('generated_sql', '')
                gold_sql = row.get('gold_sql', '')
                database_id = row.get('database_id', '')
                question = row.get('natural_language', '')
            else:
                # benchmark format
                generated_sql = row.get('generated_sql', '')
                gold_sql = row.get('gold_sql', '')
                database_id = row.get('database_id', '')
                question = row.get('question', '')
            
            if pd.isna(generated_sql) or pd.isna(gold_sql) or pd.isna(database_id):
                continue
            
            # Calculate exact match (EM)
            exact_match = self.calculate_exact_match(generated_sql, gold_sql)
            if exact_match:
                exact_matches += 1
            
            # Find database file
            db_path = self.find_database_file(dataset.lower(), database_id)
            
            if not db_path:
                results.append({
                    'query_index': idx,
                    'question': question,
                    'generated_sql': generated_sql,
                    'gold_sql': gold_sql,
                    'database_id': database_id,
                    'exact_match': exact_match,
                    'execution_match': False,
                    'execution_success': False,
                    'execution_error': 'Database not found'
                })
                continue
            
            # Execute generated SQL
            generated_execution = self.execute_sql_query(generated_sql, db_path)
            
            # Execute gold SQL
            gold_execution = self.execute_sql_query(gold_sql, db_path)
            
            # Compare results
            execution_match = self.compare_sql_results(generated_execution, gold_execution)
            
            if generated_execution['success'] and gold_execution['success']:
                successful_executions += 1
                if execution_match:
                    execution_matches += 1
            
            results.append({
                'query_index': idx,
                'question': question,
                'generated_sql': generated_sql,
                'gold_sql': gold_sql,
                'database_id': database_id,
                'exact_match': exact_match,
                'execution_match': execution_match,
                'execution_success': generated_execution['success'] and gold_execution['success'],
                'execution_error': generated_execution.get('error') or gold_execution.get('error')
            })
        
        # Calculate metrics
        total_queries = len(results)
        exact_match_rate = (exact_matches / total_queries) * 100 if total_queries > 0 else 0
        execution_accuracy = (execution_matches / total_queries) * 100 if total_queries > 0 else 0
        execution_success_rate = (successful_executions / total_queries) * 100 if total_queries > 0 else 0
        
        print(f"   Total Queries: {total_queries}")
        print(f"   Exact Matches: {exact_matches} ({exact_match_rate:.2f}%)")
        print(f"   Execution Matches: {execution_matches} ({execution_accuracy:.2f}%)")
        print(f"   Successful Executions: {successful_executions} ({execution_success_rate:.2f}%)")
        
        # Verify constraint
        if execution_accuracy >= exact_match_rate:
            print(f"   ✅ EX >= EM constraint satisfied")
        else:
            print(f"   ❌ EX < EM constraint violated!")
        
        return {
            'method': method,
            'dataset': dataset,
            'model': model,
            'data_source': str(filepath),
            'total_queries': total_queries,
            'exact_matches': exact_matches,
            'execution_matches': execution_matches,
            'successful_executions': successful_executions,
            'exact_match_rate': exact_match_rate,
            'execution_accuracy': execution_accuracy,
            'execution_success_rate': execution_success_rate,
            'constraint_satisfied': execution_accuracy >= exact_match_rate,
            'detailed_results': results
        }
    
    def run_correct_comparison(self):
        """Run the correct comparison using actual different data sources"""
        print("🔧 CORRECT METHOD COMPARISON")
        print("=" * 60)
        print("Using ACTUAL different data sources for each method:")
        print("- Graph-AST ICL: benchmark/merged_sql_full_dataset/")
        print("- Weighted-AST: extracted_sql_results/")
        print()
        
        all_results = []
        
        for method in self.method_data_sources.keys():
            print(f"\n📊 {method} METHOD")
            print("-" * 40)
            
            for dataset in self.method_data_sources[method].keys():
                for model in self.method_data_sources[method][dataset].keys():
                    result = self.analyze_method_dataset_model(method, dataset, model)
                    if result:
                        all_results.append(result)
                        
                        # Save detailed results
                        detailed_df = pd.DataFrame(result['detailed_results'])
                        output_file = self.output_dir / f"{method.replace(' ', '_').replace('-', '_')}_{dataset}_{model}_results.csv"
                        detailed_df.to_csv(output_file, index=False)
                        print(f"   💾 Saved detailed results to {output_file}")
                    print()
        
        # Create comprehensive summary
        if all_results:
            summary_data = []
            for result in all_results:
                summary_data.append({
                    'Method': result['method'],
                    'Dataset': result['dataset'],
                    'Model': result['model'],
                    'Data_Source': result['data_source'],
                    'Total_Queries': result['total_queries'],
                    'Exact_Matches': result['exact_matches'],
                    'Execution_Matches': result['execution_matches'],
                    'Successful_Executions': result['successful_executions'],
                    'EM_Rate_%': round(result['exact_match_rate'], 2),
                    'EX_Rate_%': round(result['execution_accuracy'], 2),
                    'Execution_Success_%': round(result['execution_success_rate'], 2),
                    'EX_>=_EM': result['constraint_satisfied']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.output_dir / "correct_method_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            
            print("📊 CORRECT METHOD COMPARISON SUMMARY")
            print("=" * 80)
            print(summary_df.to_string(index=False))
            
            # Create comparison table in the format of your original table
            self.create_correct_comparison_table(summary_df)
            
            # Check constraint satisfaction
            print("\n🎯 CONSTRAINT VERIFICATION:")
            print("-" * 40)
            satisfied = sum(1 for r in all_results if r['constraint_satisfied'])
            total = len(all_results)
            print(f"EX >= EM constraint satisfied: {satisfied}/{total} cases")
            
            if satisfied == total:
                print("✅ All cases satisfy EX >= EM constraint!")
            else:
                print("❌ Some cases violate EX >= EM constraint!")
                for result in all_results:
                    if not result['constraint_satisfied']:
                        print(f"   {result['method']} {result['dataset']} {result['model']}: EX={result['execution_accuracy']:.2f}% < EM={result['exact_match_rate']:.2f}%")
            
            print(f"\n💾 Summary saved to {summary_file}")
        
        print("\n✅ Correct method comparison completed!")
    
    def create_correct_comparison_table(self, summary_df: pd.DataFrame):
        """Create comparison table in the format of the original table"""
        print("\n📊 CORRECTED RESULTS TABLE (Original Format)")
        print("=" * 80)
        
        # Group by dataset and model
        for dataset in ['COSQL', 'SPARC', 'SPIDER']:
            print(f"\n{dataset}-S2T")
            print("-" * 40)
            
            dataset_data = summary_df[summary_df['Dataset'] == dataset]
            
            for model in ['CODE-LLAMA', 'GPT-J-6B', 'MISTRAL-7B']:
                model_data = dataset_data[dataset_data['Model'] == model]
                
                if len(model_data) >= 2:  # Both methods available
                    graph_ast = model_data[model_data['Method'] == 'Graph-AST ICL'].iloc[0]
                    weighted_ast = model_data[model_data['Method'] == 'Weighted-AST'].iloc[0]
                    
                    print(f"{'':>10} & {graph_ast['Total_Queries']:>3} & {model.replace('_', ' '):>12} & "
                          f"{graph_ast['Exact_Matches']:>2} & {graph_ast['EM_Rate_%']:>6.2f}% & "
                          f"{graph_ast['Execution_Matches']:>2} & {graph_ast['EX_Rate_%']:>6.2f}% & "
                          f"{weighted_ast['Exact_Matches']:>2} & {weighted_ast['EM_Rate_%']:>6.2f}% & "
                          f"{weighted_ast['Execution_Matches']:>2} & {weighted_ast['EX_Rate_%']:>6.2f}% \\\\")
        
        print("\n📊 METHOD PERFORMANCE COMPARISON")
        print("-" * 50)
        
        # Compare methods
        method_stats = summary_df.groupby('Method').agg({
            'EM_Rate_%': 'mean',
            'EX_Rate_%': 'mean',
            'Execution_Success_%': 'mean',
            'EX_>=_EM': 'all'
        }).round(2)
        
        print(method_stats.to_string())

def main():
    """Main function to run the correct comparison"""
    analyzer = CorrectMethodComparison()
    analyzer.run_correct_comparison()

if __name__ == "__main__":
    main()
