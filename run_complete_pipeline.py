#!/usr/bin/env python3
"""
Complete Credit Risk Scoring Pipeline Runner

Purpose: Executes the complete credit risk scoring pipeline on the input file
         output/credit_data_sample.csv and summarizes results to stdout.

Author: Risk Analytics Team  
Date: 2025

Pipeline Stages:
1. Data Exploration & Analysis 
2. Feature Engineering
3. Model Training
4. Model Validation  
5. New Customer Scoring

This script runs all pipeline stages sequentially and provides comprehensive
summary results for each stage.

USAGE:
    python run_complete_pipeline.py

INPUT:  
    - output/credit_data_sample.csv (existing sample data)

OUTPUT:
    - Comprehensive pipeline execution summary to stdout
    - All intermediate files and models created by each stage
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd

# Pipeline stage imports - direct imports for reliability
import subprocess
import importlib.util
from importlib import import_module

def run_pipeline_script(script_name):
    """Run a pipeline script as subprocess and capture output."""
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd='.')
        return result
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        raise

def import_and_run_main(script_name):
    """Import module and run its main function."""
    try:
        # Dynamic import
        spec = importlib.util.spec_from_file_location("temp_module", script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call main function
        if hasattr(module, 'main'):
            return module.main()
        else:
            raise AttributeError(f"Module {script_name} has no main function")
            
    except Exception as e:
        print(f"Error importing/running {script_name}: {e}")
        raise

class CreditRiskPipeline:
    """Complete Credit Risk Scoring Pipeline Orchestrator"""
    
    def __init__(self, input_file="output/credit_data_sample.csv"):
        """
        Initialize pipeline with input data file.
        
        Args:
            input_file (str): Path to input credit data CSV file
        """
        self.input_file = Path(input_file)
        self.start_time = datetime.now()
        self.stage_results = {}
        self.stage_timings = {}
        
        # Verify input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print("="*80)
        print("CREDIT RISK SCORING PIPELINE - COMPLETE EXECUTION")
        print("="*80)
        print(f"Pipeline Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input File: {self.input_file}")
        print(f"Input Records: {self._count_input_records():,}")
        print("="*80)
    
    def _count_input_records(self):
        """Count records in input file."""
        try:
            df = pd.read_csv(self.input_file)
            return len(df)
        except Exception:
            return "Unknown"
    
    def _execute_stage(self, stage_name, stage_function):
        """
        Execute a pipeline stage with error handling and timing.
        
        Args:
            stage_name (str): Name of the pipeline stage
            stage_function (callable): Function to execute
            
        Returns:
            dict: Stage execution results
        """
        print(f"\n{'='*20} STAGE: {stage_name.upper()} {'='*20}")
        stage_start = time.time()
        
        try:
            # Execute the stage
            result = stage_function()
            
            # Record timing
            duration = time.time() - stage_start
            self.stage_timings[stage_name] = duration
            
            # Store results
            self.stage_results[stage_name] = {
                'status': 'SUCCESS',
                'duration': duration,
                'result': result
            }
            
            print(f"✓ {stage_name} completed successfully in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            duration = time.time() - stage_start
            self.stage_timings[stage_name] = duration
            
            self.stage_results[stage_name] = {
                'status': 'FAILED',
                'duration': duration,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            print(f"❌ {stage_name} failed after {duration:.2f} seconds")
            print(f"Error: {str(e)}")
            raise
    
    def run_data_exploration(self):
        """Execute Stage 2: Data Exploration & Analysis"""
        return import_and_run_main('02_data_exploration.py')
    
    def run_feature_engineering(self):
        """Execute Stage 3: Feature Engineering"""
        return import_and_run_main('03_feature_engineering.py')
    
    def run_model_training(self):
        """Execute Stage 4: Model Training"""
        return import_and_run_main('04_train_credit_model.py')
    
    def run_model_validation(self):
        """Execute Stage 5: Model Validation"""
        return import_and_run_main('05_model_validation.py')
    
    def run_customer_scoring(self):
        """Execute Stage 6: New Customer Scoring"""
        return import_and_run_main('06_score_new_customers.py')
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline."""
        try:
            # Stage 1: Skip data generation (input file already exists)
            print(f"\n{'='*20} STAGE: DATA GENERATION (SKIPPED) {'='*20}")
            print("✓ Using existing input file: output/credit_data_sample.csv")
            
            # Stage 2: Data Exploration  
            exploration_results = self._execute_stage(
                "Data Exploration", 
                self.run_data_exploration
            )
            
            # Stage 3: Feature Engineering
            feature_results = self._execute_stage(
                "Feature Engineering",
                self.run_feature_engineering  
            )
            
            # Stage 4: Model Training
            training_results = self._execute_stage(
                "Model Training",
                self.run_model_training
            )
            
            # Stage 5: Model Validation
            validation_results = self._execute_stage(
                "Model Validation", 
                self.run_model_validation
            )
            
            # Stage 6: Customer Scoring
            scoring_results = self._execute_stage(
                "Customer Scoring",
                self.run_customer_scoring
            )
            
            # Generate comprehensive summary
            self.generate_pipeline_summary()
            
        except Exception as e:
            print(f"\n❌ Pipeline execution failed: {str(e)}")
            self.generate_failure_summary()
            raise
    
    def generate_pipeline_summary(self):
        """Generate comprehensive pipeline execution summary."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION COMPLETE - COMPREHENSIVE SUMMARY")
        print("="*80)
        
        # Overall execution summary
        print(f"\n📊 OVERALL EXECUTION SUMMARY")
        print("-" * 50)
        print(f"• Pipeline Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• Pipeline End Time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        print(f"• Input File: {self.input_file}")
        print(f"• Input Records: {self._count_input_records():,}")
        
        # Stage-by-stage summary
        print(f"\n📈 STAGE-BY-STAGE EXECUTION SUMMARY")
        print("-" * 50)
        
        successful_stages = 0
        failed_stages = 0
        
        for stage_name, stage_info in self.stage_results.items():
            status_icon = "✓" if stage_info['status'] == 'SUCCESS' else "❌"
            duration = stage_info['duration']
            
            print(f"{status_icon} {stage_name:<20}: {stage_info['status']:<10} ({duration:.2f}s)")
            
            if stage_info['status'] == 'SUCCESS':
                successful_stages += 1
            else:
                failed_stages += 1
        
        print(f"\n• Successful Stages: {successful_stages}")
        print(f"• Failed Stages: {failed_stages}")
        print(f"• Success Rate: {(successful_stages/(successful_stages+failed_stages)*100):.1f}%")
        
        # Detailed results summary
        self._summarize_stage_results()
        
        # Output files summary
        self._summarize_output_files()
        
        print(f"\n{'='*80}")
        print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def _summarize_stage_results(self):
        """Summarize detailed results from each stage."""
        print(f"\n📋 DETAILED STAGE RESULTS")
        print("-" * 50)
        
        # Data Exploration Results
        if 'Data Exploration' in self.stage_results and self.stage_results['Data Exploration']['status'] == 'SUCCESS':
            exploration_result = self.stage_results['Data Exploration']['result']
            if exploration_result and 'data' in exploration_result:
                df = exploration_result['data']
                print(f"\n• Data Exploration:")
                print(f"  - Records analyzed: {len(df):,}")
                print(f"  - Features analyzed: {len(df.columns)}")
                print(f"  - Default rate: {df['default_flag'].mean()*100:.2f}%")
                
                if 'correlation_matrix' in exploration_result:
                    print(f"  - Correlation analysis completed")
                if 'pca_results' in exploration_result:
                    print(f"  - PCA analysis completed")
        
        # Feature Engineering Results  
        if 'Feature Engineering' in self.stage_results and self.stage_results['Feature Engineering']['status'] == 'SUCCESS':
            feature_result = self.stage_results['Feature Engineering']['result']
            if feature_result and len(feature_result) >= 2:
                train_features, val_features = feature_result[0], feature_result[1]
                print(f"\n• Feature Engineering:")
                print(f"  - Training features: {train_features.shape}")
                print(f"  - Validation features: {val_features.shape}")
                print(f"  - Total engineered features: {train_features.shape[1] - 2}")  # Exclude ID and target
                print(f"  - Train default rate: {train_features['default_flag'].mean()*100:.2f}%")
                print(f"  - Validation default rate: {val_features['default_flag'].mean()*100:.2f}%")
        
        # Model Training Results
        if 'Model Training' in self.stage_results and self.stage_results['Model Training']['status'] == 'SUCCESS':
            training_result = self.stage_results['Model Training']['result']
            if training_result and len(training_result) >= 5:
                lr_results, dt_results, calibration_results, scored_apps, performance_summary = training_result
                print(f"\n• Model Training:")
                print(f"  - Models trained: Logistic Regression, Decision Tree")
                print(f"  - Calibration completed: {calibration_results is not None}")
                print(f"  - Scored applications: {len(scored_apps) if scored_apps is not None else 0:,}")
                if performance_summary is not None and not performance_summary.empty:
                    print(f"  - Performance metrics calculated")
        
        # Model Validation Results
        if 'Model Validation' in self.stage_results and self.stage_results['Model Validation']['status'] == 'SUCCESS':
            print(f"\n• Model Validation:")
            print(f"  - Comprehensive validation completed")
            print(f"  - Model performance metrics generated")
            print(f"  - Model validation reports created")
        
        # Customer Scoring Results
        if 'Customer Scoring' in self.stage_results and self.stage_results['Customer Scoring']['status'] == 'SUCCESS':
            scoring_result = self.stage_results['Customer Scoring']['result']
            if scoring_result and len(scoring_result) >= 3:
                applications_decisions, new_application_decisions, approval_summary = scoring_result
                print(f"\n• Customer Scoring:")
                if applications_decisions is not None:
                    print(f"  - Applications scored: {len(applications_decisions):,}")
                if new_application_decisions is not None:
                    print(f"  - New applications scored: {len(new_application_decisions):,}")
                if approval_summary is not None:
                    print(f"  - Approval summary generated")
    
    def _summarize_output_files(self):
        """Summarize output files generated by the pipeline."""
        print(f"\n📁 OUTPUT FILES GENERATED")
        print("-" * 50)
        
        output_dir = Path('output')
        if output_dir.exists():
            output_files = list(output_dir.glob('*.csv'))
            
            print(f"• Output directory: {output_dir}")
            print(f"• Total CSV files: {len(output_files)}")
            
            # List key output files
            key_files = [
                'exploration_summary.csv',
                'model_features_train.csv', 
                'model_features_validation.csv',
                'scored_applications.csv',
                'model_performance_metrics.csv',
                'validation_summary.csv',
                'new_application_decisions.csv',
                'approval_summary.csv'
            ]
            
            print(f"\n• Key output files:")
            for file_name in key_files:
                file_path = output_dir / file_name
                if file_path.exists():
                    try:
                        # Get file size and record count for CSV files
                        file_size = file_path.stat().st_size / 1024  # KB
                        df = pd.read_csv(file_path)
                        print(f"  ✓ {file_name:<30} ({len(df):,} records, {file_size:.1f}KB)")
                    except:
                        print(f"  ✓ {file_name:<30} (file exists)")
                else:
                    print(f"  ❌ {file_name:<30} (not found)")
        
        # Check for visualizations
        viz_dir = Path('output/visualizations')
        if viz_dir.exists():
            viz_files = list(viz_dir.glob('*.png'))
            print(f"\n• Visualization files: {len(viz_files)}")
            for viz_file in sorted(viz_files):
                print(f"  ✓ {viz_file.name}")
    
    def generate_failure_summary(self):
        """Generate summary when pipeline fails."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("PIPELINE EXECUTION FAILED - SUMMARY")
        print("="*80)
        
        print(f"\n📊 EXECUTION SUMMARY")
        print("-" * 50)
        print(f"• Pipeline Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• Pipeline Failed At:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"• Execution Time: {total_duration:.2f} seconds")
        
        # Show stage results
        print(f"\n📈 STAGE RESULTS")
        print("-" * 50)
        
        for stage_name, stage_info in self.stage_results.items():
            status_icon = "✓" if stage_info['status'] == 'SUCCESS' else "❌"
            duration = stage_info['duration']
            
            print(f"{status_icon} {stage_name:<20}: {stage_info['status']:<10} ({duration:.2f}s)")
            
            if stage_info['status'] == 'FAILED':
                print(f"   Error: {stage_info['error']}")


def main():
    """Main execution function."""
    try:
        # Initialize and run pipeline
        pipeline = CreditRiskPipeline("output/credit_data_sample.csv")
        pipeline.run_complete_pipeline()
        
        return 0  # Success
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        traceback.print_exc()
        return 1  # Failure


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
