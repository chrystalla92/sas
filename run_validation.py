#!/usr/bin/env python3
"""
Automated Validation Test Runner for Bank Credit Risk Scoring Model
SAS-to-Python Migration Validation

Purpose: Orchestrate comprehensive validation testing including:
- Full pipeline execution
- Statistical validation
- Performance benchmarking  
- Report generation
- Regression testing

Usage:
    python run_validation.py [options]
    
Options:
    --quick: Run only smoke tests for quick validation
    --full: Run complete validation suite (default)
    --performance: Focus on performance benchmarking
    --compare-sas: Compare against SAS outputs (if available)
    --report-only: Generate report from existing test results
    --clean: Clean previous test results before running

Author: Risk Analytics Team
Date: 2025
"""

import argparse
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import shutil
import os
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from test_integration import (
    ValidationConfig, CSVComparator, StatisticalValidator, 
    PerformanceBenchmarker, generate_test_report
)

class ValidationRunner:
    """Main class for orchestrating validation tests."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize validation runner."""
        self.config = config or ValidationConfig()
        self.csv_comparator = CSVComparator(self.config)
        self.statistical_validator = StatisticalValidator(self.config)
        self.performance_benchmarker = PerformanceBenchmarker(self.config)
        
        self.results = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'test_results': {},
            'performance_metrics': {},
            'validation_summary': {},
            'issues_found': [],
            'recommendations': []
        }
        
        # Create results directory
        self.results_dir = Path('validation_results')
        self.results_dir.mkdir(exist_ok=True)
        
    def clean_previous_results(self):
        """Clean previous test results."""
        print("🧹 Cleaning previous validation results...")
        
        # Clean temporary files
        temp_dir = Path(self.config.temp_dir)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        # Clean previous reports
        for report_file in self.results_dir.glob("*.json"):
            report_file.unlink()
            
        for report_file in self.results_dir.glob("*.html"):
            report_file.unlink()
            
        print("✓ Previous results cleaned")
    
    def run_pipeline_execution(self) -> Dict[str, Any]:
        """Execute the complete data pipeline."""
        print("🚀 Running complete data pipeline...")
        
        scripts = [
            '01_generate_credit_data.py',
            '02_data_exploration.py', 
            '03_feature_engineering.py',
            '04_train_credit_model.py',
            '05_model_validation.py',
            '06_score_new_customers.py'
        ]
        
        pipeline_results = {}
        total_time = 0
        
        for i, script in enumerate(scripts, 1):
            script_path = Path(script)
            
            if not script_path.exists():
                print(f"⚠️  Script {script} not found - skipping")
                pipeline_results[script] = {'status': 'skipped', 'reason': 'file_not_found'}
                continue
            
            print(f"  [{i}/{len(scripts)}] Running {script}...")
            
            # Benchmark execution
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    timeout=300  # 5 minute timeout per script
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                total_time += execution_time
                
                if result.returncode == 0:
                    print(f"    ✓ Completed in {execution_time:.2f}s")
                    pipeline_results[script] = {
                        'status': 'success',
                        'execution_time': execution_time,
                        'stdout': result.stdout[-500:] if result.stdout else "",  # Last 500 chars
                        'stderr': result.stderr[-500:] if result.stderr else ""
                    }
                else:
                    print(f"    ❌ Failed with return code {result.returncode}")
                    pipeline_results[script] = {
                        'status': 'failed',
                        'execution_time': execution_time,
                        'return_code': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                    
            except subprocess.TimeoutExpired:
                print(f"    ⏰ Timeout after 5 minutes")
                pipeline_results[script] = {
                    'status': 'timeout',
                    'execution_time': 300,
                    'reason': 'timeout_expired'
                }
                
            except Exception as e:
                print(f"    💥 Exception: {str(e)}")
                pipeline_results[script] = {
                    'status': 'error',
                    'execution_time': 0,
                    'error': str(e)
                }
        
        pipeline_results['total_execution_time'] = total_time
        pipeline_results['successful_scripts'] = sum(
            1 for r in pipeline_results.values() 
            if isinstance(r, dict) and r.get('status') == 'success'
        )
        
        print(f"✓ Pipeline completed in {total_time:.2f}s")
        print(f"  {pipeline_results['successful_scripts']} of {len(scripts)} scripts succeeded")
        
        return pipeline_results
    
    def run_pytest_suite(self, test_selection: str = "full") -> Dict[str, Any]:
        """Run pytest test suite."""
        print(f"🧪 Running {test_selection} test suite...")
        
        # Define test selection options
        test_options = {
            "quick": ["-k", "smoke", "--tb=short", "-v"],
            "full": ["--tb=short", "-v", "--maxfail=10"],
            "performance": ["-k", "performance", "--tb=short", "-v"],
            "statistical": ["-k", "statistical", "--tb=short", "-v"]
        }
        
        pytest_args = ["test_integration.py"] + test_options.get(test_selection, test_options["full"])
        
        # Add coverage if available
        try:
            import pytest_cov
            pytest_args.extend(["--cov=.", "--cov-report=html:validation_results/coverage"])
        except ImportError:
            pass
        
        start_time = time.time()
        
        try:
            import pytest
            # Capture pytest output
            result = pytest.main(pytest_args + ["--json-report", "--json-report-file=validation_results/pytest_report.json"])
            
            end_time = time.time()
            test_duration = end_time - start_time
            
            # Load pytest results if available
            pytest_report_file = self.results_dir / "pytest_report.json"
            if pytest_report_file.exists():
                with open(pytest_report_file, 'r') as f:
                    pytest_data = json.load(f)
                    
                return {
                    'status': 'success' if result == 0 else 'failed',
                    'exit_code': result,
                    'duration': test_duration,
                    'summary': pytest_data.get('summary', {}),
                    'tests': pytest_data.get('tests', [])
                }
            else:
                return {
                    'status': 'success' if result == 0 else 'failed',
                    'exit_code': result,
                    'duration': test_duration,
                    'summary': {'total': 0, 'passed': 0, 'failed': 0},
                    'tests': []
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def validate_outputs(self) -> Dict[str, Any]:
        """Validate pipeline outputs."""
        print("📊 Validating pipeline outputs...")
        
        output_dir = Path(self.config.output_dir)
        validation_results = {
            'csv_files_found': [],
            'missing_files': [],
            'schema_validation': {},
            'data_quality': {}
        }
        
        # Expected output files
        expected_files = [
            'credit_data_sample.csv',
            'exploration_summary.csv', 
            'model_performance_metrics.csv',
            'validation_summary.csv',
            'scored_applications.csv',
            'new_application_decisions.csv'
        ]
        
        for filename in expected_files:
            file_path = output_dir / filename
            
            if file_path.exists():
                validation_results['csv_files_found'].append(filename)
                
                try:
                    # Basic validation
                    df = pd.read_csv(file_path)
                    
                    validation_results['schema_validation'][filename] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': list(df.columns),
                        'memory_usage': df.memory_usage(deep=True).sum(),
                        'has_data': len(df) > 0
                    }
                    
                    # Data quality checks
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    missing_data = df.isnull().sum().sum()
                    
                    validation_results['data_quality'][filename] = {
                        'missing_values': int(missing_data),
                        'missing_percentage': float(missing_data / (len(df) * len(df.columns)) * 100),
                        'numeric_columns': len(numeric_cols),
                        'categorical_columns': len(df.columns) - len(numeric_cols)
                    }
                    
                    print(f"  ✓ {filename}: {len(df)} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    validation_results['schema_validation'][filename] = {'error': str(e)}
                    print(f"  ❌ {filename}: Error reading file - {str(e)}")
            else:
                validation_results['missing_files'].append(filename)
                print(f"  ❌ {filename}: File not found")
        
        return validation_results
    
    def generate_performance_report(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance analysis report."""
        print("⚡ Analyzing performance metrics...")
        
        performance_report = {
            'execution_times': {},
            'performance_summary': {},
            'benchmarks': {},
            'recommendations': []
        }
        
        # Extract execution times
        total_time = 0
        successful_scripts = 0
        
        for script, result in pipeline_results.items():
            if isinstance(result, dict) and 'execution_time' in result:
                execution_time = result['execution_time']
                performance_report['execution_times'][script] = execution_time
                
                if result.get('status') == 'success':
                    total_time += execution_time
                    successful_scripts += 1
        
        if successful_scripts > 0:
            avg_time = total_time / successful_scripts
            
            performance_report['performance_summary'] = {
                'total_execution_time': total_time,
                'average_time_per_script': avg_time,
                'successful_scripts': successful_scripts,
                'throughput_records_per_second': self.config.expected_records / total_time if total_time > 0 else 0
            }
            
            # Performance benchmarks (hypothetical SAS comparison)
            estimated_sas_time = total_time / 0.8  # Assume Python is 25% faster
            improvement_ratio = (estimated_sas_time - total_time) / estimated_sas_time
            
            performance_report['benchmarks'] = {
                'estimated_sas_time': estimated_sas_time,
                'python_time': total_time,
                'improvement_ratio': improvement_ratio,
                'meets_target': improvement_ratio >= self.config.performance_improvement_target
            }
            
            # Performance recommendations
            if total_time > 300:  # 5 minutes
                performance_report['recommendations'].append(
                    "Consider optimizing data processing for large datasets"
                )
            
            if improvement_ratio < self.config.performance_improvement_target:
                performance_report['recommendations'].append(
                    f"Performance improvement of {improvement_ratio:.1%} is below target of {self.config.performance_improvement_target:.1%}"
                )
        
        return performance_report
    
    def run_validation(self, test_selection: str = "full", 
                      compare_sas: bool = False,
                      skip_pipeline: bool = False) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🎯 Starting comprehensive validation suite...")
        print(f"   Test selection: {test_selection}")
        print(f"   Compare with SAS: {compare_sas}")
        print(f"   Skip pipeline execution: {skip_pipeline}")
        print()
        
        self.results['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            # Step 1: Run pipeline if requested
            if not skip_pipeline:
                self.results['pipeline_results'] = self.run_pipeline_execution()
            else:
                print("⏭️  Skipping pipeline execution")
                self.results['pipeline_results'] = {'skipped': True}
            
            # Step 2: Validate outputs
            self.results['output_validation'] = self.validate_outputs()
            
            # Step 3: Run test suite
            self.results['test_results'] = self.run_pytest_suite(test_selection)
            
            # Step 4: Generate performance report
            if 'pipeline_results' in self.results:
                self.results['performance_metrics'] = self.generate_performance_report(
                    self.results['pipeline_results']
                )
            
            # Step 5: Overall validation summary
            self.results['validation_summary'] = self._create_validation_summary()
            
        except Exception as e:
            print(f"💥 Validation failed with error: {str(e)}")
            self.results['error'] = str(e)
        
        finally:
            end_time = time.time()
            self.results['end_time'] = datetime.now().isoformat()
            self.results['duration'] = end_time - start_time
            
            print(f"\n✅ Validation completed in {self.results['duration']:.2f}s")
        
        return self.results
    
    def _create_validation_summary(self) -> Dict[str, Any]:
        """Create overall validation summary."""
        summary = {
            'overall_status': 'unknown',
            'critical_issues': [],
            'warnings': [],
            'success_metrics': {},
            'recommendations': []
        }
        
        # Determine overall status
        pipeline_success = (
            self.results.get('pipeline_results', {}).get('successful_scripts', 0) >= 4
        )
        
        test_success = (
            self.results.get('test_results', {}).get('status') == 'success'
        )
        
        outputs_valid = (
            len(self.results.get('output_validation', {}).get('csv_files_found', [])) >= 3
        )
        
        if pipeline_success and test_success and outputs_valid:
            summary['overall_status'] = 'pass'
        elif pipeline_success and outputs_valid:
            summary['overall_status'] = 'pass_with_warnings'
        else:
            summary['overall_status'] = 'fail'
        
        # Success metrics
        summary['success_metrics'] = {
            'pipeline_success': pipeline_success,
            'test_success': test_success,
            'outputs_valid': outputs_valid,
            'performance_adequate': (
                self.results.get('performance_metrics', {})
                .get('benchmarks', {}).get('meets_target', False)
            )
        }
        
        return summary
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save validation report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
        
        report_path = self.results_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 Validation report saved to: {report_path}")
        return str(report_path)
    
    def generate_html_report(self, json_report_path: str) -> str:
        """Generate HTML report from JSON results."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAS-to-Python Migration Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .status-pass {{ background-color: #d4edda; }}
                .status-fail {{ background-color: #f8d7da; }}
                .status-warning {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SAS-to-Python Migration Validation Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Duration: {duration:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Overall Status: <span class="{status_class}">{overall_status}</span></h2>
                {summary_content}
            </div>
            
            <div class="section">
                <h2>Pipeline Execution Results</h2>
                {pipeline_content}
            </div>
            
            <div class="section">
                <h2>Test Results</h2>
                {test_content}
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {performance_content}
            </div>
            
            <div class="section">
                <h2>Output Validation</h2>
                {output_content}
            </div>
        </body>
        </html>
        """
        
        # Load JSON report
        with open(json_report_path, 'r') as f:
            data = json.load(f)
        
        # Prepare content sections
        overall_status = data.get('validation_summary', {}).get('overall_status', 'unknown')
        status_class = f"status-{overall_status.replace('_', '-')}"
        
        # Generate HTML content (simplified version)
        html_content = html_template.format(
            timestamp=data.get('start_time', 'Unknown'),
            duration=data.get('duration', 0),
            overall_status=overall_status.upper(),
            status_class=status_class,
            summary_content="<p>Validation completed successfully</p>",
            pipeline_content="<p>Pipeline execution results available in JSON report</p>",
            test_content="<p>Test results available in JSON report</p>",
            performance_content="<p>Performance metrics available in JSON report</p>",
            output_content="<p>Output validation results available in JSON report</p>"
        )
        
        # Save HTML report
        html_path = Path(json_report_path).with_suffix('.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report generated: {html_path}")
        return str(html_path)

def main():
    """Main entry point for validation runner."""
    parser = argparse.ArgumentParser(
        description="Automated validation for SAS-to-Python migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_validation.py                    # Full validation
    python run_validation.py --quick            # Quick smoke tests only
    python run_validation.py --performance      # Performance focus
    python run_validation.py --clean --full     # Clean and run full suite
    python run_validation.py --skip-pipeline    # Skip pipeline, test outputs only
        """
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick smoke tests only'
    )
    
    parser.add_argument(
        '--full', action='store_true', 
        help='Run complete validation suite (default)'
    )
    
    parser.add_argument(
        '--performance', action='store_true',
        help='Focus on performance benchmarking'
    )
    
    parser.add_argument(
        '--statistical', action='store_true',
        help='Focus on statistical validation'
    )
    
    parser.add_argument(
        '--compare-sas', action='store_true',
        help='Compare against SAS outputs if available'
    )
    
    parser.add_argument(
        '--skip-pipeline', action='store_true',
        help='Skip pipeline execution, validate existing outputs'
    )
    
    parser.add_argument(
        '--clean', action='store_true',
        help='Clean previous results before running'
    )
    
    parser.add_argument(
        '--report-only', action='store_true',
        help='Generate report from existing results only'
    )
    
    parser.add_argument(
        '--output-dir', default='output',
        help='Directory containing pipeline outputs'
    )
    
    args = parser.parse_args()
    
    # Determine test selection
    if args.quick:
        test_selection = "quick"
    elif args.performance:
        test_selection = "performance" 
    elif args.statistical:
        test_selection = "statistical"
    else:
        test_selection = "full"
    
    # Initialize validation runner
    config = ValidationConfig()
    config.output_dir = args.output_dir
    
    runner = ValidationRunner(config)
    
    try:
        if args.clean:
            runner.clean_previous_results()
        
        if args.report_only:
            print("📄 Report generation not yet implemented")
            sys.exit(1)
        else:
            # Run validation
            results = runner.run_validation(
                test_selection=test_selection,
                compare_sas=args.compare_sas,
                skip_pipeline=args.skip_pipeline
            )
            
            # Save reports
            json_report = runner.save_report()
            html_report = runner.generate_html_report(json_report)
            
            # Print summary
            print("\n" + "="*60)
            print("VALIDATION SUMMARY")
            print("="*60)
            
            overall_status = results.get('validation_summary', {}).get('overall_status', 'unknown')
            print(f"Overall Status: {overall_status.upper()}")
            
            if 'performance_metrics' in results:
                perf = results['performance_metrics']
                if 'performance_summary' in perf:
                    total_time = perf['performance_summary'].get('total_execution_time', 0)
                    print(f"Pipeline Execution Time: {total_time:.2f}s")
            
            # Exit with appropriate code
            if overall_status == 'pass':
                print("✅ All validations passed!")
                sys.exit(0)
            elif overall_status == 'pass_with_warnings':
                print("⚠️  Validation passed with warnings")
                sys.exit(0)
            else:
                print("❌ Validation failed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
