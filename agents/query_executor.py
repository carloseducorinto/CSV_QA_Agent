"""
QueryExecutorAgent - Executa código pandas gerado dinamicamente e captura possíveis erros para fallback
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import traceback
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)

class QueryExecutorAgent:
    """Agent responsible for safely executing pandas code with error handling"""
    
    def __init__(self):
        self.execution_history: List[dict] = []
        self.safe_globals = self._setup_safe_globals()
    
    def _setup_safe_globals(self) -> dict:
        """Setup safe global environment for code execution"""
        return {
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
            # Add other safe modules as needed
        }
    
    def execute_code(self, code: str, dataframes: Dict[str, pd.DataFrame], 
                    timeout: int = 30) -> dict:
        """
        Safely execute pandas code with error handling
        
        Args:
            code: Pandas code to execute
            dataframes: Available DataFrames
            timeout: Execution timeout in seconds
            
        Returns:
            Dictionary containing execution results
        """
        execution_result = {
            'code': code,
            'success': False,
            'result': None,
            'output': '',
            'error': None,
            'execution_time': 0.0,
            'warnings': [],
            'fallback_executed': False
        }
        
        try:
            import time
            start_time = time.time()
            
            # Setup execution environment
            local_vars = self.safe_globals.copy()
            local_vars['dataframes'] = dataframes
            
            # Add individual dataframes to the namespace for convenience
            for name, df in dataframes.items():
                safe_name = self._make_safe_variable_name(name)
                local_vars[safe_name] = df
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, local_vars, local_vars)
            
            execution_time = time.time() - start_time
            execution_result['execution_time'] = execution_time
            
            # Get the result
            if 'result' in local_vars:
                execution_result['result'] = local_vars['result']
            else:
                # Try to find the last assigned variable
                execution_result['result'] = self._find_last_result(local_vars)
            
            # Capture output
            output = stdout_capture.getvalue()
            if output:
                execution_result['output'] = output
            
            # Check for warnings in stderr
            warnings = stderr_capture.getvalue()
            if warnings:
                execution_result['warnings'].append(warnings)
            
            execution_result['success'] = True
            logger.info(f"Code executed successfully in {execution_time:.3f}s")
            
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            execution_result['error'] = error_info
            
            logger.error(f"Code execution failed: {str(e)}")
            
            # Try fallback strategies
            fallback_result = self._try_fallback_strategies(code, dataframes, error_info)
            if fallback_result['success']:
                execution_result.update(fallback_result)
                execution_result['fallback_executed'] = True
        
        # Store in history
        self.execution_history.append(execution_result)
        
        return execution_result
    
    def _make_safe_variable_name(self, filename: str) -> str:
        """Convert filename to a safe Python variable name"""
        # Remove extension and special characters
        safe_name = filename.replace('.csv', '').replace('.zip', '')
        safe_name = ''.join(c if c.isalnum() else '_' for c in safe_name)
        
        # Ensure it starts with a letter
        if safe_name and safe_name[0].isdigit():
            safe_name = 'df_' + safe_name
        
        return safe_name or 'df'
    
    def _find_last_result(self, local_vars: dict) -> Any:
        """Try to find the last meaningful result from execution"""
        # Look for common result variable names
        result_candidates = ['result', 'output', 'answer', 'data', 'df']
        
        for candidate in result_candidates:
            if candidate in local_vars and candidate not in self.safe_globals:
                return local_vars[candidate]
        
        # If no explicit result, return the last DataFrame-like variable
        for name, value in local_vars.items():
            if (isinstance(value, (pd.DataFrame, pd.Series)) and 
                name not in self.safe_globals and 
                not name.startswith('_')):
                return value
        
        return None
    
    def _try_fallback_strategies(self, original_code: str, dataframes: Dict[str, pd.DataFrame], 
                               error_info: dict) -> dict:
        """Try different fallback strategies when code execution fails"""
        fallback_result = {
            'success': False,
            'result': None,
            'output': '',
            'error': error_info,
            'fallback_strategy': None
        }
        
        # Strategy 1: Basic data exploration if original code failed
        if 'result' not in original_code.lower():
            fallback_result = self._fallback_basic_exploration(dataframes)
            if fallback_result['success']:
                fallback_result['fallback_strategy'] = 'basic_exploration'
                return fallback_result
        
        # Strategy 2: Fix common syntax errors
        fixed_code = self._fix_common_errors(original_code, error_info)
        if fixed_code != original_code:
            try:
                local_vars = self.safe_globals.copy()
                local_vars['dataframes'] = dataframes
                
                exec(fixed_code, local_vars, local_vars)
                
                fallback_result.update({
                    'success': True,
                    'result': local_vars.get('result', self._find_last_result(local_vars)),
                    'fallback_strategy': 'syntax_fix',
                    'fixed_code': fixed_code
                })
                return fallback_result
            except:
                pass
        
        # Strategy 3: Simplified version of the operation
        simplified_result = self._fallback_simplified_operation(original_code, dataframes)
        if simplified_result['success']:
            simplified_result['fallback_strategy'] = 'simplified_operation'
            return simplified_result
        
        return fallback_result
    
    def _fallback_basic_exploration(self, dataframes: Dict[str, pd.DataFrame]) -> dict:
        """Fallback to basic data exploration"""
        try:
            if not dataframes:
                return {'success': False, 'result': None}
            
            # Get the first dataframe
            first_df_name = list(dataframes.keys())[0]
            df = dataframes[first_df_name]
            
            # Basic exploration
            result = {
                'dataframe': first_df_name,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'head': df.head().to_dict(),
                'describe': df.describe().to_dict() if df.select_dtypes(include=[np.number]).columns.any() else None,
                'info': {
                    'dtypes': df.dtypes.to_dict(),
                    'null_counts': df.isnull().sum().to_dict()
                }
            }
            
            return {
                'success': True,
                'result': result,
                'output': f"Exploração básica do arquivo: {first_df_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def _fix_common_errors(self, code: str, error_info: dict) -> str:
        """Try to fix common syntax errors in the code"""
        fixed_code = code
        error_message = error_info.get('message', '').lower()
        
        # Fix common column name issues
        if 'keyerror' in error_message or 'column' in error_message:
            # This would need more sophisticated column name matching
            # For now, just return original code
            pass
        
        # Fix missing result assignment
        if 'result' not in fixed_code and not any(line.strip().startswith('result') for line in fixed_code.split('\n')):
            lines = fixed_code.split('\n')
            if lines and not lines[-1].strip().startswith('#'):
                last_line = lines[-1].strip()
                if last_line and not last_line.startswith('result'):
                    lines[-1] = f"result = {last_line}"
                    fixed_code = '\n'.join(lines)
        
        return fixed_code
    
    def _fallback_simplified_operation(self, original_code: str, dataframes: Dict[str, pd.DataFrame]) -> dict:
        """Try a simplified version of the operation"""
        try:
            if not dataframes:
                return {'success': False, 'result': None}
            
            # Get the first dataframe
            first_df_name = list(dataframes.keys())[0]
            df = dataframes[first_df_name]
            
            # Detect what kind of operation was intended
            code_lower = original_code.lower()
            
            if 'mean' in code_lower or 'média' in code_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols[0]].mean()
                    return {
                        'success': True,
                        'result': result,
                        'output': f"Média da coluna {numeric_cols[0]}: {result}"
                    }
            
            elif 'sum' in code_lower or 'soma' in code_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols[0]].sum()
                    return {
                        'success': True,
                        'result': result,
                        'output': f"Soma da coluna {numeric_cols[0]}: {result}"
                    }
            
            elif 'count' in code_lower:
                result = len(df)
                return {
                    'success': True,
                    'result': result,
                    'output': f"Número de registros: {result}"
                }
            
            elif 'max' in code_lower or 'máximo' in code_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols[0]].max()
                    return {
                        'success': True,
                        'result': result,
                        'output': f"Valor máximo da coluna {numeric_cols[0]}: {result}"
                    }
            
            elif 'min' in code_lower or 'mínimo' in code_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = df[numeric_cols[0]].min()
                    return {
                        'success': True,
                        'result': result,
                        'output': f"Valor mínimo da coluna {numeric_cols[0]}: {result}"
                    }
            
            # Default: show basic info
            return self._fallback_basic_exploration(dataframes)
            
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def validate_code_safety(self, code: str) -> dict:
        """Validate if code is safe to execute"""
        validation_result = {
            'is_safe': True,
            'warnings': [],
            'blocked_operations': []
        }
        
        # List of potentially dangerous operations
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'exec(',
            'eval(',
            'open(',
            'file(',
            '__import__',
            'globals()',
            'locals()',
            'dir(',
            'getattr',
            'setattr',
            'delattr',
            'hasattr'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                validation_result['is_safe'] = False
                validation_result['blocked_operations'].append(pattern)
        
        # Check for file operations
        file_operations = ['read', 'write', 'delete', 'remove']
        for op in file_operations:
            if op in code_lower and ('file' in code_lower or 'csv' in code_lower):
                validation_result['warnings'].append(f"Potential file operation detected: {op}")
        
        return validation_result
    
    def get_execution_history(self) -> List[dict]:
        """Get history of code executions"""
        return self.execution_history.copy()
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics from execution history"""
        if not self.execution_history:
            return {}
        
        successful_executions = [ex for ex in self.execution_history if ex['success']]
        failed_executions = [ex for ex in self.execution_history if not ex['success']]
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(successful_executions) / len(self.execution_history) * 100,
            'average_execution_time': np.mean([ex['execution_time'] for ex in successful_executions]) if successful_executions else 0,
            'fallback_usage': len([ex for ex in self.execution_history if ex.get('fallback_executed')])
        }
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear() 