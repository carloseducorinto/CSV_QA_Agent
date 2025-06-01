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
import re

logger = logging.getLogger(__name__)

class QueryExecutorAgent:
    """Agent responsible for safely executing pandas code with error handling and traceability."""
    
    def __init__(self):
        self.execution_history: List[dict] = []
        self.safe_globals = self._setup_safe_globals()
        self.fallback_strategies = [
            self._fallback_basic_exploration,
            self._fallback_simplified_operation
        ]

    def _setup_safe_globals(self) -> dict:
        """Setup safe global environment for code execution."""
        return {
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
        }

    def execute_code(self, code: str, dataframes: Dict[str, pd.DataFrame], timeout: int = 30) -> dict:
        """
        Safely execute pandas code with error handling and logging.
        Logs execution environment, variables, and fallback strategies.
        """
        execution_result = {
            'code': code,
            'success': False,
            'result': None,
            'output': '',
            'error': None,
            'execution_time': 0.0,
            'warnings': [],
            'fallback_executed': False,
            'fallback_strategy': None
        }
        try:
            import time
            start_time = time.time()
            local_vars = self.safe_globals.copy()
            local_vars['dataframes'] = dataframes
            for name, df in dataframes.items():
                safe_name = self._make_safe_variable_name(name)
                local_vars[safe_name] = df
            logger.debug(f"Execution environment variables: {list(local_vars.keys())}")
            # Validate code safety
            safety = self.validate_code_safety(code)
            if not safety['safe']:
                logger.error(f"Unsafe code detected: {safety['reason']}")
                execution_result['error'] = {'type': 'UnsafeCode', 'message': safety['reason']}
                self.execution_history.append(execution_result)
                return execution_result
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, local_vars, local_vars)
            execution_time = time.time() - start_time
            execution_result['execution_time'] = execution_time
            if 'result' in local_vars:
                execution_result['result'] = local_vars['result']
            else:
                execution_result['result'] = self._find_last_result(local_vars)
            output = stdout_capture.getvalue()
            if output:
                execution_result['output'] = output
            warnings = stderr_capture.getvalue()
            if warnings:
                execution_result['warnings'].append(warnings)
            execution_result['success'] = True
            logger.info(f"Code executed successfully in {execution_time:.3f}s")
            logger.debug(f"Execution result: {execution_result['result']}")
        except (KeyError, ValueError, TypeError) as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            execution_result['error'] = error_info
            logger.error(f"Code execution failed ({type(e).__name__}): {str(e)}")
            logger.debug(f"Traceback: {error_info['traceback']}")
            fallback_result = self._try_fallback_strategies(code, dataframes, error_info)
            if fallback_result['success']:
                execution_result.update(fallback_result)
                execution_result['fallback_executed'] = True
                logger.info(f"Fallback executed: {fallback_result.get('fallback_strategy')}")
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            execution_result['error'] = error_info
            logger.error(f"Code execution failed: {str(e)}")
            logger.debug(f"Traceback: {error_info['traceback']}")
            fallback_result = self._try_fallback_strategies(code, dataframes, error_info)
            if fallback_result['success']:
                execution_result.update(fallback_result)
                execution_result['fallback_executed'] = True
                logger.info(f"Fallback executed: {fallback_result.get('fallback_strategy')}")
        self.execution_history.append(execution_result)
        return execution_result

    def _make_safe_variable_name(self, filename: str) -> str:
        """Convert filename to a safe Python variable name."""
        safe_name = filename.replace('.csv', '').replace('.zip', '')
        safe_name = ''.join(c if c.isalnum() else '_' for c in safe_name)
        if safe_name and safe_name[0].isdigit():
            safe_name = 'df_' + safe_name
        return safe_name or 'df'

    def _find_last_result(self, local_vars: dict) -> Any:
        """Try to find the last meaningful result from execution."""
        result_candidates = ['result', 'output', 'answer', 'data', 'df']
        for candidate in result_candidates:
            if candidate in local_vars and candidate not in self.safe_globals:
                return local_vars[candidate]
        for name, value in local_vars.items():
            if (isinstance(value, (pd.DataFrame, pd.Series)) and 
                name not in self.safe_globals and 
                not name.startswith('_')):
                return value
        return None

    def _try_fallback_strategies(self, original_code: str, dataframes: Dict[str, pd.DataFrame], error_info: dict) -> dict:
        """Try different fallback strategies when code execution fails. Logs strategy used."""
        for strategy in self.fallback_strategies:
            result = strategy(dataframes)
            if result.get('success'):
                result['fallback_strategy'] = strategy.__name__
                logger.info(f"Fallback strategy used: {strategy.__name__}")
                return result
        return {'success': False, 'result': None, 'output': '', 'error': error_info, 'fallback_strategy': None}

    def _fallback_basic_exploration(self, dataframes: Dict[str, pd.DataFrame]) -> dict:
        """Fallback to basic data exploration."""
        try:
            if not dataframes:
                return {'success': False, 'result': None, 'output': '', 'error': 'No dataframes available.'}
            df = list(dataframes.values())[0]
            return {'success': True, 'result': df.head(), 'output': '', 'error': None}
        except Exception as e:
            logger.error(f"Fallback basic exploration failed: {str(e)}")
            return {'success': False, 'result': None, 'output': '', 'error': str(e)}

    def _fallback_simplified_operation(self, dataframes: Dict[str, pd.DataFrame]) -> dict:
        """Fallback to a simplified operation (describe)."""
        try:
            if not dataframes:
                return {'success': False, 'result': None, 'output': '', 'error': 'No dataframes available.'}
            df = list(dataframes.values())[0]
            return {'success': True, 'result': df.describe(), 'output': '', 'error': None}
        except Exception as e:
            logger.error(f"Fallback simplified operation failed: {str(e)}")
            return {'success': False, 'result': None, 'output': '', 'error': str(e)}

    def validate_code_safety(self, code: str) -> dict:
        """Check for potentially malicious code patterns (expanded)."""
        forbidden_patterns = [
            r'__import__', r'eval\(', r'exec\(', r'open\(', r'os\.', r'sys\.', r'subprocess',
            r'socket', r'input\(', r'globals\(', r'locals\(', r'compile\(', r'breakpoint\(',
            r'pickle', r'shutil', r'ftp', r'http', r'https', r'write\(', r'read\(', r'save', r'del\s',
            r'\bimport\b', r'\bexit\b', r'\bquit\b', r'\bkill\b', r'\bremove\b', r'\bdelete\b',
            r'\bchmod\b', r'\bchown\b', r'\bmkdir\b', r'\brmdir\b', r'\bcopy\b', r'\bmove\b',
            r'\brename\b', r'\bgetenv\b', r'\benviron\b', r'\bsetattr\b', r'\bgetattr\b', r'\bexecfile\b',
            r'\bwalk\b', r'\bmakedirs\b', r'\bremovedirs\b', r'\bchdir\b', r'\bpopen\b', r'\bcall\b',
            r'\bspawn\b', r'\bstartfile\b', r'\bthread\b', r'\bthreading\b', r'\bconcurrent\b', r'\bmultiprocessing\b'
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                logger.warning(f"Unsafe code pattern detected: {pattern}")
                return {'safe': False, 'reason': f'Unsafe code pattern detected: {pattern}'}
        return {'safe': True, 'reason': ''}

    def get_execution_history(self) -> List[dict]:
        """Return the history of all code executions."""
        return self.execution_history

    def clear_history(self):
        """Clear the execution history."""
        self.execution_history.clear() 