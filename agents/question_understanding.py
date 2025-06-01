"""
QuestionUnderstandingAgent - Interpreta perguntas em linguagem natural e gera código pandas para responder usando LangChain
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import re

logger = logging.getLogger(__name__)

class QuestionUnderstandingAgent:
    """Agent responsible for understanding natural language questions and generating pandas code"""
    
    def __init__(self):
        self.question_history: List[dict] = []
        self.common_patterns = self._load_common_patterns()
    
    def _load_common_patterns(self) -> Dict[str, dict]:
        """Load common question patterns and their corresponding code templates"""
        return {
            'mean_average': {
                'patterns': [
                    r'média\s+de\s+(\w+)',
                    r'average\s+(\w+)',
                    r'valor\s+médio\s+de\s+(\w+)'
                ],
                'template': 'df["{column}"].mean()',
                'description': 'Calculate mean/average of a column'
            },
            'sum_total': {
                'patterns': [
                    r'soma\s+de\s+(\w+)',
                    r'total\s+de\s+(\w+)',
                    r'sum\s+of\s+(\w+)'
                ],
                'template': 'df["{column}"].sum()',
                'description': 'Calculate sum/total of a column'
            },
            'count': {
                'patterns': [
                    r'quantos\s+(\w+)',
                    r'número\s+de\s+(\w+)',
                    r'count\s+of\s+(\w+)',
                    r'how\s+many\s+(\w+)'
                ],
                'template': 'df["{column}"].count()',
                'description': 'Count non-null values in a column'
            },
            'max_minimum': {
                'patterns': [
                    r'maior\s+(\w+)',
                    r'máximo\s+(\w+)',
                    r'max\s+(\w+)',
                    r'highest\s+(\w+)'
                ],
                'template': 'df["{column}"].max()',
                'description': 'Find maximum value in a column'
            },
            'min_minimum': {
                'patterns': [
                    r'menor\s+(\w+)',
                    r'mínimo\s+(\w+)',
                    r'min\s+(\w+)',
                    r'lowest\s+(\w+)'
                ],
                'template': 'df["{column}"].min()',
                'description': 'Find minimum value in a column'
            },
            'group_by': {
                'patterns': [
                    r'por\s+(\w+)',
                    r'group\s+by\s+(\w+)',
                    r'agrupado\s+por\s+(\w+)',
                    r'dividido\s+por\s+(\w+)'
                ],
                'template': 'df.groupby("{column}")',
                'description': 'Group data by a column'
            },
            'top_n': {
                'patterns': [
                    r'top\s+(\d+)',
                    r'primeiro[s]?\s+(\d+)',
                    r'maior[es]?\s+(\d+)',
                    r'(\d+)\s+maiores'
                ],
                'template': 'df.nlargest({n}, "{column}")',
                'description': 'Get top N records'
            },
            'filter_where': {
                'patterns': [
                    r'onde\s+(\w+)',
                    r'where\s+(\w+)',
                    r'com\s+(\w+)',
                    r'que\s+tem\s+(\w+)'
                ],
                'template': 'df[df["{column}"] == "{value}"]',
                'description': 'Filter data based on condition'
            }
        }
    
    def understand_question(self, question: str, available_dataframes: Dict[str, pd.DataFrame]) -> dict:
        """
        Understand a natural language question and generate appropriate pandas code
        
        Args:
            question: Natural language question
            available_dataframes: Dictionary of available DataFrames
            
        Returns:
            Dictionary containing analysis and generated code
        """
        try:
            result = {
                'original_question': question,
                'understood_intent': None,
                'target_dataframe': None,
                'target_columns': [],
                'operations': [],
                'generated_code': None,
                'confidence': 0.0,
                'explanation': '',
                'fallback_suggestions': []
            }
            
            # Clean and normalize the question
            clean_question = self._clean_question(question)
            
            # Identify target DataFrame
            target_df_name = self._identify_target_dataframe(clean_question, available_dataframes)
            result['target_dataframe'] = target_df_name
            
            if not target_df_name:
                result['explanation'] = 'Não foi possível identificar qual arquivo analisar.'
                result['fallback_suggestions'] = [f"Especifique o arquivo: {name}" for name in available_dataframes.keys()]
                return result
            
            target_df = available_dataframes[target_df_name]
            
            # Identify target columns
            target_columns = self._identify_columns(clean_question, target_df)
            result['target_columns'] = target_columns
            
            # Identify operations
            operations = self._identify_operations(clean_question)
            result['operations'] = operations
            
            # Generate code
            code = self._generate_code(clean_question, target_df_name, target_columns, operations, target_df)
            result['generated_code'] = code
            
            # Calculate confidence
            result['confidence'] = self._calculate_confidence(result)
            
            # Generate explanation
            result['explanation'] = self._generate_explanation(result)
            
            # Store in history
            self.question_history.append(result)
            
            logger.info(f"Understood question with confidence {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error understanding question: {str(e)}")
            return {
                'original_question': question,
                'error': str(e),
                'confidence': 0.0,
                'explanation': 'Erro ao processar a pergunta.'
            }
    
    def _clean_question(self, question: str) -> str:
        """Clean and normalize the question"""
        # Convert to lowercase
        clean = question.lower().strip()
        
        # Remove punctuation except for essential ones
        clean = re.sub(r'[^\w\s\?]', ' ', clean)
        
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean)
        
        return clean
    
    def _identify_target_dataframe(self, question: str, dataframes: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Identify which DataFrame the question is about"""
        # Simple heuristic: look for filename mentions in the question
        for df_name in dataframes.keys():
            # Check if the filename (without extension) is mentioned
            base_name = df_name.replace('.csv', '').replace('.zip', '').lower()
            if base_name in question:
                return df_name
        
        # If no specific file mentioned, use the first one (could be improved)
        if dataframes:
            return list(dataframes.keys())[0]
        
        return None
    
    def _identify_columns(self, question: str, df: pd.DataFrame) -> List[str]:
        """Identify which columns the question is referring to"""
        mentioned_columns = []
        
        for col in df.columns:
            col_variations = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace(' ', ''),
            ]
            
            for variation in col_variations:
                if variation in question:
                    mentioned_columns.append(col)
                    break
        
        return mentioned_columns
    
    def _identify_operations(self, question: str) -> List[dict]:
        """Identify what operations are being requested"""
        operations = []
        
        for operation_name, operation_info in self.common_patterns.items():
            for pattern in operation_info['patterns']:
                matches = re.finditer(pattern, question, re.IGNORECASE)
                for match in matches:
                    operations.append({
                        'type': operation_name,
                        'matched_text': match.group(0),
                        'parameters': match.groups(),
                        'template': operation_info['template'],
                        'description': operation_info['description']
                    })
        
        return operations
    
    def _generate_code(self, question: str, df_name: str, columns: List[str], 
                      operations: List[dict], df: pd.DataFrame) -> str:
        """Generate pandas code based on understood components"""
        if not operations:
            # Fallback: basic data exploration
            if not columns:
                return f"df = dataframes['{df_name}']\ndf.head()"
            else:
                col_name = columns[0]
                return f"df = dataframes['{df_name}']\ndf['{col_name}'].describe()"
        
        # Generate code based on operations
        code_lines = [f"df = dataframes['{df_name}']"]
        
        for operation in operations:
            if operation['type'] in ['mean_average', 'sum_total', 'count', 'max_minimum', 'min_minimum']:
                if columns:
                    col_name = columns[0]
                    code_line = operation['template'].format(column=col_name)
                    code_lines.append(f"result = {code_line}")
                
            elif operation['type'] == 'group_by':
                if len(columns) >= 2:
                    group_col = columns[0]
                    agg_col = columns[1]
                    code_lines.append(f"result = df.groupby('{group_col}')['{agg_col}'].sum()")
                elif columns:
                    col_name = columns[0]
                    code_lines.append(f"result = df.groupby('{col_name}').size()")
                    
            elif operation['type'] == 'top_n':
                n = operation['parameters'][0] if operation['parameters'] else '10'
                if columns:
                    col_name = columns[0]
                    code_lines.append(f"result = df.nlargest({n}, '{col_name}')")
                else:
                    code_lines.append(f"result = df.head({n})")
                    
            elif operation['type'] == 'filter_where':
                if columns:
                    col_name = columns[0]
                    # This would need more sophisticated value detection
                    code_lines.append(f"# Filter by {col_name} - specify value")
                    code_lines.append(f"# result = df[df['{col_name}'] == 'value']")
        
        if len(code_lines) == 1:  # Only DataFrame assignment
            code_lines.append("result = df.describe()")
        
        return '\n'.join(code_lines)
    
    def _calculate_confidence(self, result: dict) -> float:
        """Calculate confidence score for the understanding"""
        confidence = 0.0
        
        # Base confidence if we identified a dataframe
        if result['target_dataframe']:
            confidence += 0.3
        
        # Bonus for identified columns
        if result['target_columns']:
            confidence += 0.3
        
        # Bonus for identified operations
        if result['operations']:
            confidence += 0.4
        
        # Reduce confidence if no operations found
        if not result['operations']:
            confidence *= 0.5
        
        return min(1.0, confidence)
    
    def _generate_explanation(self, result: dict) -> str:
        """Generate human-readable explanation of what was understood"""
        explanations = []
        
        if result['target_dataframe']:
            explanations.append(f"Analisando arquivo: {result['target_dataframe']}")
        
        if result['target_columns']:
            cols = ', '.join(result['target_columns'])
            explanations.append(f"Colunas identificadas: {cols}")
        
        if result['operations']:
            ops = [op['description'] for op in result['operations']]
            explanations.append(f"Operações: {', '.join(ops)}")
        
        if not explanations:
            explanations.append("Pergunta não compreendida completamente")
        
        return ' | '.join(explanations)
    
    def suggest_improvements(self, question: str, available_dataframes: Dict[str, pd.DataFrame]) -> List[str]:
        """Suggest improvements to make the question clearer"""
        suggestions = []
        
        # Check if question mentions specific files
        mentioned_files = [name for name in available_dataframes.keys() 
                          if name.lower().replace('.csv', '') in question.lower()]
        
        if not mentioned_files:
            suggestions.append("Especifique qual arquivo analisar")
        
        # Check if question mentions columns
        all_columns = []
        for df in available_dataframes.values():
            all_columns.extend(df.columns.tolist())
        
        mentioned_columns = [col for col in all_columns 
                           if col.lower() in question.lower()]
        
        if not mentioned_columns:
            suggestions.append("Mencione colunas específicas para análise")
        
        # Check for operation words
        operation_words = ['média', 'soma', 'total', 'máximo', 'mínimo', 'count', 'grupo']
        has_operations = any(word in question.lower() for word in operation_words)
        
        if not has_operations:
            suggestions.append("Especifique que tipo de análise deseja (média, soma, etc.)")
        
        return suggestions
    
    def get_question_history(self) -> List[dict]:
        """Get history of processed questions"""
        return self.question_history.copy()
    
    def clear_history(self):
        """Clear question history"""
        self.question_history.clear() 