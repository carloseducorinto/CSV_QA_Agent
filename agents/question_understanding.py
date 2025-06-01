"""
QuestionUnderstandingAgent - Interpreta perguntas em linguagem natural e gera código pandas para responder usando LangChain
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import re
import unicodedata
import os

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LangChain not available, using regex-only mode")

logger = logging.getLogger(__name__)

class QuestionUnderstandingAgent:
    """Agent responsible for understanding natural language questions and generating pandas code"""
    
    # Modularized pattern definitions for reuse/testing
    COMMON_PATTERNS = {
        'mean_average': {
            'patterns': [
                r'm[eé]dia\s+de\s+([\w\s_]+)',
                r'm[eé]dia\s+da\s+coluna\s+([\w_]+)',
                r'average\s+([\w\s_]+)',
                r'valor\s+m[eé]dio\s+de\s+([\w\s_]+)',
                r'average\s+of\s+([\w_]+)'
            ],
            'template': 'df["{column}"].mean()',
            'description': 'Calculate mean/average of a column'
        },
        'sum_total': {
            'patterns': [
                r'soma\s+de\s+([\w\s_]+)',
                r'soma\s+da\s+coluna\s+([\w_]+)',
                r'total\s+de\s+([\w\s_]+)',
                r'total\s+da\s+coluna\s+([\w_]+)',
                r'sum\s+of\s+([\w\s_]+)',
                r'soma\s+([\w_]+)',
                r'what\s+is\s+the\s+total\s+([\w\s_]+)',
                r'total\s+([\w\s_]+)',
                r'sum\s+([\w\s_]+)'
            ],
            'template': 'df["{column}"].sum()',
            'description': 'Calculate sum/total of a column'
        },
        'count': {
            'patterns': [
                r'quantos\s+([\w\s_]+)',
                r'n[uú]mero\s+de\s+([\w\s_]+)',
                r'count\s+of\s+([\w\s_]+)',
                r'how\s+many\s+([\w\s_]+)',
                r'contar\s+([\w_]+)'
            ],
            'template': 'df["{column}"].count()',
            'description': 'Count non-null values in a column'
        },
        'max_minimum': {
            'patterns': [
                r'maior\s+([\w\s_]+)',
                r'm[áa]ximo\s+([\w\s_]+)',
                r'm[áa]ximo\s+da\s+coluna\s+([\w_]+)',
                r'm[áa]ximo\s+da\s+([\w_]+)',
                r'max\s+([\w\s_]+)',
                r'maximum\s+([\w\s_]+)',
                r'highest\s+([\w\s_]+)'
            ],
            'template': 'df["{column}"].max()',
            'description': 'Find maximum value in a column'
        },
        'min_minimum': {
            'patterns': [
                r'menor\s+([\w\s_]+)',
                r'm[íi]nimo\s+([\w\s_]+)',
                r'm[íi]nimo\s+da\s+coluna\s+([\w_]+)',
                r'm[íi]nimo\s+da\s+([\w_]+)',
                r'min\s+([\w\s_]+)',
                r'minimum\s+([\w\s_]+)',
                r'lowest\s+([\w\s_]+)'
            ],
            'template': 'df["{column}"].min()',
            'description': 'Find minimum value in a column'
        },
        'median': {
            'patterns': [
                r'mediana\s+de\s+([\w\s_]+)',
                r'mediana\s+da\s+coluna\s+([\w_]+)',
                r'median\s+of\s+([\w\s_]+)'
            ],
            'template': 'df["{column}"].median()',
            'description': 'Calculate median of a column'
        },
        'std': {
            'patterns': [
                r'desvio\s+padr[ãa]o\s+de\s+([\w\s_]+)',
                r'desvio\s+padr[ãa]o\s+da\s+coluna\s+([\w_]+)',
                r'std\s+of\s+([\w\s_]+)',
                r'standard\s+deviation\s+of\s+([\w\s_]+)'
            ],
            'template': 'df["{column}"].std()',
            'description': 'Calculate standard deviation of a column'
        },
        'unique': {
            'patterns': [
                r'valores\s+[uú]nicos\s+de\s+([\w\s_]+)',
                r'valores\s+[uú]nicos\s+da\s+coluna\s+([\w_]+)',
                r'unique\s+values\s+of\s+([\w\s_]+)'
            ],
            'template': 'df["{column}"].unique()',
            'description': 'Get unique values of a column'
        },
        'group_by': {
            'patterns': [
                r'por\s+([\w\s]+)',
                r'group\s+by\s+([\w\s]+)',
                r'agrupado\s+por\s+([\w\s]+)',
                r'dividido\s+por\s+([\w\s]+)'
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
                r'onde\s+([\w\s]+)',
                r'where\s+([\w\s]+)',
                r'com\s+([\w\s]+)',
                r'que\s+tem\s+([\w\s]+)'
            ],
            'template': 'df[df["{column}"] == "{value}"]',
            'description': 'Filter data based on condition'
        }
    }

    def __init__(self):
        """Initialize the agent and its history."""
        self.question_history: List[dict] = []
        self.common_patterns = self.COMMON_PATTERNS
        
        # Initialize LLM if available
        self.llm = None
        if LANGCHAIN_AVAILABLE:
            try:
                # Check for OpenAI API key
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.1,
                        max_tokens=500
                    )
                    logger.info("LLM initialized successfully with OpenAI")
                else:
                    logger.warning("OpenAI API key not found, LLM unavailable")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                self.llm = None

    def _normalize(self, text: str) -> str:
        """Normalize text for flexible matching (remove accents, lowercase, etc)."""
        text = text.lower().strip()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text

    def _clean_question(self, question: str) -> str:
        """Clean and normalize the question."""
        clean = self._normalize(question)
        clean = re.sub(r'[^\w\s\?]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean

    def _generate_code_with_llm(self, question: str, df_name: str, df: pd.DataFrame) -> Optional[str]:
        """
        Generate pandas code using LLM (ChatOpenAI).
        
        Args:
            question: User's natural language question
            df_name: Name of the DataFrame file
            df: The actual DataFrame to analyze
            
        Returns:
            Generated pandas code or None if failed
        """
        if not self.llm:
            logger.debug("LLM not available, skipping LLM-based code generation")
            return None
        
        try:
            # Prepare column information
            columns_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                sample_values = df[col].dropna().head(3).tolist()
                columns_info.append(f"- {col} ({dtype}): {sample_values}")
            
            columns_str = "\n".join(columns_info)
            
            # Create the prompt
            prompt = f"""Você é um assistente de dados especializado em pandas. O usuário fez a seguinte pergunta:

{question}

Você tem acesso ao arquivo {df_name}, que possui as seguintes colunas:
{columns_str}

Gere um código pandas em Python que:
- Carregue o DataFrame com: df = dataframes['{df_name}']
- Realize a operação necessária com base na pergunta
- Armazene o resultado final na variável: result
- O código deve ser válido e executável

IMPORTANTE: 
- Responda APENAS com o código Python, sem explicações
- Use exatamente os nomes das colunas fornecidos
- Certifique-se de que o resultado final esteja na variável 'result'
- Se a pergunta não for clara, faça a melhor interpretação possível

Exemplo de formato de resposta:
df = dataframes['{df_name}']
result = df['coluna_exemplo'].sum()"""

            logger.debug(f"LLM Prompt enviado:\n{prompt}")
            
            # Send prompt to LLM
            message = HumanMessage(content=prompt)
            response = self.llm([message])
            
            generated_code = response.content.strip()
            logger.debug(f"LLM Response recebida:\n{generated_code}")
            
            # Validate the response contains valid code
            if self._validate_llm_code(generated_code, df_name):
                logger.info("✅ Código LLM validado com sucesso")
                return generated_code
            else:
                logger.warning("❌ Código LLM inválido, será usado fallback")
                return None
                
        except Exception as e:
            logger.error(f"Erro na geração de código LLM: {str(e)}")
            return None

    def _validate_llm_code(self, code: str, df_name: str) -> bool:
        """
        Validate that the LLM-generated code contains required elements.
        
        Args:
            code: Generated code string
            df_name: Expected DataFrame name
            
        Returns:
            True if code appears valid, False otherwise
        """
        if not code:
            return False
        
        # Check for required elements
        required_elements = [
            f"dataframes['{df_name}']",  # DataFrame loading
            "result =",  # Result assignment
        ]
        
        for element in required_elements:
            if element not in code:
                logger.debug(f"Elemento requerido ausente: {element}")
                return False
        
        # Check for dangerous operations (basic safety)
        dangerous_patterns = [
            'import os', 'import sys', 'exec(', 'eval(', 
            'open(', '__import__', 'subprocess'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                logger.warning(f"Padrão perigoso detectado: {pattern}")
                return False
        
        return True

    def understand_question(self, question: str, available_dataframes: Dict[str, pd.DataFrame]) -> dict:
        """
        Understand a natural language question and generate appropriate pandas code.
        Uses LLM first, then falls back to regex-based approach.
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
                'fallback_suggestions': [],
                'code_source': None  # 'llm' or 'regex'
            }
            
            clean_question = self._clean_question(question)
            logger.debug(f"Cleaned question: {clean_question}")

            # Identify target DataFrame
            target_df_name = self._identify_target_dataframe(clean_question, available_dataframes)
            result['target_dataframe'] = target_df_name
            if not target_df_name:
                result['explanation'] = 'Não foi possível identificar qual arquivo analisar.'
                result['fallback_suggestions'] = [f"Especifique o arquivo: {name}" for name in available_dataframes.keys()]
                return result
            
            target_df = available_dataframes[target_df_name]

            # 🔥 FIRST: Try LLM-based code generation
            llm_code = self._generate_code_with_llm(question, target_df_name, target_df)
            
            if llm_code:
                logger.info("✅ Usando código gerado por LLM")
                result['generated_code'] = llm_code
                result['code_source'] = 'llm'
                result['confidence'] = 0.95  # High confidence for LLM
                result['explanation'] = 'Código gerado usando LLM (ChatOpenAI)'
                result['understood_intent'] = 'Interpretação automática via LLM'
            else:
                # 🔄 FALLBACK: Use regex-based approach
                logger.info("⚡ Usando fallback: método baseado em regex")
                
                # Identify target columns
                target_columns = self._identify_columns(clean_question, target_df)
                result['target_columns'] = target_columns
                logger.debug(f"Selected columns: {target_columns}")

                # Identify operations
                operations = self._identify_operations(clean_question)
                result['operations'] = operations
                logger.debug(f"Matched operations: {operations}")

                # Generate code using regex method
                regex_code = self._generate_code(clean_question, target_df_name, target_columns, operations, target_df)
                
                if regex_code:
                    result['generated_code'] = regex_code
                    result['code_source'] = 'regex'
                    result['confidence'] = self._calculate_confidence(result)
                    result['explanation'] = self._generate_explanation(result)
                else:
                    result['explanation'] = 'Não foi possível gerar código para esta pergunta.'

            # Store in history
            self.question_history.append(result)

            logger.info(f"Question understood with confidence {result['confidence']:.2f} using {result.get('code_source', 'unknown')} method")
            return result
            
        except KeyError as e:
            logger.error(f"KeyError in understanding question: {str(e)}")
            return {
                'original_question': question,
                'error': str(e),
                'confidence': 0.0,
                'explanation': 'Erro ao processar a pergunta (KeyError).',
                'code_source': 'error'
            }
        except ValueError as e:
            logger.error(f"ValueError in understanding question: {str(e)}")
            return {
                'original_question': question,
                'error': str(e),
                'confidence': 0.0,
                'explanation': 'Erro ao processar a pergunta (ValueError).',
                'code_source': 'error'
            }
        except TypeError as e:
            logger.error(f"TypeError in understanding question: {str(e)}")
            return {
                'original_question': question,
                'error': str(e),
                'confidence': 0.0,
                'explanation': 'Erro ao processar a pergunta (TypeError).',
                'code_source': 'error'
            }
        except Exception as e:
            logger.error(f"Error understanding question: {str(e)}")
            return {
                'original_question': question,
                'error': str(e),
                'confidence': 0.0,
                'explanation': 'Erro ao processar a pergunta.',
                'code_source': 'error'
            }

    def _identify_target_dataframe(self, question: str, dataframes: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Identify which DataFrame the question is about."""
        # First try exact filename matches (including extensions)
        for df_name in dataframes.keys():
            if df_name.lower() in question.lower():
                logger.debug(f"Exact filename match found: {df_name}")
                return df_name
        
        # Then try without extensions
        for df_name in dataframes.keys():
            base_name = df_name.replace('.csv', '').replace('.zip', '').lower()
            if base_name in question:
                logger.debug(f"Base filename match found: {df_name} (base: {base_name})")
                return df_name
        
        # Try partial matches
        for df_name in dataframes.keys():
            # Split filename by underscores and check if any part matches
            parts = df_name.replace('.csv', '').replace('.zip', '').split('_')
            for part in parts:
                if len(part) > 3 and part.lower() in question.lower():
                    logger.debug(f"Partial filename match found: {df_name} (part: {part})")
                    return df_name
        
        # Default to first dataframe if only one available
        if dataframes:
            default_df = list(dataframes.keys())[0]
            logger.debug(f"Using default dataframe: {default_df}")
            return default_df
        
        return None

    def _identify_columns(self, question: str, df: pd.DataFrame) -> List[str]:
        """Identify columns in the DataFrame that match the question."""
        columns = []
        normalized_question = self._normalize(question)
        
        logger.debug(f"Searching for columns in question: '{normalized_question}'")
        logger.debug(f"Available columns: {list(df.columns)}")
        
        for col in df.columns:
            norm_col = self._normalize(str(col))
            
            # Direct match
            if norm_col in normalized_question:
                columns.append(col)
                logger.debug(f"Direct column match found: {col}")
                continue
            
            # Match individual words (for columns like 'valor_total')
            col_words = norm_col.replace('_', ' ').split()
            question_words = normalized_question.split()
            
            # Check if all column words appear in question
            if all(word in question_words for word in col_words if len(word) > 2):
                columns.append(col)
                logger.debug(f"Word-based column match found: {col} (words: {col_words})")
                continue
            
            # Check if column appears with underscores replaced by spaces
            col_spaced = norm_col.replace('_', ' ')
            if col_spaced in normalized_question:
                columns.append(col)
                logger.debug(f"Spaced column match found: {col} (as: {col_spaced})")
                continue
            
            # Check for partial matches (e.g., "receita" matches "receita_mensal")
            for word in question_words:
                if len(word) > 3 and word in norm_col:
                    columns.append(col)
                    logger.debug(f"Partial column match found: {col} (word: {word})")
                    break
        
        # Remove duplicates while preserving order
        columns = list(dict.fromkeys(columns))
        
        # If no columns found and there's only one numeric column, use it
        if not columns:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) == 1:
                columns = numeric_cols
                logger.debug(f"Using single numeric column as fallback: {columns[0]}")
        
        # If still no columns and there's only one column total, use it
        if not columns and len(df.columns) == 1:
            columns = [df.columns[0]]
            logger.debug(f"Using single available column as fallback: {columns[0]}")
        
        logger.debug(f"Final selected columns: {columns}")
        return columns

    def _identify_operations(self, question: str) -> List[dict]:
        """Identify operations in the question using multilingual regex patterns."""
        operations = []
        normalized_question = self._normalize(question)
        
        logger.debug(f"Searching for operations in question: '{normalized_question}'")
        
        for op_name, op_info in self.common_patterns.items():
            for pattern in op_info['patterns']:
                match = re.search(pattern, normalized_question, re.IGNORECASE)
                if match:
                    logger.debug(f"Pattern matched for operation '{op_name}': {pattern} | Groups: {match.groups()}")
                    operations.append({
                        'operation': op_name, 
                        'groups': match.groups(), 
                        'pattern': pattern,
                        'matched_text': match.group(0)
                    })
        
        logger.debug(f"Final identified operations: {[op['operation'] for op in operations]}")
        return operations

    def _generate_code(self, question: str, df_name: str, columns: List[str], operations: List[dict], df: pd.DataFrame) -> str:
        """Generate pandas code based on identified operations and columns."""
        if not operations:
            logger.warning("No operations identified, cannot generate code")
            return ''
        
        if not columns:
            logger.warning("No columns identified, cannot generate code")
            return ''
        
        code_lines = []
        
        # Add DataFrame selection
        code_lines.append(f"df = dataframes['{df_name}']")
        
        # Use the first operation and first column for simplicity
        op = operations[0]
        col = columns[0]
        
        # Try to extract column name from the regex groups if available
        if op.get('groups') and op['groups']:
            # Try to find a column that matches the captured group
            captured_text = op['groups'][0].strip()
            logger.debug(f"Captured text from regex: '{captured_text}'")
            
            # Check if the captured text matches any column directly
            for df_col in df.columns:
                norm_df_col = self._normalize(str(df_col))
                norm_captured = self._normalize(captured_text)
                
                if norm_df_col == norm_captured:
                    col = df_col
                    logger.debug(f"Found exact column match from regex: {col}")
                    break
                elif norm_df_col.replace('_', ' ') == norm_captured.replace('_', ' '):
                    col = df_col
                    logger.debug(f"Found spaced column match from regex: {col}")
                    break
                elif captured_text.lower() in str(df_col).lower():
                    col = df_col
                    logger.debug(f"Found partial column match from regex: {col}")
                    break
        
        op_name = op['operation']
        op_info = self.common_patterns[op_name]
        template = op_info['template']
        
        logger.debug(f"Generating code for operation '{op_name}' on column '{col}'")
        
        # Handle different operation types
        if op_name == 'top_n' and op['groups']:
            n = op['groups'][0]
            code = template.format(column=col, n=n)
        elif op_name == 'filter_where' and op['groups']:
            value = op['groups'][0]
            code = template.format(column=col, value=value)
        else:
            code = template.format(column=col)
        
        # Assign result to a variable for return
        code_lines.append(f"result = {code}")
        
        generated_code = '\n'.join(code_lines)
        logger.info(f"Generated code:\n{generated_code}")
        
        return generated_code

    def _calculate_confidence(self, result: dict) -> float:
        """Calculate a confidence score for the understanding result."""
        score = 0.5
        if result['operations']:
            score += 0.25
        if result['target_columns']:
            score += 0.15
        if result['generated_code']:
            score += 0.1
        return min(score, 1.0)

    def _generate_explanation(self, result: dict) -> str:
        """Generate a human-readable explanation for the result."""
        if not result['operations']:
            return 'Não foi possível identificar a operação desejada.'
        ops = ', '.join([op['operation'] for op in result['operations']])
        cols = ', '.join(result['target_columns'])
        return f"Operação(s) identificada(s): {ops}. Coluna(s): {cols}."

    def get_question_history(self) -> List[dict]:
        """Return the history of all processed questions."""
        return self.question_history

    def clear_history(self):
        """Clear the question history."""
        self.question_history.clear()

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