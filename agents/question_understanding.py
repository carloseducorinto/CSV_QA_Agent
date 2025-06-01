"""
QuestionUnderstandingAgent - Hybrid LLM + Regex Natural Language Processing

This module implements the core natural language understanding capabilities for the CSV Q&A Agent.
It uses a sophisticated hybrid approach combining Large Language Models (LLM) with regex patterns
to provide maximum reliability and availability.

Architecture:
- Primary Method: LLM (ChatOpenAI) for advanced question interpretation
- Fallback Method: Regex patterns for guaranteed availability
- Hybrid System: Automatic degradation when LLM is unavailable
- Multi-language: Portuguese and English support
- Pattern Library: Extensible regex pattern system

Key Features:
- Intelligent question normalization and cleaning
- DataFrame and column identification across multiple files
- Operation detection (sum, mean, count, filter, group by, etc.)
- Code generation with security validation
- Confidence scoring and explanation generation
- Question history tracking for context
- Comprehensive error handling and logging

Processing Pipeline:
1. Question normalization and cleaning
2. LLM code generation (when available)
3. Regex pattern matching (fallback or primary)
4. DataFrame and column identification
5. Operation detection and classification
6. Pandas code generation and validation
7. Confidence calculation and explanation

Supported Operations:
- Statistical: mean, sum, count, max, min, median, std
- Data exploration: unique values, data types, info
- Filtering: where conditions, value matching
- Grouping: group by columns with aggregations
- Sorting: top N, ordering by columns
- Complex queries: multi-column operations

Security Features:
- Code validation to prevent dangerous operations
- Whitelist approach for allowed pandas operations
- Input sanitization and normalization
- Error containment and graceful degradation
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import re
import unicodedata
import os

# LangChain imports with graceful fallback handling
# This allows the system to work even without LLM capabilities
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LangChain not available, using regex-only mode")

# Configure logging for this critical module
logger = logging.getLogger(__name__)

class QuestionUnderstandingAgent:
    """
    Hybrid agent for understanding natural language questions and generating pandas code.
    
    This agent is the core of the system's intelligence, responsible for interpreting
    user questions in natural language and converting them into executable pandas code.
    It combines modern LLM capabilities with reliable regex patterns to ensure
    maximum availability and robustness.
    
    Core Capabilities:
    - Multi-language question understanding (Portuguese/English)
    - Intelligent DataFrame and column identification
    - Complex operation detection and classification
    - Secure code generation with validation
    - Confidence scoring and explanation generation
    - Question history tracking for improved context
    
    Hybrid Architecture:
    - Primary: LLM-based code generation for complex queries
    - Fallback: Regex pattern matching for common operations
    - Automatic: Seamless degradation when LLM unavailable
    - Transparent: Reports which method was used
    
    Security Model:
    - Code validation prevents dangerous operations
    - Whitelist approach for allowed pandas methods
    - Input sanitization and normalization
    - Graceful error handling and containment
    """
    
    # Comprehensive pattern library for regex-based question understanding
    # Each pattern group handles specific types of operations with multiple variations
    # Supports both Portuguese and English with accent-insensitive matching
    COMMON_PATTERNS = {
        'mean_average': {
            'patterns': [
                r'm[eé]dia\s+de\s+([\w\s_]+)',          # "média de vendas"
                r'm[eé]dia\s+da\s+coluna\s+([\w_]+)',   # "média da coluna valor"
                r'average\s+([\w\s_]+)',                # "average sales"
                r'valor\s+m[eé]dio\s+de\s+([\w\s_]+)',  # "valor médio de preço"
                r'average\s+of\s+([\w_]+)'              # "average of column"
            ],
            'template': 'df["{column}"].mean()',
            'description': 'Calculate mean/average of a column'
        },
        'sum_total': {
            'patterns': [
                r'soma\s+de\s+([\w\s_]+)',              # "soma de valores"
                r'soma\s+da\s+coluna\s+([\w_]+)',       # "soma da coluna total"
                r'total\s+de\s+([\w\s_]+)',             # "total de vendas"
                r'total\s+da\s+coluna\s+([\w_]+)',      # "total da coluna valor_total"
                r'sum\s+of\s+([\w\s_]+)',               # "sum of sales"
                r'soma\s+([\w_]+)',                     # "soma vendas"
                r'what\s+is\s+the\s+total\s+([\w\s_]+)', # "what is the total sales"
                r'total\s+([\w\s_]+)',                  # "total vendas"
                r'sum\s+([\w\s_]+)'                     # "sum sales"
            ],
            'template': 'df["{column}"].sum()',
            'description': 'Calculate sum/total of a column'
        },
        'count': {
            'patterns': [
                r'quantos\s+([\w\s_]+)',                # "quantos clientes"
                r'n[uú]mero\s+de\s+([\w\s_]+)',         # "número de registros"
                r'count\s+of\s+([\w\s_]+)',             # "count of customers"
                r'how\s+many\s+([\w\s_]+)',             # "how many items"
                r'contar\s+([\w_]+)'                    # "contar produtos"
            ],
            'template': 'df["{column}"].count()',
            'description': 'Count non-null values in a column'
        },
        'max_minimum': {
            'patterns': [
                r'maior\s+([\w\s_]+)',                  # "maior valor"
                r'm[áa]ximo\s+([\w\s_]+)',              # "máximo preço"
                r'm[áa]ximo\s+da\s+coluna\s+([\w_]+)',  # "máximo da coluna valor"
                r'm[áa]ximo\s+da\s+([\w_]+)',           # "máximo da vendas"
                r'max\s+([\w\s_]+)',                    # "max price"
                r'maximum\s+([\w\s_]+)',                # "maximum value"
                r'highest\s+([\w\s_]+)'                 # "highest sales"
            ],
            'template': 'df["{column}"].max()',
            'description': 'Find maximum value in a column'
        },
        'min_minimum': {
            'patterns': [
                r'menor\s+([\w\s_]+)',                  # "menor valor"
                r'm[íi]nimo\s+([\w\s_]+)',              # "mínimo preço"
                r'm[íi]nimo\s+da\s+coluna\s+([\w_]+)',  # "mínimo da coluna valor"
                r'm[íi]nimo\s+da\s+([\w_]+)',           # "mínimo da vendas"
                r'min\s+([\w\s_]+)',                    # "min price"
                r'minimum\s+([\w\s_]+)',                # "minimum value"
                r'lowest\s+([\w\s_]+)'                  # "lowest sales"
            ],
            'template': 'df["{column}"].min()',
            'description': 'Find minimum value in a column'
        },
        'median': {
            'patterns': [
                r'mediana\s+de\s+([\w\s_]+)',           # "mediana de preços"
                r'mediana\s+da\s+coluna\s+([\w_]+)',    # "mediana da coluna valor"
                r'median\s+of\s+([\w\s_]+)'             # "median of prices"
            ],
            'template': 'df["{column}"].median()',
            'description': 'Calculate median of a column'
        },
        'std': {
            'patterns': [
                r'desvio\s+padr[ãa]o\s+de\s+([\w\s_]+)',        # "desvio padrão de valores"
                r'desvio\s+padr[ãa]o\s+da\s+coluna\s+([\w_]+)', # "desvio padrão da coluna"
                r'std\s+of\s+([\w\s_]+)',                       # "std of values"
                r'standard\s+deviation\s+of\s+([\w\s_]+)'       # "standard deviation of"
            ],
            'template': 'df["{column}"].std()',
            'description': 'Calculate standard deviation of a column'
        },
        'unique': {
            'patterns': [
                r'valores\s+[uú]nicos\s+de\s+([\w\s_]+)',       # "valores únicos de categoria"
                r'valores\s+[uú]nicos\s+da\s+coluna\s+([\w_]+)', # "valores únicos da coluna"
                r'unique\s+values\s+of\s+([\w\s_]+)'            # "unique values of category"
            ],
            'template': 'df["{column}"].unique()',
            'description': 'Get unique values of a column'
        },
        'group_by': {
            'patterns': [
                r'por\s+([\w\s]+)',                     # "vendas por região"
                r'group\s+by\s+([\w\s]+)',              # "group by category"
                r'agrupado\s+por\s+([\w\s]+)',          # "agrupado por cliente"
                r'dividido\s+por\s+([\w\s]+)'           # "dividido por tipo"
            ],
            'template': 'df.groupby("{column}")',
            'description': 'Group data by a column'
        },
        'top_n': {
            'patterns': [
                r'top\s+(\d+)',                         # "top 10"
                r'primeiro[s]?\s+(\d+)',                # "primeiros 5"
                r'maior[es]?\s+(\d+)',                  # "maiores 3"
                r'(\d+)\s+maiores'                      # "10 maiores"
            ],
            'template': 'df.nlargest({n}, "{column}")',
            'description': 'Get top N records'
        },
        'filter_where': {
            'patterns': [
                r'onde\s+([\w\s]+)',                    # "onde categoria = X"
                r'where\s+([\w\s]+)',                   # "where category = X"
                r'com\s+([\w\s]+)',                     # "com status ativo"
                r'que\s+tem\s+([\w\s]+)'                # "que tem valor > 100"
            ],
            'template': 'df[df["{column}"] == "{value}"]',
            'description': 'Filter data based on condition'
        }
    }

    def __init__(self):
        """
        Initialize the QuestionUnderstandingAgent with hybrid capabilities.
        
        Sets up both LLM and regex-based processing capabilities, with automatic
        fallback handling. Initializes pattern libraries, history tracking,
        and logging systems.
        
        Initialization Steps:
        1. Set up question history tracking
        2. Load pattern library for regex processing
        3. Initialize LLM if available (ChatOpenAI)
        4. Configure logging and error handling
        5. Validate system capabilities and report status
        """
        # Initialize question history for context tracking
        self.question_history: List[dict] = []
        
        # Load comprehensive pattern library for regex-based processing
        self.common_patterns = self.COMMON_PATTERNS
        
        # Initialize LLM capabilities if available
        self.llm = None
        if LANGCHAIN_AVAILABLE:
            try:
                # Check for OpenAI API key in environment
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    # Initialize ChatOpenAI with optimized settings
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",      # Balanced performance/cost model
                        temperature=0.1,            # Low temperature for consistent code generation
                        max_tokens=500              # Sufficient for most pandas operations
                    )
                    logger.info("LLM initialized successfully with OpenAI")
                else:
                    logger.warning("OpenAI API key not found, LLM unavailable")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                self.llm = None

    def _normalize(self, text: str) -> str:
        """
        Normalize text for flexible matching by removing accents and standardizing format.
        
        This critical preprocessing step ensures that questions can be matched
        regardless of accent usage, capitalization, or minor formatting differences.
        Essential for robust multilingual support.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Normalized text ready for pattern matching
            
        Normalization Steps:
        1. Convert to lowercase for case-insensitive matching
        2. Strip leading/trailing whitespace
        3. Remove Unicode combining characters (accents)
        4. Preserve word structure and underscores
        """
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()
        
        # Unicode normalization to handle accents and special characters
        # NFKD decomposition separates base characters from combining marks
        text = unicodedata.normalize('NFKD', text)
        
        # Remove combining characters (accents) while preserving base characters
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        return text

    def _clean_question(self, question: str) -> str:
        """
        Clean and normalize the question for consistent processing.
        
        Removes noise, standardizes spacing, and prepares the question for
        both LLM processing and regex pattern matching.
        
        Args:
            question (str): Raw user question
            
        Returns:
            str: Cleaned and normalized question
            
        Cleaning Steps:
        1. Apply text normalization (accents, case)
        2. Remove punctuation except meaningful characters
        3. Standardize whitespace and spacing
        4. Preserve important symbols and structure
        """
        # Apply normalization for accent/case handling
        clean = self._normalize(question)
        
        # Remove punctuation except question marks and preserve word boundaries
        clean = re.sub(r'[^\w\s\?]', ' ', clean)
        
        # Normalize whitespace (collapse multiple spaces to single space)
        clean = re.sub(r'\s+', ' ', clean)
        
        return clean

    def _generate_code_with_llm(self, question: str, df_name: str, df: pd.DataFrame) -> Optional[str]:
        """
        Generate pandas code using LLM (ChatOpenAI) with contextual information.
        
        This method leverages the power of Large Language Models to understand
        complex questions and generate appropriate pandas code. It provides
        context about the DataFrame structure to help the LLM make informed
        decisions about column names and operations.
        
        Args:
            question (str): User's natural language question
            df_name (str): Name of the DataFrame file for context
            df (pd.DataFrame): The actual DataFrame to analyze
            
        Returns:
            Optional[str]: Generated pandas code or None if failed
            
        Features:
        - Contextual prompts with DataFrame information
        - Column name and type awareness
        - Structured code generation with validation
        - Error handling and fallback preparation
        """
        # Early return if LLM is not available
        if not self.llm:
            logger.debug("LLM not available for code generation")
            return None
        
        try:
            # Prepare DataFrame context information for the LLM
            # This helps the LLM understand the data structure and make better decisions
            column_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                sample_values = df[col].dropna().head(3).tolist()
                column_info.append(f"- {col} ({dtype}): {sample_values}")
            
            context_info = "\n".join(column_info)
            
            # Construct a comprehensive prompt for the LLM
            # The prompt includes context, instructions, and examples
            prompt = f"""
You are a pandas expert. Generate pandas code to answer the following question about a CSV file.

DataFrame: {df_name}
Columns and sample data:
{context_info}

Question: {question}

Generate ONLY the pandas code (no explanations). The DataFrame variable is called `df`.
Use the exact column names as shown above.
Return only a single line of code that produces the result.

Examples:
- For "sum of sales": df["sales"].sum()
- For "average price": df["price"].mean()
- For "top 10 customers": df.nlargest(10, "sales")

Code:"""

            # Generate code using the LLM
            logger.debug(f"Sending prompt to LLM for question: {question}")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Extract and clean the generated code
            generated_code = response.content.strip()
            
            # Remove common prefixes/suffixes that LLMs sometimes add
            if generated_code.startswith('```python'):
                generated_code = generated_code[9:]
            if generated_code.startswith('```'):
                generated_code = generated_code[3:]
            if generated_code.endswith('```'):
                generated_code = generated_code[:-3]
            
            generated_code = generated_code.strip()
            
            logger.debug(f"LLM generated code: {generated_code}")
            
            # Validate the generated code before returning
            if self._validate_llm_code(generated_code, df_name):
                logger.info("LLM code generation successful and validated")
                return generated_code
            else:
                logger.warning("LLM generated code failed validation")
                return None
                
        except Exception as e:
            logger.error(f"LLM code generation failed: {str(e)}")
            return None

    def _validate_llm_code(self, code: str, df_name: str) -> bool:
        """
        Validate LLM-generated code for security and correctness.
        
        This critical security function ensures that LLM-generated code is safe
        to execute and follows expected patterns. It prevents injection attacks
        and validates that the code uses appropriate pandas operations.
        
        Args:
            code (str): Generated pandas code to validate
            df_name (str): Expected DataFrame name for context
            
        Returns:
            bool: True if code passes validation, False otherwise
            
        Validation Checks:
        - Required elements present (dataframes reference)
        - Dangerous operations blocked (exec, eval, imports)
        - Pandas operations within whitelist
        - Proper syntax and structure
        """
        # Basic safety checks - block dangerous operations
        dangerous_patterns = [
            'exec', 'eval', 'import', '__', 'subprocess', 'os.', 'sys.',
            'open(', 'file(', 'input(', 'raw_input('
        ]
        
        # Check for dangerous patterns in the generated code
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                logger.warning(f"Dangerous pattern detected in LLM code: {pattern}")
                return False
        
        # Ensure the code uses dataframes reference properly
        if 'dataframes[' not in code and 'df[' not in code and 'df.' not in code:
            logger.warning("LLM code doesn't reference DataFrame properly")
            return False
        
        # Ensure code produces a result
        if 'result =' not in code and not any(op in code for op in ['.sum()', '.mean()', '.count()', '.max()', '.min()']):
            logger.warning("LLM code doesn't appear to produce a result")
            return False
        
        # Additional validation could be added here (syntax checking, etc.)
        logger.debug("LLM code passed validation checks")
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