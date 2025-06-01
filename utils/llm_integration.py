"""
LLM Integration utilities for CSV Q&A Agent system
Handles OpenAI API calls and response processing for agent prompts using LangChain
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from config import Config

logger = logging.getLogger(__name__)

class LLMIntegration:
    """Handles LLM API calls for agent enhancements using LangChain"""
    
    def __init__(self):
        self.client = None
        
        # ✅ Set defaults before using them
        self.default_model = "gpt-4o"
        self.default_temperature = 1.0  # Use default temperature for model compatibility
        self.max_retries = 3
        self.retry_delay = 1.0
        self.json_parser = JsonOutputParser()
        
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize ChatOpenAI client with configuration"""
        try:
            if not Config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured. LLM features will be disabled.")
                return False
            
            self.client = ChatOpenAI(
                api_key=Config.OPENAI_API_KEY,
                model=self.default_model,
                temperature=self.default_temperature,
                max_tokens=1000
            )
            logger.info("LangChain ChatOpenAI client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI client: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Check if LLM integration is available"""
        return self.client is not None
    
    def call_llm(self, prompt: str, system_message: str = None, 
                 model: str = None, temperature: float = None,
                 max_completion_tokens: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Make a call to the LLM with the given prompt using LangChain
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
            model: Model to use (defaults to default_model)
            temperature: Temperature for response (defaults to default_temperature)
            max_completion_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with response data or None if failed
        """
        if not self.is_available():
            logger.warning("LLM not available for API call")
            return None
        
        # Create a new client instance with custom parameters if needed
        if model or temperature is not None or max_completion_tokens != 1000:
            model = model or self.default_model
            temperature = temperature if temperature is not None else self.default_temperature
            
            custom_client = ChatOpenAI(
                api_key=Config.OPENAI_API_KEY,
                model=model,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
        else:
            custom_client = self.client
            model = self.default_model
            temperature = self.default_temperature
        
        # Prepare messages
        messages = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        messages.append(HumanMessage(content=prompt))
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Make the call using LangChain
                response = custom_client.invoke(messages)
                
                response_time = time.time() - start_time
                
                result = {
                    'success': True,
                    'content': response.content,
                    'model': model,
                    'response_time': response_time,
                    'attempt': attempt + 1
                }
                
                # Add token usage if available
                if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                    result['tokens_used'] = response.response_metadata['token_usage']['total_tokens']
                else:
                    result['tokens_used'] = 0  # LangChain might not always provide this
                
                # Try to parse JSON response
                try:
                    # Clean the response content
                    content = result['content'].strip()
                    
                    # Log the raw response for debugging
                    logger.info(f"Raw LLM response (first 500 chars): {content[:500]}")
                    
                    if not content:
                        logger.warning("LLM returned empty response")
                        result['json_error'] = "Empty response from LLM"
                        result['parsed_json'] = None
                    else:
                        # Try to extract JSON if there's extra text
                        if content.startswith('```json'):
                            # Remove markdown code blocks
                            content = content.replace('```json', '').replace('```', '').strip()
                        elif content.startswith('```'):
                            # Remove any code blocks
                            lines = content.split('\n')
                            content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
                            content = content.strip()
                        
                        # Try to find JSON content if mixed with text
                        json_start = content.find('{')
                        json_end = content.rfind('}')
                        
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            json_content = content[json_start:json_end+1]
                            logger.debug(f"Extracted JSON content: {json_content[:100]}...")
                            result['parsed_json'] = json.loads(json_content)
                        else:
                            # Try parsing the whole content
                            result['parsed_json'] = json.loads(content)
                        
                        logger.debug("JSON parsing successful")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {str(e)}")
                    logger.warning(f"Raw response content: {repr(result['content'][:500])}")
                    result['json_error'] = str(e)
                    result['parsed_json'] = None
                except Exception as e:
                    logger.warning(f"Unexpected error during JSON processing: {str(e)}")
                    logger.warning(f"Raw response content: {repr(result['content'][:500])}")
                    result['json_error'] = str(e)
                    result['parsed_json'] = None
                
                logger.info(f"LangChain LLM call successful in {response_time:.2f}s")
                return result
                
            except Exception as e:
                logger.warning(f"LangChain LLM call attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"LangChain LLM call failed after {self.max_retries} attempts")
                    return {
                        'success': False,
                        'error': str(e),
                        'attempts': self.max_retries
                    }
        
        return None
    
    def analyze_column_type(self, column_name: str, sample_data: List[Any], 
                          total_count: int, unique_count: int, null_count: int,
                          current_dtype: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze column type and characteristics
        
        Args:
            column_name: Name of the column
            sample_data: Sample values from the column
            total_count: Total number of values
            unique_count: Number of unique values
            null_count: Number of null values
            current_dtype: Current pandas dtype
            
        Returns:
            Dictionary with analysis results
        """
        from templates.prompts import CSVLoaderPrompts, format_prompt
        
        prompt = format_prompt(
            CSVLoaderPrompts.COLUMN_TYPE_ANALYSIS,
            column_name=column_name,
            sample_data=str(sample_data),
            total_count=total_count,
            unique_count=unique_count,
            null_count=null_count,
            current_dtype=current_dtype
        )
        
        system_message = "Você é um especialista em análise de dados CSV. IMPORTANTE: Sempre responda APENAS com JSON válido, sem texto adicional antes ou depois do JSON."
        
        response = self.call_llm(prompt, system_message)
        
        if response and response.get('success'):
            if 'parsed_json' in response and response['parsed_json']:
                return response['parsed_json']
            else:
                logger.warning(f"Column analysis for '{column_name}' failed JSON parsing: {response.get('json_error', 'Unknown error')}")
                # Return a basic fallback response
                return {
                    'semantic_type': 'unknown',
                    'recommended_dtype': current_dtype,
                    'quality_issues': ['LLM analysis failed - JSON parsing error'],
                    'confidence': 0.0,
                    'analysis_error': response.get('json_error', 'Failed to parse JSON response')
                }
        
        logger.warning(f"Column analysis for '{column_name}' failed completely")
        return None
    
    def detect_relationships(self, table1_name: str, column1_name: str, column1_samples: List[Any],
                           table2_name: str, column2_name: str, column2_samples: List[Any],
                           overlap_count: int, overlap_percentage: float) -> Optional[Dict[str, Any]]:
        """
        Use LLM to detect relationships between columns
        
        Args:
            table1_name: Name of first table
            column1_name: Name of first column
            column1_samples: Sample values from first column
            table2_name: Name of second table
            column2_name: Name of second column
            column2_samples: Sample values from second column
            overlap_count: Number of overlapping values
            overlap_percentage: Percentage of overlap
            
        Returns:
            Dictionary with relationship analysis
        """
        from templates.prompts import CSVLoaderPrompts, format_prompt
        
        prompt = format_prompt(
            CSVLoaderPrompts.RELATIONSHIP_DETECTION,
            table1_name=table1_name,
            column1_name=column1_name,
            column1_samples=str(column1_samples),
            table2_name=table2_name,
            column2_name=column2_name,
            column2_samples=str(column2_samples),
            overlap_count=overlap_count,
            overlap_percentage=overlap_percentage
        )
        
        system_message = "Você é um especialista em modelagem de dados. IMPORTANTE: Sempre responda APENAS com JSON válido, sem texto adicional antes ou depois do JSON."
        
        response = self.call_llm(prompt, system_message)
        
        if response and response.get('success') and 'parsed_json' in response:
            return response['parsed_json']
        
        return None
    
    def analyze_encoding_error(self, filename: str, attempted_encodings: List[str],
                             error_message: str, file_bytes_hex: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze encoding issues and suggest solutions
        
        Args:
            filename: Name of the file
            attempted_encodings: List of encodings that were tried
            error_message: The encoding error message
            file_bytes_hex: First few bytes of file in hex format
            
        Returns:
            Dictionary with encoding analysis and suggestions
        """
        from templates.prompts import CSVLoaderPrompts, format_prompt
        
        prompt = format_prompt(
            CSVLoaderPrompts.ENCODING_ANALYSIS,
            filename=filename,
            attempted_encodings=str(attempted_encodings),
            error_message=error_message,
            file_bytes_hex=file_bytes_hex
        )
        
        system_message = "Você é um especialista em codificação de arquivos. IMPORTANTE: Sempre responda APENAS com JSON válido, sem texto adicional antes ou depois do JSON."
        
        response = self.call_llm(prompt, system_message)
        
        if response and response.get('success') and 'parsed_json' in response:
            return response['parsed_json']
        
        return None
    
    def analyze_parsing_error(self, filename: str, error_message: str, error_line: str,
                            delimiter: str, file_preview: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze CSV parsing errors and suggest solutions
        
        Args:
            filename: Name of the file
            error_message: The parsing error message
            error_line: The line that caused the error
            delimiter: Delimiter that was used
            file_preview: Preview of the file content
            
        Returns:
            Dictionary with parsing analysis and suggestions
        """
        from templates.prompts import CSVLoaderPrompts, format_prompt
        
        prompt = format_prompt(
            CSVLoaderPrompts.PARSING_ERROR_ANALYSIS,
            filename=filename,
            error_message=error_message,
            error_line=error_line,
            delimiter=delimiter,
            file_preview=file_preview
        )
        
        system_message = "Você é um especialista em análise de arquivos CSV. IMPORTANTE: Sempre responda APENAS com JSON válido, sem texto adicional antes ou depois do JSON."
        
        response = self.call_llm(prompt, system_message)
        
        if response and response.get('success') and 'parsed_json' in response:
            return response['parsed_json']
        
        return None
    
    def assess_data_quality(self, filename: str, rows: int, columns: int,
                          column_names: List[str], dtypes: Dict[str, str],
                          null_counts: Dict[str, int], duplicate_rows: int,
                          sample_data: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to assess overall data quality
        
        Args:
            filename: Name of the file
            rows: Number of rows
            columns: Number of columns
            column_names: List of column names
            dtypes: Dictionary of data types
            null_counts: Dictionary of null counts per column
            duplicate_rows: Number of duplicate rows
            sample_data: Sample data for each column
            
        Returns:
            Dictionary with data quality assessment
        """
        from templates.prompts import CSVLoaderPrompts, format_prompt
        
        prompt = format_prompt(
            CSVLoaderPrompts.DATA_QUALITY_ASSESSMENT,
            filename=filename,
            rows=rows,
            columns=columns,
            column_names=str(column_names),
            dtypes=str(dtypes),
            null_counts=str(null_counts),
            duplicate_rows=duplicate_rows,
            sample_data=str(sample_data)
        )
        
        system_message = "Você é um especialista em qualidade de dados. IMPORTANTE: Sempre responda APENAS com JSON válido, sem texto adicional antes ou depois do JSON."
        
        response = self.call_llm(prompt, system_message, max_completion_tokens=1500)
        
        if response and response.get('success') and 'parsed_json' in response:
            return response['parsed_json']
        
        return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for LLM calls"""
        # This would be implemented with a proper tracking system
        # For now, return placeholder data
        return {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens_used': 0,
            'average_response_time': 0.0,
            'is_available': self.is_available(),
            'integration': 'langchain_openai'
        }

# Global instance
llm_integration = LLMIntegration() 