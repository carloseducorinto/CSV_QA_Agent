"""
CSVLoaderAgent - Enhanced with LLM capabilities
Loads CSV files with intelligent analysis, encoding detection, and data quality assessment
"""

import pandas as pd
import zipfile
import io
import logging
import chardet
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

from config import Config
from utils.llm_integration import llm_integration

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

@dataclass
class LoadResult:
    """Result of CSV loading operation"""
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    schema_analysis: Optional[Dict[str, Any]] = None
    quality_assessment: Optional[Dict[str, Any]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    processing_time: Optional[float] = None
    llm_insights: Optional[Dict[str, Any]] = None

class CSVLoaderAgent:
    """Enhanced CSV Loader Agent with LLM capabilities"""
    
    def __init__(self):
        logger.info("Initializing CSVLoaderAgent...")
        
        self.max_file_size = Config.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS
        self.default_encoding = Config.DEFAULT_ENCODING
        self.max_retries = Config.MAX_RETRIES
        
        # Encoding detection settings
        self.encoding_candidates = [
            'utf-8', 'utf-8-sig', 'latin1', 'cp1252', 
            'iso-8859-1', 'cp850', 'windows-1252'
        ]
        
        # CSV parsing settings
        self.csv_delimiters = [',', ';', '\t', '|']
        self.sample_size = 10000  # Bytes for encoding detection
        
        # LLM settings
        self.use_llm_analysis = llm_integration.is_available()
        
        logger.info(f"CSVLoaderAgent initialized successfully")
        logger.info(f"Max file size: {Config.MAX_FILE_SIZE_MB}MB")
        logger.info(f"Supported extensions: {self.supported_extensions}")
        logger.info(f"LLM features: {'enabled' if self.use_llm_analysis else 'disabled'}")
        logger.info(f"Encoding candidates: {len(self.encoding_candidates)} options")
        logger.info(f"CSV delimiters: {self.csv_delimiters}")
    
    def load_files(self, uploaded_files: List[Any]) -> Dict[str, LoadResult]:
        """
        Load multiple uploaded files with enhanced analysis
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Dictionary mapping filenames to LoadResult objects
        """
        logger.info(f"Starting batch file loading process for {len(uploaded_files)} files")
        start_batch_time = time.time()
        
        results = {}
        successful_files = 0
        failed_files = 0
        
        for i, uploaded_file in enumerate(uploaded_files, 1):
            logger.info(f"Processing file {i}/{len(uploaded_files)}: {uploaded_file.name}")
            
            try:
                start_time = time.time()
                result = self._load_single_file(uploaded_file)
                result.processing_time = time.time() - start_time
                results[uploaded_file.name] = result
                
                if result.success:
                    successful_files += 1
                    logger.info(f"✅ Successfully processed {uploaded_file.name} in {result.processing_time:.2f}s")
                    if result.dataframe is not None:
                        logger.info(f"   - Shape: {result.dataframe.shape}")
                        logger.info(f"   - Memory usage: {result.metadata.get('memory_usage', 'unknown')} bytes")
                else:
                    failed_files += 1
                    logger.error(f"❌ Failed to process {uploaded_file.name}: {result.errors}")
                
            except Exception as e:
                failed_files += 1
                error_msg = f"Unexpected error processing {uploaded_file.name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results[uploaded_file.name] = LoadResult(
                    success=False,
                    errors=[error_msg],
                    processing_time=time.time() - start_time if 'start_time' in locals() else 0
                )
        
        # Analyze relationships between loaded datasets
        if len(results) > 1 and self.use_llm_analysis:
            logger.info("Starting cross-dataset relationship analysis...")
            self._analyze_cross_dataset_relationships(results)
        
        batch_time = time.time() - start_batch_time
        logger.info(f"Batch processing completed in {batch_time:.2f}s")
        logger.info(f"Results: {successful_files} successful, {failed_files} failed")
        
        return results
    
    def _load_single_file(self, uploaded_file) -> LoadResult:
        """Load a single file with comprehensive analysis"""
        logger.debug(f"Starting single file load for: {uploaded_file.name}")
        
        errors = []
        warnings = []
        
        # Security validation
        logger.debug("Performing security validation...")
        if not self._validate_file_security(uploaded_file):
            error_msg = "File failed security validation"
            logger.warning(f"Security validation failed for {uploaded_file.name}")
            return LoadResult(
                success=False,
                errors=[error_msg]
            )
        
        # Size validation
        if hasattr(uploaded_file, 'size'):
            file_size_mb = uploaded_file.size / (1024 * 1024)
            logger.debug(f"File size: {file_size_mb:.2f}MB (limit: {Config.MAX_FILE_SIZE_MB}MB)")
            
            if uploaded_file.size > self.max_file_size:
                error_msg = f"File size ({file_size_mb:.1f}MB) exceeds limit ({Config.MAX_FILE_SIZE_MB}MB)"
                logger.warning(error_msg)
                return LoadResult(
                    success=False,
                    errors=[error_msg]
                )
        
        # Determine file type and load accordingly
        file_extension = Path(uploaded_file.name).suffix.lower()
        logger.debug(f"File extension detected: {file_extension}")
        
        if file_extension == '.zip':
            logger.info("Processing as ZIP file...")
            return self._load_zip_file(uploaded_file)
        elif file_extension == '.csv':
            logger.info("Processing as CSV file...")
            return self._load_csv_file(uploaded_file)
        else:
            error_msg = f"Unsupported file type: {file_extension}"
            logger.error(error_msg)
            return LoadResult(
                success=False,
                errors=[error_msg]
            )
    
    def _load_csv_file(self, uploaded_file) -> LoadResult:
        """Load CSV file with intelligent parsing and analysis"""
        logger.debug(f"Starting CSV loading process for {uploaded_file.name}")
        
        errors = []
        warnings = []
        
        # Read file content
        logger.debug("Reading file content...")
        try:
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset for potential re-reading
            logger.debug(f"File content read successfully: {len(file_content)} bytes")
        except Exception as e:
            error_msg = f"Failed to read file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return LoadResult(
                success=False,
                errors=[error_msg]
            )
        
        # Detect encoding
        logger.debug("Starting encoding detection...")
        encoding_result = self._detect_encoding(file_content, uploaded_file.name)
        if not encoding_result['success']:
            logger.error(f"Encoding detection failed: {encoding_result.get('errors', [])}")
            return LoadResult(
                success=False,
                errors=encoding_result.get('errors', ['Encoding detection failed']),
                llm_insights=encoding_result.get('llm_analysis')
            )
        
        encoding = encoding_result['encoding']
        logger.info(f"Encoding detected: {encoding} (confidence: {encoding_result.get('confidence', 'unknown')})")
        
        # Parse CSV with intelligent delimiter detection
        logger.debug("Starting CSV parsing...")
        parsing_result = self._parse_csv_content(file_content, encoding, uploaded_file.name)
        if not parsing_result['success']:
            logger.error(f"CSV parsing failed: {parsing_result.get('errors', [])}")
            return LoadResult(
                success=False,
                errors=parsing_result.get('errors', ['CSV parsing failed']),
                warnings=parsing_result.get('warnings', []),
                llm_insights=parsing_result.get('llm_analysis')
            )
        
        df = parsing_result['dataframe']
        parsing_metadata = parsing_result['metadata']
        logger.info(f"CSV parsed successfully - Shape: {df.shape}, Delimiter: {parsing_metadata.get('delimiter')}")
        
        # Generate comprehensive metadata
        logger.debug("Generating metadata...")
        metadata = self._generate_metadata(df, uploaded_file.name, encoding, parsing_metadata)
        
        # Perform schema analysis
        logger.debug("Performing schema analysis...")
        schema_analysis = self._analyze_schema(df, uploaded_file.name)
        logger.debug(f"Schema analysis completed - {len(schema_analysis.get('columns', []))} columns analyzed")
        
        # Assess data quality
        logger.debug("Assessing data quality...")
        quality_assessment = self._assess_data_quality(df, uploaded_file.name)
        quality_score = quality_assessment.get('overall_score', 0)
        logger.info(f"Data quality assessment completed - Overall score: {quality_score}/100")
        
        # LLM-enhanced insights
        llm_insights = None
        if self.use_llm_analysis:
            logger.debug("Generating LLM insights...")
            llm_insights = self._generate_llm_insights(df, uploaded_file.name, metadata)
            logger.debug("LLM insights generated")
        else:
            logger.debug("LLM analysis disabled - skipping insights generation")
        
        logger.info(f"CSV file {uploaded_file.name} loaded and analyzed successfully")
        
        return LoadResult(
            success=True,
            dataframe=df,
            metadata=metadata,
            schema_analysis=schema_analysis,
            quality_assessment=quality_assessment,
            errors=errors,
            warnings=warnings,
            llm_insights=llm_insights
        )
    
    def _detect_encoding(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Detect file encoding with LLM fallback for complex cases"""
        logger.debug(f"Starting encoding detection for {filename}")
        
        # Try chardet first
        try:
            logger.debug("Trying chardet encoding detection...")
            detected = chardet.detect(file_content[:self.sample_size])
            confidence = detected.get('confidence', 0)
            encoding = detected.get('encoding')
            
            logger.debug(f"Chardet result: {encoding} (confidence: {confidence})")
            
            if encoding and confidence > 0.7:
                # Test the encoding by trying to decode
                try:
                    file_content.decode(encoding)
                    logger.info(f"Encoding successfully detected via chardet: {encoding}")
                    return {
                        'success': True,
                        'encoding': encoding,
                        'confidence': confidence,
                        'method': 'chardet'
                    }
                except UnicodeDecodeError:
                    logger.debug(f"Chardet suggested encoding {encoding} failed decode test")
        except Exception as e:
            logger.warning(f"Chardet failed for {filename}: {str(e)}")
        
        # Try common encodings
        logger.debug("Trying common encoding fallbacks...")
        for encoding in self.encoding_candidates:
            try:
                file_content.decode(encoding)
                logger.info(f"Encoding detected via fallback: {encoding}")
                return {
                    'success': True,
                    'encoding': encoding,
                    'confidence': 0.5,
                    'method': 'fallback'
                }
            except UnicodeDecodeError:
                logger.debug(f"Encoding {encoding} failed")
                continue
        
        # LLM-enhanced encoding analysis
        llm_analysis = None
        if self.use_llm_analysis:
            logger.debug("Attempting LLM-enhanced encoding analysis...")
            try:
                file_bytes_hex = file_content[:100].hex()
                llm_analysis = llm_integration.analyze_encoding_error(
                    filename=filename,
                    attempted_encodings=self.encoding_candidates,
                    error_message="All standard encodings failed",
                    file_bytes_hex=file_bytes_hex
                )
                
                if llm_analysis and llm_analysis.get('suggested_encodings'):
                    logger.debug(f"LLM suggested encodings: {llm_analysis['suggested_encodings']}")
                    for encoding in llm_analysis['suggested_encodings']:
                        try:
                            file_content.decode(encoding)
                            logger.info(f"Encoding detected via LLM suggestion: {encoding}")
                            return {
                                'success': True,
                                'encoding': encoding,
                                'confidence': llm_analysis.get('confidence', 0.3),
                                'method': 'llm_suggestion',
                                'llm_analysis': llm_analysis
                            }
                        except UnicodeDecodeError:
                            logger.debug(f"LLM suggested encoding {encoding} failed")
                            continue
            except Exception as e:
                logger.warning(f"LLM encoding analysis failed: {str(e)}")
        
        logger.error(f"All encoding detection methods failed for {filename}")
        return {
            'success': False,
            'errors': ['Could not determine file encoding'],
            'llm_analysis': llm_analysis
        }
    
    def _parse_csv_content(self, file_content: bytes, encoding: str, filename: str) -> Dict[str, Any]:
        """Parse CSV content with intelligent delimiter detection"""
        logger.debug(f"Starting CSV content parsing for {filename} with encoding {encoding}")
        
        try:
            text_content = file_content.decode(encoding)
            logger.debug(f"Content decoded successfully: {len(text_content)} characters")
        except UnicodeDecodeError as e:
            error_msg = f"Failed to decode with {encoding}: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'errors': [error_msg]
            }
        
        # Try different parsing options
        logger.debug("Attempting CSV parsing with different delimiters...")
        for delimiter in self.csv_delimiters:
            logger.debug(f"Trying delimiter: '{delimiter}'")
            try:
                # Try with different quote characters
                for quotechar in ['"', "'", None]:
                    logger.debug(f"  Trying quotechar: {repr(quotechar)}")
                    try:
                        df = pd.read_csv(
                            io.StringIO(text_content),
                            delimiter=delimiter,
                            quotechar=quotechar,
                            on_bad_lines='skip'  # Updated parameter name
                        )
                        
                        # Validate the result
                        if len(df.columns) > 1 and len(df) > 0:
                            metadata = {
                                'delimiter': delimiter,
                                'quotechar': quotechar,
                                'encoding': encoding,
                                'parsing_method': 'standard'
                            }
                            
                            logger.info(f"CSV parsing successful with delimiter '{delimiter}' and quotechar {repr(quotechar)}")
                            logger.debug(f"Parsed DataFrame shape: {df.shape}")
                            
                            return {
                                'success': True,
                                'dataframe': df,
                                'metadata': metadata
                            }
                    
                    except Exception as parse_error:
                        logger.debug(f"    Parsing failed with {repr(quotechar)}: {str(parse_error)}")
                        continue
            
            except Exception as e:
                logger.debug(f"  Delimiter '{delimiter}' failed: {str(e)}")
                continue
        
        # LLM-enhanced parsing error analysis
        llm_analysis = None
        if self.use_llm_analysis:
            logger.debug("Attempting LLM-enhanced parsing analysis...")
            try:
                preview = text_content[:500]
                llm_analysis = llm_integration.analyze_parsing_error(
                    filename=filename,
                    error_message="Standard parsing methods failed",
                    error_line="N/A",
                    delimiter="auto-detect",
                    file_preview=preview
                )
                
                if llm_analysis and llm_analysis.get('parsing_options'):
                    options = llm_analysis['parsing_options']
                    logger.debug(f"LLM suggested parsing options: {options}")
                    
                    try:
                        df = pd.read_csv(
                            io.StringIO(text_content),
                            delimiter=options.get('delimiter', ','),
                            quotechar=options.get('quotechar', '"'),
                            skiprows=options.get('skiprows', 0),
                            header=options.get('header', 0),
                            on_bad_lines='skip'
                        )
                        
                        metadata = {
                            'delimiter': options.get('delimiter', ','),
                            'quotechar': options.get('quotechar', '"'),
                            'encoding': encoding,
                            'parsing_method': 'llm_assisted',
                            'skiprows': options.get('skiprows', 0),
                            'header': options.get('header', 0)
                        }
                        
                        logger.info(f"CSV parsing successful using LLM suggestions")
                        logger.debug(f"LLM-assisted DataFrame shape: {df.shape}")
                        
                        return {
                            'success': True,
                            'dataframe': df,
                            'metadata': metadata,
                            'warnings': ['Used LLM-assisted parsing'],
                            'llm_analysis': llm_analysis
                        }
                    
                    except Exception as e:
                        logger.warning(f"LLM-suggested parsing failed: {str(e)}")
            except Exception as e:
                logger.warning(f"LLM parsing analysis failed: {str(e)}")
        
        logger.error(f"All CSV parsing methods failed for {filename}")
        return {
            'success': False,
            'errors': ['Could not parse CSV file with any standard method'],
            'llm_analysis': llm_analysis
        }
    
    def _analyze_schema(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze DataFrame schema with LLM enhancement"""
        logger.debug(f"Starting schema analysis for {filename}")
        
        schema_info = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'shape': df.shape,
            'column_analysis': {}
        }
        
        logger.debug(f"Analyzing {len(df.columns)} columns: {df.columns.tolist()}")
        
        # Analyze each column
        for i, column in enumerate(df.columns, 1):
            logger.debug(f"Analyzing column {i}/{len(df.columns)}: {column}")
            
            col_analysis = {
                'dtype': str(df[column].dtype),
                'null_count': int(df[column].isnull().sum()),
                'null_percentage': float(df[column].isnull().sum() / len(df) * 100),
                'unique_count': int(df[column].nunique()),
                'unique_percentage': float(df[column].nunique() / len(df) * 100),
                'sample_values': df[column].dropna().head(10).tolist()
            }
            
            logger.debug(f"  Column {column}: {col_analysis['dtype']}, {col_analysis['unique_count']} unique, {col_analysis['null_count']} nulls")
            
            # LLM-enhanced column type analysis
            if self.use_llm_analysis and len(df) > 0:
                try:
                    logger.debug(f"  Running LLM analysis for column {column}")
                    llm_analysis = llm_integration.analyze_column_type(
                        column_name=column,
                        sample_data=col_analysis['sample_values'],
                        total_count=len(df),
                        unique_count=col_analysis['unique_count'],
                        null_count=col_analysis['null_count'],
                        current_dtype=str(df[column].dtype)
                    )
                    
                    if llm_analysis:
                        col_analysis['llm_analysis'] = llm_analysis
                        col_analysis['semantic_type'] = llm_analysis.get('semantic_type', 'unknown')
                        col_analysis['recommended_dtype'] = llm_analysis.get('recommended_dtype', str(df[column].dtype))
                        col_analysis['quality_issues'] = llm_analysis.get('quality_issues', [])
                        logger.debug(f"  LLM detected semantic type: {col_analysis['semantic_type']}")
                except Exception as e:
                    logger.warning(f"LLM column analysis failed for {column}: {str(e)}")
            
            schema_info['column_analysis'][column] = col_analysis
        
        logger.info(f"Schema analysis completed for {filename} - {len(df.columns)} columns analyzed")
        return schema_info
    
    def _assess_data_quality(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Assess data quality with comprehensive metrics"""
        logger.debug(f"Starting data quality assessment for {filename}")
        
        # Basic quality metrics
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        completeness = float((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0
        
        quality_metrics = {
            'completeness': completeness,
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': float(df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0,
            'empty_columns': [col for col in df.columns if df[col].isnull().all()],
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
            'high_cardinality_columns': [col for col in df.columns if df[col].nunique() > len(df) * 0.9]
        }
        
        logger.debug(f"Basic quality metrics calculated:")
        logger.debug(f"  Completeness: {completeness:.1f}%")
        logger.debug(f"  Duplicate rows: {quality_metrics['duplicate_rows']}")
        logger.debug(f"  Empty columns: {len(quality_metrics['empty_columns'])}")
        logger.debug(f"  Constant columns: {len(quality_metrics['constant_columns'])}")
        logger.debug(f"  High cardinality columns: {len(quality_metrics['high_cardinality_columns'])}")
        
        # Calculate overall quality score
        completeness_score = quality_metrics['completeness']
        duplicate_penalty = min(quality_metrics['duplicate_percentage'] * 2, 20)
        empty_col_penalty = len(quality_metrics['empty_columns']) * 10
        constant_col_penalty = len(quality_metrics['constant_columns']) * 5
        
        quality_score = max(0, completeness_score - duplicate_penalty - empty_col_penalty - constant_col_penalty)
        quality_metrics['overall_score'] = round(quality_score, 2)
        
        logger.debug(f"Quality score calculation: {completeness_score:.1f} - {duplicate_penalty:.1f} - {empty_col_penalty:.1f} - {constant_col_penalty:.1f} = {quality_score:.1f}")
        
        # LLM-enhanced quality assessment
        if self.use_llm_analysis:
            try:
                logger.debug("Running LLM-enhanced quality assessment...")
                sample_data = {}
                for col in df.columns[:10]:  # Limit to first 10 columns for API efficiency
                    sample_data[col] = df[col].dropna().head(5).tolist()
                
                llm_assessment = llm_integration.assess_data_quality(
                    filename=filename,
                    rows=len(df),
                    columns=len(df.columns),
                    column_names=list(df.columns),
                    dtypes=df.dtypes.astype(str).to_dict(),
                    null_counts=df.isnull().sum().to_dict(),
                    duplicate_rows=int(df.duplicated().sum()),
                    sample_data=sample_data
                )
                
                if llm_assessment:
                    quality_metrics['llm_assessment'] = llm_assessment
                    quality_metrics['quality_issues'] = llm_assessment.get('quality_issues', [])
                    quality_metrics['recommendations'] = llm_assessment.get('recommended_actions', [])
                    quality_metrics['insights'] = llm_assessment.get('data_insights', [])
                    logger.debug(f"LLM found {len(quality_metrics['quality_issues'])} quality issues")
            except Exception as e:
                logger.warning(f"LLM quality assessment failed: {str(e)}")
        
        logger.info(f"Data quality assessment completed for {filename} - Overall score: {quality_score:.1f}/100")
        return quality_metrics
    
    def _generate_llm_insights(self, df: pd.DataFrame, filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive LLM insights about the dataset"""
        logger.debug(f"Generating LLM insights for {filename}")
        
        if not self.use_llm_analysis:
            logger.debug("LLM analysis disabled, skipping insights generation")
            return None
        
        try:
            # Generate basic insights
            insights = {
                'summary': f"Dataset {filename} contains {len(df)} rows and {len(df.columns)} columns",
                'key_observations': [],
                'recommended_next_steps': [],
                'potential_use_cases': []
            }
            
            # Add column type insights
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            logger.debug(f"Column types: {len(numeric_cols)} numeric, {len(text_cols)} text")
            
            if numeric_cols:
                insights['key_observations'].append(f"Contains {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}")
            
            if text_cols:
                insights['key_observations'].append(f"Contains {len(text_cols)} text columns: {', '.join(text_cols[:3])}{'...' if len(text_cols) > 3 else ''}")
            
            # Add data quality insights
            null_percentage = (df.isnull().sum().sum() / df.size) * 100
            if null_percentage > 10:
                insights['key_observations'].append(f"Data has {null_percentage:.1f}% missing values - consider data cleaning")
            
            # Suggest next steps
            if numeric_cols:
                insights['recommended_next_steps'].append("Perform statistical analysis on numeric columns")
            
            if text_cols:
                insights['recommended_next_steps'].append("Analyze text patterns and categories")
            
            if len(df) > 1000:
                insights['recommended_next_steps'].append("Consider sampling for exploratory analysis")
            
            # Suggest use cases
            if len(numeric_cols) > 2:
                insights['potential_use_cases'].append("Regression analysis or predictive modeling")
            
            if text_cols and numeric_cols:
                insights['potential_use_cases'].append("Classification or clustering analysis")
            
            logger.info(f"LLM insights generated for {filename}: {len(insights['key_observations'])} observations, {len(insights['recommended_next_steps'])} recommendations")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate LLM insights for {filename}: {str(e)}", exc_info=True)
            return {
                'summary': f"Dataset {filename} contains {len(df)} rows and {len(df.columns)} columns",
                'error': f"Insight generation failed: {str(e)}"
            }
    
    def _analyze_cross_dataset_relationships(self, results: Dict[str, LoadResult]):
        """Analyze relationships between multiple loaded datasets"""
        logger.info("Starting cross-dataset relationship analysis...")
        
        if not self.use_llm_analysis:
            logger.debug("LLM analysis disabled, skipping relationship analysis")
            return
        
        successful_results = {name: result for name, result in results.items() if result.success}
        
        if len(successful_results) < 2:
            logger.debug(f"Only {len(successful_results)} successful results, need at least 2 for relationship analysis")
            return
        
        logger.info(f"Analyzing relationships between {len(successful_results)} datasets")
        
        try:
            # Compare columns across datasets to identify potential relationships
            datasets = list(successful_results.items())
            relationships_found = 0
            
            for i in range(len(datasets)):
                for j in range(i + 1, len(datasets)):
                    name1, result1 = datasets[i]
                    name2, result2 = datasets[j]
                    
                    logger.debug(f"Comparing {name1} with {name2}")
                    
                    # Find potential relationship columns
                    relationships = self._find_column_relationships(
                        name1, result1.dataframe,
                        name2, result2.dataframe
                    )
                    
                    if relationships:
                        relationships_found += len(relationships)
                        logger.info(f"Found {len(relationships)} relationships between {name1} and {name2}")
                        
                        if not result1.relationships:
                            result1.relationships = []
                        if not result2.relationships:
                            result2.relationships = []
                        
                        result1.relationships.extend(relationships)
                        result2.relationships.extend(relationships)
            
            logger.info(f"Cross-dataset relationship analysis completed - {relationships_found} relationships found")
            
        except Exception as e:
            logger.error(f"Cross-dataset relationship analysis failed: {str(e)}", exc_info=True)
    
    def _find_column_relationships(self, table1_name: str, df1: pd.DataFrame,
                                  table2_name: str, df2: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find relationships between columns in two DataFrames"""
        logger.debug(f"Finding column relationships between {table1_name} and {table2_name}")
        
        relationships = []
        comparisons_made = 0
        
        try:
            for col1 in df1.columns:
                for col2 in df2.columns:
                    comparisons_made += 1
                    logger.debug(f"Comparing {table1_name}.{col1} with {table2_name}.{col2}")
                    
                    # Calculate overlap
                    values1 = set(df1[col1].dropna().astype(str))
                    values2 = set(df2[col2].dropna().astype(str))
                    
                    overlap = values1.intersection(values2)
                    overlap_count = len(overlap)
                    
                    if overlap_count > 0:
                        overlap_percentage = (overlap_count / max(len(values1), len(values2))) * 100
                        logger.debug(f"  Overlap found: {overlap_count} values ({overlap_percentage:.1f}%)")
                        
                        if overlap_percentage > 10:  # Only analyze significant overlaps
                            logger.debug(f"  Significant overlap detected, running LLM analysis...")
                            # LLM analysis of relationship
                            try:
                                llm_analysis = llm_integration.detect_relationships(
                                    table1_name=table1_name,
                                    column1_name=col1,
                                    column1_samples=list(values1)[:10],
                                    table2_name=table2_name,
                                    column2_name=col2,
                                    column2_samples=list(values2)[:10],
                                    overlap_count=overlap_count,
                                    overlap_percentage=overlap_percentage
                                )
                                
                                if llm_analysis and llm_analysis.get('has_relationship'):
                                    relationship = {
                                        'table1': table1_name,
                                        'column1': col1,
                                        'table2': table2_name,
                                        'column2': col2,
                                        'overlap_count': overlap_count,
                                        'overlap_percentage': overlap_percentage,
                                        'relationship_type': llm_analysis.get('relationship_type'),
                                        'strength': llm_analysis.get('strength'),
                                        'confidence': llm_analysis.get('confidence'),
                                        'explanation': llm_analysis.get('explanation'),
                                        'recommendations': llm_analysis.get('recommendations', [])
                                    }
                                    relationships.append(relationship)
                                    logger.info(f"  ✅ Relationship detected: {col1} ↔ {col2} ({llm_analysis.get('relationship_type', 'unknown')})")
                            except Exception as e:
                                logger.warning(f"LLM relationship analysis failed for {col1} ↔ {col2}: {str(e)}")
                    else:
                        logger.debug(f"  No overlap found")
            
            logger.debug(f"Column relationship analysis completed: {comparisons_made} comparisons, {len(relationships)} relationships found")
            
        except Exception as e:
            logger.error(f"Column relationship analysis failed: {str(e)}", exc_info=True)
        
        return relationships
    
    def _load_zip_file(self, uploaded_file) -> LoadResult:
        """Load ZIP file containing CSV files"""
        logger.info(f"Loading ZIP file: {uploaded_file.name}")
        
        try:
            zip_content = uploaded_file.read()
            uploaded_file.seek(0)
            logger.debug(f"ZIP content read: {len(zip_content)} bytes")
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                all_files = zip_file.namelist()
                csv_files = [name for name in all_files if name.lower().endswith('.csv')]
                
                logger.debug(f"ZIP contains {len(all_files)} files, {len(csv_files)} CSV files")
                logger.debug(f"CSV files found: {csv_files}")
                
                if not csv_files:
                    error_msg = "No CSV files found in ZIP archive"
                    logger.error(error_msg)
                    return LoadResult(
                        success=False,
                        errors=[error_msg]
                    )
                
                if len(csv_files) > 1:
                    error_msg = "ZIP contains multiple CSV files. Please upload individual files for multi-file analysis."
                    logger.warning(error_msg)
                    return LoadResult(
                        success=False,
                        errors=[error_msg]
                    )
                
                # Extract and process the single CSV file
                csv_name = csv_files[0]
                logger.info(f"Extracting CSV file: {csv_name}")
                csv_content = zip_file.read(csv_name)
                logger.debug(f"Extracted CSV content: {len(csv_content)} bytes")
                
                # Create a mock uploaded file object
                class MockUploadedFile:
                    def __init__(self, content, name):
                        self.content = content
                        self.name = name
                        self.size = len(content)
                    
                    def read(self):
                        return self.content
                    
                    def seek(self, pos):
                        pass
                
                mock_file = MockUploadedFile(csv_content, csv_name)
                logger.debug(f"Created mock file object for {csv_name}")
                
                result = self._load_csv_file(mock_file)
                logger.info(f"ZIP file processed {'successfully' if result.success else 'with errors'}")
                return result
                
        except zipfile.BadZipFile:
            error_msg = "Invalid ZIP file"
            logger.error(error_msg)
            return LoadResult(
                success=False,
                errors=[error_msg]
            )
        except Exception as e:
            error_msg = f"Failed to process ZIP file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return LoadResult(
                success=False,
                errors=[error_msg]
            )
    
    def _validate_file_security(self, uploaded_file) -> bool:
        """Validate file for security concerns"""
        logger.debug(f"Validating file security for: {uploaded_file.name}")
        
        # Basic filename validation
        if '..' in uploaded_file.name or uploaded_file.name.startswith('/'):
            logger.warning(f"Security violation - suspicious filename: {uploaded_file.name}")
            return False
        
        # Extension validation
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in self.supported_extensions:
            logger.warning(f"Security violation - unsupported extension: {file_extension}")
            return False
        
        logger.debug(f"File security validation passed for: {uploaded_file.name}")
        return True
    
    def _generate_metadata(self, df: pd.DataFrame, filename: str, encoding: str, parsing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata for the loaded DataFrame"""
        logger.debug(f"Generating metadata for {filename}")
        
        memory_usage = df.memory_usage(deep=True).sum()
        
        metadata = {
            'filename': filename,
            'encoding': encoding,
            'parsing_info': parsing_metadata,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': memory_usage,
            'load_timestamp': time.time()
        }
        
        logger.debug(f"Metadata generated - Memory usage: {memory_usage} bytes, Shape: {df.shape}")
        return metadata
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        logger.debug(f"Returning supported formats: {self.supported_extensions}")
        return self.supported_extensions.copy()
    
    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM integration status and usage statistics"""
        logger.debug("Getting LLM status and usage statistics")
        
        status = {
            'enabled': self.use_llm_analysis,
            'available': llm_integration.is_available(),
            'usage_stats': llm_integration.get_usage_stats()
        }
        
        logger.debug(f"LLM status: enabled={status['enabled']}, available={status['available']}")
        return status