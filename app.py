"""
CSV Q&A Agent - Main Streamlit Application

This is the main entry point for the CSV Q&A Agent system, providing a web interface
for uploading CSV files, analyzing data, and asking questions in natural language.

The application follows a multi-agent architecture:
- CSVLoaderAgent: Handles file upload and data loading
- QuestionUnderstandingAgent: Interprets natural language questions
- QueryExecutorAgent: Executes generated pandas code safely
- AnswerFormatterAgent: Formats responses with visualizations

Features:
- Hybrid LLM + Regex system for maximum reliability
- Advanced data quality analysis
- Relationship detection between datasets
- Multilingual support (Portuguese/English)
- Interactive visualizations with Plotly
- Comprehensive error handling and logging
"""

# Import required libraries for the Streamlit web interface
import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import logging

# Add the project root to the Python path to import our agents
# This ensures our custom agents can be imported regardless of working directory
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our specialized agents for the CSV Q&A pipeline
from agents.csv_loader import CSVLoaderAgent
from agents.question_understanding import QuestionUnderstandingAgent
from agents.query_executor import QueryExecutorAgent
from agents.answer_formatter import AnswerFormatterAgent

# Import configuration for feature flags
from config import Config

# Configure logging for this module
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,  # 👈 Necessário para permitir mensagens DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# LangGraph integration (optional - only if enabled)
try:
    if Config.ENABLE_LANGGRAPH:
        from agents.langgraph_workflow import answer_question_langgraph
        LANGGRAPH_INTEGRATION_AVAILABLE = True
    else:
        LANGGRAPH_INTEGRATION_AVAILABLE = False
except ImportError:
    LANGGRAPH_INTEGRATION_AVAILABLE = False
    logger.info("LangGraph integration not available (install with: pip install langgraph)")

def analyze_uploaded_files(uploaded_files):
    """
    Analyze uploaded files using CSVLoaderAgent with comprehensive error handling.
    
    This function orchestrates the complete file analysis process:
    1. Initializes the CSV loader agent
    2. Checks LLM availability status
    3. Processes uploaded files with timing metrics
    4. Stores results in session state for later use
    5. Displays processing status and LLM availability
    
    Args:
        uploaded_files (list): List of Streamlit uploaded file objects
        
    Returns:
        dict: Analysis results for each file, or None if analysis fails
    """
    # Early return if no files provided
    if not uploaded_files:
        return None
    
    try:
        # Initialize the CSV loader agent which handles file processing
        csv_loader = CSVLoaderAgent()
        
        # Check if LLM (OpenAI) is available for enhanced analysis
        # This affects what insights and features are available
        llm_status = csv_loader.get_llm_status()
        
        # Show loading spinner while processing files
        with st.spinner("🔍 Analyzing files with AI-powered insights..."):
            # Track processing time for performance monitoring
            start_time = time.time()
            
            # Load and analyze files using the CSV loader agent
            # This includes: file parsing, schema analysis, quality assessment
            results = csv_loader.load_files(uploaded_files)
            
            # Calculate total processing time
            processing_time = time.time() - start_time
        
        # Display system status in two columns for better UX
        col1, col2 = st.columns([1, 1])
        
        # Show LLM availability status
        with col1:
            if llm_status['available']:
                st.success("🤖 AI Analysis: Enabled")
            else:
                st.warning("🤖 AI Analysis: Limited (API key needed)")
        
        # Show processing performance metrics
        with col2:
            st.info(f"⏱️ Processing time: {processing_time:.1f}s")
        
        # Store results in Streamlit session state for persistence across interactions
        st.session_state.analysis_results = results
        st.session_state.processing_time = processing_time
        
        return results
        
    except ImportError as e:
        # Handle missing dependencies gracefully
        st.error(f"❌ Import error: {str(e)}")
        st.error("Please check that all required dependencies are installed.")
        return None
    except Exception as e:
        # Handle all other errors with detailed debugging information
        st.error(f"❌ Analysis failed: {str(e)}")
        st.error("Please check the console for detailed error information.")
        
        # Log the full error traceback for debugging
        import traceback
        error_details = traceback.format_exc()
        with st.expander("🔧 Debug Information"):
            st.code(error_details)
        
        return None

def display_file_analysis(filename, result):
    """
    Display comprehensive analysis for a single file in organized tabs.
    
    This function creates a rich, tabbed interface showing:
    - Data preview with download options
    - Schema analysis with column details
    - Data quality assessment with recommendations
    - AI-powered insights (when available)
    - Relationship detection with other datasets
    
    Args:
        filename (str): Name of the analyzed file
        result: Analysis result object containing all analysis data
    """
    
    # Handle failed analysis gracefully
    if not result.success:
        st.error(f"❌ Failed to process {filename}")
        if result.errors:
            for error in result.errors:
                st.error(f"• {error}")
        return
    
    # Extract the DataFrame for analysis display
    df = result.dataframe
    
    # Display file overview header
    st.subheader(f"📊 Analysis: {filename}")
    
    # Show key metrics in a visually appealing grid layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Row count metric with formatting for large numbers
    with col1:
        st.metric("Rows", f"{len(df):,}")
    
    # Column count metric
    with col2:
        st.metric("Columns", len(df.columns))
    
    # Data quality score (0-100 scale)
    with col3:
        if result.quality_assessment:
            quality_score = result.quality_assessment.get('overall_score', 0)
            st.metric("Quality Score", f"{quality_score}/100")
        else:
            st.metric("Quality Score", "N/A")
    
    # Data completeness percentage
    with col4:
        completeness = result.quality_assessment.get('completeness', 100) if result.quality_assessment else 100
        st.metric("Completeness", f"{completeness:.1f}%")
    
    # Create organized tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data", "🔍 Schema", "📈 Quality", "🤖 AI Insights", "🔗 Relationships"])
    
    # TAB 1: Data Preview and Download
    with tab1:
        st.subheader("Data Preview")
        # Display first 100 rows to avoid overwhelming the interface
        st.dataframe(df.head(100))
        
        # Inform user if data is truncated
        if len(df) > 100:
            st.info(f"Showing first 100 rows of {len(df):,} total rows")
        
        # Provide download option for processed data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download processed data as CSV",
            data=csv,
            file_name=f"processed_{filename}",
            mime="text/csv"
        )
    
    # TAB 2: Schema Analysis and Column Details
    with tab2:
        st.subheader("Schema Analysis")
        
        if result.schema_analysis:
            schema = result.schema_analysis
            
            # Create comprehensive column summary table
            st.write("**Column Overview:**")
            col_df = pd.DataFrame([
                {
                    'Column': col,
                    'Type': analysis['dtype'],
                    'Unique Values': analysis['unique_count'],
                    'Null Count': analysis['null_count'],
                    'Null %': f"{analysis['null_percentage']:.1f}%",
                    'Semantic Type': analysis.get('semantic_type', 'Unknown')
                }
                for col, analysis in schema['column_analysis'].items()
            ])
            
            st.dataframe(col_df)
            
            # Visualize data type distribution with interactive pie chart
            dtype_counts = pd.Series(schema['dtypes']).value_counts()
            fig_dtype = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title="Data Type Distribution"
            )
            st.plotly_chart(fig_dtype)
    
    # TAB 3: Data Quality Assessment and Recommendations
    with tab3:
        st.subheader("Data Quality Assessment")
        
        if result.quality_assessment:
            quality = result.quality_assessment
            
            # Display quality metrics in organized columns
            col1, col2 = st.columns(2)
            
            # Left column: Core quality metrics
            with col1:
                st.metric("Overall Score", f"{quality['overall_score']}/100")
                st.metric("Completeness", f"{quality['completeness']:.1f}%")
                st.metric("Duplicate Rows", quality['duplicate_rows'])
            
            # Right column: Quality warnings and insights
            with col2:
                # Highlight problematic columns
                if quality.get('empty_columns'):
                    st.warning(f"Empty columns: {', '.join(quality['empty_columns'])}")
                
                if quality.get('constant_columns'):
                    st.warning(f"Constant columns: {', '.join(quality['constant_columns'])}")
                
                if quality.get('high_cardinality_columns'):
                    st.info(f"High cardinality: {', '.join(quality['high_cardinality_columns'])}")
            
            # Display actionable quality issues
            if quality.get('quality_issues'):
                st.subheader("🚨 Quality Issues Detected:")
                for issue in quality['quality_issues']:
                    st.warning(f"• {issue}")
            
            # Show data improvement recommendations
            if quality.get('recommendations'):
                st.subheader("💡 Recommendations:")
                for rec in quality['recommendations']:
                    st.info(f"• {rec}")
            
            # Visualize missing values pattern
            if df.isnull().sum().sum() > 0:
                st.subheader("Missing Values Heatmap")
                null_df = df.isnull().sum().to_frame('Null Count')
                null_df = null_df[null_df['Null Count'] > 0]
                
                if len(null_df) > 0:
                    # Create interactive bar chart for missing values
                    fig_null = px.bar(
                        x=null_df.index,
                        y=null_df['Null Count'],
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig_null)
    
    # TAB 4: AI-Powered Insights (when LLM is available)
    with tab4:
        st.subheader("🤖 AI-Powered Insights")
        
        if result.llm_insights:
            insights = result.llm_insights
            
            # Display AI-generated summary
            st.write("**Summary:**")
            st.info(insights.get('summary', 'No summary available'))
            
            # Show key observations from AI analysis
            if insights.get('key_observations'):
                st.write("**Key Observations:**")
                for obs in insights['key_observations']:
                    st.write(f"• {obs}")
            
            # Display recommended actions
            if insights.get('recommended_next_steps'):
                st.write("**Recommended Next Steps:**")
                for step in insights['recommended_next_steps']:
                    st.write(f"• {step}")
            
            # Show potential business use cases
            if insights.get('potential_use_cases'):
                st.write("**Potential Use Cases:**")
                for use_case in insights['potential_use_cases']:
                    st.write(f"• {use_case}")
        else:
            # Fallback message when LLM is not available
            st.info("AI insights require a valid OpenAI API key. The analysis above provides comprehensive data insights.")
    
    # TAB 5: Data Relationships (cross-dataset analysis)
    with tab5:
        st.subheader("🔗 Data Relationships")
        
        if result.relationships:
            st.write("**Detected Relationships with Other Datasets:**")
            
            # Display each relationship with detailed information
            for i, rel in enumerate(result.relationships, 1):
                # Create clear relationship header
                st.markdown(f"**#{i} - {rel['table1']}.{rel['column1']} ↔ {rel['table2']}.{rel['column2']}**")
                
                # Use container for better visual organization
                with st.container():
                    # Add custom styling for relationship details
                    st.markdown("""
                    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
                    """, unsafe_allow_html=True)
                    
                    # Show relationship metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Overlap Count", rel['overlap_count'])
                        st.metric("Overlap %", f"{rel['overlap_percentage']:.1f}%")
                    
                    with col2:
                        if rel.get('relationship_type'):
                            st.write(f"**Type:** {rel['relationship_type']}")
                        if rel.get('strength'):
                            st.write(f"**Strength:** {rel['strength']}")
                        if rel.get('confidence'):
                            st.write(f"**Confidence:** {rel['confidence']}")
                    
                    # Display relationship explanation
                    if rel.get('explanation'):
                        st.write(f"**Explanation:** {rel['explanation']}")
                    
                    # Show actionable recommendations
                    if rel.get('recommendations'):
                        st.write("**Recommendations:**")
                        for rec in rel['recommendations']:
                            st.write(f"• {rec}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # Add visual separator between relationships
                if i < len(result.relationships):
                    st.markdown("---")
        else:
            # Contextual messages based on number of uploaded files
            if len(st.session_state.get('analysis_results', {})) > 1:
                st.info("No relationships detected between uploaded datasets.")
            else:
                st.info("Upload multiple files to detect relationships between datasets.")

def is_simple_question(question: str) -> bool:
    """
    Determine if a question is simple enough for LangGraph testing.
    
    During the experimental phase, we only route simple questions to LangGraph
    to minimize risk while validating the new approach.
    
    Args:
        question (str): User's question
        
    Returns:
        bool: True if question contains simple operation patterns
    """
    simple_patterns = [
        'média', 'mean', 'average',
        'soma', 'sum', 'total', 
        'contar', 'count', 'quantos',
        'máximo', 'max', 'maior',
        'mínimo', 'min', 'menor'
    ]
    
    question_lower = question.lower()
    return any(pattern in question_lower for pattern in simple_patterns)

def answer_question_with_langgraph_option(question: str, analysis_results: dict) -> dict:
    """
    Enhanced answer_question that can optionally use LangGraph.
    
    This function wraps the existing answer_question functionality with optional
    LangGraph integration. It ensures 100% backward compatibility while allowing
    safe testing of the new LangGraph approach.
    
    Args:
        question (str): User's natural language question
        analysis_results (dict): Results from file analysis
        
    Returns:
        dict: Response in exact same format as current system
        
    Safety Features:
    - Always falls back to current system if LangGraph fails
    - Only uses LangGraph if explicitly enabled via configuration
    - Can filter questions to only test LangGraph on simple cases
    - Optional A/B testing to compare both approaches
    """
    import time
    start_time = time.time()
    
    # Determine if we should try LangGraph for this question
    should_try_langgraph = (
        Config.ENABLE_LANGGRAPH and 
        LANGGRAPH_INTEGRATION_AVAILABLE and
        (not Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY or is_simple_question(question))
    )
    
    # LOG: Decision point
    logger.info("=" * 60)
    logger.info(f"🔍 PROCESSING QUESTION: '{question[:50]}{'...' if len(question) > 50 else ''}'")
    logger.info(f"📊 Available DataFrames: {list(analysis_results.keys())}")
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   - ENABLE_LANGGRAPH: {Config.ENABLE_LANGGRAPH}")
    logger.info(f"   - LANGGRAPH_AVAILABLE: {LANGGRAPH_INTEGRATION_AVAILABLE}")
    logger.info(f"   - SIMPLE_QUESTIONS_ONLY: {Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY}")
    logger.info(f"   - Is Simple Question: {is_simple_question(question) if Config.LANGGRAPH_SIMPLE_QUESTIONS_ONLY else 'N/A'}")
    logger.info(f"🎯 EXECUTION PATH: {'LangGraph' if should_try_langgraph else 'Current System'}")
    logger.info("=" * 60)
    
    # Try LangGraph if enabled and appropriate
    if should_try_langgraph:
        try:
            logger.info("🚀 STARTING LANGGRAPH EXECUTION")
            logger.info(f"   Question: {question}")
            logger.info(f"   DataFrames: {len(analysis_results)} file(s)")
            
            langgraph_start = time.time()
            langgraph_result = answer_question_langgraph(question, analysis_results)
            langgraph_time = time.time() - langgraph_start
            
            logger.info(f"⏱️  LangGraph execution time: {langgraph_time:.3f}s")
            logger.info(f"✅ LangGraph result: Success={langgraph_result.get('success')}")
            
            # Optionally run comparison with current system
            if Config.ENABLE_LANGGRAPH_COMPARISON:
                try:
                    logger.info("🔄 RUNNING COMPARISON WITH CURRENT SYSTEM")
                    comparison_start = time.time()
                    current_result = answer_question_current(question, analysis_results)
                    comparison_time = time.time() - comparison_start
                    
                    _log_comparison_results(question, current_result, langgraph_result, comparison_time, langgraph_time)
                except Exception as e:
                    logger.warning(f"❌ Comparison with current system failed: {e}")
            
            # Return LangGraph result if successful
            if langgraph_result.get('success'):
                total_time = time.time() - start_time
                logger.info(f"🎉 LANGGRAPH EXECUTION COMPLETED SUCCESSFULLY")
                logger.info(f"   Total time: {total_time:.3f}s")
                logger.info(f"   Answer: {langgraph_result.get('answer', '')[:100]}{'...' if len(langgraph_result.get('answer', '')) > 100 else ''}")
                logger.info("=" * 60)
                return langgraph_result
            elif Config.LANGGRAPH_ROLLBACK_ON_ERROR:
                logger.warning(f"⚠️  LangGraph failed, rolling back to current system")
                logger.warning(f"   Error: {langgraph_result.get('error', 'Unknown error')}")
            else:
                logger.error(f"❌ LangGraph failed and rollback disabled")
                logger.error(f"   Error: {langgraph_result.get('error', 'Unknown error')}")
                total_time = time.time() - start_time
                logger.info(f"   Total time: {total_time:.3f}s")
                logger.info("=" * 60)
                return langgraph_result
                
        except Exception as e:
            logger.error(f"❌ LANGGRAPH EXECUTION EXCEPTION: {str(e)}")
            if not Config.LANGGRAPH_ROLLBACK_ON_ERROR:
                total_time = time.time() - start_time
                logger.info(f"   Total time: {total_time:.3f}s")
                logger.info("=" * 60)
                return {
                    'success': False,
                    'error': f'LangGraph processing failed: {str(e)}',
                    'answer': f'Erro no processamento LangGraph: {str(e)}'
                }
            logger.warning(f"   Rolling back to current system...")
    
    # Use current system (default behavior or fallback)
    logger.info("🔧 STARTING CURRENT SYSTEM EXECUTION")
    logger.info(f"   Reason: {'Fallback from LangGraph' if should_try_langgraph else 'Default execution path'}")
    logger.info(f"   Question: {question}")
    
    current_start = time.time()
    result = answer_question_current(question, analysis_results)
    current_time = time.time() - current_start
    total_time = time.time() - start_time
    
    logger.info(f"⏱️  Current system execution time: {current_time:.3f}s")
    logger.info(f"✅ Current system result: Success={result.get('success')}")
    logger.info(f"🎉 CURRENT SYSTEM EXECUTION COMPLETED")
    logger.info(f"   Total time: {total_time:.3f}s")
    logger.info(f"   Answer: {result.get('answer', '')[:100]}{'...' if len(result.get('answer', '')) > 100 else ''}")
    logger.info("=" * 60)
    
    return result

def _log_comparison_results(question: str, current_result: dict, langgraph_result: dict, current_time: float, langgraph_time: float):
    """Enhanced comparison logging with detailed metrics"""
    logger.info("🔍 COMPARISON ANALYSIS")
    logger.info(f"   Question: {question[:50]}{'...' if len(question) > 50 else ''}")
    logger.info(f"   📈 Performance:")
    logger.info(f"     - Current System: {current_time:.3f}s")
    logger.info(f"     - LangGraph: {langgraph_time:.3f}s")
    logger.info(f"     - Speed Difference: {((langgraph_time - current_time) / current_time * 100):+.1f}%")
    logger.info(f"   ✅ Success Rates:")
    logger.info(f"     - Current System: {current_result.get('success')}")
    logger.info(f"     - LangGraph: {langgraph_result.get('success')}")
    logger.info(f"   📝 Answers:")
    logger.info(f"     - Current: {current_result.get('answer', '')[:50]}{'...' if len(current_result.get('answer', '')) > 50 else ''}")
    logger.info(f"     - LangGraph: {langgraph_result.get('answer', '')[:50]}{'...' if len(langgraph_result.get('answer', '')) > 50 else ''}")
    logger.info(f"     - Answers Match: {current_result.get('answer') == langgraph_result.get('answer')}")
    
    # Log any differences in detail
    if current_result.get('success') != langgraph_result.get('success'):
        logger.warning(f"   ⚠️  SUCCESS MISMATCH detected!")
    
    if current_result.get('answer') != langgraph_result.get('answer'):
        logger.warning(f"   ⚠️  ANSWER MISMATCH detected!")
        logger.warning(f"     Current answer: {current_result.get('answer', 'None')}")
        logger.warning(f"     LangGraph answer: {langgraph_result.get('answer', 'None')}")
    
    logger.info("🔍 END COMPARISON ANALYSIS")

def answer_question_current(question: str, analysis_results: dict) -> dict:
    """
    Complete question-answering pipeline using all enhanced agents (CURRENT SYSTEM).
    
    This is the original, working implementation that has been renamed to allow
    side-by-side testing with LangGraph. The logic is exactly the same as before.
    
    This function orchestrates the entire Q&A process:
    1. Initialize all required agents (understanding, execution, formatting)
    2. Extract DataFrames from analysis results
    3. Use QuestionUnderstandingAgent to interpret the question
    4. Execute generated pandas code safely with QueryExecutorAgent
    5. Format the response with AnswerFormatterAgent
    6. Return comprehensive results with error handling
    
    Args:
        question (str): Natural language question from user
        analysis_results (dict): Results from file analysis containing DataFrames
        
    Returns:
        dict: Complete response including success status, answer, and debugging info
    """
    logger.info("🔧 CURRENT SYSTEM: Starting execution pipeline")
    logger.debug(f"   Question: {question}")
    logger.debug(f"   Available files: {list(analysis_results.keys())}")
    
    try:
        # Initialize all agents for the Q&A pipeline
        logger.info("   🏗️  Initializing agents...")
        question_agent = QuestionUnderstandingAgent()  # Interprets natural language
        executor_agent = QueryExecutorAgent()          # Executes pandas code safely
        formatter_agent = AnswerFormatterAgent()       # Formats responses with visualizations
        logger.info("   ✅ Agents initialized successfully")
        
        # Extract DataFrames from analysis results for processing
        # Only include successfully analyzed files with valid DataFrames
        dataframes = {}
        for filename, result in analysis_results.items():
            if result.success and result.dataframe is not None:
                dataframes[filename] = result.dataframe
        
        logger.info(f"   📊 Extracted {len(dataframes)} valid DataFrames")
        
        # Validate that we have data to work with
        if not dataframes:
            logger.warning("   ❌ No valid DataFrames available for analysis")
            return {
                'success': False,
                'error': 'No valid DataFrames available for analysis.',
                'answer': 'Nenhum dado válido disponível para análise.'
            }
        
        # STEP 1: Understand the question using hybrid LLM+Regex approach
        logger.info("   🧠 STEP 1: Understanding question...")
        understanding_result = question_agent.understand_question(question, dataframes)
        
        logger.info(f"   📋 Understanding completed:")
        logger.info(f"     - Code source: {understanding_result.get('code_source', 'unknown')}")
        logger.info(f"     - Confidence: {understanding_result.get('confidence', 0):.2f}")
        logger.info(f"     - Generated code: {'Yes' if understanding_result.get('generated_code') else 'No'}")
        
        # Check if question understanding failed
        if understanding_result.get('error'):
            logger.warning(f"   ❌ Understanding failed: {understanding_result['error']}")
            return {
                'success': False,
                'error': understanding_result['error'],
                'answer': f"Erro ao entender a pergunta: {understanding_result['explanation']}"
            }
        
        # STEP 2: Execute the generated pandas code safely
        logger.info("   ⚙️  STEP 2: Executing generated code...")
        if understanding_result.get('generated_code'):
            logger.debug(f"     Code preview: {understanding_result['generated_code'][:100]}...")
            
            execution_result = executor_agent.execute_code(
                understanding_result['generated_code'], 
                dataframes
            )
            
            logger.info(f"   📋 Execution completed:")
            logger.info(f"     - Success: {execution_result.get('success')}")
            logger.info(f"     - Execution time: {execution_result.get('execution_time', 0):.3f}s")
            logger.info(f"     - Fallback used: {execution_result.get('fallback_executed', False)}")
            
            if execution_result.get('error'):
                logger.warning(f"     - Error: {execution_result['error']}")
        else:
            # No code was generated - likely question too ambiguous
            logger.warning("   ❌ No code was generated for execution")
            return {
                'success': False,
                'error': 'No code generated',
                'answer': 'Não foi possível gerar código para esta pergunta. Tente reformular de forma mais específica.'
            }
        
        # STEP 3: Format the response with visualizations and insights
        logger.info("   📝 STEP 3: Formatting response...")
        formatted_response = formatter_agent.format_response(
            execution_result, 
            question, 
            understanding_result
        )
        
        logger.info(f"   📋 Formatting completed:")
        logger.info(f"     - Natural language answer generated: {'Yes' if formatted_response.get('natural_language_answer') else 'No'}")
        logger.info(f"     - Visualizations: {len(formatted_response.get('visualizations', []))}")
        logger.info(f"     - Data insights: {len(formatted_response.get('data_insights', []))}")
        
        final_answer = formatted_response.get('natural_language_answer', 'Resposta não disponível.')
        logger.info(f"   💬 Final answer: {final_answer[:100]}{'...' if len(final_answer) > 100 else ''}")
        
        # Return comprehensive successful result
        logger.info("   ✅ Current system execution successful")
        return {
            'success': True,
            'understanding': understanding_result,     # Question interpretation details
            'execution': execution_result,             # Code execution results
            'formatted_response': formatted_response,  # Final formatted answer
            'answer': final_answer
        }
        
    except Exception as e:
        # Handle any unexpected errors with full debugging information
        logger.error(f"   ❌ Current system exception: {str(e)}")
        logger.error(f"     Exception type: {type(e).__name__}")
        
        import traceback
        error_details = traceback.format_exc()
        logger.debug(f"     Traceback: {error_details}")
        
        return {
            'success': False,
            'error': str(e),
            'error_details': error_details,
            'answer': f'Erro interno: {str(e)}'
        }

def answer_question(question: str, analysis_results: dict) -> dict:
    """
    Main answer_question function with optional LangGraph integration.
    
    This function maintains backward compatibility while allowing optional
    LangGraph testing. By default, it uses the current working system.
    
    Args:
        question (str): Natural language question from user
        analysis_results (dict): Results from file analysis containing DataFrames
        
    Returns:
        dict: Complete response including success status, answer, and debugging info
    """
    return answer_question_with_langgraph_option(question, analysis_results)

# Configure Streamlit page settings for optimal user experience
st.set_page_config(
    page_title="Agente Aprende -  CSV Q&A Inteligente", 
    layout="wide",                    # Use full browser width
    page_icon="📊",                   # Custom page icon
    initial_sidebar_state="expanded"  # Show sidebar by default
)

# Custom CSS for enhanced visual appeal and better UX
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .question-section {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Streamlit session state for data persistence across interactions
# This ensures data persists when users interact with the interface
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main application header with gradient styling
st.markdown('<h1 class="main-header">📊 Agente Aprende - CSV Q&A Inteligente</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for file management and application settings
with st.sidebar:
    st.header("🗂️ Gerenciamento de Arquivos")
    
    # Clear session button for starting fresh
    if st.button("🗑️ Limpar Sessão", type="secondary"):
        # Reset all session state variables
        st.session_state.uploaded_files_data = {}
        st.session_state.analysis_results = {}
        st.session_state.chat_history = []
        st.success("Sessão limpa!")
        st.rerun()  # Refresh the interface
    
    # Display currently loaded files for user reference
    if st.session_state.uploaded_files_data:
        st.subheader("📋 Arquivos Carregados")
        for filename in st.session_state.uploaded_files_data.keys():
            st.markdown(f"✅ {filename}")
    
    # Application configuration options
    st.header("⚙️ Configurações")
    st.slider("Nível de Detalhamento", 1, 5, 3, help="Controla o nível de detalhamento das respostas")
    st.checkbox("Mostrar código gerado", help="Exibe o código pandas gerado internamente")

# Main content area split into two columns for better organization
col1, col2 = st.columns([1, 1])

# LEFT COLUMN: File Upload and Analysis Section
with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload de Arquivos")
    
    # File uploader with support for multiple formats
    uploaded_files = st.file_uploader(
        "Envie aqui seus arquivos CSV ou ZIP",
        type=["csv", "zip"],                    # Supported file types
        accept_multiple_files=True,             # Allow multiple file selection
        help="Suporte para múltiplos arquivos CSV e arquivos ZIP contendo CSVs"
    )
    
    # Validate and display uploaded files with comprehensive error checking
    if uploaded_files:
        st.markdown("### 📊 Arquivos Carregados")
        
        valid_files = []
        for file in uploaded_files:
            try:
                # Calculate and display file size information
                file_size_mb = file.size / (1024 * 1024)
                st.write(f"📄 **{file.name}** ({file.size:,} bytes / {file_size_mb:.2f} MB)")
                
                # Validate file size constraints
                if file.size > 100 * 1024 * 1024:  # 100MB limit
                    st.error(f"⚠️ {file.name} is too large ({file_size_mb:.1f}MB). Maximum size is 100MB.")
                elif file.size == 0:
                    st.error(f"⚠️ {file.name} is empty.")
                else:
                    # File passed validation
                    valid_files.append(file)
                    st.success(f"✅ {file.name} ready for analysis")
                    
            except Exception as e:
                # Handle file reading errors gracefully
                st.error(f"❌ Error reading {file.name}: {str(e)}")
        
        # Update uploaded_files to only include validated files
        uploaded_files = valid_files if valid_files else None
    
    # Analysis trigger button with proper validation
    if st.button("🚀 Analisar Dados", type="primary", disabled=not uploaded_files, use_container_width=True):
        if uploaded_files:
            # Analyze the files using our CSVLoaderAgent
            results = analyze_uploaded_files(uploaded_files)
            
            if results:
                # Display success message with file count
                st.success(f"✅ Successfully analyzed {len(results)} file(s)!")
                
                # Display detailed results for each analyzed file
                for filename, result in results.items():
                    with st.expander(f"📊 {filename}", expanded=True):
                        display_file_analysis(filename, result)
            else:
                # Analysis failed - show error message
                st.error("❌ Analysis failed. Please check your files and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN: Question Input and Processing Section
with col2:
    st.markdown('<div class="question-section">', unsafe_allow_html=True)
    st.subheader("❓ Faça sua Pergunta")
    
    # Question input area with helpful placeholder text
    user_question = st.text_area(
        "Digite sua pergunta em linguagem natural sobre os dados",
        placeholder="Ex: Qual é a média de vendas por região? Quais são os produtos mais vendidos?",
        height=100
    )
    
    # Question processing button with comprehensive validation
    if st.button("🔍 Responder Pergunta", type="secondary", use_container_width=True):
        # Validate prerequisites before processing
        if not st.session_state.get('analysis_results'):
            st.error("❌ Primeiro analise os dados usando o botão 'Analisar Dados'.")
        elif not user_question.strip():
            st.error("❌ Digite uma pergunta para iniciar a análise.")
        else:
            # Process the question through the complete pipeline
            with st.spinner("🤖 Processando pergunta..."):
                # Execute the complete question-answering pipeline
                qa_result = answer_question(user_question, st.session_state.analysis_results)
                
                if qa_result['success']:
                    # Display successful result with comprehensive details
                    st.success("✅ Pergunta processada com sucesso!")
                    
                    # Main answer prominently displayed
                    st.markdown("### 💬 Resposta:")
                    st.markdown(qa_result['answer'])
                    
                    # Show additional details in organized expandable sections
                    col1, col2, col3 = st.columns(3)
                    
                    # Question understanding details
                    with col1:
                        if qa_result.get('understanding'):
                            with st.expander("🧠 Entendimento da Pergunta"):
                                understanding = qa_result['understanding']
                                st.write(f"**Confiança:** {understanding.get('confidence', 0):.2f}")
                                st.write(f"**Operações:** {', '.join([op['operation'] for op in understanding.get('operations', [])])}")
                                st.write(f"**Colunas:** {', '.join(understanding.get('target_columns', []))}")
                                # Show generated code for transparency
                                if understanding.get('generated_code'):
                                    st.code(understanding['generated_code'], language='python')
                    
                    # Code execution details
                    with col2:
                        if qa_result.get('execution'):
                            with st.expander("⚙️ Execução"):
                                execution = qa_result['execution']
                                st.write(f"**Sucesso:** {'✅' if execution.get('success') else '❌'}")
                                st.write(f"**Tempo:** {execution.get('execution_time', 0):.3f}s")
                                # Show if fallback strategy was used
                                if execution.get('fallback_executed'):
                                    st.warning(f"Fallback usado: {execution.get('fallback_strategy', 'N/A')}")
                    
                    # Response formatting details
                    with col3:
                        if qa_result.get('formatted_response'):
                            with st.expander("📊 Detalhes da Resposta"):
                                response = qa_result['formatted_response']
                                st.write(f"**Confiança:** {response.get('confidence_score', 0):.2f}")
                                # Display data insights if available
                                if response.get('data_insights'):
                                    st.write("**Insights:**")
                                    for insight in response['data_insights']:
                                        st.write(f"• {insight}")
                                
                                # Render interactive visualizations
                                if response.get('visualizations'):
                                    st.write("**Visualizações:**")
                                    for viz in response['visualizations']:
                                        if viz.get('type') == 'plotly' and viz.get('data'):
                                            st.plotly_chart(viz['data'], use_container_width=True)
                    
                else:
                    # Handle and display processing errors
                    st.error("❌ Erro ao processar a pergunta.")
                    st.write(qa_result['answer'])
                    
                    # Show detailed debugging information if available
                    if qa_result.get('error_details'):
                        with st.expander("🔧 Detalhes do Erro"):
                            st.code(qa_result['error_details'])
                
                # Add interaction to chat history for reference
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': qa_result['answer'],
                    'success': qa_result['success'],
                    'timestamp': time.time(),
                    'details': qa_result
                })
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat History Section - Shows previous questions and answers
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Histórico de Perguntas")
    
    # Display chat history in reverse chronological order (newest first)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        success_icon = "✅" if chat.get('success', False) else "❌"
        
        # Create collapsible expander for each chat entry
        with st.expander(f"{success_icon} Pergunta {len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
            st.markdown(f"**Pergunta:** {chat['question']}")
            st.markdown(f"**Resposta:** {chat['answer']}")

            # Show additional technical details for successful queries
            if chat.get('details') and chat['details'].get('success'):
                col1, col2 = st.columns(2)
                with col1:
                    if chat['details'].get('understanding'):
                        understanding = chat['details']['understanding']
                        st.caption(f"Confiança: {understanding.get('confidence', 0):.2f}")
                with col2:
                    if chat['details'].get('execution'):
                        execution = chat['details']['execution']
                        st.caption(f"Tempo execução: {execution.get('execution_time', 0):.3f}s")

# Footer Section - System Status and Implementation Details
st.markdown("---")
st.markdown("""
### 🚀 Sistema Completo e Funcional

**Agentes implementados e funcionais:**
- ✅ **CSVLoaderAgent**: Carregamento e análise completa de arquivos CSV/ZIP
- ✅ **QuestionUnderstandingAgent**: Interpretação avançada de perguntas em linguagem natural (pt-BR + en-US)
- ✅ **QueryExecutorAgent**: Execução segura de código pandas com fallbacks inteligentes
- ✅ **AnswerFormatterAgent**: Formatação de respostas com visualizações e localização

**Recursos disponíveis:**
- 🔍 Análise automática de dados com insights de IA
- 💬 Perguntas em linguagem natural (português e inglês)
- 📊 Visualizações interativas automáticas
- 🔒 Execução segura de código com validação
- 📈 Análise de qualidade e relacionamentos entre datasets
- 🌐 Suporte multilíngue (pt-BR/en-US)
- 📋 Histórico completo de perguntas e respostas
""")

# Developer Mode - Additional debugging information (typically hidden in production)
if st.checkbox("🔧 Modo Desenvolvedor"):
    # Display current session state for debugging
    st.json({
        "session_state": {
            "files_count": len(st.session_state.uploaded_files_data),
            "chat_history_count": len(st.session_state.chat_history)
        }
    })

# Add temporary debug section for column identification issues
try:
    import debug_actual_csv
    debug_actual_csv.add_debug_section()
except ImportError:
    pass
