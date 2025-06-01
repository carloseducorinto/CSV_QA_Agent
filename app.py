import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add the project root to the Python path to import our agents
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our agents
from agents.csv_loader import CSVLoaderAgent
from agents.question_understanding import QuestionUnderstandingAgent
from agents.query_executor import QueryExecutorAgent
from agents.answer_formatter import AnswerFormatterAgent

def analyze_uploaded_files(uploaded_files):
    """Analyze uploaded files using CSVLoaderAgent"""
    if not uploaded_files:
        return None
    
    try:
        # Initialize the CSV loader agent
        csv_loader = CSVLoaderAgent()
        
        # Check LLM status
        llm_status = csv_loader.get_llm_status()
        
        with st.spinner("🔍 Analyzing files with AI-powered insights..."):
            start_time = time.time()
            
            # Load and analyze files
            results = csv_loader.load_files(uploaded_files)
            
            processing_time = time.time() - start_time
        
        # Display LLM status
        col1, col2 = st.columns([1, 1])
        with col1:
            if llm_status['available']:
                st.success("🤖 AI Analysis: Enabled")
            else:
                st.warning("🤖 AI Analysis: Limited (API key needed)")
        
        with col2:
            st.info(f"⏱️ Processing time: {processing_time:.1f}s")
        
        # Store results in session state
        st.session_state.analysis_results = results
        st.session_state.processing_time = processing_time
        
        return results
        
    except ImportError as e:
        st.error(f"❌ Import error: {str(e)}")
        st.error("Please check that all required dependencies are installed.")
        return None
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.error("Please check the console for detailed error information.")
        
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        with st.expander("🔧 Debug Information"):
            st.code(error_details)
        
        return None

def display_file_analysis(filename, result):
    """Display comprehensive analysis for a single file"""
    
    if not result.success:
        st.error(f"❌ Failed to process {filename}")
        if result.errors:
            for error in result.errors:
                st.error(f"• {error}")
        return
    
    df = result.dataframe
    
    # File overview
    st.subheader(f"📊 Analysis: {filename}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        if result.quality_assessment:
            quality_score = result.quality_assessment.get('overall_score', 0)
            st.metric("Quality Score", f"{quality_score}/100")
        else:
            st.metric("Quality Score", "N/A")
    
    with col4:
        completeness = result.quality_assessment.get('completeness', 100) if result.quality_assessment else 100
        st.metric("Completeness", f"{completeness:.1f}%")
    
    # Tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data", "🔍 Schema", "📈 Quality", "🤖 AI Insights", "🔗 Relationships"])
    
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head(100))  # Show first 100 rows
        
        if len(df) > 100:
            st.info(f"Showing first 100 rows of {len(df):,} total rows")
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download processed data as CSV",
            data=csv,
            file_name=f"processed_{filename}",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Schema Analysis")
        
        if result.schema_analysis:
            schema = result.schema_analysis
            
            # Column summary
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
            
            # Data type distribution
            dtype_counts = pd.Series(schema['dtypes']).value_counts()
            fig_dtype = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title="Data Type Distribution"
            )
            st.plotly_chart(fig_dtype)
    
    with tab3:
        st.subheader("Data Quality Assessment")
        
        if result.quality_assessment:
            quality = result.quality_assessment
            
            # Quality metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Overall Score", f"{quality['overall_score']}/100")
                st.metric("Completeness", f"{quality['completeness']:.1f}%")
                st.metric("Duplicate Rows", quality['duplicate_rows'])
            
            with col2:
                if quality.get('empty_columns'):
                    st.warning(f"Empty columns: {', '.join(quality['empty_columns'])}")
                
                if quality.get('constant_columns'):
                    st.warning(f"Constant columns: {', '.join(quality['constant_columns'])}")
                
                if quality.get('high_cardinality_columns'):
                    st.info(f"High cardinality: {', '.join(quality['high_cardinality_columns'])}")
            
            # Quality issues and recommendations
            if quality.get('quality_issues'):
                st.subheader("🚨 Quality Issues Detected:")
                for issue in quality['quality_issues']:
                    st.warning(f"• {issue}")
            
            if quality.get('recommendations'):
                st.subheader("💡 Recommendations:")
                for rec in quality['recommendations']:
                    st.info(f"• {rec}")
            
            # Null value heatmap
            if df.isnull().sum().sum() > 0:
                st.subheader("Missing Values Heatmap")
                null_df = df.isnull().sum().to_frame('Null Count')
                null_df = null_df[null_df['Null Count'] > 0]
                
                if len(null_df) > 0:
                    fig_null = px.bar(
                        x=null_df.index,
                        y=null_df['Null Count'],
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig_null)
    
    with tab4:
        st.subheader("🤖 AI-Powered Insights")
        
        if result.llm_insights:
            insights = result.llm_insights
            
            # Summary
            st.write("**Summary:**")
            st.info(insights.get('summary', 'No summary available'))
            
            # Key observations
            if insights.get('key_observations'):
                st.write("**Key Observations:**")
                for obs in insights['key_observations']:
                    st.write(f"• {obs}")
            
            # Recommended next steps
            if insights.get('recommended_next_steps'):
                st.write("**Recommended Next Steps:**")
                for step in insights['recommended_next_steps']:
                    st.write(f"• {step}")
            
            # Potential use cases
            if insights.get('potential_use_cases'):
                st.write("**Potential Use Cases:**")
                for use_case in insights['potential_use_cases']:
                    st.write(f"• {use_case}")
        else:
            st.info("AI insights require a valid OpenAI API key. The analysis above provides comprehensive data insights.")
    
    with tab5:
        st.subheader("🔗 Data Relationships")
        
        if result.relationships:
            st.write("**Detected Relationships with Other Datasets:**")
            
            for i, rel in enumerate(result.relationships, 1):
                # Use container instead of expander to avoid nesting
                st.markdown(f"**#{i} - {rel['table1']}.{rel['column1']} ↔ {rel['table2']}.{rel['column2']}**")
                
                # Create a bordered container using markdown
                with st.container():
                    # Add some styling
                    st.markdown("""
                    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
                    """, unsafe_allow_html=True)
                    
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
                    
                    if rel.get('explanation'):
                        st.write(f"**Explanation:** {rel['explanation']}")
                    
                    if rel.get('recommendations'):
                        st.write("**Recommendations:**")
                        for rec in rel['recommendations']:
                            st.write(f"• {rec}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                # Add separator between relationships
                if i < len(result.relationships):
                    st.markdown("---")
        else:
            if len(st.session_state.get('analysis_results', {})) > 1:
                st.info("No relationships detected between uploaded datasets.")
            else:
                st.info("Upload multiple files to detect relationships between datasets.")

def answer_question(question: str, analysis_results: dict) -> dict:
    """Complete question-answering pipeline using all enhanced agents"""
    try:
        # Initialize all agents
        question_agent = QuestionUnderstandingAgent()
        executor_agent = QueryExecutorAgent()
        formatter_agent = AnswerFormatterAgent()
        
        # Extract DataFrames from analysis results
        dataframes = {}
        for filename, result in analysis_results.items():
            if result.success and result.dataframe is not None:
                dataframes[filename] = result.dataframe
        
        if not dataframes:
            return {
                'success': False,
                'error': 'No valid DataFrames available for analysis.',
                'answer': 'Nenhum dado válido disponível para análise.'
            }
        
        # Step 1: Understand the question
        understanding_result = question_agent.understand_question(question, dataframes)
        
        if understanding_result.get('error'):
            return {
                'success': False,
                'error': understanding_result['error'],
                'answer': f"Erro ao entender a pergunta: {understanding_result['explanation']}"
            }
        
        # Step 2: Execute the generated code
        if understanding_result.get('generated_code'):
            execution_result = executor_agent.execute_code(
                understanding_result['generated_code'], 
                dataframes
            )
        else:
            return {
                'success': False,
                'error': 'No code generated',
                'answer': 'Não foi possível gerar código para esta pergunta. Tente reformular de forma mais específica.'
            }
        
        # Step 3: Format the response
        formatted_response = formatter_agent.format_response(
            execution_result, 
            question, 
            understanding_result
        )
        
        return {
            'success': True,
            'understanding': understanding_result,
            'execution': execution_result,
            'formatted_response': formatted_response,
            'answer': formatted_response.get('natural_language_answer', 'Resposta não disponível.')
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            'success': False,
            'error': str(e),
            'error_details': error_details,
            'answer': f'Erro interno: {str(e)}'
        }

# Configure the page
st.set_page_config(
    page_title="Agente Aprende -  CSV Q&A Inteligente", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Initialize session state
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<h1 class="main-header">📊 Agente Aprende - CSV Q&A Inteligente</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for file management and settings
with st.sidebar:
    st.header("🗂️ Gerenciamento de Arquivos")
    
    # Clear session button
    if st.button("🗑️ Limpar Sessão", type="secondary"):
        st.session_state.uploaded_files_data = {}
        st.session_state.analysis_results = {}
        st.session_state.chat_history = []
        st.success("Sessão limpa!")
        st.rerun()
    
    # Show loaded files
    if st.session_state.uploaded_files_data:
        st.subheader("📋 Arquivos Carregados")
        for filename in st.session_state.uploaded_files_data.keys():
            st.markdown(f"✅ {filename}")
    
    st.header("⚙️ Configurações")
    st.slider("Nível de Detalhamento", 1, 5, 3, help="Controla o nível de detalhamento das respostas")
    st.checkbox("Mostrar código gerado", help="Exibe o código pandas gerado internamente")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload de Arquivos")
    
    uploaded_files = st.file_uploader(
        "Envie aqui seus arquivos CSV ou ZIP",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="Suporte para múltiplos arquivos CSV e arquivos ZIP contendo CSVs"
    )
    
    # Show uploaded files with better error handling
    if uploaded_files:
        st.markdown("### 📊 Arquivos Carregados")
        
        valid_files = []
        for file in uploaded_files:
            try:
                # Display file info
                file_size_mb = file.size / (1024 * 1024)
                st.write(f"📄 **{file.name}** ({file.size:,} bytes / {file_size_mb:.2f} MB)")
                
                # Check file size
                if file.size > 100 * 1024 * 1024:  # 100MB limit
                    st.error(f"⚠️ {file.name} is too large ({file_size_mb:.1f}MB). Maximum size is 100MB.")
                elif file.size == 0:
                    st.error(f"⚠️ {file.name} is empty.")
                else:
                    valid_files.append(file)
                    st.success(f"✅ {file.name} ready for analysis")
                    
            except Exception as e:
                st.error(f"❌ Error reading {file.name}: {str(e)}")
        
        # Update uploaded_files to only include valid files
        uploaded_files = valid_files if valid_files else None
    
    # Analysis button
    if st.button("🚀 Analisar Dados", type="primary", disabled=not uploaded_files, use_container_width=True):
        if uploaded_files:
            # Analyze the files using our CSVLoaderAgent
            results = analyze_uploaded_files(uploaded_files)
            
            if results:
                st.success(f"✅ Successfully analyzed {len(results)} file(s)!")
                
                # Display results for each file
                for filename, result in results.items():
                    with st.expander(f"📊 {filename}", expanded=True):
                        display_file_analysis(filename, result)
            else:
                st.error("❌ Analysis failed. Please check your files and try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="question-section">', unsafe_allow_html=True)
    st.subheader("❓ Faça sua Pergunta")
    
    # Question input with better UX
    user_question = st.text_area(
        "Digite sua pergunta em linguagem natural sobre os dados",
        placeholder="Ex: Qual é a média de vendas por região? Quais são os produtos mais vendidos?",
        height=100
    )
    
    # Analysis button with proper validation
    if st.button("🔍 Responder Pergunta", type="secondary", use_container_width=True):
        if not st.session_state.get('analysis_results'):
            st.error("❌ Primeiro analise os dados usando o botão 'Analisar Dados'.")
        elif not user_question.strip():
            st.error("❌ Digite uma pergunta para iniciar a análise.")
        else:
            with st.spinner("🤖 Processando pergunta..."):
                # Complete question-answering pipeline
                qa_result = answer_question(user_question, st.session_state.analysis_results)
                
                if qa_result['success']:
                    # Display the answer
                    st.success("✅ Pergunta processada com sucesso!")
                    
                    # Main answer
                    st.markdown("### 💬 Resposta:")
                    st.markdown(qa_result['answer'])
                    
                    # Show additional details in expanders
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if qa_result.get('understanding'):
                            with st.expander("🧠 Entendimento da Pergunta"):
                                understanding = qa_result['understanding']
                                st.write(f"**Confiança:** {understanding.get('confidence', 0):.2f}")
                                st.write(f"**Operações:** {', '.join([op['operation'] for op in understanding.get('operations', [])])}")
                                st.write(f"**Colunas:** {', '.join(understanding.get('target_columns', []))}")
                                if understanding.get('generated_code'):
                                    st.code(understanding['generated_code'], language='python')
                    
                    with col2:
                        if qa_result.get('execution'):
                            with st.expander("⚙️ Execução"):
                                execution = qa_result['execution']
                                st.write(f"**Sucesso:** {'✅' if execution.get('success') else '❌'}")
                                st.write(f"**Tempo:** {execution.get('execution_time', 0):.3f}s")
                                if execution.get('fallback_executed'):
                                    st.warning(f"Fallback usado: {execution.get('fallback_strategy', 'N/A')}")
                    
                    with col3:
                        if qa_result.get('formatted_response'):
                            with st.expander("📊 Detalhes da Resposta"):
                                response = qa_result['formatted_response']
                                st.write(f"**Confiança:** {response.get('confidence_score', 0):.2f}")
                                if response.get('data_insights'):
                                    st.write("**Insights:**")
                                    for insight in response['data_insights']:
                                        st.write(f"• {insight}")
                                
                                # Show visualizations if available
                                if response.get('visualizations'):
                                    st.write("**Visualizações:**")
                                    for viz in response['visualizations']:
                                        if viz.get('type') == 'plotly' and viz.get('data'):
                                            st.plotly_chart(viz['data'], use_container_width=True)
                    
                else:
                    # Show error
                    st.error("❌ Erro ao processar a pergunta.")
                    st.write(qa_result['answer'])
                    
                    # Show debug info if available
                    if qa_result.get('error_details'):
                        with st.expander("🔧 Detalhes do Erro"):
                            st.code(qa_result['error_details'])
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': qa_result['answer'],
                    'success': qa_result['success'],
                    'timestamp': time.time(),
                    'details': qa_result
                })
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat history section
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Histórico de Perguntas")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        success_icon = "✅" if chat.get('success', False) else "❌"
        with st.expander(f"{success_icon} Pergunta {len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
            st.markdown(f"**Pergunta:** {chat['question']}")
            st.markdown(f"**Resposta:** {chat['answer']}")
            
            # Show additional details if available
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

# Footer with implementation status
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

# Development info (remove in production)
if st.checkbox("🔧 Modo Desenvolvedor"):
    st.json({
        "session_state": {
            "files_count": len(st.session_state.uploaded_files_data),
            "chat_history_count": len(st.session_state.chat_history)
        }
    })
