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

# Configure the page
st.set_page_config(
    page_title="Agente CSV Q&A Inteligente", 
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
st.markdown('<h1 class="main-header">📊 Agente CSV Q&A Inteligente</h1>', unsafe_allow_html=True)
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
                # TODO: Implement QuestionUnderstandingAgent pipeline
                st.info("🚧 Funcionalidade de Q&A em desenvolvimento. Por enquanto, use a análise de dados acima.")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': "Funcionalidade em desenvolvimento - use a análise de dados completa disponível acima.",
                    'timestamp': time.time()
                })
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat history section
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Histórico de Perguntas")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Pergunta {len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
            st.markdown(f"**Pergunta:** {chat['question']}")
            st.markdown(f"**Resposta:** {chat['answer']}")

# Footer with next steps
st.markdown("---")
st.markdown("""
### 🚀 Próximos Passos para Implementação Completa

**Agentes a serem implementados:**
- **CSVLoaderAgent**: Carregamento e parsing de arquivos CSV/ZIP
- **SchemaAnalyzerAgent**: Análise de schema e sugestão de relações
- **QuestionUnderstandingAgent**: Interpretação de perguntas em linguagem natural
- **QueryExecutorAgent**: Execução de código pandas gerado
- **AnswerFormatterAgent**: Formatação de respostas

**Recursos adicionais:**
- Visualizações interativas com plotly
- Export de resultados
- Histórico persistente
- Suporte a múltiplos idiomas
""")

# Development info (remove in production)
if st.checkbox("🔧 Modo Desenvolvedor"):
    st.json({
        "session_state": {
            "files_count": len(st.session_state.uploaded_files_data),
            "chat_history_count": len(st.session_state.chat_history)
        }
    })
