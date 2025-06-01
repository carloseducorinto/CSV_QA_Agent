"""
AnswerFormatterAgent - Formata a resposta para o usuário em linguagem natural com base no resultado da análise
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AnswerFormatterAgent:
    """Agent responsible for formatting responses in natural language with visualizations"""
    
    def __init__(self):
        self.formatting_history: List[dict] = []
        self.visualization_templates = self._load_visualization_templates()
    
    def _load_visualization_templates(self) -> Dict[str, dict]:
        """Load templates for different types of visualizations"""
        return {
            'numeric_summary': {
                'chart_type': 'bar',
                'description': 'Resumo estatístico de valores numéricos'
            },
            'categorical_distribution': {
                'chart_type': 'pie',
                'description': 'Distribuição de categorias'
            },
            'time_series': {
                'chart_type': 'line',
                'description': 'Evolução temporal'
            },
            'comparison': {
                'chart_type': 'bar',
                'description': 'Comparação entre grupos'
            },
            'correlation': {
                'chart_type': 'heatmap',
                'description': 'Correlação entre variáveis'
            }
        }
    
    def format_response(self, execution_result: dict, question: str, 
                       understanding_result: dict = None) -> dict:
        """
        Format execution results into a user-friendly response
        
        Args:
            execution_result: Result from QueryExecutorAgent
            question: Original user question
            understanding_result: Result from QuestionUnderstandingAgent
            
        Returns:
            Dictionary containing formatted response
        """
        formatted_response = {
            'original_question': question,
            'success': execution_result.get('success', False),
            'natural_language_answer': '',
            'visualizations': [],
            'data_insights': [],
            'recommendations': [],
            'technical_details': {},
            'confidence_score': 0.0
        }
        
        try:
            if execution_result.get('success'):
                formatted_response = self._format_successful_response(
                    execution_result, question, understanding_result, formatted_response
                )
            else:
                formatted_response = self._format_error_response(
                    execution_result, question, formatted_response
                )
            
            # Store in history
            self.formatting_history.append(formatted_response)
            
            logger.info(f"Response formatted with confidence {formatted_response['confidence_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            formatted_response.update({
                'natural_language_answer': 'Ocorreu um erro ao formatar a resposta.',
                'technical_details': {'error': str(e)}
            })
        
        return formatted_response
    
    def _format_successful_response(self, execution_result: dict, question: str,
                                  understanding_result: dict, response: dict) -> dict:
        """Format a successful execution result"""
        result_data = execution_result.get('result')
        
        if result_data is None:
            response['natural_language_answer'] = "A análise foi executada, mas não produziu resultados visíveis."
            return response
        
        # Determine response type based on result data
        if isinstance(result_data, (int, float)):
            response = self._format_numeric_result(result_data, question, execution_result, response)
        elif isinstance(result_data, pd.Series):
            response = self._format_series_result(result_data, question, execution_result, response)
        elif isinstance(result_data, pd.DataFrame):
            response = self._format_dataframe_result(result_data, question, execution_result, response)
        elif isinstance(result_data, dict):
            response = self._format_dict_result(result_data, question, execution_result, response)
        else:
            response = self._format_generic_result(result_data, question, execution_result, response)
        
        # Add technical details
        response['technical_details'] = {
            'execution_time': execution_result.get('execution_time', 0),
            'code_executed': execution_result.get('code'),
            'fallback_used': execution_result.get('fallback_executed', False)
        }
        
        # Calculate confidence
        response['confidence_score'] = self._calculate_response_confidence(
            execution_result, understanding_result, response
        )
        
        return response
    
    def _format_numeric_result(self, result: Union[int, float], question: str,
                             execution_result: dict, response: dict) -> dict:
        """Format a single numeric result"""
        # Format the number appropriately
        if isinstance(result, float):
            if result.is_integer():
                formatted_value = f"{int(result):,}"
            else:
                formatted_value = f"{result:,.2f}"
        else:
            formatted_value = f"{result:,}"
        
        # Generate natural language response
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['média', 'average', 'mean']):
            response['natural_language_answer'] = f"A média calculada é **{formatted_value}**."
        elif any(word in question_lower for word in ['soma', 'total', 'sum']):
            response['natural_language_answer'] = f"O total calculado é **{formatted_value}**."
        elif any(word in question_lower for word in ['máximo', 'max', 'maior']):
            response['natural_language_answer'] = f"O valor máximo encontrado é **{formatted_value}**."
        elif any(word in question_lower for word in ['mínimo', 'min', 'menor']):
            response['natural_language_answer'] = f"O valor mínimo encontrado é **{formatted_value}**."
        elif any(word in question_lower for word in ['count', 'quantos', 'número']):
            response['natural_language_answer'] = f"O total de registros é **{formatted_value}**."
        else:
            response['natural_language_answer'] = f"O resultado da análise é **{formatted_value}**."
        
        # Create a simple visualization
        viz = self._create_single_value_chart(result, question)
        if viz:
            response['visualizations'].append(viz)
        
        # Add insights
        response['data_insights'] = self._generate_numeric_insights(result, question)
        
        return response
    
    def _format_series_result(self, result: pd.Series, question: str,
                            execution_result: dict, response: dict) -> dict:
        """Format a pandas Series result"""
        series_length = len(result)
        
        if series_length == 0:
            response['natural_language_answer'] = "A análise não retornou dados."
            return response
        
        # Create natural language description
        if series_length == 1:
            value = result.iloc[0]
            response['natural_language_answer'] = f"O resultado da análise é **{value}** para {result.index[0]}."
        else:
            response['natural_language_answer'] = f"A análise retornou **{series_length}** resultados."
            
            # Add top values description
            if result.dtype in ['int64', 'float64']:
                top_value = result.max()
                top_index = result.idxmax()
                response['natural_language_answer'] += f" O maior valor é **{top_value:,.2f}** para {top_index}."
        
        # Create visualization
        viz = self._create_series_chart(result, question)
        if viz:
            response['visualizations'].append(viz)
        
        # Add insights
        response['data_insights'] = self._generate_series_insights(result, question)
        
        return response
    
    def _format_dataframe_result(self, result: pd.DataFrame, question: str,
                               execution_result: dict, response: dict) -> dict:
        """Format a pandas DataFrame result"""
        rows, cols = result.shape
        
        if rows == 0:
            response['natural_language_answer'] = "A análise não retornou dados."
            return response
        
        # Create natural language description
        response['natural_language_answer'] = f"A análise retornou uma tabela com **{rows}** registro(s) e **{cols}** coluna(s)."
        
        # Add summary information
        if rows <= 10:
            response['natural_language_answer'] += " Os resultados são mostrados na tabela abaixo."
        else:
            response['natural_language_answer'] += f" Mostrando os primeiros 10 registros de {rows} total."
        
        # Create visualization
        viz = self._create_dataframe_chart(result, question)
        if viz:
            response['visualizations'].append(viz)
        
        # Add insights
        response['data_insights'] = self._generate_dataframe_insights(result, question)
        
        return response
    
    def _format_dict_result(self, result: dict, question: str,
                          execution_result: dict, response: dict) -> dict:
        """Format a dictionary result (often from fallback operations)"""
        if 'dataframe' in result:
            # This is likely a basic exploration result
            response['natural_language_answer'] = f"Análise do arquivo **{result['dataframe']}**:"
            response['natural_language_answer'] += f"\n- Dimensões: {result['shape'][0]} linhas × {result['shape'][1]} colunas"
            
            if 'columns' in result:
                response['natural_language_answer'] += f"\n- Colunas: {', '.join(result['columns'][:5])}"
                if len(result['columns']) > 5:
                    response['natural_language_answer'] += f" (e mais {len(result['columns']) - 5})"
        else:
            response['natural_language_answer'] = "Resultado da análise obtido com sucesso."
        
        return response
    
    def _format_generic_result(self, result: Any, question: str,
                             execution_result: dict, response: dict) -> dict:
        """Format any other type of result"""
        response['natural_language_answer'] = f"A análise foi concluída. Resultado: {str(result)}"
        return response
    
    def _format_error_response(self, execution_result: dict, question: str, response: dict) -> dict:
        """Format an error response"""
        error_info = execution_result.get('error', {})
        
        if execution_result.get('fallback_executed'):
            response['natural_language_answer'] = "A pergunta não pôde ser processada completamente, mas aqui está uma análise básica dos dados."
            response['confidence_score'] = 0.3
        else:
            response['natural_language_answer'] = "Não foi possível processar a pergunta. "
            
            # Provide helpful suggestions based on error type
            if 'KeyError' in str(error_info.get('type', '')):
                response['natural_language_answer'] += "Verifique se os nomes das colunas estão corretos."
            elif 'TypeError' in str(error_info.get('type', '')):
                response['natural_language_answer'] += "Pode haver incompatibilidade entre tipos de dados."
            else:
                response['natural_language_answer'] += "Tente reformular a pergunta de forma mais específica."
        
        response['technical_details'] = {
            'error_type': error_info.get('type'),
            'error_message': error_info.get('message')
        }
        
        return response
    
    def _create_single_value_chart(self, value: Union[int, float], question: str) -> Optional[dict]:
        """Create a simple chart for a single value"""
        try:
            # Create a simple gauge or bar chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = value,
                title = {'text': "Resultado"},
                gauge = {
                    'axis': {'range': [None, value * 1.2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, value * 0.5], 'color': "lightgray"},
                        {'range': [value * 0.5, value], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            ))
            
            return {
                'type': 'plotly',
                'data': fig.to_dict(),
                'title': 'Resultado da Análise'
            }
        except:
            return None
    
    def _create_series_chart(self, series: pd.Series, question: str) -> Optional[dict]:
        """Create a chart for Series data"""
        try:
            # Determine chart type based on data
            if len(series) <= 20:
                # Bar chart for small series
                fig = px.bar(
                    x=series.index.astype(str),
                    y=series.values,
                    title="Resultados da Análise"
                )
            else:
                # Line chart for larger series
                fig = px.line(
                    x=series.index,
                    y=series.values,
                    title="Resultados da Análise"
                )
            
            fig.update_layout(
                xaxis_title="Categorias",
                yaxis_title="Valores",
                showlegend=False
            )
            
            return {
                'type': 'plotly',
                'data': fig.to_dict(),
                'title': 'Visualização dos Resultados'
            }
        except:
            return None
    
    def _create_dataframe_chart(self, df: pd.DataFrame, question: str) -> Optional[dict]:
        """Create a chart for DataFrame data"""
        try:
            # Simple heuristic for chart type
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Scatter plot for numeric columns
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title="Visualização dos Dados"
                )
            elif len(numeric_cols) == 1:
                # Histogram for single numeric column
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title="Distribuição dos Dados"
                )
            else:
                # Count plot for categorical data
                first_col = df.columns[0]
                value_counts = df[first_col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    title="Distribuição dos Dados"
                )
            
            return {
                'type': 'plotly',
                'data': fig.to_dict(),
                'title': 'Visualização dos Dados'
            }
        except:
            return None
    
    def _generate_numeric_insights(self, value: Union[int, float], question: str) -> List[str]:
        """Generate insights for numeric results"""
        insights = []
        
        if isinstance(value, float) and value != int(value):
            insights.append("O resultado contém valores decimais.")
        
        if value == 0:
            insights.append("O resultado é zero, o que pode indicar ausência de dados ou condição não atendida.")
        elif value < 0:
            insights.append("O resultado é negativo.")
        
        return insights
    
    def _generate_series_insights(self, series: pd.Series, question: str) -> List[str]:
        """Generate insights for Series results"""
        insights = []
        
        if series.dtype in ['int64', 'float64']:
            std_dev = series.std()
            mean_val = series.mean()
            
            if std_dev > mean_val:
                insights.append("Os dados apresentam alta variabilidade.")
            else:
                insights.append("Os dados são relativamente homogêneos.")
        
        if len(series) > 1:
            insights.append(f"A análise abrange {len(series)} categorias/períodos.")
        
        return insights
    
    def _generate_dataframe_insights(self, df: pd.DataFrame, question: str) -> List[str]:
        """Generate insights for DataFrame results"""
        insights = []
        
        rows, cols = df.shape
        insights.append(f"Dataset com {rows} registros e {cols} variáveis.")
        
        # Check for missing data
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            insights.append(f"Há {missing_data} valores ausentes nos dados.")
        
        return insights
    
    def _calculate_response_confidence(self, execution_result: dict,
                                     understanding_result: dict, response: dict) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.0
        
        # Base confidence from execution success
        if execution_result.get('success'):
            confidence += 0.5
        
        # Reduce confidence if fallback was used
        if execution_result.get('fallback_executed'):
            confidence *= 0.7
        
        # Add confidence from understanding if available
        if understanding_result and understanding_result.get('confidence'):
            confidence += understanding_result['confidence'] * 0.3
        else:
            confidence += 0.2  # Default understanding confidence
        
        # Boost confidence if we have visualizations
        if response.get('visualizations'):
            confidence += 0.1
        
        # Boost confidence if we have insights
        if response.get('data_insights'):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def get_formatting_history(self) -> List[dict]:
        """Get history of formatted responses"""
        return self.formatting_history.copy()
    
    def clear_history(self):
        """Clear formatting history"""
        self.formatting_history.clear() 