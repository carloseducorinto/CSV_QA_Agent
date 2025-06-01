# 📋 Especificação Funcional e Técnica
## Sistema CSV Q&A Agent com Integração LLM

**Versão:** 2.0  
**Data:** Maio 2025  
**Autor:** Sistema de Análise Inteligente de Dados CSV  

---

## 🎯 1. ESPECIFICAÇÃO FUNCIONAL

### 1.1 Visão Geral do Sistema

O **CSV Q&A Agent** é uma aplicação web inteligente que permite aos usuários fazer perguntas em linguagem natural sobre dados contidos em arquivos CSV, obtendo respostas automatizadas através de análise de dados e visualizações interativas.

**Objetivo Principal:** Democratizar o acesso à análise de dados, permitindo que usuários não técnicos obtenham insights de dados CSV através de perguntas em linguagem natural.

### 1.2 Requisitos Funcionais

#### RF001 - Upload e Processamento de Arquivos
- **Descrição:** Sistema deve aceitar upload de arquivos CSV e ZIP
- **Critérios de Aceitação:**
  - Suporte a múltiplos arquivos simultaneamente
  - Detecção automática de encoding (UTF-8, ISO-8859-1, etc.)
  - Validação de integridade dos arquivos
  - Limite máximo de 100MB por arquivo
  - Suporte a arquivos ZIP contendo CSVs

#### RF002 - Análise Automática de Schema
- **Descrição:** Análise automática da estrutura dos dados carregados
- **Critérios de Aceitação:**
  - Identificação de tipos de dados (numérico, texto, data)
  - Detecção de valores nulos e duplicados
  - Cálculo de métricas de qualidade dos dados
  - Identificação de relacionamentos entre datasets
  - Score de qualidade geral (0-100)

#### RF003 - Interpretação de Perguntas em Linguagem Natural
- **Descrição:** Sistema deve interpretar perguntas em português e inglês
- **Critérios de Aceitação:**
  - Suporte bilíngue (pt-BR e en-US)
  - Identificação automática de operações (soma, média, máximo, etc.)
  - Mapeamento de colunas mencionadas nas perguntas
  - Detecção de arquivo de destino quando especificado
  - Sistema híbrido LLM + Regex para máxima cobertura

#### RF004 - Geração Automática de Código de Análise
- **Descrição:** Geração de código pandas executável baseado nas perguntas
- **Critérios de Aceitação:**
  - Código pandas válido e executável
  - Validação de segurança (bloqueio de operações perigosas)
  - Fallback automático entre LLM e sistema baseado em padrões
  - Transparência sobre o método utilizado (LLM vs Regex)

#### RF005 - Execução Segura de Código
- **Descrição:** Execução controlada do código gerado com tratamento de erros
- **Critérios de Aceitação:**
  - Ambiente de execução isolado
  - Validação de código antes da execução
  - Tratamento de exceções com fallbacks inteligentes
  - Timeout para operações longas (30 segundos)
  - Logging completo de execução

#### RF006 - Formatação de Respostas
- **Descrição:** Apresentação de resultados em linguagem natural com visualizações
- **Critérios de Aceitação:**
  - Respostas em linguagem natural
  - Geração automática de gráficos quando apropriado
  - Insights sobre os dados analisados
  - Localização em português e inglês
  - Indicadores de confiança das respostas

#### RF007 - Interface Web Interativa
- **Descrição:** Interface amigável para upload de arquivos e interação
- **Critérios de Aceitação:**
  - Upload drag-and-drop de arquivos
  - Visualização prévia dos dados carregados
  - Campo de pergunta com sugestões
  - Histórico de perguntas e respostas
  - Exportação de resultados
  - Design responsivo para diferentes dispositivos

#### RF008 - Análise com IA Avançada (Opcional)
- **Descrição:** Insights automáticos usando LLM quando API key disponível
- **Critérios de Aceitação:**
  - Sumário automático dos dados
  - Identificação de padrões e anomalias
  - Sugestões de próximos passos de análise
  - Casos de uso potenciais identificados
  - Degradação graceful quando LLM indisponível

### 1.3 Casos de Uso Principais

#### CU001 - Análise Financeira Simples
**Ator:** Analista Financeiro  
**Fluxo:**
1. Upload de arquivo "vendas_2024.csv"
2. Pergunta: "Qual é o total de vendas no primeiro semestre?"
3. Sistema gera código: `df[df['data'].str.contains('2024-0[1-6]')]['vendas'].sum()`
4. Exibe resultado com gráfico de evolução mensal

#### CU002 - Comparação Entre Datasets
**Ator:** Gerente de Vendas  
**Fluxo:**
1. Upload de múltiplos arquivos CSV (vendas por região)
2. Pergunta: "Compare as vendas entre São Paulo e Rio de Janeiro"
3. Sistema identifica relacionamentos entre arquivos
4. Gera análise comparativa com visualização

#### CU003 - Análise Exploratória Automática
**Ator:** Cientista de Dados  
**Fluxo:**
1. Upload de dataset complexo
2. Sistema gera insights automáticos via LLM
3. Sugestões de análises adicionais
4. Identificação de padrões interessantes

### 1.4 Regras de Negócio

#### RN001 - Processamento de Arquivos
- Arquivos com mais de 100MB são rejeitados
- Encoding é detectado automaticamente com fallback para UTF-8
- Arquivos corrompidos geram mensagem de erro específica

#### RN002 - Segurança de Execução
- Código gerado é validado antes da execução
- Operações de sistema são bloqueadas
- Timeout de 30 segundos para operações longas

#### RN003 - Uso de LLM
- LLM é utilizado apenas quando API key está disponível
- Sistema funciona completamente sem LLM (modo fallback)
- Custo de API é considerado (modelo gpt-3.5-turbo padrão)

#### RN004 - Qualidade dos Dados
- Score de qualidade é calculado automaticamente
- Alertas são gerados para problemas de qualidade críticos
- Sugestões de limpeza são fornecidas quando apropriado

---

## ⚙️ 2. ESPECIFICAÇÃO TÉCNICA

### 2.1 Arquitetura do Sistema

#### 2.1.1 Visão Geral da Arquitetura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (Streamlit)   │◄──►│   (Python)      │◄──►│   (OpenAI API)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Storage  │    │   Data Processing│    │   LLM Service   │
│   (Local/Cloud) │    │   (Pandas)      │    │   (ChatOpenAI)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### 2.1.2 Componentes Principais

1. **Presentation Layer**
   - `app.py` - Interface Streamlit principal
   - Gerenciamento de estado da sessão
   - Upload e visualização de arquivos

2. **Business Logic Layer**
   - `CSVLoaderAgent` - Carregamento e validação de arquivos
   - `SchemaAnalyzerAgent` - Análise de estrutura de dados
   - `QuestionUnderstandingAgent` - Interpretação de perguntas
   - `QueryExecutorAgent` - Execução segura de código
   - `AnswerFormatterAgent` - Formatação de respostas

3. **Data Layer**
   - DataFrames em memória (pandas)
   - Cache de resultados de análise
   - Logs de execução

4. **External Services**
   - OpenAI API (ChatOpenAI)
   - LangChain framework

### 2.2 Especificação dos Agentes

#### 2.2.1 CSVLoaderAgent

**Responsabilidades:**
- Carregamento de arquivos CSV/ZIP
- Detecção de encoding
- Validação de integridade
- Análise inicial de qualidade

**Interface Pública:**
```python
class CSVLoaderAgent:
    def load_files(self, uploaded_files: List[UploadedFile]) -> Dict[str, LoadResult]
    def get_llm_status(self) -> Dict[str, Any]
```

**Estrutura LoadResult:**
```python
@dataclass
class LoadResult:
    success: bool
    dataframe: Optional[pd.DataFrame]
    filename: str
    schema_analysis: Optional[Dict]
    quality_assessment: Optional[Dict]
    llm_insights: Optional[Dict]
    relationships: Optional[List[Dict]]
    errors: List[str]
```

#### 2.2.2 QuestionUnderstandingAgent

**Responsabilidades:**
- Interpretação de perguntas em linguagem natural
- Geração de código pandas via LLM ou regex
- Validação de código gerado
- Sistema híbrido com fallback

**Interface Pública:**
```python
class QuestionUnderstandingAgent:
    def understand_question(self, question: str, dataframes: Dict[str, pd.DataFrame]) -> dict
    def get_question_history(self) -> List[dict]
    def clear_history(self) -> None
```

**Métodos Privados Principais:**
```python
def _generate_code_with_llm(self, question: str, df_name: str, df: pd.DataFrame) -> Optional[str]
def _validate_llm_code(self, code: str, df_name: str) -> bool
def _identify_target_dataframe(self, question: str, dataframes: Dict) -> Optional[str]
def _identify_columns(self, question: str, df: pd.DataFrame) -> List[str]
def _identify_operations(self, question: str) -> List[dict]
```

#### 2.2.3 QueryExecutorAgent

**Responsabilidades:**
- Execução segura de código pandas
- Validação de segurança pré-execução
- Tratamento de erros com fallbacks
- Logging detalhado de execução

**Interface Pública:**
```python
class QueryExecutorAgent:
    def execute_code(self, code: str, dataframes: Dict[str, pd.DataFrame]) -> dict
    def validate_code_safety(self, code: str) -> bool
    def get_execution_history(self) -> List[dict]
```

### 2.3 Estruturas de Dados

#### 2.3.1 Estrutura de Resposta Unificada

```python
{
    'original_question': str,           # Pergunta original do usuário
    'target_dataframe': Optional[str],  # Nome do arquivo identificado
    'target_columns': List[str],        # Colunas identificadas
    'operations': List[dict],           # Operações detectadas
    'generated_code': Optional[str],    # Código pandas gerado
    'confidence': float,                # Score de confiança (0.0-1.0)
    'explanation': str,                 # Explicação em linguagem natural
    'code_source': str,                 # 'llm', 'regex', ou 'error'
    'understood_intent': Optional[str], # Intenção interpretada
    'fallback_suggestions': List[str],  # Sugestões em caso de falha
    'error': Optional[str]              # Mensagem de erro se houver
}
```

#### 2.3.2 Estrutura de Execução

```python
{
    'success': bool,                    # Status de sucesso
    'result': Any,                      # Resultado da execução
    'execution_time': float,            # Tempo de execução em segundos
    'code': str,                        # Código executado
    'output': str,                      # Output capturado
    'error': Optional[str],             # Erro se houver
    'fallback_executed': bool,          # Se fallback foi usado
    'fallback_strategy': Optional[str], # Estratégia de fallback utilizada
    'variables_available': List[str]    # Variáveis disponíveis na execução
}
```

### 2.4 Algoritmos Principais

#### 2.4.1 Algoritmo de Interpretação Híbrida

```python
def understand_question(question, dataframes):
    # 1. Pré-processamento
    clean_question = normalize_and_clean(question)
    target_df = identify_target_dataframe(clean_question, dataframes)
    
    # 2. Tentativa LLM (se disponível)
    if llm_available:
        llm_code = generate_code_with_llm(question, target_df, dataframes[target_df])
        if validate_llm_code(llm_code):
            return create_response(llm_code, 'llm', confidence=0.95)
    
    # 3. Fallback Regex
    columns = identify_columns(clean_question, dataframes[target_df])
    operations = identify_operations(clean_question)
    regex_code = generate_code_regex(target_df, columns, operations)
    
    if regex_code:
        return create_response(regex_code, 'regex', confidence=calculate_confidence())
    else:
        return create_error_response("Não foi possível gerar código")
```

#### 2.4.2 Algoritmo de Validação de Segurança

```python
def validate_code_safety(code):
    # 1. Verificações estruturais
    required_elements = ["dataframes[", "result ="]
    if not all(element in code for element in required_elements):
        return False
    
    # 2. Verificações de segurança
    dangerous_patterns = [
        'import os', 'import sys', 'exec(', 'eval(',
        'open(', '__import__', 'subprocess', 'os.system'
    ]
    if any(pattern in code.lower() for pattern in dangerous_patterns):
        return False
    
    # 3. Verificação de sintaxe
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
```

### 2.5 Performance e Escalabilidade

#### 2.5.1 Métricas de Performance

| Operação | Tempo Esperado | Limite Máximo |
|----------|----------------|---------------|
| Upload de arquivo (10MB) | < 2s | 5s |
| Análise de schema | < 1s | 3s |
| Interpretação LLM | < 3s | 10s |
| Interpretação Regex | < 0.1s | 0.5s |
| Execução de código | < 1s | 30s |
| Formatação de resposta | < 0.5s | 2s |

#### 2.5.2 Limitações de Recursos

- **Memória:** Máximo 2GB por sessão de usuário
- **Armazenamento:** Arquivos removidos após 24h de inatividade
- **CPU:** Timeout de 30s para operações de análise
- **API Calls:** Rate limiting para OpenAI API (50 requests/min)

### 2.6 Segurança

#### 2.6.1 Validação de Entrada

```python
SECURITY_VALIDATIONS = {
    'file_upload': {
        'max_size': 100 * 1024 * 1024,  # 100MB
        'allowed_extensions': ['.csv', '.zip'],
        'virus_scan': False  # Implementar se necessário
    },
    'code_execution': {
        'sandbox': True,
        'timeout': 30,
        'blocked_imports': ['os', 'sys', 'subprocess'],
        'blocked_functions': ['exec', 'eval', 'open']
    },
    'llm_integration': {
        'input_sanitization': True,
        'output_validation': True,
        'api_key_encryption': True
    }
}
```

#### 2.6.2 Tratamento de Dados Sensíveis

- **Logs:** Não registrar dados de usuário em logs
- **Temporário:** Dados em memória apenas durante a sessão
- **API:** Chaves de API armazenadas como variáveis de ambiente
- **Transmissão:** Dados não transmitidos para serviços externos (exceto LLM opt-in)

### 2.7 Monitoramento e Logging

#### 2.7.1 Níveis de Log

```python
LOGGING_CONFIG = {
    'DEBUG': 'Detalhes de execução, prompts LLM, códigos gerados',
    'INFO': 'Operações principais, tempos de execução, métodos utilizados',
    'WARNING': 'Fallbacks executados, validações falharam',
    'ERROR': 'Falhas de execução, erros de API, exceções',
    'CRITICAL': 'Falhas de sistema, problemas de segurança'
}
```

#### 2.7.2 Métricas de Monitoramento

- Taxa de sucesso de interpretação (LLM vs Regex)
- Tempo médio de resposta por tipo de pergunta
- Uso de recursos (CPU, memória) por sessão
- Frequência de fallbacks executados
- Erros de API do OpenAI

### 2.8 Deployment e Configuração

#### 2.8.1 Dependências

```python
DEPENDENCIES = {
    'core': [
        'streamlit>=1.45.0',
        'pandas>=2.2.0',
        'plotly>=6.1.0',
        'numpy>=2.2.0'
    ],
    'llm': [
        'langchain-openai>=0.3.0',
        'langchain>=0.3.0'
    ],
    'optional': [
        'openpyxl>=3.1.0',  # Para arquivos Excel
        'chardet>=5.2.0'    # Detecção de encoding
    ]
}
```

#### 2.8.2 Variáveis de Ambiente

```bash
# Obrigatórias
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Opcionais
OPENAI_API_KEY=sk-...              # Para funcionalidade LLM
LOG_LEVEL=INFO                     # DEBUG, INFO, WARNING, ERROR
MAX_FILE_SIZE_MB=100               # Limite de upload
SESSION_TIMEOUT_HOURS=24           # Timeout de sessão
ENABLE_LLM_INSIGHTS=true           # Habilitar insights automáticos
```

#### 2.8.3 Comando de Execução

```bash
# Desenvolvimento
streamlit run app.py --server.port 8501

# Produção (com Docker)
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY csv-qa-agent

# Com configurações customizadas
streamlit run app.py --server.headless true --server.port 80 --server.address 0.0.0.0
```

### 2.9 Testes e Qualidade

#### 2.9.1 Estratégia de Testes

- **Testes Unitários:** Cada agente e método principal
- **Testes de Integração:** Fluxo completo de perguntas
- **Testes de Performance:** Tempo de resposta e uso de recursos
- **Testes de Segurança:** Validação de código malicioso
- **Testes de Usabilidade:** Interface e experiência do usuário

#### 2.9.2 Cobertura de Testes

```python
TEST_COVERAGE_TARGETS = {
    'agents': 90,           # Cobertura mínima dos agentes
    'core_functions': 95,   # Funções principais
    'security': 100,        # Validações de segurança
    'error_handling': 85    # Tratamento de erros
}
```

---

## 🎯 3. CONSIDERAÇÕES DE IMPLEMENTAÇÃO

### 3.1 Roadmap de Desenvolvimento

#### Fase 1 - Core (Concluída)
- ✅ Agentes básicos implementados
- ✅ Sistema de regex funcionando
- ✅ Interface Streamlit operacional
- ✅ Integração LLM com fallback

#### Fase 2 - Melhorias (Próxima)
- 🔄 Cache de respostas LLM
- 🔄 Métricas de performance
- 🔄 Testes automatizados abrangentes
- 🔄 Documentação de API

#### Fase 3 - Avançada (Futuro)
- 📋 Suporte a outros formatos (Excel, JSON)
- 📋 API REST para integração
- 📋 Deployment em cloud
- 📋 Multi-tenancy

### 3.2 Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Falha da API OpenAI | Média | Alto | Sistema de fallback robusto |
| Performance com arquivos grandes | Alta | Médio | Limites de tamanho e otimizações |
| Código malicioso gerado pelo LLM | Baixa | Alto | Validação rigorosa de segurança |
| Interpretação incorreta de perguntas | Média | Médio | Sistema híbrido e logging detalhado |

### 3.3 Manutenibilidade

- **Código Modular:** Cada agente é independente e testável
- **Logging Abrangente:** Rastreabilidade completa de operações
- **Configuração Externa:** Parâmetros em variáveis de ambiente
- **Documentação:** Código autodocumentado e guias de uso
- **Versionamento:** Controle de versão rigoroso com tags

---

## 📊 4. CONCLUSÃO

O sistema CSV Q&A Agent representa uma solução robusta e escalável para análise de dados através de linguagem natural. A arquitetura híbrida garante alta disponibilidade e performance, enquanto a integração LLM oferece capacidades avançadas de interpretação.

**Pontos Fortes:**
- Sistema híbrido com fallback confiável
- Segurança robusta na execução de código
- Interface intuitiva e amigável
- Suporte multilíngue
- Arquitetura modular e extensível

**Áreas de Melhoria Futura:**
- Cache inteligente para otimização de performance
- Suporte a formatos adicionais de dados
- API REST para integração com outros sistemas
- Deployment em cloud com alta disponibilidade

O sistema está pronto para uso em produção e oferece uma base sólida para futuras expansões e melhorias. 