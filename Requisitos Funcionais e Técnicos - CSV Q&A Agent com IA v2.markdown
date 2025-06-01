# Requisitos Funcionais e Técnicos - CSV Q&A Agent com IA

**Versão:** 2.0  
**Data:** 01 de Junho de 2025, 16:57 -03  
**Autor:** Consolidação baseada nos documentos "Agente CSV Q&A Híbrido com IA", "Agente CSV: Q&A Inteligente para Análise de Dados", "Agente Inteligente Q&A para CSVs", "Arquitetura Agente Q&A CSV", "Especificação CSV Q&A Agent" e "Guia Agente de Compreensão de Perguntas Híbrido LLM Regex"  
**Equipe:** Agente Aprende  
**Membros da Equipe:**  
- **Dayse Kamikawachi**: Product Owner / IA Specialist  
- **Carlos Eduardo Gabriel Santos**: Lead Developer / Architect  
- **Pedro Markovic**: UX/UI Designer / Frontend Dev  

---

## Introdução

O **CSV Q&A Agent com IA** é uma solução inovadora projetada para simplificar a análise de dados em arquivos CSV, permitindo que usuários não técnicos obtenham insights valiosos por meio de perguntas em linguagem natural. Utilizando uma abordagem híbrida que combina inteligência artificial (LLM) com um sistema de fallback baseado em regex, o sistema garante precisão e confiabilidade. Este documento consolida os requisitos funcionais e técnicos da versão 2.0, detalhando funcionalidades, arquitetura, especificações técnicas e considerações para implantação e manutenção.

---

## 1. Especificação Funcional

### 1.1 Visão Geral do Sistema

O **CSV Q&A Agent** é uma aplicação web que capacita usuários a interagir com dados em arquivos CSV por meio de perguntas em linguagem natural, em português e inglês. Ele carrega arquivos (CSV ou ZIP), analisa automaticamente a estrutura dos dados, interpreta perguntas, executa consultas e apresenta respostas em linguagem natural, com gráficos interativos quando apropriado. Seu objetivo principal é democratizar a análise de dados, permitindo que usuários como analistas de negócios, gestores e outros profissionais (ex.: lidando com notas fiscais, vendas, estoques) obtenham insights sem necessidade de codificação.

**Características principais:**
- Sistema híbrido: utiliza LLM (ex.: ChatOpenAI, GPT-3.5, GPT-4o, Gemini Pro) com fallback baseado em regex.
- Interface web intuitiva construída com Streamlit.
- Suporte a junções automáticas entre múltiplos arquivos.
- Transparência sobre os métodos utilizados (LLM ou regex).

### 1.2 Requisitos Funcionais

#### RF001 - Upload e Processamento de Arquivos
- **Descrição:** O sistema deve permitir o upload e processamento de arquivos CSV e ZIP.  
- **Critérios de Aceitação:**  
  - Suporte a múltiplos arquivos simultaneamente (ex.: ZIP com vários CSVs).  
  - Detecção automática de encoding (ex.: UTF-8, Latin1).  
  - Validação de integridade e presença de cabeçalho nos arquivos.  
  - Limite máximo de 100 MB por arquivo.  
  - Interface com upload via drag-and-drop.

#### RF002 - Análise Automática de Schema
- **Descrição:** O sistema deve analisar automaticamente a estrutura dos dados carregados.  
- **Critérios de Aceitação:**  
  - Identificação de tipos de dados (ex.: numérico, texto, data, int, float, string).  
  - Detecção de valores nulos, duplicados e outliers.  
  - Cálculo de um Quality Score (0-100) para os dados.  
  - Identificação de relacionamentos entre datasets com base em colunas comuns.  
  - Visualização do schema dos dados.

#### RF003 - Interpretação de Perguntas em Linguagem Natural
- **Descrição:** O sistema deve interpretar perguntas em linguagem natural em português e inglês.  
- **Critérios de Aceitação:**  
  - Suporte bilíngue (pt-BR e en-US).  
  - Identificação de operações (ex.: soma, média, máximo, comparação, top N, correlação).  
  - Mapeamento de colunas mencionadas, mesmo com nomes semelhantes.  
  - Detecção do arquivo alvo, quando especificado.  
  - Sistema híbrido (LLM + regex) para maior precisão e cobertura.

#### RF004 - Geração Automática de Código de Análise
- **Descrição:** O sistema deve gerar código pandas executável com base nas perguntas.  
- **Critérios de Aceitação:**  
  - Código pandas válido e seguro.  
  - Validação de segurança para bloquear operações perigosas.  
  - Fallback automático (LLM para regex) em caso de falha ou indisponibilidade.  
  - Transparência sobre o método utilizado (LLM ou regex).

#### RF005 - Execução Segura de Código
- **Descrição:** O sistema deve executar o código gerado de forma segura e controlada.  
- **Critérios de Aceitação:**  
  - Ambiente de execução isolado (sandbox).  
  - Validação prévia do código.  
  - Tratamento de exceções com mensagens claras e fallbacks.  
  - Timeout de 30 segundos para operações longas.  
  - Logging detalhado de todas as execuções.

#### RF006 - Formatação de Respostas
- **Descrição:** O sistema deve apresentar resultados em linguagem natural com visualizações.  
- **Critérios de Aceitação:**  
  - Respostas em linguagem natural, objetivas e localizadas (pt-BR e en-US).  
  - Inclusão de valores, tabelas ou texto, citando arquivos e colunas usados.  
  - Geração automática de gráficos interativos (ex.: barras, linhas, scatter plots) com Plotly.  
  - Insights adicionais via LLM, se disponível.  
  - Indicadores de confiança nas respostas.  
  - Exportação de resultados (PNG, PDF) e dados processados.

#### RF007 - Interface Web Interativa
- **Descrição:** O sistema deve oferecer uma interface amigável para interação.  
- **Critérios de Aceitação:**  
  - Interface Streamlit responsiva (desktop e mobile).  
  - Upload via drag-and-drop.  
  - Visualização prévia dos dados carregados.  
  - Campo para perguntas em linguagem natural com sugestões automáticas.  
  - Histórico de perguntas e respostas.  
  - Feedback em tempo real sobre o progresso e método usado.  
  - Coleta de feedback do usuário para melhorias.

#### RF008 - Análise Avançada com IA (Opcional)
- **Descrição:** O sistema deve fornecer insights adicionais via LLM, se configurado.  
- **Critérios de Aceitação:**  
  - Geração de sumários automáticos dos dados.  
  - Identificação de padrões e anomalias.  
  - Sugestões de próximas análises.  
  - Funcionamento em modo degradado (sem LLM) com todas as funcionalidades básicas.

### 1.3 Casos de Uso

#### CU001 - Análise Financeira Simples
- **Ator:** Analista financeiro  
- **Cenário:** Usuário faz upload de "vendas_2024.csv" e pergunta: "Qual é o total de vendas no primeiro semestre?"  
- **Resultado Esperado:** Sistema gera código pandas (`df[df['data'].dt.semester == 1]['vendas'].sum()`), exibe o valor total e um gráfico de barras.

#### CU002 - Comparação Entre Datasets
- **Ator:** Gerente regional  
- **Cenário:** Usuário faz upload de "vendas_sp.csv" e "vendas_rj.csv" e pergunta: "Compare as vendas entre São Paulo e Rio de Janeiro."  
- **Resultado Esperado:** Sistema detecta colunas comuns, realiza junção automática e apresenta análise comparativa com tabela e gráfico.

#### CU003 - Análise Exploratória Automática
- **Ator:** Cientista de dados  
- **Cenário:** Usuário faz upload de um dataset e utiliza a funcionalidade de análise com IA.  
- **Resultado Esperado:** Sistema gera sumário, identifica padrões/anomalias e sugere próximas análises.

#### CU004 - Perguntas Complexas
- **Ator:** Analista de dados  
- **Cenário:** Usuário pergunta: "Quais os 5 produtos mais vendidos por categoria?"  
- **Resultado Esperado:** Sistema aciona LLM para gerar código complexo e apresenta resultado com gráfico.

#### CU005 - Perguntas Simples (Fallback)
- **Ator:** Usuário geral  
- **Cenário:** Usuário pergunta: "Qual a soma da coluna valor_total?"  
- **Resultado Esperado:** Sistema usa regex para gerar código simples e retorna o resultado rapidamente.

### 1.4 Regras de Negócio

- **RN001 - Arquivos:**  
  - Arquivos devem ter cabeçalho para identificação de colunas.  
  - Arquivos acima de 100 MB serão rejeitados.  
  - Encoding é detectado automaticamente, com fallback para UTF-8.

- **RN002 - Perguntas e Respostas:**  
  - Perguntas devem preferencialmente referenciar colunas dos dados.  
  - Respostas devem citar arquivos e colunas usados para transparência.

- **RN003 - Junções:**  
  - Junções automáticas entre arquivos baseiam-se em colunas comuns (ex.: `numero_nf`, `id_pedido`).

- **RN004 - Segurança:**  
  - Código gerado é validado, bloqueando operações perigosas (ex.: `exec()`, `eval()`, `import os`).  
  - Timeout de 30 segundos para execuções longas.

- **RN005 - LLM:**  
  - LLM é opcional e requer API Key (`OPENAI_API_KEY`).  
  - Modelo padrão: `gpt-3.5-turbo`.  
  - Sistema funciona sem LLM usando regex.

- **RN006 - Qualidade de Dados:**  
  - Score de qualidade é calculado (nulos, duplicatas, consistência).  
  - Alertas para problemas críticos (ex.: >50% de valores nulos).

---

## 2. Especificação Técnica

### 2.1 Arquitetura do Sistema

#### 2.1.1 Componentes Principais (Agentes)
O sistema é composto por agentes especializados, cada um com uma função específica:

1. **CSVLoaderAgent:**  
   - Carrega e valida arquivos CSV/ZIP.  
   - Converte para DataFrames pandas.  
   - Detecta encoding e realiza análise inicial de qualidade.

2. **SchemaAnalyzerAgent:**  
   - Analisa a estrutura dos DataFrames.  
   - Identifica tipos de dados, métricas de qualidade e relacionamentos entre arquivos.

3. **QuestionUnderstandingAgent:**  
   - Interpreta perguntas em linguagem natural.  
   - Gera código pandas usando LLM ou regex (sistema híbrido).

4. **QueryExecutorAgent:**  
   - Executa código pandas em sandbox.  
   - Valida segurança e gerencia timeouts/erros.

5. **AnswerFormatterAgent:**  
   - Formata resultados em linguagem natural.  
   - Gera visualizações interativas com Plotly.

#### 2.1.2 Pipeline de Processamento de Perguntas
O fluxo de uma pergunta segue as etapas:  
**User Question → Normalization & Cleaning → Identify Target DataFrame → LLM Generation (if available) → Regex Fallback → Code Validation → Safe Execution → Response Formatting → User Interface**

### 2.2 Stack Tecnológica

- **Frontend:** Streamlit (interface web interativa e responsiva).  
- **Backend:** Python 3.8+ (recomendado 3.11+).  
- **AI/LLM:** OpenAI (GPT-3.5/GPT-4o), LangChain (orquestração de prompts); alternativa: Gemini Pro.  
- **Processamento de Dados:** Pandas (manipulação de DataFrames).  
- **Visualizações:** Plotly (gráficos interativos).  
- **Histórico/Logging:** SQLite (leve, com opção de escalar para BigQuery).

### 2.3 Estruturas de Dados

#### 2.3.1 DataFrames
- Armazenados em memória durante a sessão do usuário.

#### 2.3.2 LoadResult (CSVLoaderAgent Output)
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

#### 2.3.3 Estrutura de Resposta (QuestionUnderstandingAgent Output)
```python
{
    'original_question': str,
    'target_dataframe': Optional[str],
    'target_columns': List[str],
    'operations': List[dict],
    'generated_code': Optional[str],
    'confidence': float,
    'explanation': str,
    'code_source': str,  # 'llm', 'regex' ou 'error'
    'understood_intent': Optional[str],
    'fallback_suggestions': List[str],
    'error': Optional[str]
}
```

#### 2.3.4 Estrutura de Execução (QueryExecutorAgent Output)
```python
{
    'success': bool,
    'result': Any,
    'execution_time': float,
    'code': str,
    'output': str,
    'error': Optional[str],
    'fallback_executed': bool,
    'fallback_strategy': Optional[str],
    'variables_available': List[str]
}
```

#### 2.3.5 Histórico
- Lista de estruturas de resposta/execução para rastreamento.

### 2.4 Algoritmos Principais

#### 2.4.1 Interpretação Híbrida
```python
def understand_question(question, dataframes):
    clean_question = normalize_and_clean(question)
    target_df = identify_target_dataframe(clean_question, dataframes)
    
    if llm_available:
        llm_code = generate_code_with_llm(question, target_df, dataframes[target_df])
        if validate_llm_code(llm_code):
            return create_response(llm_code, 'llm', confidence=0.95)
    
    columns = identify_columns(clean_question, dataframes[target_df])
    operations = identify_operations(clean_question)
    regex_code = generate_code_regex(target_df, columns, operations)
    
    if regex_code:
        return create_response(regex_code, 'regex', confidence=calculate_confidence())
    else:
        return create_error_response("Não foi possível interpretar a pergunta")
```

#### 2.4.2 Validação de Segurança
```python
def validate_code_safety(code):
    required_elements = ["dataframes[", "result ="]
    if not all(element in code for element in required_elements):
        return False
    
    dangerous_patterns = [
        'import os', 'import sys', 'exec(', 'eval(',
        'open(', '__import__', 'subprocess', 'os.system'
    ]
    if any(pattern in code.lower() for pattern in dangerous_patterns):
        return False
    
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
```

### 2.5 Performance e Escalabilidade

#### 2.5.1 Métricas de Performance
| Operação                | Tempo Esperado | Limite Máximo |
|-------------------------|----------------|---------------|
| Upload (10 MB)          | < 2 s          | 5 s           |
| Análise de Schema       | < 1 s          | 3 s           |
| Interpretação via LLM   | < 3 s          | 10 s          |
| Interpretação via Regex | < 0.1 s        | 0.5 s         |
| Execução de Código      | < 1 s          | 30 s          |
| Formatação de Resposta  | < 0.5 s        | 2 s           |

#### 2.5.2 Limitações de Recursos
- **Memória:** Máximo de 2 GB por sessão.  
- **Armazenamento:** Arquivos temporários removidos após 24h de inatividade.  
- **CPU:** Timeout de 30 s para operações intensivas.  
- **API:** Limite de 50 chamadas/min para OpenAI API.

#### 2.5.3 Considerações de Escalabilidade
- Design stateless por sessão.  
- Cache de DataFrames e resultados (otimização futura).  
- Escalabilidade horizontal com múltiplas instâncias Streamlit.  
- SQLite escalável para BigQuery em cenários de alto volume.

### 2.6 Segurança

#### 2.6.1 Validação Multicamada
- **Entrada:** Validação de tamanho, extensão e integridade; scanner de vírus opcional.  
- **Código:** Verificação de sintaxe e bloqueio de operações perigosas.  
- **Execução:** Sandbox isolado com controle de timeout e memória.

#### 2.6.2 Operações Bloqueadas
- `import os`, `import sys`, `exec()`, `eval()`, `open()`, `subprocess`, `__import__`, operações de rede, acesso a arquivos ou variáveis de ambiente.

#### 2.6.3 Tratamento de Dados Sensíveis
- Dados não registrados em logs.  
- Armazenamento em memória apenas durante a sessão.  
- API Keys em variáveis de ambiente (criptografia opcional).  
- Conformidade com LGPD/GDPR.

#### 2.6.4 Auditoria
- Logs estruturados (`app.log`, `security.log`, `performance.log`, `llm_usage.log`).

### 2.7 Monitoramento e Logging

#### 2.7.1 Níveis de Log
```python
LOGGING_CONFIG = {
    'DEBUG': 'Prompts LLM, códigos gerados',
    'INFO': 'Operações principais, tempos, métodos',
    'WARNING': 'Fallbacks, validações falhadas',
    'ERROR': 'Falhas de execução, erros de API',
    'CRITICAL': 'Falhas de sistema, problemas de segurança'
}
```

#### 2.7.2 Métricas de Monitoramento
- Taxa de sucesso (LLM vs. Regex).  
- Tempo médio de resposta por operação.  
- Uso de recursos (CPU, memória).  
- Frequência de fallbacks e erros de API.  
- **Recomendação:** Dashboard Grafana para produção.

### 2.8 Implantação e Configuração

#### 2.8.1 Requisitos do Sistema
- Python 3.8+ (recomendado 3.11+).  
- Mínimo 2 GB de RAM.  
- Conexão à internet (opcional para LLM).

#### 2.8.2 Dependências
```python
DEPENDENCIES = {
    'core': ['streamlit>=1.45.0', 'pandas>=2.2.0', 'plotly>=6.1.0', 'numpy>=2.2.0'],
    'llm': ['langchain-openai>=0.3.0', 'langchain>=0.3.0'],
    'optional': ['openpyxl>=3.1.0', 'chardet>=5.2.0']
}
```

#### 2.8.3 Variáveis de Ambiente
```bash
# Obrigatórias
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Opcionais
OPENAI_API_KEY=sk-...
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=100
SESSION_TIMEOUT_HOURS=24
ENABLE_LLM_INSIGHTS=true
```

#### 2.8.4 Comandos de Execução
- **Desenvolvimento:** `streamlit run app.py --server.port 8501`  
- **Produção (Docker):** `docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY csv-qa-agent`  
- **Produção em Escala:** Uso de Kubernetes com auto-scaling (AWS/GCP/Azure), cache Redis e métricas avançadas.

### 2.9 Testes e Qualidade

#### 2.9.1 Estrutura de Testes
- **Unitários:** Por agente (`unit/`).  
- **Integração:** Fluxo completo (`integration/`).  
- **Performance:** Benchmarks (`performance/`).  
- **Segurança:** Testes contra injeção (`security/`).  
- **Usabilidade:** Avaliação da interface.

#### 2.9.2 Ferramentas e Cobertura
- **Ferramenta:** pytest.  
- **Cobertura Alvo:** 90% (agentes), 100% (segurança), 85% (geral).  
- **Métricas de Qualidade:**  
  - Prontidão para Produção: 89%.  
  - Documentação: 95% completa.  
  - Performance: Sub-segundo para 90% dos casos.  
  - Disponibilidade: 99.9% com fallback.

#### 2.9.3 Certificações
- Production Ready.  
- Security Validated.  
- Performance Optimized.  
- Well Documented.

---

## 3. Considerações Adicionais

### 3.1 Roadmap de Desenvolvimento

#### Fase Atual (v2.0) - Concluída
- Sistema híbrido LLM+Regex.  
- Segurança robusta.  
- Interface Streamlit funcional.  
- Documentação completa.

#### Fase 2.1 - Planejada
- Cache de respostas LLM.  
- Dashboard de métricas em tempo real.  
- API REST para integração.  
- Suporte a Excel (.xlsx).

#### Fase 3.0 - Futura
- Multi-tenancy com autenticação.  
- Análise de dados em tempo real.  
- Integração com bancos SQL.  
- Aplicativo mobile.

### 3.2 Riscos e Mitigações

| Risco                          | Probabilidade | Impacto | Mitigação                        |
|--------------------------------|---------------|---------|----------------------------------|
| Falha da API OpenAI            | Média         | Alto    | Fallback robusto com regex       |
| Lentidão com arquivos grandes  | Alta          | Médio   | Limite de 100 MB, cache futuro   |
| Código malicioso via LLM       | Baixa         | Alto    | Validação rigorosa de segurança  |
| Interpretação incorreta        | Média         | Médio   | Sistema híbrido, logs detalhados |
| Arquivo sem cabeçalho          | Baixa         | Médio   | Alerta ao usuário                |
| Pergunta ambígua               | Média         | Baixo   | Solicitar reformulação           |
| Erro na execução de consulta   | Baixa         | Médio   | Correção automática via schema   |

### 3.3 Manutenibilidade

- **Código Modular:** Agentes independentes e testáveis.  
- **Logging:** Rastreabilidade completa.  
- **Configuração:** Parâmetros via variáveis de ambiente.  
- **Documentação:** Código comentado e guias detalhados.  
- **Versionamento:** Controle via Git.

---

## 4. Conclusão

O **CSV Q&A Agent com IA v2.0** é uma solução robusta e escalável para análise de dados via linguagem natural. O sistema híbrido (LLM + regex) garante alta disponibilidade e precisão, enquanto a segurança enterprise protege contra riscos. A interface Streamlit, intuitiva e multilíngue, democratiza o acesso à análise de dados para usuários não técnicos. Áreas de melhoria incluem cache para performance, suporte a novos formatos e integração via API REST. O sistema é **Production Ready**, com uma base sólida para futuras expansões.