# 📊 CSV Q&A Agent Inteligente

Um sistema inteligente baseado em agentes especializados para análise de dados CSV e resposta a perguntas em linguagem natural.

## 🎯 Status Atual: **TOTALMENTE FUNCIONAL** ✅

### ✅ **Recursos Implementados e Funcionando:**

#### 🤖 **CSVLoaderAgent (100% Funcional)**
- ✅ **Carregamento inteligente de CSV/ZIP** com detecção automática de encoding
- ✅ **Análise de schema avançada** com detecção de tipos semânticos
- ✅ **Avaliação de qualidade de dados** com pontuação automática
- ✅ **Detecção de relacionamentos** entre múltiplos datasets
- ✅ **Integração LLM** para insights avançados (quando API key disponível)
- ✅ **Tratamento robusto de erros** e fallback gracioso
- ✅ **Suporte a encoding complexo** (UTF-8, Latin1, etc.)
- ✅ **Validação de segurança** de arquivos

#### 🎨 **Interface Streamlit (100% Funcional)**
- ✅ **Upload de múltiplos arquivos** CSV e ZIP
- ✅ **Análise completa com visualizações** em tabs organizadas
- ✅ **Métricas de qualidade** e completude dos dados
- ✅ **Visualizações interativas** com Plotly
- ✅ **Download de dados processados**
- ✅ **Interface responsiva** e moderna

#### 🔧 **Recursos Técnicos Avançados**
- ✅ **Logging abrangente** para debugging
- ✅ **Gestão de memória** eficiente
- ✅ **Tratamento de timeout** para arquivos grandes
- ✅ **Cache de resultados** para performance
- ✅ **Monitoramento de uso** de LLM

## 🚀 Funcionalidades

- **Upload Inteligente**: Suporte para arquivos CSV individuais e arquivos ZIP contendo múltiplos CSVs
- **Análise Automática**: Detecção automática de encoding, tipos de dados e estrutura dos arquivos
- **Perguntas em Linguagem Natural**: Faça perguntas sobre seus dados em português
- **Interface Moderna**: Interface web construída com Streamlit
- **Histórico de Conversas**: Mantenha um histórico das perguntas e respostas
- **Arquitetura Modular**: Sistema baseado em agentes especializados

## 🏗️ Arquitetura

O sistema é composto por 5 agentes especializados:

### 🔄 CSVLoaderAgent
- **Arquivo**: `agents/csv_loader.py`
- **Função**: Carrega arquivos CSV e ZIP em DataFrames pandas
- **Recursos**: 
  - Detecção automática de encoding
  - Suporte a arquivos ZIP
  - Metadata detalhada dos arquivos

### 📊 SchemaAnalyzerAgent
- **Arquivo**: `agents/schema_analyzer.py`
- **Função**: Analisa estrutura dos dados e sugere relações entre tabelas
- **Recursos**: 
  - Estatísticas descritivas
  - Detecção de chaves potenciais
  - Sugestões de relacionamentos

### 🧠 QuestionUnderstandingAgent
- **Arquivo**: `agents/question_understanding.py`
- **Função**: Interpreta perguntas em linguagem natural e gera código pandas
- **Recursos**: 
  - Processamento de linguagem natural
  - Geração de código pandas
  - Integração com LangChain

### ⚡ QueryExecutorAgent
- **Arquivo**: `agents/query_executor.py`
- **Função**: Executa código pandas com tratamento de erros
- **Recursos**: 
  - Execução segura de código
  - Tratamento de exceções
  - Fallback strategies

### 📝 AnswerFormatterAgent
- **Arquivo**: `agents/answer_formatter.py`
- **Função**: Formata respostas para o usuário
- **Recursos**: 
  - Respostas em linguagem natural
  - Visualizações automáticas
  - Formatação contextual

## 📦 Instalação

1. **Clone o repositório**:
```bash
git clone <repository-url>
cd CSV_QA_Agent
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **Configure as variáveis de ambiente** (crie um arquivo `.env`):
```env
OPENAI_API_KEY=sua_chave_openai_aqui
LANGCHAIN_API_KEY=sua_chave_langchain_aqui
```

## 🚀 Como Usar

1. **Execute a aplicação**:
```bash
streamlit run app.py
```

2. **Acesse a interface web**:
   - Abra seu navegador em `http://localhost:8501`

3. **Faça upload dos seus dados**:
   - Clique em "Browse files" para selecionar arquivos CSV ou ZIP
   - Aguarde o processamento automático

4. **Faça perguntas sobre seus dados**:
   - Digite perguntas em linguagem natural
   - Exemplos:
     - "Qual é a média de vendas por região?"
     - "Quais são os 10 produtos mais vendidos?"
     - "Mostre a distribuição de idades dos clientes"

## 📁 Estrutura do Projeto

```
CSV_QA_Agent/
├── app.py                     # Interface Streamlit principal
├── requirements.txt           # Dependências do projeto
├── README.md                  # Este arquivo
├── .cursorrules              # Regras específicas do projeto
├── agents/                    # Agentes especializados
│   ├── __init__.py
│   ├── csv_loader.py         # Carregamento de arquivos
│   ├── schema_analyzer.py    # Análise de schema
│   ├── question_understanding.py  # Entendimento de perguntas
│   ├── query_executor.py     # Execução de queries
│   └── answer_formatter.py   # Formatação de respostas
├── utils/                     # Utilitários auxiliares
├── templates/                 # Templates e prompts
├── data/                      # Dados de exemplo
└── logs/                      # Logs da aplicação
```

## 🔧 Configurações Avançadas

### Configurações da Interface
- **Nível de Detalhamento**: Controla a verbosidade das respostas
- **Mostrar Código Gerado**: Exibe o código pandas gerado internamente
- **Modo Desenvolvedor**: Informações técnicas para debug

### Variáveis de Ambiente
```env
# Obrigatórias
OPENAI_API_KEY=your_openai_api_key_here

# Opcionais
LANGCHAIN_API_KEY=your_langchain_api_key_here
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=100
DEFAULT_ENCODING=utf-8
```

## 🎯 Exemplos de Uso

### Análise de Vendas
```
Pergunta: "Qual foi o total de vendas por mês no último ano?"
Resposta: Gráfico de barras + valor numérico + insights
```

### Análise de Clientes
```
Pergunta: "Quais são as características dos clientes mais valiosos?"
Resposta: Tabela com segmentação + estatísticas + recomendações
```

### Análise Temporal
```
Pergunta: "Como evoluíram as vendas ao longo do tempo?"
Resposta: Gráfico de linha + tendências + sazonalidade
```

## 🛠️ Desenvolvimento

### Adicionando Novos Agentes
1. Crie um novo arquivo em `agents/`
2. Implemente a classe do agente
3. Adicione ao `__init__.py`
4. Integre na pipeline principal

### Contribuindo
1. Fork o projeto
2. Crie uma branch para sua feature
3. Faça commit das mudanças
4. Abra um Pull Request

## 📋 TODO

- [ ] Implementar todos os agentes especializados
- [ ] Adicionar suporte a Excel (.xlsx)
- [ ] Implementar cache de resultados
- [ ] Adicionar exportação de relatórios
- [ ] Suporte a múltiplos idiomas
- [ ] API REST para integração externa
- [ ] Dashboard de métricas
- [ ] Testes automatizados

## 🐛 Problemas Conhecidos

- Arquivos muito grandes podem causar lentidão
- Alguns tipos de encoding podem não ser detectados automaticamente
- Perguntas muito complexas podem gerar código inválido

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Suporte

Para suporte, abra uma issue no GitHub ou entre em contato através do email de suporte.

---

**Feito com ❤️ usando Python, Streamlit, Pandas e LangChain** 