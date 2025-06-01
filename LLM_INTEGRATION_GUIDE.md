# 🤖 Guia de Integração LLM - QuestionUnderstandingAgent

## 📋 Visão Geral

A classe `QuestionUnderstandingAgent` foi refatorada para incluir um mecanismo híbrido de interpretação de perguntas:

1. **🔥 Primeira tentativa**: LLM (ChatOpenAI via LangChain)
2. **🔄 Fallback**: Sistema baseado em regex (método original)

---

## 🚀 Configuração

### Pré-requisitos

```bash
pip install langchain-openai
```

### Configuração da API Key

```bash
# No Windows
set OPENAI_API_KEY=sua_api_key_aqui

# No Linux/Mac
export OPENAI_API_KEY=sua_api_key_aqui
```

### Ou no código Python:

```python
import os
os.environ['OPENAI_API_KEY'] = 'sua_api_key_aqui'
```

---

## 💡 Como Funciona

### 1. Inicialização do Agente

```python
from agents.question_understanding import QuestionUnderstandingAgent

# O agente detecta automaticamente se LLM está disponível
agent = QuestionUnderstandingAgent()

# Verificar status do LLM
if agent.llm:
    print("✅ LLM disponível")
else:
    print("⚡ Usando apenas regex")
```

### 2. Processamento de Perguntas

```python
import pandas as pd

# Dados de exemplo
df = pd.DataFrame({
    'valor_total': [100, 200, 300],
    'produto': ['A', 'B', 'C']
})

dataframes = {'vendas.csv': df}

# Fazer pergunta
result = agent.understand_question(
    "Qual é a soma dos valores totais?", 
    dataframes
)

print(f"Método usado: {result['code_source']}")  # 'llm' ou 'regex'
print(f"Código gerado: {result['generated_code']}")
```

---

## 🔧 Funcionalidades Implementadas

### ✅ Método `_generate_code_with_llm()`

**Assinatura:**
```python
def _generate_code_with_llm(self, question: str, df_name: str, df: pd.DataFrame) -> Optional[str]
```

**Funcionalidades:**
- 📊 Analisa colunas e tipos de dados
- 🎯 Gera prompt contextualizado
- 🔍 Valida código gerado
- 🛡️ Verificações básicas de segurança

### ✅ Método `_validate_llm_code()`

**Validações realizadas:**
- ✅ Presença de `dataframes['arquivo']`
- ✅ Presença de `result = ...`
- 🚫 Detecção de código perigoso
- 🔍 Verificação de estrutura básica

### ✅ Sistema Híbrido

**Fluxo de processamento:**
1. 🤖 **LLM**: Tenta gerar código usando ChatOpenAI
2. ✅ **Validação**: Verifica se código é válido e seguro
3. 🔄 **Fallback**: Se falhar, usa sistema regex
4. 📊 **Resultado**: Retorna código com metadados

---

## 📊 Estrutura de Resposta

```python
{
    'original_question': 'Qual é a soma dos valores?',
    'target_dataframe': 'vendas.csv',
    'generated_code': 'df = dataframes["vendas.csv"]\nresult = df["valor_total"].sum()',
    'confidence': 0.95,
    'explanation': 'Código gerado usando LLM (ChatOpenAI)',
    'code_source': 'llm',  # ou 'regex'
    'understood_intent': 'Interpretação automática via LLM'
}
```

---

## 🎯 Exemplos Práticos

### Exemplo 1: LLM Disponível

```python
# Com API key configurada
agent = QuestionUnderstandingAgent()

result = agent.understand_question(
    "Mostre os 5 produtos com maior valor total",
    dataframes
)

# Resultado esperado:
# code_source: 'llm'
# confidence: 0.95
# generated_code: código sofisticado gerado pelo LLM
```

### Exemplo 2: Fallback para Regex

```python
# Sem API key ou LLM indisponível
agent = QuestionUnderstandingAgent()

result = agent.understand_question(
    "Soma da coluna valor_total",
    dataframes
)

# Resultado esperado:
# code_source: 'regex'
# confidence: 1.0
# generated_code: código baseado em padrões regex
```

### Exemplo 3: Tratamento de Erros

```python
result = agent.understand_question(
    "Pergunta muito complexa que falha em ambos os métodos",
    dataframes
)

if result.get('error'):
    print(f"Erro: {result['explanation']}")
    print(f"Sugestões: {result.get('fallback_suggestions', [])}")
```

---

## 📈 Vantagens da Implementação

### 🤖 LLM (Primeira Opção)
- ✅ Interpretação mais inteligente
- ✅ Suporte a perguntas complexas
- ✅ Flexibilidade para casos não previstos
- ✅ Geração de código mais sofisticado

### ⚡ Regex Fallback
- ✅ Sempre disponível (não depende de API)
- ✅ Rápido e confiável
- ✅ Padrões otimizados e testados
- ✅ Zero custo operacional

### 🔄 Sistema Híbrido
- ✅ Máxima disponibilidade
- ✅ Balanceamento custo/performance
- ✅ Degradação graceful
- ✅ Transparência para o usuário

---

## 🔒 Segurança

### Validações Implementadas

```python
# Elementos obrigatórios
required_elements = [
    f"dataframes['{df_name}']",  # Carregamento do DataFrame
    "result =",                   # Atribuição do resultado
]

# Padrões perigosos bloqueados
dangerous_patterns = [
    'import os', 'import sys', 'exec(', 'eval(', 
    'open(', '__import__', 'subprocess'
]
```

---

## 📊 Monitoramento e Logs

### Logs Disponíveis

```python
# Logs de inicialização
logger.info("LLM initialized successfully with OpenAI")
logger.warning("OpenAI API key not found, LLM unavailable")

# Logs de processamento
logger.debug("LLM Prompt enviado: ...")
logger.debug("LLM Response recebida: ...")
logger.info("✅ Usando código gerado por LLM")
logger.info("⚡ Usando fallback: método baseado em regex")

# Logs de validação
logger.info("✅ Código LLM validado com sucesso")
logger.warning("❌ Código LLM inválido, será usado fallback")
```

### Histórico de Perguntas

```python
# Acessar histórico
history = agent.get_question_history()

for item in history:
    print(f"Pergunta: {item['original_question']}")
    print(f"Método: {item['code_source']}")
    print(f"Confiança: {item['confidence']}")
```

---

## 🎯 Casos de Uso

### 1. Perguntas Simples (Regex)
- "Soma da coluna valor_total"
- "Média de vendas"
- "Máximo da receita"

### 2. Perguntas Complexas (LLM)
- "Mostre os 5 produtos mais vendidos por categoria"
- "Compare vendas do primeiro trimestre com o segundo"
- "Quais categorias têm receita acima da média?"

### 3. Perguntas Multilíngues
- **Português**: "Qual é a soma dos valores?"
- **Inglês**: "What is the total sales?"

---

## 🚀 Próximos Passos

### Possíveis Melhorias

1. **🔧 Cache de Respostas LLM**: Evitar chamadas repetidas
2. **📊 Métricas de Performance**: Tempo de resposta LLM vs Regex
3. **🎯 Prompt Engineering**: Otimizar prompts para melhores resultados
4. **🔄 Modelos Alternativos**: Suporte a outros modelos além do GPT-3.5

### Configurações Avançadas

```python
# Personalizar modelo LLM
agent = QuestionUnderstandingAgent()
if agent.llm:
    agent.llm.model = "gpt-4"  # Usar GPT-4 para maior precisão
    agent.llm.temperature = 0.0  # Respostas mais determinísticas
```

---

## 🎉 Conclusão

A refatoração da `QuestionUnderstandingAgent` oferece:

- 🤖 **Inteligência Aumentada**: LLM para casos complexos
- ⚡ **Confiabilidade**: Fallback regex sempre disponível
- 🔒 **Segurança**: Validação robusta de código gerado
- 📊 **Transparência**: Rastreabilidade completa do processo
- 🚀 **Flexibilidade**: Adaptável a diferentes cenários de uso

O sistema está pronto para produção e oferece uma experiência superior de interpretação de perguntas em linguagem natural! 