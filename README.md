# 🏥 Tech Challenge — Fase 04 | Predição de Obesidade

> **FIAP PosTech — Data Analytics**
> Desenvolvimento de um sistema preditivo de Machine Learning para auxiliar equipes médicas no diagnóstico de níveis de obesidade.
---

## ‍💻 Autor

**Gustavo Henrique Lisboa do Nascimento** — FIAP PosTech Data Analytics

---

## 🔗 Links Rápidos

| Recurso | Link |
|---|---|
| 🚀 **Aplicação Streamlit (deploy)** | [Acessar app](https://SEU-APP.streamlit.app) |
| 🎥 **Vídeo de apresentação** | [Assistir no YouTube](https://youtu.be/SEU-VIDEO) |
| 📓 **Notebook completo** | [Abrir notebook](notebooks/TechChallenge_Fase04_Obesidade_v3.ipynb) |

---

## 🎯 O Problema

A obesidade é uma condição multifatorial — envolve genética, hábitos alimentares, atividade física e fatores ambientais. O desafio: **construir um modelo capaz de prever o nível de obesidade de um paciente** a partir de variáveis comportamentais e biométricas, apoiando a tomada de decisão clínica.

### Dataset
- **Fonte:** `Obesity.csv` — 2.111 registros, 17 variáveis
- **Variável-alvo:** `Obesity` (7 classes: de `Insufficient_Weight` a `Obesity_Type_III`)
- **Dicionário completo:** [`docs/dicionario_obesity.pdf`](docs/dicionario_obesity.pdf)

---

## 🧪 Estratégia de Modelagem — Dois Modelos, Duas Histórias

Um dos achados centrais deste projeto foi identificar **vazamento de informação (data leakage)** ao usar Peso/Altura/IMC como features — variáveis que **definem matematicamente** o rótulo. Por isso, treinei dois modelos:

| Modelo | Features | Acurácia | F1-macro | Propósito |
|---|---|---|---|---|
| **A — com IMC** | Inclui peso, altura e IMC | **98,8%** | 0,988 | Baseline / sanity check (vazamento conhecido) |
| **B — sem IMC** ⭐ | Apenas variáveis comportamentais e idade/gênero | **77,5%** | 0,767 | **Modelo de valor clínico real** |

> 💡 **Insight de negócio:** O Modelo B é o que importa. Ele prevê obesidade *antes* de o paciente estar obeso, com base no estilo de vida — exatamente o tipo de triagem que um médico de atenção primária precisa.

**Algoritmo escolhido:** `XGBoost` com hiperparâmetros otimizados via GridSearchCV.

---

## 🛠️ Pipeline de Machine Learning

```
Dados Brutos (Obesity.csv)
        │
        ▼
[1] Análise Exploratória ──► Detecção de vazamento (peso/altura/IMC)
        │
        ▼
[2] Pré-processamento ──► Renomeação PT-BR, encoding categórico,
        │                     arredondamento de escalas Likert
        ▼
[3] Feature Engineering ──► IMC (Modelo A), tratamento de ordinais
        │
        ▼
[4] Split Estratificado ──► Treino (80%) / Teste (20%)
        │
        ▼
[5] Treinamento ──► XGBoost + GridSearchCV (5-fold CV)
        │
        ▼
[6] Avaliação ──► Accuracy, F1-macro, Matriz de Confusão
        │
        ▼
[7] Serialização ──► .pkl (joblib) + metadata.json
        │
        ▼
[8] Deploy ──► Streamlit App
```

Detalhes em [`notebooks/TechChallenge_Fase04_Obesidade_v3.ipynb`](notebooks/).

---

## 📊 Principais Insights para a Equipe Médica

1. **Histórico familiar** é o preditor não-biométrico mais forte — pacientes com histórico têm risco multiplicado.
2. **Lanches entre refeições** (`CAEC`) tem correlação alta com sobrepeso, especialmente em jovens adultos.
3. **Sedentarismo + transporte motorizado** combinados elevam significativamente o risco.
4. **Consumo de água** mostra padrão inverso esperado (mais água → menor IMC), reforçando recomendação clínica.
5. **Idade 20–35** concentra os casos de Obesidade Tipo I — janela crítica para intervenção.

> Análises completas no [Dashboard](dashboard/) e no notebook.

---

## 📁 Estrutura do Repositório

```
tech-challenge-fase4-obesidade/
├── data/raw/Obesity.csv          # Dataset original
├── notebooks/                    # Análise e treinamento
├── src/                          # Código modular
├── models/                       # Modelos serializados + metadata
├── app/                          # Aplicação Streamlit
├── dashboard/                    # Painel analítico
└── docs/                         # Briefing, dicionário, imagens
```

---

## 🚀 Como Executar Localmente

### Pré-requisitos
- Python 3.10+
- pip ou conda

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/SEU-USUARIO/tech-challenge-fase4-obesidade.git
cd tech-challenge-fase4-obesidade

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Rode o app Streamlit
streamlit run app/app_v7.py
```

Acesse `http://localhost:8501` no navegador.

### Para reproduzir o treinamento

```bash
jupyter notebook notebooks/TechChallenge_Fase04_Obesidade_v3.ipynb
```

---

## 📦 Stack Tecnológico

| Camada | Ferramentas |
|---|---|
| Linguagem | Python 3.10 |
| Análise | pandas, numpy, matplotlib, seaborn |
| ML | scikit-learn, XGBoost, joblib |
| App | Streamlit |
| Dashboard | Plotly / Power BI |
| Versionamento | Git + GitHub |

---

## 📄 Licença

Este projeto está sob a licença MIT — veja [LICENSE](LICENSE).

---

<p align="center">
  <em>Desenvolvido como Tech Challenge da Fase 04 — FIAP PosTech Data Analytics</em>
</p>
