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
| 🚀 **Aplicação Streamlit** | [Acessar app](https://tech-challenge---fase-4-mcgtzwyczcwcguygblcusk.streamlit.app/) |
| 📓 **Notebook renderizado** | [Acessar Notebook](https://nbviewer.org/github/gnascimento9/Tech-Challenge---Fase-4/blob/main/notebooks/TechChallenge_Fase04_Obesidade.ipynb) |

---

## 🎯 O Problema

A obesidade é uma condição multifatorial — envolve genética, hábitos alimentares, atividade física e fatores ambientais. O desafio: **construir um modelo capaz de prever o nível de obesidade de um paciente** a partir de variáveis comportamentais e biométricas, apoiando a tomada de decisão clínica.

---

## 🧪 Estratégia de Modelagem — Dois Modelos, Duas Histórias

Um dos achados centrais deste projeto foi identificar **vazamento de informação (data leakage)** ao usar Peso/Altura/IMC como features — variáveis que **definem matematicamente** o rótulo. Por isso, treinei dois modelos:

| Modelo | Features | Acurácia | F1-macro | Propósito |
|---|---|---|---|---|
| **A — com IMC** | Inclui peso, altura e IMC | **98,8%** | 0,988 | Baseline / sanity check (vazamento conhecido) |
| **B — sem IMC** | Apenas variáveis comportamentais e idade/gênero | **77,5%** | 0,767 | **Modelo de valor clínico real** |


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

---

## 📊 Principais Insights para a Equipe Médica

1. **Histórico familiar** é o preditor não-biométrico mais forte — pacientes com histórico têm risco multiplicado.
2. **Lanches entre refeições** (`CAEC`) tem correlação alta com sobrepeso, especialmente em jovens adultos.
3. **Sedentarismo + transporte motorizado** combinados elevam significativamente o risco.
4. **Consumo de água** mostra padrão inverso esperado (mais água → menor IMC), reforçando recomendação clínica.
5. **Idade 20–35** concentra os casos de Obesidade Tipo I — janela crítica para intervenção.

---

## 📁 Estrutura do Repositório

```
Tech-Challenge---Fase-4
├── data/raw/Obesity.csv
├── notebooks/                                          
├── models/                     
├── app/                          
└── docs/                       
```

---

## 🚀 Como Executar Localmente

### Pré-requisitos
- Python 3.10+

### Passo a passo

```bash
# 1. Clone o repositório
git clone 

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Rode o app Streamlit
streamlit run app/app.py
```

Acesse `http://localhost:8501` no navegador.

---

## 📄 Licença

Este projeto está sob a licença MIT — veja [LICENSE](LICENSE).

---

<p align="center">
  <em>Desenvolvido como Tech Challenge da Fase 04 — FIAP PosTech Data Analytics</em>
</p>
