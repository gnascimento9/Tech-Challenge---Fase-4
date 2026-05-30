import json
from pathlib import Path

import joblib

_APP_DIR = Path(__file__).resolve().parent
_ROOT    = _APP_DIR.parent
import pandas as pd
import plotly.express as px
import streamlit as st

# CONFIG E CONSTANTES
st.set_page_config(
    page_title="Predição de Obesidade — Apoio Clínico",
    page_icon="🏥", layout="wide", initial_sidebar_state="expanded",
)

AZUL = "#2874A6"
AZUL_ESCURO = "#1A5276"
CINZA = "#9AA5B1"

ORDEM_EN = ["Insufficient_Weight", "Normal_Weight",
            "Overweight_Level_I", "Overweight_Level_II",
            "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]

LABEL_PT = dict(zip(ORDEM_EN, [
    "Abaixo do peso", "Peso normal", "Sobrepeso I", "Sobrepeso II",
    "Obesidade tipo I", "Obesidade tipo II", "Obesidade tipo III",
]))
ORDEM_PT = list(LABEL_PT.values())

CORES = dict(zip(ORDEM_PT, [
    "#7FB3D5", "#27AE60", "#F4D03F", "#E67E22",
    "#E74C3C", "#C0392B", "#7B241C",
]))

SIMNAO = {"yes": "Sim", "no": "Não"}
FREQ = {"no": "Não", "Sometimes": "Às vezes",
        "Frequently": "Frequentemente", "Always": "Sempre"}
TRANSPORTE = {"Automobile": "Automóvel", "Motorbike": "Moto", "Bike": "Bicicleta",
              "Public_Transportation": "Transp. público", "Walking": "A pé"}
GENERO = {"Female": "Feminino", "Male": "Masculino"}

NOVOS_NOMES = {
    'Gender': 'genero', 'Age': 'idade', 'Height': 'altura_m', 'Weight': 'peso_kg',
    'family_history': 'historia_familiar_sobrepeso',
    'FAVC': 'come_comida_calorica_freq', 'FCVC': 'freq_consumo_vegetais',
    'NCP': 'num_refeicoes_principais', 'CAEC': 'come_entre_refeicoes',
    'SMOKE': 'fumante', 'CH2O': 'consumo_agua_litros',
    'SCC': 'monitora_calorias', 'FAF': 'freq_atividade_fisica',
    'TUE': 'tempo_uso_dispositivos', 'CALC': 'freq_consumo_alcool',
    'MTRANS': 'meio_transporte', 'Obesity': 'nivel_obesidade',
}

# DADOS E MODELOS
@st.cache_resource
def carregar_artefatos():
    base = _ROOT / "models"          # repo/models/
    return (
        joblib.load(base / "modelo_A_com_imc.pkl"),
        joblib.load(base / "modelo_B_sem_imc.pkl"),
        joblib.load(base / "label_encoder.pkl"),
        json.loads((base / "metadata.json").read_text()),
    )

@st.cache_data
def carregar_dados():
    df = pd.read_csv(_ROOT / "data" / "raw" / "Obesity.csv").rename(columns=NOVOS_NOMES).drop_duplicates().reset_index(drop=True)
    for col in ['freq_consumo_vegetais', 'num_refeicoes_principais',
                'consumo_agua_litros', 'freq_atividade_fisica', 'tempo_uso_dispositivos']:
        df[col] = df[col].round().astype(int)
    df["imc"] = df["peso_kg"] / df["altura_m"] ** 2
    df["obesidade_pt"] = df["nivel_obesidade"].map(LABEL_PT)
    df["genero_pt"] = df["genero"].map(GENERO)
    df["historia_familiar_pt"] = df["historia_familiar_sobrepeso"].map(SIMNAO)
    df["transporte_pt"] = df["meio_transporte"].map(TRANSPORTE)
    df["lanche_pt"] = df["come_entre_refeicoes"].map(FREQ)
    return df

modelo_A, modelo_B, encoder, meta = carregar_artefatos()
df = carregar_dados()

# HELPERS DE UI
def secao(rotulo, titulo):
    st.markdown(
        f"<div style='border-left:3px solid {AZUL}; padding:4px 0 4px 18px; margin:28px 0 12px 0;'>"
        f"<p style='color:{CINZA}; font-size:12px; letter-spacing:1.5px; text-transform:uppercase; margin:0 0 4px 0;'>{rotulo}</p>"
        f"<h3 style='color:{AZUL}; margin:0; font-weight:600;'>{titulo}</h3></div>",
        unsafe_allow_html=True,
    )

def tema(fig, legenda=None):
    fig.update_layout(
        font=dict(family="Inter, Arial, sans-serif", size=12, color=CINZA),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=40),
        legend=dict(title=dict(text=legenda or ""), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(154,165,177,0.15)", linecolor="rgba(154,165,177,0.35)")
    fig.update_yaxes(gridcolor="rgba(154,165,177,0.15)", linecolor="rgba(154,165,177,0.35)")
    return fig

def barra_empilhada_pct(df_, col_x, label_x, ordem_x=None, mapa_cor=None, legenda="Classe"):
    ct = pd.crosstab(df_[col_x], df_["obesidade_pt"], normalize="index")
    if ordem_x is not None:
        ct = ct.reindex(index=ordem_x)
    ct = (ct.reindex(columns=ORDEM_PT) * 100).reset_index()
    longo = ct.melt(id_vars=col_x, var_name=legenda, value_name="%")
    fig = px.bar(
        longo, x=col_x, y="%", color=legenda, barmode="stack",
        color_discrete_map=mapa_cor or CORES,
        category_orders={legenda: ORDEM_PT, **({col_x: ordem_x} if ordem_x else {})},
        labels={col_x: label_x, "%": "Proporção (%)"},
    )
    return tema(fig, legenda=legenda)

def card_etapa(n, titulo, desc):
    return (
        f"<div style='display:flex; gap:16px; align-items:flex-start;"
        f" background-color:rgba(40,116,166,0.06); border-left:2px solid {AZUL};"
        f" padding:16px 18px; margin-bottom:14px; border-radius:0 6px 6px 0;'>"
        f"<div style='color:{AZUL}; font-size:28px; font-weight:700; line-height:1; min-width:36px; opacity:0.85;'>{n}</div>"
        f"<div style='flex:1;'>"
        f"<p style='color:#E4E9EE; font-size:15px; font-weight:600; margin:0 0 6px 0; line-height:1.3;'>{titulo}</p>"
        f"<p style='color:#D1D9E0; font-size:14.5px; margin:0; line-height:1.7;'>{desc}</p>"
        f"</div></div>"
    )

# PÁGINA 1 — VISÃO GERAL
def pagina_visao_geral():
    st.markdown(
        f"<h1 style='color:{AZUL}; margin-bottom:6px; font-weight:600;'>Sistema de Apoio à Decisão Clínica</h1>"
        f"<p style='color:{CINZA}; font-size:17px; margin-top:0; margin-bottom:24px;'>"
        "Predição de níveis de obesidade a partir de hábitos do paciente</p>",
        unsafe_allow_html=True,
    )

    secao("Contexto", "O Desafio")
    st.markdown(
        """
        A obesidade é uma condição multifatorial — envolve **genética, ambiente e
        comportamento** — e sua prevalência cresce globalmente. Hospitais precisam
        de ferramentas que apoiem a **identificação precoce de pacientes em risco**,
        antes que a condição evolua para quadros graves.

        Este projeto entrega um sistema preditivo que auxilia a equipe médica na
        **triagem preventiva de obesidade** a partir de dados sobre hábitos
        alimentares, atividade física e estilo de vida do paciente.
        """
    )

    secao("Metodologia", "Abordagem Adotada")
    etapas = [
        ("01", "Entendimento do problema",
         "Análise de 2.111 registros de pacientes, com 16 variáveis sobre hábitos alimentares, atividade física e estilo de vida."),
        ("02", "Preparação dos dados",
         "Remoção de duplicatas, padronização de variáveis ordinais e tradução das colunas para português."),
        ("03", "Engenharia de atributos",
         "Criação de variáveis derivadas (IMC, faixa etária) e codificação apropriada para classificação multiclasse."),
        ("04", "Treinamento e validação",
         "Comparação de Regressão Logística, Random Forest e XGBoost com validação cruzada estratificada (5 folds)."),
        ("05", "Seleção do modelo",
         "Avaliação por acurácia e F1-macro. Modelo selecionado supera a meta de 75% de acurácia do desafio."),
    ]
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("".join(card_etapa(*e) for e in etapas[:3]), unsafe_allow_html=True)
    with col_b:
        st.markdown("".join(card_etapa(*e) for e in etapas[3:]), unsafe_allow_html=True)

    secao("Resultado", "Indicadores do Projeto")
    c1, c2, c3, _ = st.columns(4)
    c1.metric("Pacientes analisados", f"{len(df):,}")
    c2.metric("Variáveis preditoras", len(meta["modelo_B"]["features"]))
    c3.metric("Classes de obesidade", len(meta["classes"]))

    secao("Base de Dados", "Distribuição da População Estudada")
    counts = df["obesidade_pt"].value_counts().reindex(ORDEM_PT).reset_index()
    counts.columns = ["Classe", "Quantidade"]
    fig = px.bar(
        counts, x="Classe", y="Quantidade", color="Classe",
        color_discrete_map=CORES, category_orders={"Classe": ORDEM_PT}, text="Quantidade",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Classe de Obesidade", yaxis_title="Número de Pacientes")
    tema(fig)
    st.plotly_chart(fig, use_container_width=True)


# PÁGINA 2 — PAINEL ANALÍTICO
def pagina_painel():
    st.markdown(
        f"<h1 style='color:{AZUL}; margin-bottom:6px; font-weight:600;'>Painel Analítico</h1>"
        f"<p style='color:{CINZA}; font-size:15px; margin-top:0; margin-bottom:18px;'>"
        f"Insights populacionais para a equipe médica</p>",
        unsafe_allow_html=True,
    )

    # Filtros
    st.sidebar.markdown(
        f"<h4 style='color:{AZUL}; margin-bottom:10px; font-weight:600;'>Filtros do painel</h4>",
        unsafe_allow_html=True,
    )
    gen_opts = sorted(df["genero_pt"].unique())
    gen_sel = st.sidebar.multiselect("Gênero", gen_opts, gen_opts, key="flt_gen")
    idade_min, idade_max = int(df["idade"].min()), int(df["idade"].max())
    age_range = st.sidebar.slider("Faixa etária (anos)", idade_min, idade_max,
                                  (idade_min, idade_max), key="flt_idade")
    hist_sel = st.sidebar.multiselect("Histórico familiar de excesso de peso",
                                       ["Sim", "Não"], ["Sim", "Não"], key="flt_hist")
    faf_min, faf_max = st.sidebar.select_slider(
        "Frequência de atividade física",
        options=[0, 1, 2, 3], value=(0, 3),
        format_func=lambda v: {0: "Nenhuma", 1: "1–2x/sem", 2: "3–4x/sem", 3: "5x+/sem"}[v],
        key="flt_faf",
    )

    df_f = df[
        df["genero_pt"].isin(gen_sel)
        & df["idade"].between(*age_range)
        & df["historia_familiar_pt"].isin(hist_sel)
        & df["freq_atividade_fisica"].between(faf_min, faf_max)
    ]

    st.markdown(
        f"<div style='background-color:rgba(40,116,166,0.12); padding:12px 18px;"
        f" border-left:3px solid {AZUL}; border-radius:4px; margin-bottom:18px;'>"
        f"<span style='color:{CINZA}; font-size:13px;'>Amostra atual: </span>"
        f"<span style='color:{AZUL}; font-weight:600; font-size:15px;'>{len(df_f):,} pacientes</span>"
        f"<span style='color:{CINZA}; font-size:13px;'> selecionados pelos filtros aplicados</span></div>",
        unsafe_allow_html=True,
    )
    if df_f.empty:
        st.warning("Nenhum paciente corresponde aos filtros. Ajuste a barra lateral.")
        return

    # KPIs
    obeso = df_f["nivel_obesidade"].str.startswith("Obesity").mean() * 100
    sobre = df_f["nivel_obesidade"].str.startswith("Overweight").mean() * 100
    fam = (df_f["historia_familiar_sobrepeso"] == "yes").mean() * 100
    inativos = (df_f["freq_atividade_fisica"] == 0).mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Com obesidade", f"{obeso:.1f}%")
    k2.metric("Com sobrepeso", f"{sobre:.1f}%")
    k3.metric("Com histórico familiar", f"{fam:.1f}%")
    k4.metric("Sem atividade física", f"{inativos:.1f}%")
    st.markdown("---")

    def box_por_classe(y, label_y, x_rot=30):
        d = df_f.copy()
        d["obesidade_pt"] = pd.Categorical(d["obesidade_pt"], categories=ORDEM_PT, ordered=True)
        fig = px.box(
            d.sort_values("obesidade_pt"), x="obesidade_pt", y=y, color="obesidade_pt",
            color_discrete_map=CORES, category_orders={"obesidade_pt": ORDEM_PT},
            labels={"obesidade_pt": "Classe", y: label_y},
        )
        fig.update_layout(showlegend=False, xaxis_title="")
        fig.update_xaxes(tickangle=x_rot)
        return tema(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### IMC por classe de obesidade")
        st.plotly_chart(box_por_classe("imc", "IMC (kg/m²)"), use_container_width=True)
    with c2:
        st.markdown("##### Idade vs. IMC")
        fig = px.scatter(
            df_f, x="idade", y="imc", color="obesidade_pt",
            color_discrete_map=CORES, category_orders={"obesidade_pt": ORDEM_PT},
            opacity=0.7,
            labels={"idade": "Idade (anos)", "imc": "IMC (kg/m²)", "obesidade_pt": "Classe"},
        )
        st.plotly_chart(tema(fig, legenda="Classe"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Histórico familiar × obesidade")
        ct = pd.crosstab(df_f["obesidade_pt"], df_f["historia_familiar_pt"], normalize="index")
        ct = (ct.reindex(ORDEM_PT) * 100).reset_index()
        longo = ct.melt(id_vars="obesidade_pt", var_name="Histórico familiar", value_name="%")
        fig = px.bar(
            longo, x="obesidade_pt", y="%", color="Histórico familiar", barmode="stack",
            color_discrete_map={"Sim": AZUL_ESCURO, "Não": "#AED6F1"},
            category_orders={"obesidade_pt": ORDEM_PT},
            labels={"obesidade_pt": "Classe", "%": "Proporção (%)"},
        )
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(tema(fig, legenda="Histórico familiar"), use_container_width=True)
    with c4:
        st.markdown("##### Atividade física por classe")
        st.plotly_chart(box_por_classe("freq_atividade_fisica",
                                        "Frequência (0=nenhuma · 3=5x+/sem)"),
                        use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("##### Meio de transporte × obesidade")
        st.plotly_chart(barra_empilhada_pct(df_f, "transporte_pt", "Meio de transporte"),
                        use_container_width=True)
    with c6:
        st.markdown("##### Lanche entre refeições × obesidade")
        ordem_lanche = ["Não", "Às vezes", "Frequentemente", "Sempre"]
        st.plotly_chart(barra_empilhada_pct(df_f, "lanche_pt", "Frequência de lanche",
                                             ordem_x=ordem_lanche),
                        use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<div style='background-color:rgba(40,116,166,0.10); padding:24px;"
        f" border-radius:6px; border-left:3px solid {AZUL}; margin-top:8px;'>"
        f"<p style='color:{CINZA}; font-size:12px; letter-spacing:1.5px;"
        f" text-transform:uppercase; margin:0 0 4px 0;'>Resumo Clínico</p>"
        f"<h4 style='color:{AZUL}; margin:0 0 14px 0; font-weight:600;'>"
        f"Principais insights para a equipe médica</h4>"
        f"<ul style='color:#D1D9E0; line-height:1.8; margin:0; padding-left:20px; font-size:14.5px;'>"
        f"<li><strong style='color:#E4E9EE;'>Histórico familiar</strong> é o fator não comportamental mais determinante para risco de obesidade.</li>"
        f"<li><strong style='color:#E4E9EE;'>Atividade física baixa</strong> está fortemente associada às classes mais graves de obesidade.</li>"
        f"<li><strong style='color:#E4E9EE;'>Consumo frequente de alimentos calóricos</strong> combinado com <strong style='color:#E4E9EE;'>lanches frequentes</strong> ampliam significativamente o risco.</li>"
        f"<li><strong style='color:#E4E9EE;'>Transporte ativo</strong> (a pé ou bicicleta) tem efeito protetor; <strong style='color:#E4E9EE;'>uso de automóvel</strong> é fator de risco associado.</li>"
        f"</ul></div>",
        unsafe_allow_html=True,
    )


# PÁGINA 3 — SISTEMA PREDITIVO
def pagina_preditivo():
    st.markdown(
        f"<h1 style='color:{AZUL}; margin-bottom:6px; font-weight:600;'>Sistema Preditivo</h1>"
        f"<p style='color:{CINZA}; font-size:15px; margin-top:0; margin-bottom:18px;'>"
        f"Avaliação individual de paciente</p>",
        unsafe_allow_html=True,
    )

    modelo_escolhido = st.radio(
        "Modelo a utilizar",
        ["B — Modelo Baseado em Hábitos", "A — Modelo Baseado no IMC"],
        horizontal=True, key="rad_modelo",
        help=("Modelo B usa apenas hábitos — ideal para triagem antes da consulta. "
              "Modelo A inclui peso/altura, alcançando precisão quase perfeita por "
              "aprender a fórmula do IMC, mas com pouco valor clínico adicional."),
    )
    usar_A = modelo_escolhido.startswith("A")
    st.markdown("Preencha os dados do paciente. O sistema retornará o **nível de obesidade estimado**.")

    INV_SIMNAO = {v: k for k, v in SIMNAO.items()}
    INV_GENERO = {v: k for k, v in GENERO.items()}
    INV_FREQ = {v: k for k, v in FREQ.items()}
    INV_TRANSP = {v: k for k, v in TRANSPORTE.items()}

    with st.form("form_paciente"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### Perfil do paciente")
            gen_in = st.selectbox("Gênero", list(INV_GENERO))
            age_in = st.slider("Idade", 14, 70, 25)
            fam_in = st.selectbox("Histórico familiar de excesso de peso", list(INV_SIMNAO))
            if usar_A:
                st.markdown("**Medidas**")
                alt_in = st.slider("Altura (m)", 1.40, 2.00, 1.70, 0.01)
                pes_in = st.slider("Peso (kg)", 35.0, 200.0, 70.0, 0.5)
        with c2:
            st.markdown("##### Hábitos alimentares")
            favc_in = st.selectbox("Come alimentos calóricos com frequência?", list(INV_SIMNAO))
            fcvc_in = st.slider("Consumo de vegetais (1=raro · 3=sempre)", 1, 3, 2)
            ncp_in = st.slider("Refeições principais por dia", 1, 4, 3)
            caec_in = st.selectbox("Come entre as refeições?", list(INV_FREQ))
            calc_in = st.selectbox("Consumo de álcool", list(INV_FREQ))
        with c3:
            st.markdown("##### Estilo de vida")
            smk_in = st.selectbox("Fuma?", list(INV_SIMNAO))
            agua_in = st.slider("Água por dia (1=<1L · 3=>2L)", 1, 3, 2)
            scc_in = st.selectbox("Monitora as calorias ingeridas?", list(INV_SIMNAO))
            faf_in = st.slider("Atividade física (0=nenhuma · 3=5x+/sem)", 0, 3, 1)
            tue_in = st.slider("Tempo em dispositivos (0=0–2h · 2=>5h)", 0, 2, 1)
            mtr_in = st.selectbox("Meio de transporte habitual", list(INV_TRANSP))
        submit = st.form_submit_button(
            f"Realizar predição (Modelo {'A' if usar_A else 'B'})",
            use_container_width=True, type="primary",
        )

    if not submit:
        return

    base = {
        "genero": INV_GENERO[gen_in],
        "idade": age_in,
        "historia_familiar_sobrepeso": INV_SIMNAO[fam_in],
        "come_comida_calorica_freq": INV_SIMNAO[favc_in],
        "freq_consumo_vegetais": fcvc_in,
        "num_refeicoes_principais": ncp_in,
        "come_entre_refeicoes": INV_FREQ[caec_in],
        "fumante": INV_SIMNAO[smk_in],
        "consumo_agua_litros": agua_in,
        "monitora_calorias": INV_SIMNAO[scc_in],
        "freq_atividade_fisica": faf_in,
        "tempo_uso_dispositivos": tue_in,
        "freq_consumo_alcool": INV_FREQ[calc_in],
        "meio_transporte": INV_TRANSP[mtr_in],
    }
    if usar_A:
        base["altura_m"] = alt_in
        base["peso_kg"] = pes_in
        base["imc"] = pes_in / alt_in ** 2
        entrada = pd.DataFrame([base])[meta["modelo_A"]["features"]]
        modelo_usado = modelo_A
    else:
        entrada = pd.DataFrame([base])[meta["modelo_B"]["features"]]
        modelo_usado = modelo_B

    pred = modelo_usado.predict(entrada)[0]
    classe = encoder.inverse_transform([pred])[0]
    classe_pt = LABEL_PT[classe]
    cor = CORES[classe_pt]

    st.markdown("---")
    st.markdown(
        f"<div style='background-color:{cor}; padding:25px; border-radius:10px;"
        f" text-align:center; color:white;'>"
        f"<p style='margin:0; font-size:16px; opacity:0.9;'>Resultado da predição</p>"
        f"<h1 style='margin:8px 0; color:white;'>{classe_pt}</h1></div>",
        unsafe_allow_html=True,
    )


# SIDEBAR
st.sidebar.markdown(
    f"<h2 style='color:{AZUL}; margin-bottom:0; font-weight:600;'>Apoio Clínico</h2>"
    f"<p style='color:{CINZA}; margin-top:4px; font-size:13px;'>Predição de Obesidade</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

PAGINAS = {
    "Visão geral":       pagina_visao_geral,
    "Painel analítico":  pagina_painel,
    "Sistema preditivo": pagina_preditivo,
}

pagina = st.sidebar.radio("Navegação", list(PAGINAS.keys()), key="nav")
st.sidebar.markdown("---")
st.sidebar.caption("Tech Challenge · Fase 04")
st.sidebar.caption("Pós Tech FIAP — Data Analytics")
st.sidebar.caption("**Aluno:** Gustavo Henrique Lisboa do Nascimento")

PAGINAS[pagina]()