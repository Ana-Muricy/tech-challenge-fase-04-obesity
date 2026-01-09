import streamlit as st
import pandas as pd
import joblib

# ================================
# Configuração da página
# ================================
st.set_page_config(
    page_title="Predição de Nível de Obesidade",
    page_icon="🏥",
    layout="centered"
)

# ================================
# Título e descrição
# ================================
st.title("🏥 Sistema Preditivo de Obesidade")
st.markdown(
    """
    Esta aplicação utiliza **Machine Learning** para auxiliar a equipe médica
    na **predição do nível de obesidade** de um paciente, considerando dados físicos
    e comportamentais.
    """
)

st.divider()

# ================================
# Carregamento do modelo
# ================================
@st.cache_resource
def load_model():
    return joblib.load("models/obesity_model.pkl")

model = load_model()

# ================================
# Entrada de dados do usuário
# ================================
st.subheader("📋 Dados do Paciente")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.number_input("Idade", min_value=1, max_value=120, value=25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0)
    family_history = st.selectbox("Algum membro da família sofreu ou sofre de excesso de peso?", ["yes", "no"])

with col2:
    favc = st.selectbox("Você come alimentos altamente calóricos com frequência?", ["yes", "no"])
    fcvc = st.slider("Você costuma comer vegetais nas suas refeições? (1=raramente; 2=às vezes; 3=sempre)", 1.0, 3.0, 2.0)
    ncp = st.slider("Quantas refeições principais você faz diariamente? (1=uma refeição; 2=duas; 3=três; 4=quatro ou mais)", 1.0, 4.0, 3.0)
    caec = st.selectbox("Você come alguma coisa entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Você fuma?", ["yes", "no"])

col3, col4 = st.columns(2)

with col3:
    ch2o = st.slider("Qual seu consumo diário de água? (1=1L/dia; 2=1-2L/dia; 3= mais que 2L/dia)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Você monitora as calorias que ingere diariamente?", ["yes", "no"])
    faf = st.slider("Com que frequência você pratica atividade física (0=nenhuma; 1=1-2x/sem; 2=3-4x/sem; 3=5x/sem ou mais)", 0.0, 3.0, 1.0)

with col4:
    tue = st.slider("Quanto tempo você usa dispositivos tecnológicos como celular, videogame, televisão, computador e outros? (0=0-2h/dia; 1=3-5h/dia; 2=mais que 5h/dia)", 0.0, 2.0, 1.0)
    calc = st.selectbox("Com que frequência você bebe álcool?", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox(
        "Qual meio de transporte você costuma usar?",
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
    )

# ================================
# Predição
# ================================
st.divider()

if st.button("🔍 Prever Nível de Obesidade"):

    bmi = weight / (height ** 2)

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history": [family_history],
        "FAVC": [favc],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CAEC": [caec],
        "SMOKE": [smoke],
        "CH2O": [ch2o],
        "SCC": [scc],
        "FAF": [faf],
        "TUE": [tue],
        "CALC": [calc],
        "MTRANS": [mtrans],
        "BMI": [bmi]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"🩺 **Nível de obesidade previsto:** {prediction}")

    st.markdown(
        """
        **⚠️ Aviso:** Este sistema é uma ferramenta de apoio à decisão e **não substitui
        a avaliação clínica de um profissional de saúde**.
        """
    )

# ================================
# Rodapé
# ================================
st.divider()
st.caption("Tech Challenge – Fase 04 | Data Analytics")
