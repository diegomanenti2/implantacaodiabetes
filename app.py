import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import util
import data_handler
import pickle

print("Abriu a pagina")

# verifica se a senha de acesso está correta
if not util.check_password():
    # se a senha estiver errada, para o processamento do app
    print("Usuario nao logado")
    st.stop()
    

print("Carregou a pagina")

# Aqui começa a estrutura do App que vai ser executado em produção (nuvem AWS)

# primeiro de tudo, carrega os dados do diabetes para um dataframe
dados = data_handler.load_data()

# carrega o modelo de predição já treinado e validado
model = pickle.load(open('./models/diabetes_model.pkl', 'rb')) 

# começa a estrutura da interface do sistema
st.title('App dos dados Diabetes')

data_analyses_on = st.toggle('Exibir análise dos dados')

if data_analyses_on:
    # essa parte é só um exmplo de que é possível realizar diversas visualizações e plotagens com o streamlit
    st.header('Dados do Diabetes - Dataframe')
    
    # exibe todo o dataframe dos dados do diabetes
    st.dataframe(dados)

    # plota um histograma das idades dos passageiros
    st.header('Histograma das idades')
    fig = plt.figure()
    plt.hist(dados['Age'], bins=10)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)

    # plota um gráfico de barras com a contagem dos sobreviventes
    st.header('Diabéticos')
    st.bar_chart(dados.Outcome.value_counts())
    
# daqui em diante vamos montar a inteface para capturar os dados de input do usuário para realizar a predição
# que vai identificar se um passageiro sobreviveu ou não
st.header('Preditor da Chance de ter Diabetes')

# ler as seguintes informações de input:
#Pregnancies - int
#Glucose - int
#BloodPressure - int
#SkinThickness - int
#Insulin - int
#BMI - float
#DiabetesPedigreeFunction - float
#Age - int

# essas foram as informações utilizadas para treinar o modelo
# assim, todas essas informações também devem ser passadas para o modelo realizar a predição

# define a linha 1 de inputs com 3 colunas
col1, col2, col3 = st.columns(3)

# captura a p_class do passageiro, com base na lista de classes dispobilizadas 
with col1:
    # Pregnancies: 
    pregnancies = st.number_input('Quantidade de Gestações (QTD)', step=1)

# captura Glucose
with col2:
    glocose = st.number_input('Glicose (mg/dL)', step=1)

    
# BloodPressure
with col3:
    # age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
    bloodPressure = st.number_input('Pressão Sanguinea (mm HG)', step=1)
    

# define a linha 2 de inputs, também com 3 colunas
col1, col2, col3 = st.columns(3)

# SkinThickness
with col1:
    skinThickness = st.number_input('Dobra Cutânea do Tríceps (mm)', step=1)

# Insulin
with col2:
    insulin = st.number_input('Insulina (UI)', step=1)

# BMI
with col3:
    bmi = st.number_input('Indice de Massa Corporal (IMC)', step=0.01)
    
    
# define a linha 3 de inputs, com 2 colunas
col1, col2,col3 = st.columns(3)

# DiabetesPedigreeFunction
with col1:
    diabetesPedigreeFunction = st.number_input('Probabilidade diabetes (0.00 - 1.00)', step=0.01)
    
# Age
with col2:
    age = st.number_input('Idade (anos)', step=1)


# define a linha 3 de inputs, com 2 colunas
col1, col2,col3 = st.columns(3)

# define o botão de verificar, que deverá ser pressionado para o sistema realizar a predição
with col1:
    submit = st.button('Verificar')

# armazena todos os dados do paciente nesse dict
paciente = {}
  
# verifica se o botão submit foi pressionado e se o campo Diabético está em cache
if submit or 'Diabético' in st.session_state:
    
    #Validação das entradas
    if pregnancies < 0 :
        st.error("Número de Gravidez inválido!")
        st.stop()
    
    if glocose <= 0 :
        st.error("Valor de Glicose invalido!")
        st.stop()

    if bloodPressure <= 0 :
        st.error("Valor de pressão sanguinea invalido!")
        st.stop()

    if skinThickness <= 0 :
        st.error("Valor de Dobra Cutânea do Tríceps invalido!")
        st.stop()

    if insulin <= 0 :
        st.error("Valor de Insulina invalido!")
        st.stop()
   
    if bmi <= 0 :
        st.error("Valor de IMC invalido!")
        st.stop()    

    if (diabetesPedigreeFunction<0 or diabetesPedigreeFunction>1  ) :
        st.error("Probabilidade diabetes invalido!")
        st.stop()
    
    if age < 0 :
        st.error("Idade invalida!")
        st.stop() 
 
    # seta todos os attrs do paciente. Nesse caso não temos tratamentos a fazer nem vriáveis categóricas.
    paciente = {
        "Pregnancies": pregnancies,
        "Glucose": glocose,
        "BloodPressure": bloodPressure,
        "SkinThickness": skinThickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetesPedigreeFunction,
        "Age": age
    }
    print(paciente)
    
    # converte o paciente para um pandas dataframe
    # isso é feito para igualar ao tipo de dado que foi utilizado para treinar o modelo
    values = pd.DataFrame([paciente])
    print(values) 

    # realiza a predição relacionadas a diabetes
    results = model.predict(values)
    print(results)
    
    # o modelo foi treinado para retornar uma lista com 0 ou 1, onde cada posição da lista indica se o paciente tem diabetes (1) ou não (0)
    # como estamos realizando a predição de somente um paciente por vez, o modelo deverá retornar somente um elemento na lista
    if len(results) == 1:
        # converte o valor retornado para inteiro
        diabetico = int(results[0])
        
        # verifica se o paciente tem diabetes
        if diabetico == 1:
            # se sim, exibe uma mensagem que o paciente tem diabetes
            st.subheader('Paciente DIABÉTICO! 😢')
            if 'Diabético' not in st.session_state:
                st.snow()
    
        else:
            # se não, exibe uma mensagem que o paciente não tem diabétes
            st.subheader('Paciente NÃO DIABÉTICO! 😃🙌🏻')
            if 'Diabético' not in st.session_state:
                st.balloons()
                
        
        # salva no cache da aplicação se o paciente tem diabetes
        st.session_state['Diabético'] = diabetico
    
    # verifica se existe um paciente e se já foi verificado se ele sobreviveu ou não
    if paciente and 'Diabético' in st.session_state:
        # se sim, pergunta ao usuário se a predição está certa e salva essa informação
        st.write("A predição está correta?")
        col1, col2, col3 = st.columns([1,1,5])
        with col1:
            correct_prediction = st.button('👍🏻')
        with col2:
            wrong_prediction = st.button('👎🏻')
        
        # exibe uma mensagem para o usuário agradecendo o feedback
        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback"
            if wrong_prediction:
                message += ", iremos usar esses dados para melhorar as predições"
            message += "."
            
            # adiciona no dict do passageiro se a predição está correta ou não
            if correct_prediction:
                paciente['CorrectPrediction'] = True
            elif wrong_prediction:
                paciente['CorrectPrediction'] = False
                
            # adiciona no dict do passageiro se ele sobreviveu ou não
            paciente['Diabético'] = st.session_state['Diabético']
            
            # escreve a mensagem na tela
            st.write(message)
            print(message)
            
            # salva a predição no JSON para cálculo das métricas de avaliação do sistema
            data_handler.save_prediction(paciente)
            
    st.write('')
    # adiciona um botão para permitir o usuário realizar uma nova análise
    col1, col2, col3 = st.columns(3)
    with col2:
        new_test = st.button('Iniciar Nova Análise')
        
        # se o usuário pressionar no botão e já existe um passageiro, remove ele do cache
        if new_test and 'Diabético' in st.session_state:
            del st.session_state['Diabético']
            st.rerun()

# calcula e exibe as métricas de avaliação do modelo
# aqui, somente a acurária está sendo usada
accuracy_predictions_on = st.toggle('Exibir acurácia')

if accuracy_predictions_on:
    # pega todas as predições salvas no JSON
    predictions = data_handler.get_all_predictions()
    # salva o número total de predições realizadas
    num_total_predictions = len(predictions)
    
    # calcula o número de predições corretas e salva os resultados conforme as predições foram sendo realizadas
    accuracy_hist = [0]
    # salva o numero de predições corretas
    correct_predictions = 0
    # percorre cada uma das predições, salvando o total móvel e o número de predições corretas
    for index, paciente in enumerate(predictions):
        total = index + 1
        if paciente['CorrectPrediction'] == True:
            correct_predictions += 1
            
        # calcula a acurracia movel
        temp_accuracy = correct_predictions / total if total else 0
        # salva o valor na lista de historico de acuracias
        accuracy_hist.append(round(temp_accuracy, 2)) 
    
    # calcula a acuracia atual
    accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0
    
    # exibe a acuracia atual para o usuário
    st.metric(label='Acurácia', value=round(accuracy, 2))
    # TODO: usar o attr delta do st.metric para exibir a diferença na variação da acurácia
    
    # exibe o histórico da acurácia
    st.subheader("Histórico de acurácia")
    st.line_chart(accuracy_hist)
    
