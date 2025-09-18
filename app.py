import streamlit as st
import pandas as pd

model = pd.read_pickle

st.markdown("# Descubra a felicidade")

redes_option = ['LinkedIn', 'Twitch', 'YouTube', 'Instagram', 'Amigos',
       'Twitter / X', 'Outra rede social']
redes = st.selectbox("Como conheceu o Téo Me Why?", options=redes_option)

cursos_opt = ['0', 
              '1', 
              '2', 
              '3',
              'Mais que 3']

cursos = st.selectbox("Quantos cursos acompanhou do Téo Me Why?",options=cursos_opt)

estado_opt = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", 
    "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", 
    "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
]

formacao_opt = [
    'Exatas', 'Biológicas', 'Humanas'
]

col1, col2, col3 =st.columns(3)

with col1:
    video_game = st.radio("Curte games?",['Sim',"Não"])
    board_game = st.radio("Curte jogos de tabuleiro?",['Sim',"Não"])
    idade = st.number_input("Sua idade", 12,100)

with col2:
    futebol = st.radio("Curte futebol?",['Sim',"Não"])
    f1 = st.radio("Curte jogos de fórmula 1?",['Sim',"Não"])
    estado = st.selectbox("Estado que mora atualmente",options=estado_opt)


with col3:
    livros = st.radio("Curte livros?",['Sim',"Não"])
    mma = st.radio("Curte jogos de MMA?",['Sim',"Não"])
    formacao = st.selectbox("Área de Formação", options=formacao_opt)

temp_dados_opt = [
    'Não atuo', 
    'De 0 a 6 meses', 
    'De 6 meses a 1 ano'
    'De 1 ano a 2 anos',
    'de 2 anos a 4 anos', 
    'Mais de 4 anos', 
]

temp_dados =  st.selectbox("Tempo que atua na área de dados", options=temp_dados_opt)

 
senioridade_opt = [
    'Iniciante', 
    'Júnior', 
    'Pleno', 
    'Sênior', 
    'Especialista', 
    'Coordenação', 
    'Gerência',
    'Diretoria', 
    'C-Level'
]

senioridade = st.selectbox("Posição da cadeira (senioridade)",options=senioridade_opt)

data = { 
        'Como conheceu o Téo Me Why?' :redes,
        'Quantos cursos acompanhou do Téo Me Why?': cursos, 
        'Curte games?':video_game,
        'Curte futebol?':futebol, 
        'Curte livros?': livros, 
        'Curte jogos de tabuleiro?': board_game,
        'Curte jogos de fórmula 1?': f1, 
        'Curte jogos de MMA?':mma, 
        'Idade':idade,
        'Estado que mora atualmente':estado, 
        'Área de Formação':formacao,
        'Tempo que atua na área de dados':temp_dados, 
        'Posição da cadeira (senioridade)':senioridade}

df = pd.DataFrame([data])

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)",
]

df =pd.get_dummies(df[dummy_vars]).astype(int)

df_template = pd.DataFrame(columns=['Como conheceu o Téo Me Why?_Amigos',
       'Como conheceu o Téo Me Why?_Instagram',
       'Como conheceu o Téo Me Why?_LinkedIn',
       'Como conheceu o Téo Me Why?_Outra rede social',
       'Como conheceu o Téo Me Why?_Twitch',
       'Como conheceu o Téo Me Why?_Twitter / X',
       'Como conheceu o Téo Me Why?_YouTube',
       'Quantos cursos acompanhou do Téo Me Why?_0',
       'Quantos cursos acompanhou do Téo Me Why?_1',
       'Quantos cursos acompanhou do Téo Me Why?_2',
       'Quantos cursos acompanhou do Téo Me Why?_3',
       'Quantos cursos acompanhou do Téo Me Why?_Mais que 3',
       'Estado que mora atualmente_AM', 'Estado que mora atualmente_BA',
       'Estado que mora atualmente_CE', 'Estado que mora atualmente_DF',
       'Estado que mora atualmente_ES', 'Estado que mora atualmente_GO',
       'Estado que mora atualmente_MA', 'Estado que mora atualmente_MG',
       'Estado que mora atualmente_MT', 'Estado que mora atualmente_PA',
       'Estado que mora atualmente_PB', 'Estado que mora atualmente_PE',
       'Estado que mora atualmente_PR', 'Estado que mora atualmente_RJ',
       'Estado que mora atualmente_RN', 'Estado que mora atualmente_RS',
       'Estado que mora atualmente_SC', 'Estado que mora atualmente_SP',
       'Área de Formação_Biológicas', 'Área de Formação_Exatas',
       'Área de Formação_Humanas',
       'Tempo que atua na área de dados_De 0 a 6 meses',
       'Tempo que atua na área de dados_De 1 ano a 2 anos',
       'Tempo que atua na área de dados_De 6 meses a 1 ano',
       'Tempo que atua na área de dados_Mais de 4 anos',
       'Tempo que atua na área de dados_Não atuo',
       'Tempo que atua na área de dados_de 2 anos a 4 anos',
       'Posição da cadeira (senioridade)_C-Level',
       'Posição da cadeira (senioridade)_Coordenação',
       'Posição da cadeira (senioridade)_Diretoria',
       'Posição da cadeira (senioridade)_Especialista',
       'Posição da cadeira (senioridade)_Gerência',
       'Posição da cadeira (senioridade)_Iniciante',
       'Posição da cadeira (senioridade)_Júnior',
       'Posição da cadeira (senioridade)_Pleno',
       'Posição da cadeira (senioridade)_Sênior', 'Curte games?',
       'Curte futebol?', 'Curte livros?', 'Curte jogos de tabuleiro?',
       'Curte jogos de fórmula 1?', 'Curte jogos de MMA?', 'Idade'])

df = pd.concat([df_template,df]).fillna(0)

st.dataframe(df)