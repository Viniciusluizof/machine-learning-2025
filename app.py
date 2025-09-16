import streamlit as st

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

"""
features = [
    redes, 
    cursos,
    video_game, 
    futebol, 
    livros, 
    board_game, 
    f1, 
    mma, 
    idade, 
    estado, 
    formacao,
    temp_dados,
    senioridade]"""