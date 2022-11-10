from tkinter import filedialog
from turtle import filling
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

### QUATRO INPUTS:

### PERCENTUAL MAIS RECENTE DAQUELE CALIBRE AMOSTRADO 
### MEDIA DO PESO DAQUELE CALIBRE NA AMOSTRAGEM MAIS RECENTE
### DIAS ATE A EMBALAGEM DA AMOSTRAGEM MAIS RECENTE
### ORDEM DO CONTROLE


@st.experimental_memo
def get_data():
    url = 'http://sia:3000/backend/busca_generica/buscaGenerica?view=MGCLI.DXDW_HISTORICO_CALIBRES'
    dataframe = pd.read_json(url)
    dataframe.isnull().sum()
    dataframe = dataframe.dropna()
    return dataframe
dataframe = get_data()

filtro_cal = dataframe['VALOR_CALIBRE'] != 'CALIBRE_11'
dataframe = dataframe[filtro_cal]

filtro_cal = dataframe['VALOR_CALIBRE'] != 'CALIBRE_13'
dataframe = dataframe[filtro_cal]

filtro_cal = dataframe['VALOR_CALIBRE'] != 'CALIBRE_4'
dataframe = dataframe[filtro_cal]

dataframe['talhao_calibre'] =   dataframe['VALOR_CALIBRE']  + dataframe['TALH_ST_DESCRICAO'] 



### TENHO QUE JUNTAR AS INFOS COM O 'dataframe' 

###procv de talhao calibre


@st.experimental_memo
def get_data2():
    url_amostragem = 'http://sia:3000/backend/busca_generica/buscaGenerica?view=MGAGR.AGR_VW_DX_CALIBRES_CAMPO%20WHERE%201=1%20AND%20SYSDATE%20-%20to_date(DATA,%27yyyy-mm-dd%27)%20%3C=%2060%20AND%20DATA%20IS%20NOT%20NULL'
    df_amostragem_1 = pd.read_json(url_amostragem)
    return df_amostragem_1

df_amostragem_1 = get_data2()


dados = df_amostragem_1.groupby(['TALHAO','CALIBRE'])['DATA'].max()
dados2 = pd.DataFrame(dados)
dados2 = dados2.reset_index()

dados2['coluna_merg'] = dados2['TALHAO'] + dados2['CALIBRE'] + dados2['DATA']
df_amostragem_1['coluna_merge'] = df_amostragem_1['TALHAO']+ df_amostragem_1['CALIBRE']+df_amostragem_1['DATA']
dataset_merge = df_amostragem_1.merge(dados2, left_on = 'coluna_merge', right_on = 'coluna_merg')


df_amostragem_2 = dataset_merge[['TALHAO_x','FRUTO','PESO','CALIBRE_x']]

df_amostragem_1_piv = pd.pivot_table(dataset_merge, values = ['FRUTO','PESO'], index=['CALIBRE_x','TALHAO_x'],
                aggfunc={'FRUTO': np.sum,
                            'PESO': np.mean})
df_amostragem_1_piv = df_amostragem_1_piv.reset_index()

amostragem_percents = df_amostragem_1_piv.groupby(['TALHAO_x','CALIBRE_x'])['FRUTO'].sum() / df_amostragem_1_piv.groupby(['TALHAO_x'])['FRUTO'].sum() 
amostragem_percents = pd.DataFrame(amostragem_percents)
amostragem_percents = amostragem_percents.reset_index()




amostragem_percents['VALOR_CALIBRE'] = 'CALIBRE_' + amostragem_percents['CALIBRE_x']
amostragem_percents['talhao_calibre'] = amostragem_percents['VALOR_CALIBRE'] + amostragem_percents['TALHAO_x']





df_amostragem_1_piv['VALOR_CALIBRE'] = 'CALIBRE_' + df_amostragem_1_piv['CALIBRE_x']
df_amostragem_1_piv['talhao_calibre'] = df_amostragem_1_piv['VALOR_CALIBRE'] + df_amostragem_1_piv['TALHAO_x']




dataset_merge1 = dataframe.merge(amostragem_percents, left_on = 'talhao_calibre', right_on = 'talhao_calibre')
dataset_merge2 = dataset_merge1.merge(df_amostragem_1_piv, left_on = 'talhao_calibre', right_on = 'talhao_calibre')

dataset_merge3 = dataset_merge2[['TALH_ST_DESCRICAO','CPROC_IN_CODIGO', 
'DATA_EMBALAGEM', 'ORDEM','VALOR_CALIBRE_x','CALIBRE','talhao_calibre','FRUTO_x','PESO','VARIEDADE']]

dataset_merge3.rename(columns = {'FRUTO_x':'Perc_cal_amostr'}, inplace = True)

### AQUI JA TENHO O PESO E O PERCENTUAL DO CALIBRE E A ORDEM

### FALTA AGORA OS DIAS DA AMOSTRAGEM ATE EMBALAGEM
dados2['VALOR_CALIBRE'] = 'CALIBRE_' + dados2['CALIBRE']

dados2['talhao_calibre'] = dados2['VALOR_CALIBRE'] + dados2['TALHAO']


dataset_merge4 = dataset_merge3.merge(dados2, left_on = 'talhao_calibre', right_on = 'talhao_calibre')

dataset_merge4 = dataset_merge4[['TALH_ST_DESCRICAO','CPROC_IN_CODIGO', 
'DATA_EMBALAGEM', 'ORDEM','VALOR_CALIBRE_x','CALIBRE_x','talhao_calibre','Perc_cal_amostr','PESO','DATA','VARIEDADE']]
dataset_merge4.rename(columns = {'VALOR_CALIBRE_x':'REAL','DATA':'DATA_AMOSTRAGEM'}, inplace = True)


from datetime import datetime, timedelta
dataset_merge4['DATA_EMBALAGEM'] = pd.to_datetime(dataset_merge4['DATA_EMBALAGEM'], format="%Y/%m/%d")
dataset_merge4['DATA_AMOSTRAGEM'] = pd.to_datetime(dataset_merge4['DATA_AMOSTRAGEM'], format="%Y/%m/%d")


dataset_merge4['DIAS_ATE_EMBALA'] = (dataset_merge4['DATA_EMBALAGEM'] - dataset_merge4['DATA_AMOSTRAGEM'])
dataset_merge4['DIAS_ATE_EMBALA'] = dataset_merge4['DIAS_ATE_EMBALA'] / np.timedelta64(1,'D')
dataset_merge4['DIAS_ATE_EMBALA'] = dataset_merge4['DIAS_ATE_EMBALA'].astype(int)







dataframe_final = dataset_merge4[['TALH_ST_DESCRICAO','CPROC_IN_CODIGO','DATA_EMBALAGEM','DATA_AMOSTRAGEM','REAL','CALIBRE_x','ORDEM','Perc_cal_amostr','DIAS_ATE_EMBALA','PESO','VARIEDADE']]



# dataframe_final = dataframe_final[dataframe_final['ORDEM'].between(1, 11)]
# dataframe_final = dataframe_final[dataframe_final['Perc_cal_amostr'].between(0, 0.42)]
dataframe_final = dataframe_final[dataframe_final['DIAS_ATE_EMBALA'].between(5, 30)]
# dataframe_final = dataframe_final[dataframe_final['PESO'].between(0, 791)]

dataframe_final = dataframe_final[dataframe_final['CALIBRE_x'].between(0, 0.5)]


dataframe_final['ERROR_AMOSTRA'] = abs(dataframe_final['Perc_cal_amostr'] - dataframe_final['CALIBRE_x'])


dados_model = dataframe_final[['ORDEM','Perc_cal_amostr','DIAS_ATE_EMBALA','PESO']]


with open('Modelo_final.pkl', 'rb') as f:
    modelo = pickle.load(f)

pred_new = modelo.predict(dados_model)


dataframe_final['Pred'] = pred_new
dataframe_final['Erro'] =  abs(dataframe_final['CALIBRE_x'] - dataframe_final['Pred'])

dataframe_final['Erro_neg'] =  dataframe_final['Pred'] - dataframe_final['CALIBRE_x'] 





dataframe_final = dataframe_final[dataframe_final['Erro'].between(0, 0.5)]


def check(dataframe_final):
    if dataframe_final['Erro'] >  0.11:
        return 'ERROU'
    else:
        return 'ACERTOU'

dataframe_final['Check'] = dataframe_final.apply(check, axis = 1)



def check(dataframe_final):
    if dataframe_final['ERROR_AMOSTRA'] >  0.11:
        return 'ERROU'
    else:
        return 'ACERTOU'

dataframe_final['Check Amostra'] = dataframe_final.apply(check, axis = 1)

### CRIAR AGR OS ST FILTROS

cnt = len(dataframe_final['CPROC_IN_CODIGO'].value_counts())


st.error(f'### Desempenho Geral - Novos dados ({cnt} controles)')


coluna1, coluna2 = st.columns(2)

with coluna1:
    st.error('Acurácia com threshold de 10%')
    fig = px.bar(dataframe_final, x = 'REAL', color = 'Check', category_orders = {'REAL':['CALIBRE_5','CALIBRE_6','CALIBRE_7','CALIBRE_8','CALIBRE_9','CALIBRE_10','CALIBRE_12','CALIBRE_14'], 'Check':['ACERTOU','ERROU']})
    st.plotly_chart(fig)


with coluna2:
    st.error('Probabilidade de erros nas previsões por Calibre')
    # fig = px.bar(dataframe_final, x = 'REAL', color = 'Check Amostra', category_orders = {'REAL':['CALIBRE_5','CALIBRE_6','CALIBRE_7','CALIBRE_8','CALIBRE_9','CALIBRE_10','CALIBRE_12','CALIBRE_14'], 'Check Amostra':['ACERTOU','ERROU']})
    
    # st.plotly_chart(fig)

    fig = px.histogram(dataframe_final, x = 'Erro_neg', color = 'REAL', marginal = 'violin', category_orders = {'REAL':['CALIBRE_5','CALIBRE_6','CALIBRE_7','CALIBRE_8','CALIBRE_9','CALIBRE_10','CALIBRE_12','CALIBRE_14'], 'Check':['ACERTOU','ERROU']})
    fig.add_vline(0.11,line_color = 'red', line_dash="dot",  annotation_text=" + 10 % ", annotation_font_color="red")
    fig.add_vline(-0.11,line_color = 'red', line_dash="dot",  annotation_text=" - 10 % ", annotation_font_color="red")
    fig
    
    st.write("""
    - Os erros para todos os calibres se concentram em torno de zero e dentro da tolerancia de 10%
        - Exceto calibre 5 que ficou em torno de 16% 
    
    """)
    st.write('___')


coluna1, coluna2 = st.columns(2)


with coluna1:

    st.info("Percentuais dos totais de percentuais de cada calibre")
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(labels=dataframe_final['REAL'] , pull=[0, 0, 0, 0,0, 0.2, 0, 0], marker_colors = px.colors.cyclical.Twilight, values=dataframe_final['Pred'], name="Previsto"),
                1, 1)
    fig.add_trace(go.Pie(labels=dataframe_final['REAL'], pull=[0, 0, 0, 0,0,0.2, 0, 0, 0], values=dataframe_final['CALIBRE_x'], name="Real"),
                1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name", marker=dict(line = dict(color = '#000000', width = 1)))
    
    fig.update_layout( 
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Pred', x=0.17, y=0.5, font_size=20, showarrow=False),
                    dict(text='Real', x=0.80, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig)

with coluna2:

    st.info("Médias previstas por calibre")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x = dataframe_final['REAL'],  marker_color='darkblue', y = dataframe_final['CALIBRE_x'], histfunc = 'avg',name = 'Real'))

    fig.add_trace(go.Histogram(x = dataframe_final['REAL'],   marker_color='royalblue',y = dataframe_final['Pred'], histfunc = 'avg', name = 'Previsto'))

    # fig.add_trace(go.Histogram(x = dataframe_final['REAL'],   marker_color='green',y = dataframe_final['Perc_cal_amostr'], histfunc = 'avg', name = 'P_AMOS'))

    fig.update_traces(marker_line_color = 'rgb(0,0,0)',marker_line_width = 1.5)

    fig.update_yaxes(range = [0,0.5])

    fig.update_layout( height = 400, width = 650)
    fig.update_xaxes(categoryorder='array', categoryarray= ['CALIBRE_5','CALIBRE_6','CALIBRE_7','CALIBRE_8', 'CALIBRE_9','CALIBRE_10','CALIBRE_12','CALIBRE_14'])

    st.plotly_chart(fig)

    
coluna1, coluna2 = st.columns([1,0.01])

with coluna1:

    st.info('Erro do modelo por influência do erro da amostragem')
    fig = go.Figure()

    fig.add_trace(go.Bar(y = dataframe_final['Erro']*100, name = 'Error'))
    fig.add_trace(go.Scatter(y = dataframe_final['ERROR_AMOSTRA']*100, name = 'DIF AMOSTRA'))

    fig.add_hline(10, line_color = 'darkblue', line_dash="dot",  annotation_text=" + 10 % ",annotation_font_color="darkblue")

    fig.update_layout( title = 'Caliber (%) forecasting')
    fig.update_layout(height = 500, width = 1400)
    fig


coluna1, coluna2 = st.columns([1,0.1])

st.error('### Análise das previsões nível Talhão:Controle')

dd = dataframe_final['TALH_ST_DESCRICAO'].value_counts()
ee = pd.DataFrame(dd)
ee = ee.drop(columns = ['TALH_ST_DESCRICAO'])
ee = ee.reset_index()
lista_talhoes = ee
st.write('___')
colu1, colu2, colu3 = st.columns([0.3,0.3,1])

input_talhao = colu1.selectbox('Escolha um talhão:', lista_talhoes, key = 'Escolha de talhao')


filtro_talhao = dataframe_final['TALH_ST_DESCRICAO'] == input_talhao
dataframe_final = dataframe_final[filtro_talhao]




dd = dataframe_final['CPROC_IN_CODIGO'].value_counts()
ee = pd.DataFrame(dd)
ee = ee.drop(columns = ['CPROC_IN_CODIGO'])
ee = ee.reset_index()
lista_controles= ee


input_control = colu2.selectbox('Escolha um controle:', options = lista_controles, key = 'Escolha de controle')



filtro_control = dataframe_final['CPROC_IN_CODIGO'] == input_control
dataframe_final = dataframe_final[filtro_control]

variety = dataframe_final.reset_index()
variety = variety['VARIEDADE'][0]

colu3.write(' ')
colu3.write(' ')
colu3.write(' ')

colu3.write(f'Variedade do talhão: {variety}')

#dataframe_final.query('CPROC_IN_CODIGO in @input_control')

st.write('___')
colunaxx, coluna1, colunax, coluna2, colunaxxx  = st.columns([0.01,1,0.1,1,0.1])

with coluna1:
    
    st.info('Percentuais dos totais de percentuais de cada calibre')

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(labels=dataframe_final['REAL'] , pull=[0, 0, 0, 0,0, 0.2, 0, 0], marker_colors = px.colors.cyclical.Twilight, values=dataframe_final['Pred'], name="Previsto"),
                1, 1)
    fig.add_trace(go.Pie(labels=dataframe_final['REAL'], pull=[0, 0, 0, 0,0,0.2, 0, 0, 0], values=dataframe_final['CALIBRE_x'], name="Real"),
                1, 2)

    fig.update_traces(hole=.4, hoverinfo="label+percent+name", marker=dict(line = dict(color = '#000000', width = 1)))
    fig.update_layout( height = 400, width = 650,

        annotations=[dict(text='Pred', x=0.17, y=0.5, font_size=20, showarrow=False),
                    dict(text='Real', x=0.82, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig)


with coluna2:
    
    st.info('Comparativo de médias por calibre')
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = dataframe_final['REAL'],  marker_color='darkblue', y = dataframe_final['CALIBRE_x'], histfunc = 'avg',name = 'Real'))

    fig.add_trace(go.Histogram(x = dataframe_final['REAL'],   marker_color='royalblue',y = dataframe_final['Pred'], histfunc = 'avg', name = 'Previsto'))

    #fig.add_trace(go.Scatter(y = dataframe_final['Check'],   marker_color='royalblue',y = dataframe_final['Pred'], name = 'Check'))

        

    # fig.add_trace(go.Histogram(x = dataframe_final['REAL'],   marker_color='green',y = dataframe_final['Perc_cal_amostr'], histfunc = 'avg', name = 'P_AMOS'))

    fig.update_traces(marker_line_color = 'rgb(0,0,0)',marker_line_width = 1.5)

    fig.update_layout(height = 400, width = 650)
    st.plotly_chart(fig)
    #### TALHAO CACHM 9.4 e controle 932,936,974 o modelo foi melhor que a amostragem
    
with coluna1:
    st.info('Acurácia no controle com threshold de 10%')


    fig = px.bar(dataframe_final, x = 'REAL', color = 'Check', category_orders = {'REAL':['CALIBRE_5','CALIBRE_6','CALIBRE_7','CALIBRE_8','CALIBRE_9','CALIBRE_10','CALIBRE_12','CALIBRE_14'],'Check':['ACERTOU','ERROU']})
    fig.update_layout(height = 400, width = 500)
    st.plotly_chart(fig)

    

with coluna2:
    st.info('Caliber (%) forecasting')
    fig = go.Figure()


    fig.add_trace(go.Bar(x =dataframe_final['REAL'] ,y = dataframe_final['Erro']*100, name = 'Error'))

    fig.add_trace(go.Scatter(x =dataframe_final['REAL'] , y = dataframe_final['ERROR_AMOSTRA']*100, name = 'DIF AMOSTRA'))

    fig.add_trace(go.Scatter(x =dataframe_final['REAL'] ,y = dataframe_final['DIAS_ATE_EMBALA'], name = 'DIAS'))

    

    fig.add_hline(11, line_color = 'red', line_dash="dot",  annotation_text=" + 10 % ", annotation_font_color="red")

   
    fig.update_layout(height = 400, width = 650)
    fig
    

st.write('Modelo erra em situações em que um calibre tem um percentual muito alto para sua característica, Exemplo: próximo de 50% calibre 6 ou altos percentuais em calibres que normalmente são 0 ou próximos de zero')

st.write('E também tende a errar mais com o avanço de dias em alguns calibres e quando a amostragem é muito divergente do real')

st.write('Exemplo controle 897 CACH M9.4 onde o calibre 6 teve 53 % e a amostragem do calibre 7 foi bem distante do real ')


# if variety == 'KEITT':
#     st.write('Melhor espaço de amostragem do Calibre 5 da Keitt é  de 16 e 26 Dias')
#     st.write('Melhor espaço de amostragem do Calibre 6 da Keitt é  de 18 e 23 Dias')
#     st.write('Melhor espaço de amostragem do Calibre 7 da Keitt é  de 22 e 23')
#     st.write('Melhor espaço de amostragem do Calibre 8 da Keitt é  de 16:17 e 22:23 Dias')
#     st.write('Melhor espaço de amostragem do Calibre 9 da Keitt é  de 6:11 e 14:19, com percentual de em média de 8.5 e um maximo de até 10')

#     st.write('Melhor espaço de amostragem do Calibre 10 da Keitt é  de 6 e 23, com percentual de em média de 5 e um maximo de até 10')

#     st.write('Melhor espaço de amostragem do Calibre 12 da Keitt é  de 6 e 25, com percentual de em média de 5 e um maximo de até 10')

#     st.write('Sem dados calibre 14')









######################## NDVI RASCUNHO ########################


    # import plotly.express as px
    # df2 = pd.read_csv('sentinel_agd.csv')
    # df2 = df2.dropna()

    # df2['NDVI'] = (df2['band_8'] - df2['band_4']) / (df2['band_8'] + df2['band_4'])
    
    # fig = px.scatter_mapbox(df2, lat="lat", lon="long", color="NDVI", size="NDVI",
    #                 color_continuous_scale=px.colors.cyclical.mygbm, size_max=15, zoom=10,
    #                 mapbox_style="carto-positron")
    # fig
    