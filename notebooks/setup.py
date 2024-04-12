import data_base as db
import estrutura as es
import datetime

# Sql das bases
query_treino = """select * from cscforms.tbl_cad_treino_predicao_pedidos_embarques;"""
query_predicao = """select * from cscforms.tbl_dad_predicao_pedidos_embarques;"""

# Parâmetros de conexão
dbname = "postgres"
user = "userrpaplanejamento"
password = "Planejamento2022*"
host = "dbweb.cpntaziiu66w.sa-east-1.rds.amazonaws.com"
port = "5432"

db_connector = db.DatabaseConnector(dbname, user, password, host, port)
df_loader = db.DataFrameLoader(db_connector)

df_treino = df_loader.load_dataframe(query_treino)
df_predicao = df_loader.load_dataframe(query_predicao)

diretorio_previsao = 'previsoes/'
aprendizado = es.Aprendizado(df_treino, df_predicao, diretorio_previsao)

aprendizado.dividir_dados_treino()
aprendizado.identificar_melhor_modelo()
aprendizado.imprimir_informacoes_modelo()
aprendizado.otimizar_modelo_com_hiperparametros()

aprendizado.dividir_dados_predicao()

hoje = datetime.date.today()
proxima_semana = hoje + datetime.timedelta(7)

#proxima_semana = proxima_semana - 1 

numero_proxima_semana = proxima_semana.isocalendar()[1]
ano_proxima_semana = proxima_semana.isocalendar()[0]

print(f"O número da próxima semana é {numero_proxima_semana-1} e o ano é {ano_proxima_semana}.")

aprendizado.prever_e_salvar()
aprendizado.prever_e_salvar_data(ano_proxima_semana, numero_proxima_semana)
r = input("Digite algo: ")