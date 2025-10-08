# @title Dashboard Gerencial para Comparação de Estratégias (V4 - Gráfico de Barras)
# @markdown ---
# @markdown ###  Instruções:
# @markdown 1. Execute esta célula.
# @markdown 2. O dashboard será carregado com dados de exemplo.
# @markdown 3. Para analisar seu próprio log, clique no botão **"Carregar e Analisar Arquivo de Log (.txt)"**.

# ==============================================================================
# SEÇÃO 1: IMPORTAÇÕES E CONFIGURAÇÕES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
import re

pio.renderers.default = "colab"

# ==============================================================================
# SEÇÃO 2: FUNÇÃO PRINCIPAL DE ANÁLISE E GERAÇÃO DE GRÁFICOS
# ==============================================================================

def analisar_solucoes_destaque(log_content: str, nome_arquivo: str):
    """
    Função principal que lê o log, extrai as soluções de destaque e gera um
    gráfico de barras comparativo para análise gerencial.
    """
    print(f"Analisando o arquivo de log: '{nome_arquivo}'...")
    print("="*80)

    # --- 2.1: Processamento e Extração dos Dados do Log ---
    linhas = log_content.split('\n')
    solucoes = {}
    pattern = re.compile(r"INFO - (.*?)\s+:\s+Espera=([\d.]+)h,\s+Utilização=([\d.]+)%,\s+Risco=([\d.]+),\s+Custo=([\d.]+)")

    for linha in linhas:
        match = pattern.search(linha)
        if match:
            estrategia = match.group(1).strip()
            # Usar um dicionário para evitar duplicatas automaticamente
            solucoes[estrategia] = {
                'Espera (h)': float(match.group(2)),
                'Utilização (%)': float(match.group(3)),
                'Risco Clínico': float(match.group(4)),
                'Custo Terminal': float(match.group(5))
            }

    if not solucoes:
        print("\n❌ Erro: Nenhuma solução de destaque foi encontrada no arquivo de log.")
        return

    df_solucoes = pd.DataFrame.from_dict(solucoes, orient='index')

    print("📈 SOLUÇÕES DE DESTAQUE ENCONTRADAS:")
    display(df_solucoes)
    print("="*80)

    # --- 2.2: GRÁFICO DE BARRAS AGRUPADAS (ABORDAGEM GERENCIAL) ---
    print("\n📊 Gráfico Comparativo de Estratégias de Otimização")

    # Prepara os dados para o formato de plotagem
    df_plot = df_solucoes.reset_index().rename(columns={'index': 'Estratégia'})

    # Métricas a serem plotadas
    metricas = ['Espera (h)', 'Risco Clínico', 'Custo Terminal', 'Utilização (%)']

    # Cria os subplots (um para cada métrica)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten() # Transforma a matriz 2x2 em uma lista para facilitar o loop

    for i, metrica in enumerate(metricas):
        ax = axes[i]
        valores = df_plot[metrica]

        # Define se a barra menor é melhor (verde) ou pior (vermelho)
        if metrica == 'Utilização (%)':
            # Para Utilização, maior é melhor
            melhor_valor = valores.max()
            cores = ['#2ca02c' if v == melhor_valor else '#d62728' for v in valores]
            ax.set_title(f'Comparativo de {metrica} (Maior é Melhor)', fontsize=14)
        else:
            # Para as outras métricas, menor é melhor
            melhor_valor = valores.min()
            cores = ['#2ca02c' if v == melhor_valor else '#d62728' for v in valores]
            ax.set_title(f'Comparativo de {metrica} (Menor é Melhor)', fontsize=14)

        bars = ax.bar(df_plot['Estratégia'], valores, color=cores, zorder=3)
        ax.set_ylabel(metrica, fontsize=12)
        ax.tick_params(axis='x', labelrotation=15, labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        # Adiciona os valores no topo das barras
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, weight='bold')

    fig.suptitle(f'Dashboard de Análise de Estratégias ({nome_arquivo})', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ==============================================================================
# SEÇÃO 3: WIDGETS E INTERFACE DO USUÁRIO
# ==============================================================================

upload_button_log = widgets.Button(
    description="Carregar e Analisar Arquivo de Log (.txt)",
    disabled=False,
    button_style='primary',
    tooltip='Clique para selecionar um arquivo .txt com o log das soluções',
    icon='upload'
)

output_area_log = widgets.Output()

exemplo_log_string = """
2025-09-22 17:51:59,814 - INFO - ----------------------------------------------------------------------------------------------------
2025-09-22 17:51:59,814 - INFO - MENOR TEMPO ESPERA       : Espera=606h, Utilização=85.8%, Risco=2812, Custo=109
2025-09-22 17:51:59,814 - INFO - MAIOR UTILIZACAO         : Espera=641h, Utilização=86.5%, Risco=2469, Custo=43
2025-09-22 17:51:59,814 - INFO - MENOR RISCO CLINICO      : Espera=653h, Utilização=85.7%, Risco=2329, Custo=113
2025-09-22 17:51:59,814 - INFO - MENOR CUSTO TERMINAL     : Espera=641h, Utilização=86.5%, Risco=2469, Custo=43
2025-09-22 17:51:59,814 - INFO - SOLUCAO BALANCEADA       : Espera=641h, Utilização=86.5%, Risco=2469, Custo=43
"""

def on_log_button_clicked(b):
    with output_area_log:
        clear_output(wait=True)
        print("Aguardando upload do arquivo de log (.txt)...")
        uploaded = files.upload()

        if uploaded:
            file_name = next(iter(uploaded))
            file_content = uploaded[file_name].decode('utf-8')

            try:
                analisar_solucoes_destaque(file_content, file_name)
            except Exception as e:
                print(f"\n❌ Erro ao processar o arquivo de log: {e}")
                print("Verifique o formato do arquivo.")
        else:
            print("\nNenhum arquivo foi selecionado.")

upload_button_log.on_click(on_log_button_clicked)

# Exibe a interface
print("Pressione o botão para analisar um arquivo de log ou veja os resultados do exemplo abaixo.")
display(upload_button_log, output_area_log)

# Executa uma análise inicial com os dados de exemplo
with output_area_log:
    clear_output(wait=True)
    print("Carregando análise com dados de exemplo...")
    analisar_solucoes_destaque(exemplo_log_string, "exemplo_log.txt")
