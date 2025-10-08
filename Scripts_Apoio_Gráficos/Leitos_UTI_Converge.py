# @title Dashboard de Análise de Convergência (V2 - Matplotlib)
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
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
import re

# ==============================================================================
# SEÇÃO 2: FUNÇÃO PRINCIPAL DE ANÁLISE E GERAÇÃO DE GRÁFICOS
# ==============================================================================

def analisar_log_convergencia(log_content: str, nome_arquivo: str):
    """
    Função principal que lê o conteúdo de um log do Pymoo e gera gráficos
    de convergência com Matplotlib.
    """
    print(f"Analisando o arquivo de log: '{nome_arquivo}'...")
    print("="*80)

    # --- 2.1: Processamento e Limpeza do Arquivo de Log ---
    linhas = log_content.split('\n')
    dados_log = []
    header_found = False

    line_pattern = re.compile(r"\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*([\d.E+-]+)\s+\|\s*([\d.E+-]+)")

    for linha in linhas:
        if 'n_gen' in linha and 'n_eval' in linha:
            header_found = True
            continue
        if '====' in linha and header_found:
            continue

        if header_found:
            match = line_pattern.match(linha)
            if match:
                try:
                    dados_log.append([
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                        float(match.group(4)),
                        float(match.group(5))
                    ])
                except (ValueError, IndexError):
                    continue

    if not dados_log:
        print("\n❌ Erro: Nenhum dado de convergência válido foi encontrado no arquivo.")
        return

    df_log = pd.DataFrame(dados_log, columns=['n_gen', 'n_eval', 'n_nds', 'cv_min', 'cv_avg'])

    # --- 2.2: Análise e Interpretação dos Resultados ---
    geracao_viabilidade = df_log[df_log['cv_avg'] == 0]['n_gen'].min()
    max_nds = df_log['n_nds'].max()
    # Usamos .idxmax() que retorna o índice da primeira ocorrência do máximo
    geracao_max_nds_idx = df_log['n_nds'].idxmax()
    geracao_max_nds = df_log.loc[geracao_max_nds_idx, 'n_gen']

    print("📈 MÉTRICAS DE CONVERGÊNCIA:")
    if pd.isna(geracao_viabilidade):
        print(f"  - ⚠️  Viabilidade não alcançada. Menor Violação Média: {df_log['cv_avg'].min():.4f}")
    else:
        print(f"  - ✅ Viabilidade alcançada na Geração: {int(geracao_viabilidade)}")

    print(f"  - Pico de Diversidade da Solução: {max_nds} soluções na Fronteira de Pareto")
    print(f"  - Pico de Diversidade alcançado na Geração: {int(geracao_max_nds)}")
    print("="*80)

    # --- 2.3: GRÁFICO DE CONVERGÊNCIA COM MATPLOTLIB ---
    print("\n📊 Gráficos de Desempenho do Algoritmo ao Longo das Gerações")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Gráfico 1: Convergência para a Viabilidade
    ax1.plot(df_log['n_gen'], df_log['cv_avg'], color='crimson', linewidth=2.5, label='Violação Média (cv_avg)')
    ax1.set_title('Convergência para a Viabilidade', fontsize=16)
    ax1.set_ylabel('Violação Média da Restrição', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1) # Linha de viabilidade

    if not pd.isna(geracao_viabilidade):
        ax1.axvline(x=geracao_viabilidade, color='green', linestyle='--', linewidth=2, label=f'Viabilidade na Gen {int(geracao_viabilidade)}')
        ax1.legend()

    # Gráfico 2: Evolução da Fronteira de Pareto
    ax2.plot(df_log['n_gen'], df_log['n_nds'], color='royalblue', linewidth=2.5, label='Nº de Soluções (n_nds)')
    ax2.set_title('Evolução da Fronteira de Pareto', fontsize=16)
    ax2.set_xlabel('Geração', fontsize=12)
    ax2.set_ylabel('Nº de Soluções na Fronteira', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax2.axvline(x=geracao_max_nds, color='orange', linestyle='--', linewidth=2, label=f'Pico de {max_nds} Soluções')
    ax2.legend()

    fig.suptitle(f'Análise de Convergência do Algoritmo ({nome_arquivo})', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta para o supertítulo
    plt.show()

# ==============================================================================
# SEÇÃO 3: WIDGETS E INTERFACE DO USUÁRIO
# ==============================================================================

upload_button_log = widgets.Button(
    description="Carregar e Analisar Arquivo de Log (.txt)",
    disabled=False,
    button_style='primary',
    tooltip='Clique para selecionar um arquivo .txt com o log de execução do Pymoo',
    icon='upload'
)

output_area_log = widgets.Output()

exemplo_log_string = """
n_gen  |  n_eval  | n_nds  |     cv_min    |     cv_avg    |      eps      |   indicator
==========================================================================================
     1 |      350 |      5 |  0.000000E+00 |  2.6200000000 |             - |             -
     2 |      700 |      4 |  0.000000E+00 |  1.2742857143 |  0.5286343612 |         ideal
     3 |     1050 |      7 |  0.000000E+00 |  0.6571428571 |  0.4122807018 |         ideal
     4 |     1400 |     11 |  0.000000E+00 |  0.3485714286 |  0.3100000000 |         ideal
     5 |     1750 |     10 |  0.000000E+00 |  0.000000E+00 |  0.2182254197 |         ideal
     6 |     2100 |      4 |  0.000000E+00 |  0.000000E+00 |  0.7181818182 |         ideal
     7 |     2450 |      4 |  0.000000E+00 |  0.000000E+00 |  0.6544831525 |         ideal
     8 |     2800 |      6 |  0.000000E+00 |  0.000000E+00 |  0.0367231638 |         ideal
     9 |     3150 |     10 |  0.000000E+00 |  0.000000E+00 |  0.1201005025 |         ideal
    10 |     3500 |     12 |  0.000000E+00 |  0.000000E+00 |  0.1557580779 |         ideal
    50 |    17500 |     30 |  0.000000E+00 |  0.000000E+00 |  0.1658536585 |         ideal
   100 |    35000 |     37 |  0.000000E+00 |  0.000000E+00 |  0.0045248869 |         ideal
   150 |    52500 |     55 |  0.000000E+00 |  0.000000E+00 |  0.0122950820 |         ideal
   200 |    70000 |     41 |  0.000000E+00 |  0.000000E+00 |  0.0045180723 |         ideal
   250 |    87500 |     39 |  0.000000E+00 |  0.000000E+00 |  0.0035699178 |             f
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
                analisar_log_convergencia(file_content, file_name)
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
    analisar_log_convergencia(exemplo_log_string, "exemplo_log.txt")
