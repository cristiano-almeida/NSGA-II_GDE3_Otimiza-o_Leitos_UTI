# @title Dashboard Interativo para Análise de Agendamento de Leitos de UTI (V5 - Gantt com Matplotlib)
# @markdown ---
# @markdown ###  Instruções:
# @markdown 1. Execute esta célula (clicando no botão de play à esquerda ou pressionando Shift+Enter).
# @markdown 2. O dashboard será carregado inicialmente com dados de exemplo.
# @markdown 3. Para analisar seus próprios dados, clique no botão **"Carregar e Analisar Arquivo (.csv)"**.

# ==============================================================================
# SEÇÃO 1: IMPORTAÇÕES E CONFIGURAÇÕES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
import io

# ==============================================================================
# SEÇÃO 2: FUNÇÃO PRINCIPAL DE ANÁLISE E GERAÇÃO DE GRÁFICOS
# ==============================================================================

def analisar_agendamento_uti(df: pd.DataFrame, nome_arquivo: str):
    """
    Função principal que recebe um DataFrame de resultado e gera um dashboard
    completo de métricas e visualizações gerenciais.
    """
    print(f"Analisando o arquivo: '{nome_arquivo}'...")
    print("="*80)

    # --- 2.1: Cálculo das Métricas Chave (KPIs) ---
    numero_leitos = 12
    horizonte_tempo = 168

    for col in ['Tempo_Espera_Horas', 'Gravidade_Score', 'Custo_Terminal', 'Tempo_UTI_Estimado', 'Hora_Internacao_Otimizada', 'Hora_Estimada_Alta']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    tempo_espera_total = df['Tempo_Espera_Horas'].sum()
    risco_clinico_total = (df['Tempo_Espera_Horas'] * df['Gravidade_Score']).sum()
    custo_terminal_total = df['Custo_Terminal'].sum()
    pacientes_fora_horizonte = (df['Status'] == 'Fora do Horizonte').sum()

    print("📈 MÉTRICAS CHAVE DO AGENDAMENTO:")
    print(f"  - Tempo Total de Espera: {tempo_espera_total:.0f} horas")
    print(f"  - Média de Espera por Paciente: {df['Tempo_Espera_Horas'].mean():.1f} horas")
    print(f"  - Risco Clínico Agregado: {risco_clinico_total:.0f}")
    print(f"  - Custo Terminal (Extrapolação): {custo_terminal_total:.0f}")
    print(f"  - Pacientes com Alta Fora do Horizonte: {pacientes_fora_horizonte} de {len(df)}")
    print("="*80)

    # --- 2.2: GRÁFICO 1 - LINHA DO TEMPO DE OCUPAÇÃO POR LEITO (GRÁFICO DE GANTT COM MATPLOTLIB) ---
    print("\n📊 1. Linha do Tempo de Ocupação por Leito (Gráfico de Gantt)")

    fig, ax = plt.subplots(figsize=(18, 8))

    # Configuração de cores baseada na gravidade
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=df['Gravidade_Score'].min(), vmax=df['Gravidade_Score'].max())

    leitos_disponiveis_em = [0] * numero_leitos
    df_sorted = df.sort_values(by='Hora_Internacao_Otimizada').copy()

    for _, paciente in df_sorted.iterrows():
        hora_agendada = paciente['Hora_Internacao_Otimizada']
        duracao = paciente['Tempo_UTI_Estimado']

        # Encontra o leito que ficará vago mais cedo
        leito_alocado = np.argmin(leitos_disponiveis_em)
        hora_disponibilidade_leito = leitos_disponiveis_em[leito_alocado]

        # O início real é o mais tarde entre a hora agendada e a disponibilidade do leito
        inicio_real = max(hora_agendada, hora_disponibilidade_leito)

        # Desenha a barra para o paciente
        ax.barh(y=leito_alocado, width=duracao, left=inicio_real, height=0.8,
                color=cmap(norm(paciente['Gravidade_Score'])), edgecolor='black', alpha=0.8)

        # Adiciona o ID do paciente dentro da barra
        ax.text(inicio_real + duracao / 2, leito_alocado, f"ID {int(paciente['ID_Paciente'])}",
                ha='center', va='center', color='black', fontsize=8, weight='bold')

        # Atualiza a hora de disponibilidade do leito
        leitos_disponiveis_em[leito_alocado] = inicio_real + duracao

    # Configuração do gráfico
    ax.set_yticks(range(numero_leitos))
    ax.set_yticklabels([f"Leito {i+1}" for i in range(numero_leitos)])
    ax.invert_yaxis()
    ax.set_xlabel("Horas desde o início do Planejamento", fontsize=12)
    ax.set_ylabel("Leitos", fontsize=12)
    ax.set_title("Linha do Tempo de Ocupação dos Leitos de UTI", fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.set_xlim(0, horizonte_tempo)

    # Adiciona a barra de cores como legenda
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.01)
    cbar.set_label('Gravidade do Paciente', fontsize=12)

    plt.tight_layout()
    plt.show()

    # --- 2.3: GRÁFICO 2 - NÍVEL DE OCUPAÇÃO DA UTI AO LONGO DO TEMPO ---
    print("\n📊 2. Nível de Ocupação da UTI (Hora a Hora)")

    ocupacao_horaria = np.zeros(horizonte_tempo)
    for _, row in df.iterrows():
        inicio = int(row['Hora_Internacao_Otimizada'])
        fim = int(row['Hora_Estimada_Alta'])
        ocupacao_horaria[max(0, inicio):min(horizonte_tempo, fim)] += 1

    plt.figure(figsize=(15, 6))
    plt.plot(range(horizonte_tempo), ocupacao_horaria, label='Leitos Ocupados', color='#007acc', linewidth=2)
    plt.axhline(y=numero_leitos, color='red', linestyle='--', label=f'Capacidade Máxima ({numero_leitos} Leitos)')
    plt.title('Nível de Ocupação da UTI ao Longo do Tempo', fontsize=16)
    plt.xlabel('Hora', fontsize=12)
    plt.ylabel('Número de Leitos Ocupados', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, numero_leitos + 2)
    plt.xlim(0, horizonte_tempo)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2.4 E 2.5 (Demais gráficos) ---
    print("\n📊 3. Distribuição do Tempo de Espera dos Pacientes")

    tempos_de_espera = df[df['Tempo_Espera_Horas'] > 0]['Tempo_Espera_Horas']

    plt.figure(figsize=(10, 6))
    if not tempos_de_espera.empty:
        plt.hist(tempos_de_espera, bins=20, color='#2ca02c', edgecolor='black')
        plt.title('Distribuição do Tempo de Espera (para pacientes que esperaram)', fontsize=16)
        plt.xlabel('Tempo de Espera (Horas)', fontsize=12)
        plt.ylabel('Número de Pacientes', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'Nenhum paciente precisou esperar!', ha='center', va='center', fontsize=18)
        plt.title('Distribuição do Tempo de Espera', fontsize=16)

    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.show()

    print("\n📊 4. Análise de Priorização: Relação entre Gravidade e Tempo de Espera")
    print("   (Idealmente, pontos devem se concentrar no canto inferior direito)")

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['Gravidade_Score'],
        df['Tempo_Espera_Horas'],
        c=df['Gravidade_Score'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='black'
    )
    plt.title('Análise de Priorização: Gravidade vs. Tempo de Espera', fontsize=16)
    plt.xlabel('Gravidade do Paciente (Score)', fontsize=12)
    plt.ylabel('Tempo de Espera (Horas)', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Nível de Gravidade', fontsize=12)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# SEÇÃO 3: WIDGETS E INTERFACE DO USUÁRIO
# ==============================================================================

upload_button = widgets.Button(
    description="Carregar e Analisar Arquivo (.csv)",
    disabled=False,
    button_style='success',
    tooltip='Clique para selecionar um arquivo CSV de resultado da otimização',
    icon='upload'
)

output_area = widgets.Output()

exemplo_csv_string = """ID_Paciente,Hora_Chegada,Gravidade_Score,Tempo_UTI_Estimado,Hora_Internacao_Otimizada,Tempo_Espera_Horas,Hora_Estimada_Alta,Custo_Terminal,Status,Internacao_Imediata
0,11,9,80,60,49,140,0,Dentro do Horizonte,Não
1,4,3,120,49,45,169,3,Fora do Horizonte,Não
2,1,8,120,37,36,157,0,Dentro do Horizonte,Não
3,0,10,91,0,0,91,0,Dentro do Horizonte,Sim
4,42,3,26,60,18,86,0,Dentro do Horizonte,Não
5,8,5,28,8,0,36,0,Dentro do Horizonte,Sim
6,13,3,24,13,0,37,0,Dentro do Horizonte,Sim
7,8,4,51,37,29,88,0,Dentro do Horizonte,Não
8,14,7,47,14,0,61,0,Dentro do Horizonte,Sim
9,21,1,24,32,11,56,0,Dentro do Horizonte,Não
10,22,3,24,25,3,49,0,Dentro do Horizonte,Não
11,27,4,31,36,9,67,0,Dentro do Horizonte,Não
12,3,5,79,88,85,167,0,Dentro do Horizonte,Não
13,8,5,24,8,0,32,0,Dentro do Horizonte,Sim
14,18,3,68,86,68,154,0,Dentro do Horizonte,Não
15,2,3,43,2,0,45,0,Dentro do Horizonte,Sim
16,1,4,24,1,0,25,0,Dentro do Horizonte,Sim
17,42,4,24,42,0,66,0,Dentro do Horizonte,Sim
18,7,5,53,7,0,60,0,Dentro do Horizonte,Sim
19,1,10,36,1,0,37,0,Dentro do Horizonte,Sim
20,35,3,86,71,36,157,0,Dentro do Horizonte,Não
21,35,2,76,86,51,162,0,Dentro do Horizonte,Não
22,10,2,47,93,83,140,0,Dentro do Horizonte,Não
23,9,2,79,66,57,145,0,Dentro do Horizonte,Não
24,8,4,76,8,0,84,0,Dentro do Horizonte,Sim
25,52,5,120,56,4,176,40,Fora do Horizonte,Não
26,3,6,57,3,0,60,0,Dentro do Horizonte,Sim
27,35,5,120,47,12,167,0,Dentro do Horizonte,Não
28,17,4,42,62,45,104,0,Dentro do Horizonte,Não
29,9,5,32,9,0,41,0,Dentro do Horizonte,Sim
"""

def on_button_clicked(b):
    with output_area:
        clear_output(wait=True)
        print("Aguardando upload do arquivo CSV...")
        uploaded = files.upload()

        if uploaded:
            file_name = next(iter(uploaded))
            file_content = uploaded[file_name]

            try:
                df = pd.read_csv(io.BytesIO(file_content))
                analisar_agendamento_uti(df, file_name)
            except Exception as e:
                print(f"\n❌ Erro ao processar o arquivo: {e}")
                print("Verifique se o arquivo é um CSV válido com as colunas esperadas.")
        else:
            print("\nNenhum arquivo foi selecionado.")

upload_button.on_click(on_button_clicked)

# Exibe a interface do usuário
print("Pressione o botão para iniciar a análise ou veja os resultados do exemplo abaixo.")
display(upload_button, output_area)

# Executa uma análise inicial com os dados de exemplo para demonstração
with output_area:
    clear_output(wait=True)
    print("Carregando análise com dados de exemplo...")
    df_exemplo = pd.read_csv(io.StringIO(exemplo_csv_string))
    analisar_agendamento_uti(df_exemplo, "exemplo_incorporado.csv")
