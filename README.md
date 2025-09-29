# Otimização de Alocação de Leitos de UTI

Este projeto realiza a otimização da alocação de leitos de UTI utilizando algoritmos evolutivos multiobjetivo (`GDE3` e `NSGA-II`). Inclui scripts de execução, arquivos de requisitos, base de dados sintética de pacientes e pastas para armazenar resultados e gráficos.

---

## Estrutura de Pastas e Arquivos Principais

```
.
├── Bases/                            # Pasta com bases de dados (Central, Underload e Overload)
├── GDE3_Resultados/                  # Resultados do algoritmo GDE3
├── NSGA-II_Resultados/               # Resultados do algoritmo NSGA-II
├── Scripts_Apoio_Gráficos/           # Scripts auxiliares para visualização de gráficos
├── UTI_GDE3.py                       # Script principal para otimização usando GDE3
├── UTI_NSGA-II.py                    # Script principal para otimização usando NSGA-II
├── requirements_GDE3.txt             # Dependências específicas para o GDE3
├── requirements_NSGA-II.txt          # Dependências específicas para o NSGA-II
├── README.md                         # Este arquivo
├── base_pacientes_uti_realista.csv	  # Nome do arquivo na raiz do projeto relacionado aos scripts Principais
├── Manual_Técnico.pdf				  # Manual resumido das principais funções
```

---

## Requisitos

- Python 3.8 – 3.12 (recomendado: 3.12)
- Pip >= 22.0
- Sistema operacional: Windows, macOS ou Linux

> ⚠️ Observações importantes:
> - Python 3.13 ainda não possui wheels pré-compiladas para o pacote `pymoode`.
> - Caso utilize Python 3.13, será necessário compilar o pacote a partir do código-fonte com Microsoft Visual C++ Build Tools (Windows).
> - Para garantir reprodutibilidade, recomenda-se Python 3.12.

---

## Instalação e Configuração

### 1️⃣ Navegar até a pasta do projeto

**Windows (cmd/PowerShell):**
```powershell
cd caminho\para\projeto
```

**macOS / Linux (bash/zsh):**
```bash
cd caminho/para/projeto
```

### 2️⃣ Criar o ambiente virtual

**Windows:**
```powershell
py -3.12 -m venv .venv
```

**macOS / Linux:**
```bash
python3.12 -m venv .venv
```

### 3️⃣ Ativar o ambiente virtual

**Windows:**
```powershell
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 4️⃣ Instalar pacotes e dependências

```bash
pip install -r requirements_NSGA-II.txt
pip install -r requirements_GDE3.txt
pip install pymoode
```

### 5️⃣ Desativar o ambiente virtual

```bash
deactivate
```

---

## Execução dos Scripts

- **GDE3:**
```bash
python UTI_GDE3.py
```

- **NSGA-II:**
```bash
python UTI_NSGA-II.py
```

Os scripts geram resultados e relatórios nas pastas `GDE3_Resultados` e `NSGA-II_Resultados`, além de gráficos auxiliares em `Scripts_Apoio_Gráficos`.

---

## Observações Acadêmicas

1. Recomenda-se criar ambientes virtuais isolados para cada algoritmo (`.venv_GDE3`, `.venv_NSGA-II`) para evitar conflitos de dependências.
2. A base sintética `base_pacientes_uti_realista.csv` pode ser substituída por bases reais, desde que o formato de coluna seja mantido.
3. Este README é aplicável a **Windows, macOS e Linux**, garantindo reprodutibilidade e consistência em experimentos.

---

## Referências

- Documentação do [`pymoode`](https://pypi.org/project/pymoode/)
- Artigos acadêmicos sobre otimização multiobjetivo em alocação de leitos de UTI
- Python 3.12 Release Notes: [https://www.python.org/downloads/release/python-3127/](https://www.python.org/downloads/release/python-3127/)
