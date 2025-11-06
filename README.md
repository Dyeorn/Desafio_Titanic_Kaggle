# 🚢 Desafio Titanic - Previsão de Sobrevivência (Kaggle)

Este projeto é uma solução para o desafio "Titanic - Machine Learning from Disaster" do Kaggle.

O objetivo é prever se um passageiro no navio Titanic sobreviveu (1) ou não (0) com base em variáveis como sexo, classe da passagem, idade e outras características.

## 🎯 Objetivo

Criar um modelo de Machine Learning que preveja a coluna `Survived` no conjunto de dados de teste, utilizando as features fornecidas no conjunto de treino.

## 🛠️ Tecnologias e Bibliotecas

| Categoria | Biblioteca/Ferramenta | Uso no Projeto |
| :--- | :--- | :--- |
| **Linguagem** | Python | Linguagem principal para análise e modelagem. |
| **Notebook** | Jupyter Notebook | Desenvolvimento e execução passo a passo do código. |
| **DataFrames** | Pandas | Manipulação, leitura (`read_csv`) e pré-processamento dos dados. |
| **Cálculo Numérico** | NumPy | Funções matemáticas e manipulação de arrays. |
| **Machine Learning**| Scikit-learn (sklearn) | Implementação dos modelos de `LogisticRegression` e `RandomForestClassifier` e validação cruzada (`RepeatedKFold`). |
| **Visualização** | Matplotlib / Pylab | Geração de histogramas para avaliar a distribuição dos resultados. |

## 🚀 Metodologia

A solução final utiliza uma abordagem de **Regressão Logística** após uma etapa de Engenharia de Features, que comprovou ser a mais eficaz em comparação com a abordagem inicial de Random Forest.

### 1\. Pré-processamento

  * **Binarização do Gênero (Sex):** A coluna `Sex` foi convertida em uma variável binária (`Sex_binario`), onde `female` é `1` e `male` é `0`.
  * **Tratamento de Valores Ausentes (NaN):** Valores ausentes (NaN) nas colunas numéricas, como `Age` e `Fare`, foram preenchidos com o valor **`-1`** para que o modelo pudesse processar os dados.

### 2\. Engenharia de Features (Feature Engineering)

Foram criadas novas variáveis binárias (`0` ou `1`) para capturar informações importantes, melhorando a acurácia do modelo:

  * **Porto de Embarque (`Embarked`):** Foram criadas variáveis binárias para os portos 'S' e 'C' (`Embarked_S` e `Embarked_C`).
  * **Informação de Cabine (`Cabin`):** Uma feature chamada `Cabine_nula` foi criada para indicar se a informação da cabine estava faltando (`1` se ausente, `0` se presente).
  * **Títulos do Nome (`Name`):** Foram extraídos os títulos (Mr., Miss, Mrs., Master, Col., Major) para capturar o status social do passageiro, que é um forte preditor de sobrevivência.

### 3\. Modelo e Validação

  * **Modelo Escolhido:** **Regressão Logística (`LogisticRegression`)**.
  * **Validação:** Foi utilizada a **Validação Cruzada Repetida** (`RepeatedKFold` com `n_splits=2` e `n_repeats=10`) para estimar a acurácia.
  * **Performance (Validação Interna):** A acurácia média obtida com esta metodologia foi de aproximadamente **0.8190**.

## ⚙️ Como Reproduzir

Para rodar este projeto, você precisará dos arquivos de dados do Kaggle (`train.csv` e `test.csv`) no mesmo diretório do seu notebook.

### 1\. Ambiente

Crie um ambiente Python e instale as bibliotecas necessárias:

```bash
# Crie e ative um ambiente virtual
# python -m venv titanic-env

# 2. Ative o ambiente
# macOS/Linux:
source .titanic-env/bin/activate
# Windows CMD:
.titanic-env\Scripts\activate.bat
# Windows PowerShell:
.titanic-env\Scripts\Activate.ps1

# Instale as dependências
pip install pandas numpy scikit-learn matplotlib
```

## 📈 Resultados no Kaggle

Com esta metodologia, o resultado na plataforma da Kaggle foi em torno de **0.76 - 0.77 de acurácia**.
