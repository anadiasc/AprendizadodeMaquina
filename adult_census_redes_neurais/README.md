# Bancada de testes — Redes Neurais no Adult Census Income

Bancada de experimentos para encontrar as configurações de rede neural (MLP), as técnicas de tratamento de dados e a engenharia de features que maximizam as métricas de classificação no dataset [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) (prever se a renda anual é `>50K`).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `bancada_redes_neurais_adult.ipynb` | notebook com todas as etapas, resultados, gráficos e análises |
| `analise_dataset_resultados.ipynb` | gráficos das características do dataset, análise dos principais resultados da rede e importância das features (permutação e dependência parcial) |
| `bancada.py` | código da bancada: carregamento, pipeline de pré-processamento, rede, validação cruzada e registro dos experimentos |
| `resultados/resultados_experimentos.csv` | todas as métricas de todos os experimentos |

## Como rodar

```bash
pip install scikit-learn imbalanced-learn pandas matplotlib kagglehub jupyter
jupyter notebook bancada_redes_neurais_adult.ipynb
jupyter notebook analise_dataset_resultados.ipynb   # ~2 min; usa a melhor configuração (bancada.MELHOR_CONFIG)
```

O dataset é baixado do Kaggle (`kagglehub`). Sem acesso ao Kaggle, é usado um espelho do UCI no GitHub com as mesmas 32.561 linhas. A execução completa leva cerca de 40 minutos num processador de 4 núcleos (os folds rodam em paralelo). Os resultados de cada configuração ficam salvos em `resultados/cache/`, então uma execução interrompida continua de onde parou.

## Protocolo

- Teste final de 20% (estratificado), separado no início e usado só na avaliação final.
- Validação cruzada estratificada de 5 folds no treino, com o pré-processamento ajustado só no treino de cada fold (sem vazamento).
- Parada antecipada pela perda (log loss) numa fatia de 10% do treino de cada fold, com os pesos da melhor época restaurados.
- Busca gulosa em 12 etapas (faltantes → codificação → escala → features → PCA → arquitetura → regularização → ativação → otimização → balanceamento), seguida de uma busca aleatória e do ajuste do limiar de decisão.
- Métrica de escolha: ROC-AUC. Todas as métricas são registradas: acurácia, acurácia balanceada, precisão, recall, F1, MCC, ROC-AUC, PR-AUC e log loss.

## Principais resultados

| Modelo (teste) | ROC-AUC | PR-AUC | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|
| MLP linha de base (`LabelEncoder` + `StandardScaler`, 64-32) | 0,9048 | 0,7703 | 0,8494 | 0,7125 | 0,6276 | 0,6673 |
| **MLP final** (ensemble de 5 sementes), limiar 0,5 | **0,9212** | **0,8078** | **0,8658** | **0,7601** | 0,6467 | 0,6988 |
| **MLP final**, limiar 0,40 (máx. F1) | **0,9212** | **0,8078** | 0,8575 | 0,6863 | **0,7519** | **0,7176** |
| Regressão logística (como no notebook de PCA) | 0,8616 | 0,7018 | 0,8274 | 0,7220 | 0,4605 | 0,5623 |
| Random Forest (como no tutorial) | 0,9114 | 0,7926 | 0,8610 | 0,7407 | 0,6505 | 0,6927 |
| HistGradientBoosting (referência) | 0,9294 | 0,8342 | 0,8743 | 0,7797 | 0,6658 | 0,7183 |

Melhor configuração: faltantes pela moda, *one-hot*, `StandardScaler`, indicadores de ganho/perda de capital, sem PCA, 1 camada oculta de 256 neurônios (ReLU), L2 = 0,3, Adam (taxa 3e-4, lote 256, redução da taxa em platôs), sem balanceamento e limiar ajustado.

O que mais pesou: a codificação *one-hot* (≈ 70% do ganho de ROC-AUC na validação), o indicador de ganho de capital e o ajuste do limiar (que muda F1 e recall mais do que qualquer hiperparâmetro). PCA, log nas colunas de capital, `MinMaxScaler`, ativação sigmoide e lotes pequenos pioraram as métricas. As técnicas de balanceamento não melhoraram a ROC-AUC, apenas trocaram precisão por recall, o que o ajuste do limiar faz sem custo extra. O HistGradientBoosting ainda supera a melhor MLP neste dataset tabular.

## Importância das features

Por permutação no teste (queda da ROC-AUC ao embaralhar cada coluna): `marital.status` (0,057) > `capital.gain` (0,044) > `education.num` (0,031) ≈ `age` (0,030) > `hours.per.week` (0,016) > `occupation` (0,014). `fnlwgt`, `race`, `native.country` e `education` têm importância quase nula — esta última porque repete `education.num`. Embaralhando juntas as colunas redundantes, a **situação familiar** (`relationship` + `marital.status` + `sex`, queda de 0,114) aparece como a informação mais importante para a rede.
