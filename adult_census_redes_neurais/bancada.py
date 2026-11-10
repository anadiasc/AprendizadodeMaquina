"""Bancada de testes de Redes Neurais (MLP) para o dataset Adult Census Income.

Este módulo reúne todo o código reutilizável do notebook
``bancada_redes_neurais_adult.ipynb``:

- carregamento dos dados (Kaggle ou espelho do UCI no GitHub);
- tratamento de valores faltantes, engenharia de features, codificação,
  escala e PCA, tudo dentro de um ``Pipeline`` do scikit-learn (ajustado
  SOMENTE nos dados de treino de cada fold, para não vazar informação);
- construção e treino da rede neural (``MLPClassifier`` do scikit-learn, com
  laço de treino próprio para ter parada antecipada pela perda de validação);
- validação cruzada estratificada com todas as métricas de avaliação;
- execução de "etapas" de experimentos e registro dos resultados.

Cada experimento é descrito por um dicionário de configuração (``cfg``).
A configuração ``CONFIG_BASE`` define o ponto de partida; cada experimento
altera apenas algumas chaves.
"""

from __future__ import annotations

import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler, TargetEncoder)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

COLUNAS = ["age", "workclass", "fnlwgt", "education", "education.num",
           "marital.status", "occupation", "relationship", "race", "sex",
           "capital.gain", "capital.loss", "hours.per.week", "native.country",
           "income"]

COLUNAS_NUMERICAS = ["age", "fnlwgt", "education.num", "capital.gain",
                     "capital.loss", "hours.per.week"]
COLUNAS_CATEGORICAS = ["workclass", "education", "marital.status", "occupation",
                       "relationship", "race", "sex", "native.country"]

# Espelho público do UCI Adult (adult.data + adult.test, sem cabeçalho).
# As primeiras 32.561 linhas são exatamente o arquivo adult.data, que é o
# mesmo conteúdo do adult.csv do Kaggle (uciml/adult-census-income).
URL_ESPELHO = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
N_LINHAS_KAGGLE = 32561


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def carregar_dados() -> pd.DataFrame:
    """Carrega o Adult Census Income com '?' convertido em NaN.

    Tenta primeiro o Kaggle (kagglehub). Se não houver acesso, usa o espelho
    do UCI no GitHub, que contém as mesmas 32.561 linhas.
    """
    try:
        import kagglehub
        pasta = kagglehub.dataset_download("uciml/adult-census-income")
        df = pd.read_csv(glob.glob(os.path.join(pasta, "*.csv"))[0])
        origem = "Kaggle (uciml/adult-census-income)"
    except Exception:  # sem credenciais/rede para o Kaggle
        df = pd.read_csv(URL_ESPELHO, header=None, names=COLUNAS,
                         skipinitialspace=True).iloc[:N_LINHAS_KAGGLE]
        origem = "espelho UCI no GitHub (mesmas linhas do Kaggle)"

    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()
    df = df.replace("?", np.nan)
    print(f"Dados carregados de: {origem} -> {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def separar_x_y(df: pd.DataFrame):
    """Separa as features e o alvo (1 = renda > 50K, 0 = renda <= 50K)."""
    X = df.drop(columns=["income"])
    y = (df["income"] == ">50K").astype(int).to_numpy()
    return X, y


# --------------------------------------------------------------------------
# Engenharia de features
# --------------------------------------------------------------------------
ESTADO_CIVIL = {
    "Married-civ-spouse": "Casado", "Married-AF-spouse": "Casado",
    "Married-spouse-absent": "Separado", "Separated": "Separado",
    "Divorced": "Separado", "Widowed": "Viuvo", "Never-married": "Solteiro",
}
SETOR = {
    "Federal-gov": "Governo", "State-gov": "Governo", "Local-gov": "Governo",
    "Self-emp-inc": "Autonomo-inc", "Self-emp-not-inc": "Autonomo",
    "Private": "Privado", "Without-pay": "Sem-renda", "Never-worked": "Sem-renda",
}


def _log_sinal(x):
    """log(1 + |x|) mantendo o sinal (serve para valores negativos)."""
    return np.sign(x) * np.log1p(np.abs(x))


class EngenhariaFeatures(BaseEstimator, TransformerMixin):
    """Cria, transforma ou remove colunas conforme a lista ``opcoes``.

    Opções disponíveis:
      - ``remover_education``: remove ``education`` (redundante com ``education.num``);
      - ``remover_fnlwgt``: remove ``fnlwgt`` (peso amostral do censo);
      - ``capital_liquido``: cria ``capital.net = capital.gain - capital.loss``;
      - ``flags_capital``: cria indicadores 0/1 de ganho e de perda de capital;
      - ``log_capital``: aplica log(1 + x) às colunas de capital (muito assimétricas);
      - ``agrupar_pais``: ``native.country`` vira Estados Unidos x Outro;
      - ``agrupar_estado_civil``: agrupa ``marital.status`` em 4 categorias;
      - ``agrupar_setor``: agrupa ``workclass`` em 5 categorias;
      - ``faixas_idade_horas``: cria faixas de idade e de horas semanais (categóricas);
      - ``interacoes``: cria idade x escolaridade e horas x escolaridade.
    """

    def __init__(self, opcoes=()):
        self.opcoes = opcoes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        op = set(self.opcoes)
        if "capital_liquido" in op:
            X["capital.net"] = X["capital.gain"] - X["capital.loss"]
        if "flags_capital" in op:
            X["tem_ganho_capital"] = (X["capital.gain"] > 0).astype(float)
            X["tem_perda_capital"] = (X["capital.loss"] > 0).astype(float)
        if "log_capital" in op:
            for col in ["capital.gain", "capital.loss", "capital.net"]:
                if col in X:
                    X[col] = _log_sinal(X[col].astype(float))
        if "agrupar_pais" in op:
            pais = X["native.country"]
            X["native.country"] = pais.where(pais.isna() | (pais == "United-States"), "Outro")
        if "agrupar_estado_civil" in op:
            X["marital.status"] = X["marital.status"].map(ESTADO_CIVIL)
        if "agrupar_setor" in op:
            X["workclass"] = X["workclass"].map(SETOR)  # NaN continua NaN
        if "faixas_idade_horas" in op:
            X["faixa_idade"] = pd.cut(X["age"], [0, 25, 35, 45, 55, 65, 120]).astype(str)
            X["faixa_horas"] = pd.cut(X["hours.per.week"], [0, 30, 40, 50, 200]).astype(str)
        if "interacoes" in op:
            X["idade_x_educ"] = X["age"] * X["education.num"]
            X["horas_x_educ"] = X["hours.per.week"] * X["education.num"]
        if "remover_education" in op:
            X = X.drop(columns=["education"])
        if "remover_fnlwgt" in op:
            X = X.drop(columns=["fnlwgt"])
        return X


# --------------------------------------------------------------------------
# Pré-processamento
# --------------------------------------------------------------------------
ESCALAS = {
    "standard": lambda: StandardScaler(),
    "minmax": lambda: MinMaxScaler(),
    "robust": lambda: RobustScaler(),
    "quantile": lambda: QuantileTransformer(output_distribution="normal",
                                            n_quantiles=1000, random_state=RANDOM_STATE),
    "power": lambda: PowerTransformer(method="yeo-johnson"),
}


def construir_preprocessador(cfg: dict) -> Pipeline:
    """Monta o pipeline de pré-processamento a partir da configuração.

    Chaves usadas: ``faltantes``, ``features``, ``codificacao``, ``escala``, ``pca``.
    """
    # 1) faltantes nas categóricas: moda ou categoria própria.
    #    ("remover" também imputa a moda aqui, para os dados de validação/teste;
    #    as linhas com NaN são retiradas apenas do treino, em ``_remover_faltantes``.)
    if cfg["faltantes"] == "categoria":
        imputador = SimpleImputer(strategy="constant", fill_value="Desconhecido")
    else:
        imputador = SimpleImputer(strategy="most_frequent")

    # 2) codificação das categóricas
    cod = cfg["codificacao"]
    escala = ESCALAS[cfg["escala"]]
    if cod == "label":
        # inteiros 0, 1, 2... (impõe uma ordem que não existe) + escala
        codificador = [("cod", OrdinalEncoder(handle_unknown="use_encoded_value",
                                              unknown_value=-1)),
                       ("esc", StandardScaler())]
    elif cod == "onehot":
        codificador = [("cod", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    elif cod == "onehot_raras":
        # categorias com menos de 0,5% das linhas são agrupadas em "infrequent"
        codificador = [("cod", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                             min_frequency=0.005, sparse_output=False))]
    elif cod == "target":
        # cada categoria vira a taxa média de renda > 50K (com validação cruzada interna)
        codificador = [("cod", TargetEncoder(target_type="binary", random_state=RANDOM_STATE)),
                       ("esc", StandardScaler())]
    else:
        raise ValueError(f"codificação desconhecida: {cod}")

    pipe_cat = Pipeline([("imp", imputador)] + codificador)
    pipe_num = Pipeline([("esc", escala())])

    colunas = ColumnTransformer([
        ("num", pipe_num, make_column_selector(dtype_include="number")),
        ("cat", pipe_cat, make_column_selector(dtype_exclude="number")),
    ])

    etapas = [("features", EngenhariaFeatures(tuple(cfg["features"]))),
              ("colunas", colunas)]
    if cfg.get("pca"):
        etapas.append(("pca", PCA(n_components=cfg["pca"], svd_solver="full",
                                  random_state=RANDOM_STATE)))
    return Pipeline(etapas)


# --------------------------------------------------------------------------
# Rede neural
# --------------------------------------------------------------------------
def construir_rede(cfg: dict, seed: int = RANDOM_STATE) -> MLPClassifier:
    """Cria um perceptron multicamadas (MLP) do scikit-learn.

    Chaves usadas: ``camadas`` (neurônios por camada oculta), ``ativacao``,
    ``l2`` (penalidade alpha), ``otimizador``, ``lr`` e ``batch_size``.
    A saída é um neurônio sigmoide (probabilidade de renda > 50K) treinado
    com entropia cruzada binária (log loss).
    """
    return MLPClassifier(hidden_layer_sizes=tuple(cfg["camadas"]),
                         activation=cfg["ativacao"], solver=cfg["otimizador"],
                         alpha=cfg["l2"], batch_size=cfg["batch_size"],
                         learning_rate_init=cfg["lr"], momentum=0.9,
                         nesterovs_momentum=True, shuffle=True, random_state=seed)


def _reduzir_lr(rede: MLPClassifier, fator: float = 0.5, minimo: float = 1e-5):
    """Divide a taxa de aprendizado do otimizador interno do MLPClassifier."""
    otim = rede._optimizer
    novo = max(otim.learning_rate_init * fator, minimo)
    otim.learning_rate_init = novo
    otim.learning_rate = novo  # SGD usa este atributo; o Adam recalcula a partir do init


def treinar_rede(Xp_fit, y_fit, Xp_es, y_es, cfg: dict, seed: int, pesos=None):
    """Treina a rede época por época (``partial_fit``) com parada antecipada.

    A cada época mede a perda (log loss) no conjunto de parada ``Xp_es``.
    Se a perda não melhora por ``paciencia`` épocas, o treino para e os pesos
    da melhor época são restaurados (como o ``EarlyStopping`` do Keras com
    ``restore_best_weights=True``). Com ``reduzir_lr``, a taxa de aprendizado
    cai pela metade após ``paciencia // 3`` épocas sem melhora.
    """
    rede = construir_rede(cfg, seed)
    melhor_perda, melhor_epoca, sem_melhora, sem_melhora_lr = np.inf, 0, 0, 0
    melhores_pesos = None
    paciencia_lr = max(2, cfg["paciencia"] // 3)
    curva = []
    for epoca in range(1, cfg["epocas"] + 1):
        rede.partial_fit(Xp_fit, y_fit, sample_weight=pesos, classes=np.array([0, 1]))
        perda = log_loss(y_es, np.clip(rede.predict_proba(Xp_es)[:, 1], 1e-7, 1 - 1e-7))
        curva.append(perda)
        if perda < melhor_perda - 1e-5:
            melhor_perda, melhor_epoca = perda, epoca
            sem_melhora = sem_melhora_lr = 0
            melhores_pesos = ([w.copy() for w in rede.coefs_],
                              [b.copy() for b in rede.intercepts_])
        else:
            sem_melhora += 1
            sem_melhora_lr += 1
            if sem_melhora >= cfg["paciencia"]:
                break
            if cfg.get("reduzir_lr") and sem_melhora_lr >= paciencia_lr:
                _reduzir_lr(rede)
                sem_melhora_lr = 0
    rede.coefs_, rede.intercepts_ = melhores_pesos
    rede.curva_perda_ = curva
    return rede, melhor_epoca


def _remover_faltantes(X, y):
    mask = X[["workclass", "occupation", "native.country"]].notna().all(axis=1).to_numpy()
    return X[mask], y[mask]


def _balancear(Xp, y, metodo, seed):
    """Reamostra os dados JÁ pré-processados (somente treino)."""
    if metodo in (None, "nenhum", "class_weight"):
        return Xp, y
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    amostrador = {"smote": SMOTE(random_state=seed),
                  "oversampling": RandomOverSampler(random_state=seed),
                  "undersampling": RandomUnderSampler(random_state=seed)}[metodo]
    return amostrador.fit_resample(Xp, y)


def treinar_modelo(X_tr: pd.DataFrame, y_tr: np.ndarray, cfg: dict, seed: int = RANDOM_STATE):
    """Ajusta o pré-processamento e treina a rede. Devolve (preprocessador, rede, nº de épocas).

    Uma fatia estratificada (``frac_parada``) do treino é separada para a
    parada antecipada. Ela não é reamostrada, então a perda de parada é
    medida na distribuição real das classes.
    """
    if cfg["faltantes"] == "remover":
        X_tr, y_tr = _remover_faltantes(X_tr, y_tr)

    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr, y_tr, test_size=cfg["frac_parada"], stratify=y_tr, random_state=seed)

    prep = construir_preprocessador(cfg)
    Xp_fit = prep.fit_transform(X_fit, y_fit)
    Xp_es = prep.transform(X_es)
    Xp_fit, y_fit = _balancear(Xp_fit, y_fit, cfg["balanceamento"], seed)

    pesos = None
    if cfg["balanceamento"] == "class_weight":
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_fit)
        pesos = w[y_fit]  # peso de cada exemplo = peso da sua classe

    rede, n_epocas = treinar_rede(Xp_fit, y_fit, Xp_es, y_es, cfg, seed, pesos)
    return prep, rede, n_epocas


def prever_proba(prep, rede, X: pd.DataFrame) -> np.ndarray:
    """Probabilidade de renda > 50K para cada linha de ``X``."""
    return rede.predict_proba(prep.transform(X))[:, 1]


# --------------------------------------------------------------------------
# Métricas e validação cruzada
# --------------------------------------------------------------------------
METRICAS = ["acuracia", "acuracia_bal", "precisao", "recall", "f1", "mcc", "roc_auc", "pr_auc", "log_loss"]


def calcular_metricas(y_true, proba, limiar: float = 0.5) -> dict:
    pred = (proba >= limiar).astype(int)
    return {
        "acuracia": accuracy_score(y_true, pred),
        "acuracia_bal": balanced_accuracy_score(y_true, pred),
        "precisao": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7)),
    }


def _rodar_fold(X_tr, y_tr, X_va, cfg, seed):
    """Treina em um fold e devolve (probabilidades da validação, nº de épocas).

    Cada processo usa 1 thread de álgebra linear: para redes pequenas isso é
    tão rápido quanto usar várias, e permite rodar os folds em paralelo.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        prep, rede, n_ep = treinar_modelo(X_tr, y_tr, cfg, seed=seed)
        return prever_proba(prep, rede, X_va), n_ep


def avaliar_cv(X: pd.DataFrame, y: np.ndarray, cfg: dict, n_folds: int = 5,
               seed: int = RANDOM_STATE, n_jobs: int = -1) -> dict:
    """Validação cruzada estratificada. Os folds são sempre os mesmos (mesma
    semente), então todas as configurações são comparadas nas mesmas divisões.
    Os folds são treinados em paralelo (``n_jobs`` processos; -1 = todos os núcleos).

    Devolve as métricas médias, os desvios-padrão, o nº médio de épocas, o
    tempo e as probabilidades fora-do-fold (OOF) de cada exemplo.
    """
    from joblib import Parallel, delayed
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    divisoes = list(skf.split(X, y))
    inicio = time.time()
    saidas = Parallel(n_jobs=n_jobs)(
        delayed(_rodar_fold)(X.iloc[tr], y[tr], X.iloc[va], cfg, seed + i)
        for i, (tr, va) in enumerate(divisoes))
    oof = np.zeros(len(y), dtype=float)
    por_fold, epocas = [], []
    for (tr, va), (proba, n_ep) in zip(divisoes, saidas):
        oof[va] = proba
        por_fold.append(calcular_metricas(y[va], proba))
        epocas.append(n_ep)
    df_folds = pd.DataFrame(por_fold)
    res = {m: df_folds[m].mean() for m in METRICAS}
    res.update({f"{m}_dp": df_folds[m].std() for m in METRICAS})
    res.update({"epocas": float(np.mean(epocas)), "tempo_s": time.time() - inicio, "oof": oof})
    return res


def treinar_sementes(X_tr, y_tr, cfg: dict, sementes, n_jobs: int = -1):
    """Treina uma rede por semente (em paralelo). Devolve [(prep, rede, nº de épocas), ...]."""
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    def _um(seed):
        with threadpool_limits(1):
            return treinar_modelo(X_tr, y_tr, cfg, seed=seed)
    return Parallel(n_jobs=n_jobs)(delayed(_um)(s) for s in sementes)


# --------------------------------------------------------------------------
# Configuração base e execução das etapas
# --------------------------------------------------------------------------
CONFIG_BASE = {
    # dados
    "faltantes": "moda",          # moda | categoria | remover
    "features": (),               # opções de EngenhariaFeatures
    "codificacao": "label",       # label | onehot | onehot_raras | target
    "escala": "standard",         # standard | minmax | robust | quantile | power
    "pca": None,                  # None ou fração de variância (ex.: 0.95)
    "balanceamento": "nenhum",    # nenhum | class_weight | smote | oversampling | undersampling
    # rede
    "camadas": (64, 32),          # neurônios em cada camada oculta
    "ativacao": "relu",           # relu | tanh | logistic
    "l2": 1e-4,                   # penalidade L2 (alpha do scikit-learn)
    # treino
    "otimizador": "adam",         # adam | sgd (com momentum de Nesterov)
    "lr": 1e-3,
    "batch_size": 256,
    "epocas": 200,
    "paciencia": 10,
    "reduzir_lr": False,
    "frac_parada": 0.1,
}

# Melhor configuração encontrada pela bancada (notebook bancada_redes_neurais_adult.ipynb)
MELHOR_CONFIG = {
    **CONFIG_BASE,
    "features": ("flags_capital",),
    "codificacao": "onehot",
    "camadas": (256,),
    "l2": 0.3,
    "lr": 3e-4,
    "paciencia": 30,
    "reduzir_lr": True,
}
# Limiar que maximizou o F1 nas previsões fora-do-fold da melhor configuração
LIMIAR_F1 = 0.40


class Bancada:
    """Executa etapas de experimentos e guarda o histórico de todos os resultados.

    Em cada etapa, as variações são aplicadas sobre a melhor configuração
    encontrada até o momento (busca gulosa, etapa por etapa). A melhor
    variação da etapa é escolhida pela métrica ``metrica_escolha``.
    """

    def __init__(self, X, y, n_folds=5, metrica_escolha="roc_auc", seed=RANDOM_STATE,
                 pasta_cache=None):
        self.X, self.y = X, y
        self.n_folds, self.metrica, self.seed = n_folds, metrica_escolha, seed
        self.melhor_cfg = dict(CONFIG_BASE)
        self.historico = []      # uma linha por experimento
        self.oof = {}            # probabilidades OOF por experimento
        self._cache = {}
        # Cache em disco: se o notebook for interrompido, ao rodar de novo as
        # configurações já avaliadas são lidas do disco em vez de retreinadas.
        self.pasta_cache = pasta_cache
        if pasta_cache:
            os.makedirs(pasta_cache, exist_ok=True)

    def _chave(self, cfg):
        return repr(sorted(cfg.items()))

    def _arquivo_cache(self, chave):
        import hashlib
        texto = f"{chave}|{self.n_folds}|{self.seed}|{len(self.y)}"
        return os.path.join(self.pasta_cache, hashlib.md5(texto.encode()).hexdigest() + ".joblib")

    def _avaliar_com_cache(self, cfg):
        import joblib
        chave = self._chave(cfg)
        if chave in self._cache:  # não repete configurações já avaliadas
            return self._cache[chave]
        arquivo = self._arquivo_cache(chave) if self.pasta_cache else None
        if arquivo and os.path.exists(arquivo):
            res = joblib.load(arquivo)
        else:
            res = avaliar_cv(self.X, self.y, cfg, self.n_folds, self.seed)
            if arquivo:
                joblib.dump(res, arquivo)
        self._cache[chave] = res
        return res

    def avaliar(self, cfg, etapa, nome):
        res = self._avaliar_com_cache(cfg)
        self.oof[f"{etapa} | {nome}"] = res["oof"]
        linha = {"etapa": etapa, "experimento": nome}
        linha.update({k: v for k, v in res.items() if k != "oof"})
        linha["cfg"] = dict(cfg)
        self.historico.append(linha)
        return linha

    def rodar_etapa(self, etapa: str, variacoes: dict, adotar: bool = True) -> pd.DataFrame:
        """Avalia cada variação (dicionário de chaves alteradas) sobre a melhor
        configuração atual. ``variacoes`` deve incluir a opção "atual" ({}).
        """
        linhas = []
        for nome, mudancas in variacoes.items():
            cfg = {**self.melhor_cfg, **mudancas}
            linha = self.avaliar(cfg, etapa, nome)
            linhas.append(linha)
            print(f"  {nome:<38} ROC-AUC={linha['roc_auc']:.4f}  F1={linha['f1']:.4f}  "
                  f"Acc={linha['acuracia']:.4f}  ({linha['tempo_s']:.0f}s)")
        tabela = pd.DataFrame(linhas).sort_values(self.metrica, ascending=False)
        if adotar:
            vencedora = tabela.iloc[0]
            self.melhor_cfg = dict(vencedora["cfg"])
            print(f"\n>> Escolhida: {vencedora['experimento']} "
                  f"({self.metrica} = {vencedora[self.metrica]:.4f})")
        return formatar_tabela(tabela)

    def oof_melhor(self) -> np.ndarray:
        """Probabilidades fora-do-fold da melhor configuração atual."""
        return self._avaliar_com_cache(self.melhor_cfg)["oof"]

    def resultados(self) -> pd.DataFrame:
        return pd.DataFrame(self.historico)


def formatar_tabela(tabela: pd.DataFrame) -> pd.DataFrame:
    colunas = ["experimento", "roc_auc", "roc_auc_dp", "pr_auc", "acuracia", "precisao",
               "recall", "f1", "mcc", "log_loss", "epocas", "tempo_s"]
    return tabela[colunas].set_index("experimento").round(4)
