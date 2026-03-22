# =============================================================================
#  COVID-19 Deaths Prediction Pipeline
#  Auteur  : Souheil DABBABI — Data Scientist
#  Outil   : Dataiku DSS (pipeline original) | Python (reproduction)
#  Objectif: Prédire le nombre de décès COVID-19 via un modèle de régression
# =============================================================================
#
#  Pipeline steps :
#   1. Import & exploration des données
#   2. Jointure LEFT JOIN par pays (Country)
#   3. Nettoyage & feature engineering
#   4. Filtrage sur les États-Unis (US)
#   5. Division Train / Test
#   6. Entraînement AutoML (Ridge, Random Forest, Gradient Boosting)
#   7. Scoring sur le dataset Test → Test_scored
#   8. Évaluation des performances (R², RMSE, MAE)
#
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Plot style ────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2E75B6", "#1F4E79", "#BDD7EE", "#FF6B35", "#2AB1AC"]

# =============================================================================
# ÉTAPE 1 — Chargement des données
# =============================================================================
# Pour utiliser les vrais datasets, remplace par :
#   df_covid     = pd.read_csv("02_05_2021.csv")
#   continent_map = pd.read_csv("Continent_Country_Mapping.csv")

print("=" * 60)
print("  ÉTAPE 1 — Chargement des données")
print("=" * 60)

np.random.seed(42)
N = 5000

countries_pool = [
    "US", "Brazil", "India", "France", "Germany",
    "Italy", "Spain", "UK", "Mexico", "Russia", "Turkey"
]
country_weights = [0.30, 0.10, 0.10, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05, 0.05, 0.05]

country_col = np.random.choice(countries_pool, N, p=country_weights)
confirmed   = np.random.randint(1_000, 500_000, N)
recovered   = (confirmed * np.random.uniform(0.4, 0.9, N)).astype(int)
active      = (confirmed - recovered - np.random.randint(0, 5_000, N)).clip(0)
deaths      = (confirmed * np.random.uniform(0.005, 0.04, N)).astype(int)

df_covid = pd.DataFrame({
    "Country":   country_col,
    "Confirmed": confirmed,
    "Recovered": recovered,
    "Active":    active,
    "Deaths":    deaths,
    "Date":      pd.date_range("2020-01-01", periods=N, freq="h").date,
})

continent_map = pd.DataFrame({
    "Country":   countries_pool,
    "Continent": [
        "North America", "South America", "Asia", "Europe", "Europe",
        "Europe", "Europe", "Europe", "North America", "Europe", "Asia"
    ],
})

print(f"  Dataset COVID    : {df_covid.shape[0]:,} lignes × {df_covid.shape[1]} colonnes")
print(f"  Dataset Mapping  : {continent_map.shape[0]} pays × {continent_map.shape[1]} colonnes")
print()


# =============================================================================
# ÉTAPE 2 — Jointure LEFT JOIN (Dataiku : recette Join)
# =============================================================================

print("=" * 60)
print("  ÉTAPE 2 — Jointure des datasets (LEFT JOIN sur Country)")
print("=" * 60)

df_joined = pd.merge(df_covid, continent_map, on="Country", how="left")

print(f"  Avant jointure  : {df_covid.shape}")
print(f"  Après jointure  : {df_joined.shape}")
print(f"  Colonnes ajoutées : {set(df_joined.columns) - set(df_covid.columns)}")
print()


# =============================================================================
# ÉTAPE 3 — Préparation des données (Dataiku : recette Prepare)
# =============================================================================

print("=" * 60)
print("  ÉTAPE 3 — Nettoyage & Feature Engineering")
print("=" * 60)

df_prep = df_joined.copy()

# Valeurs manquantes
df_prep["Continent"].fillna("Unknown", inplace=True)

# Feature engineering
df_prep["Mortality_Rate"] = df_prep["Deaths"]    / (df_prep["Confirmed"] + 1)
df_prep["Recovery_Rate"]  = df_prep["Recovered"] / (df_prep["Confirmed"] + 1)
df_prep["Active_Rate"]    = df_prep["Active"]     / (df_prep["Confirmed"] + 1)

# Encodage des variables catégorielles
le_country   = LabelEncoder()
le_continent = LabelEncoder()
df_prep["Country_enc"]   = le_country.fit_transform(df_prep["Country"])
df_prep["Continent_enc"] = le_continent.fit_transform(df_prep["Continent"])

# Suppression des colonnes non utiles
df_prep.drop(columns=["Date"], inplace=True)

print(f"  Shape final     : {df_prep.shape}")
print(f"  Nouvelles features : Mortality_Rate, Recovery_Rate, Active_Rate")
print(f"  Encodages ajoutés  : Country_enc, Continent_enc")
print()


# =============================================================================
# ÉTAPE 4 — Filtrage US (Dataiku : recette Filter)
# =============================================================================

print("=" * 60)
print("  ÉTAPE 4 — Filtrage sur les États-Unis (Country == 'US')")
print("=" * 60)

df_us = df_prep[df_prep["Country"] == "US"].copy()

print(f"  Avant filtre : {len(df_prep):,} lignes")
print(f"  Après filtre : {len(df_us):,} lignes (US uniquement)")
print()


# =============================================================================
# ÉTAPE 5 — Division Train / Test (Dataiku : recette Split)
# =============================================================================

print("=" * 60)
print("  ÉTAPE 5 — Division Train / Test (80% / 20%)")
print("=" * 60)

FEATURES = [
    "Confirmed", "Recovered", "Active",
    "Mortality_Rate", "Recovery_Rate", "Active_Rate",
    "Country_enc", "Continent_enc",
]
TARGET = "Deaths"

X = df_us[FEATURES]
y = df_us[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Train_Dataset : {X_train.shape[0]:,} lignes ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test          : {X_test.shape[0]:,} lignes ({X_test.shape[0]/len(X)*100:.0f}%)")
print()


# =============================================================================
# ÉTAPE 6 — Entraînement AutoML (Dataiku : Lab AutoML Regression)
# =============================================================================

print("=" * 60)
print("  ÉTAPE 6 — Entraînement du modèle (AutoML Regression)")
print("  Variable cible : Deaths")
print("=" * 60)

models = {
    "Ridge Regression":   Ridge(alpha=1.0),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results        = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    results[name] = {
        "R² train": round(r2_score(y_train, y_pred_tr), 4),
        "R² test":  round(r2_score(y_test,  y_pred_te), 4),
        "RMSE":     round(np.sqrt(mean_squared_error(y_test, y_pred_te)), 2),
        "MAE":      round(mean_absolute_error(y_test, y_pred_te), 2),
    }
    trained_models[name] = model
    print(f"  ✓ {name:<25} R²={results[name]['R² test']:.4f}  "
          f"RMSE={results[name]['RMSE']:.1f}  MAE={results[name]['MAE']:.1f}")

df_results = pd.DataFrame(results).T
best_name  = df_results["R² test"].idxmax()
best_model = trained_models[best_name]

print()
print(f"  Meilleur modèle sélectionné : {best_name}")
print()


# =============================================================================
# ÉTAPE 7 — Scoring sur le dataset Test (Dataiku : recette Score)
# =============================================================================

print("=" * 60)
print("  ÉTAPE 7 — Scoring → génération de Test_scored")
print("=" * 60)

y_pred = best_model.predict(X_test)

test_scored = X_test.copy()
test_scored["Deaths_actual"]    = y_test.values
test_scored["Deaths_predicted"] = y_pred.round().astype(int)
test_scored["Residual"]         = test_scored["Deaths_actual"] - test_scored["Deaths_predicted"]
test_scored["Abs_Error"]        = test_scored["Residual"].abs()

print(f"  Dataset Test_scored généré : {test_scored.shape}")
print()
print(test_scored[["Deaths_actual", "Deaths_predicted", "Residual", "Abs_Error"]].head(8).to_string())
print()


# =============================================================================
# ÉTAPE 8 — Évaluation (Dataiku : Evaluate Recipe)
# =============================================================================

print("=" * 60)
print(f"  ÉTAPE 8 — Évaluation — {best_name}")
print("=" * 60)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"  R²   : {r2:.4f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print()

# ── Dashboard d'évaluation ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Predicted vs Actual
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y_test, y_pred, alpha=0.45, color=PALETTE[0], s=18, label="Prédictions")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax1.plot(lims, lims, "r--", lw=1.5, label="Parfait (y=x)")
ax1.set_xlabel("Valeurs réelles (Deaths)", fontsize=11)
ax1.set_ylabel("Valeurs prédites (Deaths)", fontsize=11)
ax1.set_title(f"Prédictions vs Valeurs Réelles  |  R² = {r2:.4f}",
              fontsize=13, fontweight="bold", color="#1F4E79")
ax1.legend()

# 2. Metrics panel
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
ax2.text(0.05, 0.5,
         f"MÉTRIQUES\n\n"
         f"  R²   =  {r2:.4f}\n"
         f"  RMSE =  {rmse:.2f}\n"
         f"  MAE  =  {mae:.2f}\n\n"
         f"  Modèle :\n  {best_name}",
         transform=ax2.transAxes, fontsize=11,
         verticalalignment="center", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor=PALETTE[2], alpha=0.35))

# 3. Distribution des résidus
residuals = y_test.values - y_pred
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(residuals, bins=40, color=PALETTE[3], edgecolor="white", alpha=0.8)
ax3.axvline(0, color="red", lw=1.5, linestyle="--")
ax3.set_title("Distribution des Résidus", fontsize=12, fontweight="bold", color="#1F4E79")
ax3.set_xlabel("Résidu")
ax3.set_ylabel("Fréquence")

# 4. Feature Importance
ax4 = fig.add_subplot(gs[1, 1])
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    importances.plot(kind="barh", ax=ax4, color=PALETTE[0], edgecolor="white")
    ax4.set_title("Feature Importance", fontsize=12, fontweight="bold", color="#1F4E79")
    ax4.set_xlabel("Importance")
else:
    ax4.text(0.5, 0.5, "Feature importance\nnon disponible",
             ha="center", va="center", transform=ax4.transAxes, fontsize=11)
    ax4.axis("off")

# 5. Comparaison R² des modèles
ax5 = fig.add_subplot(gs[1, 2])
r2_scores = {k: v["R² test"] for k, v in results.items()}
bar_colors = [PALETTE[1] if k == best_name else PALETTE[2] for k in r2_scores]
bars = ax5.bar(range(len(r2_scores)), list(r2_scores.values()),
               color=bar_colors, edgecolor="white")
ax5.set_xticks(range(len(r2_scores)))
ax5.set_xticklabels([k.replace(" ", "\n") for k in r2_scores], fontsize=9)
ax5.set_title("Comparaison R² (Test)", fontsize=12, fontweight="bold", color="#1F4E79")
ax5.set_ylabel("R²")
ax5.set_ylim(0, 1.05)
for bar, val in zip(bars, r2_scores.values()):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle(
    "Tableau de Bord — Évaluation du Modèle COVID-19 Deaths Prediction",
    fontsize=14, fontweight="bold", color="#1F4E79", y=1.01
)

plt.savefig("model_evaluation.png", dpi=130, bbox_inches="tight")
plt.show()

print("  Graphique sauvegardé → model_evaluation.png")
print()
print("=" * 60)
print("  PIPELINE TERMINÉ")
print(f"  Modèle final : {best_name}")
print(f"  R² = {r2:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}")
print("=" * 60)
