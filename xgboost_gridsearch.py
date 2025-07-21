import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, \
    precision_score
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder


def process_data_file(data_file_csv, base_path, prefix, roc_axes):
    # Criar diretório para salvar resultados
    output_dir = base_path / data_file_csv.stem / prefix
    os.makedirs(output_dir, exist_ok=True)

    # Leitura dos arquivos de dados
    data = pd.read_csv(data_file_csv)

    is_multiclass = False
    roc_label = ""
    # cenarios 1, 2 e 3
    if "1" in data_file_csv.stem:
        roc_label = "Sobrevida 1 ano"
    if "2" in data_file_csv.stem:
        roc_label = "Sobrevida 3 anos"
    if "3" in data_file_csv.stem:
        roc_label = "Sobrevida 5 anos"
    if "4" in data_file_csv.stem:
        roc_label = "Faixas de Sobrevida"
        is_multiclass = True

    # dados variável alvo
    y = data["sobrevida"]

    # transforma os dados em numeros
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # dados de entrada filtrados
    X = data.drop(columns=["sobrevida"])
    # transforma os dados em numeros
    X_encoded = pd.get_dummies(X)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded )

    # Modelo base
    mc = XGBClassifier(random_state=42, use_label_encoder=False)

    # Grade de hiperparâmetros para o grid search
    param_grid = {}
    if is_multiclass: # cenario 4
        param_grid = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [2, 4, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'objective': ['multi:softmax'],  # softmax = classes
            'eval_metric': ['mlogloss', 'merror']  # mlogloss = log-loss, merror = erro de classificação
        }
    else: # cenarios 1,2 e 3
        param_grid = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [2, 4, 6],
            'learning_rate':[0.01, 0.05, 0.1],
            'objective': ['binary:logistic'],
            'eval_metric': ['logloss', 'error', 'auc']
        }

    # numero de folds a ser usado na validação cruzada
    n_folds = 3

    # Grid Search com validação cruzada
    grid_search = GridSearchCV(
        estimator=mc,
        param_grid=param_grid,
        scoring='accuracy',
        cv=n_folds,
        verbose=1,
        n_jobs=-1
    )

    # Executa a busca
    grid_search.fit(X_train, y_train)

    # Resultados da validação cruzada
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Salvar as médias e desvios padrão da acurácia para cada alternativa testada pelo grid search
    cv_summary = cv_results[['params', 'mean_test_score', 'std_test_score']]
    cv_summary = cv_summary.sort_values(by='mean_test_score', ascending=False)
    cv_summary.to_csv(f"{output_dir}/cv_resultados.csv", index=False)

    # Obtém os resultados fold-a-fold do melhor modelo para realizar o teste de friedman
    best_index = cv_results['rank_test_score'].idxmin()
    fold_scores = []
    for i in range(n_folds):
        fold_scores.append(cv_results.loc[best_index, f'split{i}_test_score'])

    # Salva o modelo
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"{output_dir}/melhor_modelo.pkl")

    # Previsões
    y_train_pred = np.argmax(best_model.predict_proba(X_train), axis=1)
    y_score_test = best_model.predict_proba(X_test)
    y_pred_test = np.argmax(y_score_test, axis=1)

    # Métricas - Treino
    acc_train = accuracy_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred, average='macro')
    recall_train = recall_score(y_train, y_train_pred, average='macro')
    precision_train = precision_score(y_train, y_train_pred, average='macro')

    # Métricas - Teste
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    recall_test = recall_score(y_test, y_pred_test, average='macro')
    precision_test = precision_score(y_test, y_pred_test, average='macro')

    # Salvar as métricas em arquivo
    with open(f"{output_dir}/metricas_avaliacao.txt", "w", encoding='utf-8') as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(grid_search.best_params_))
        f.write("\n\nMétricas - Treinamento:\n")
        f.write(f"Acurácia:  {acc_train:.4f}\n")
        f.write(f"F1-score:  {f1_train:.4f}\n")
        f.write(f"Recall:    {recall_train:.4f}\n")
        f.write(f"Precision: {precision_train:.4f}\n")
        f.write("\nMétricas - Teste:\n")
        f.write(f"Acurácia:  {acc_test:.4f}\n")
        f.write(f"F1-score:  {f1_test:.4f}\n")
        f.write(f"Recall:    {recall_test:.4f}\n")
        f.write(f"Precision: {precision_test:.4f}\n")
        f.write(f"\n\nMédias de Acurácia para melhor fold: {fold_scores}\n")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Matriz de Confusão")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/matriz_confusao.png")
    plt.close()

    plt.figure(figsize=(10, 6))

    # Curva ROC
    plt.figure(figsize=(10, 6))
    if is_multiclass:
        # Curva ROC por classe (para o caso do modelo cenario 4)
        y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
        fpr, tpr, roc_auc = {}, {}, {}
        n_classes = len(le.classes_)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_test[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Classe {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
    else:
        # Curva ROC (para o caso do modelo cenarios 1, 2 e 3)
        y_score = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Curva ROC {roc_label} (AUC = {roc_auc:.2f})')

        # grafico com os cenarios 1, 2 e 3 integrados
        roc_axes.plot(fpr, tpr, label=f'Curva ROC {roc_label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR (Falsos Positivos)')
    plt.ylabel('TPR (Verdadeiros Positivos)')
    plt.title('Curvas ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/curva_roc.png")
    plt.close()

    print("Gráficos e métricas salvos em:", output_dir)
    print("Modelo salvo em 'melhor_modelo.pkl'")


# Grafico ROC dos cenários integrados
roc_fig, roc_axes = plt.subplots()

# diretório de resultados
data_dir = Path("data/datasets_gerados/")
result_dir = Path("resultados")

data_files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix == '.csv']
for data_file in data_files:
    process_data_file(data_file, result_dir, "_xgboost", roc_axes)

# Grafico ROC dos cenários integrados
roc_axes.plot([0, 1], [0, 1], 'k--')
roc_axes.set_xlabel('FPR (Falsos Positivos)')
roc_axes.set_ylabel('TPR (Verdadeiros Positivos)')
roc_axes.set_title(f'Curva ROC')
roc_axes.legend(loc='lower right')
roc_axes.grid(True)
roc_fig.tight_layout()
roc_fig.savefig(f"{result_dir}/curva_roc_cenarios_123_integrados_xgboost.png")
plt.close(roc_fig)


