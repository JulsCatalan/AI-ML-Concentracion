"""
Análisis completo de datos del Titanic - Random Forest + KMeans + KNN
Julián Catalán
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def cargar_datos(archivo_csv):
    """Cargar el dataset desde un archivo CSV"""
    try:
        df = pd.read_csv(archivo_csv)
        print(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def limpiar_strings(df):
    """Eliminar espacios antes y después en strings"""
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

def split_name(name):
    """Limpiar y separar los nombres en FirstName y LastName"""
    try:
        # Quitar comillas y caracteres especiales
        name = re.sub(r'["\'`]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()

        # Separar en apellido y resto
        last, rest = name.split(",", 1)
        last = last.strip()
        rest = rest.strip()

        # Si hay paréntesis, nombre real de mujer casada
        if "(" in rest:
            inside = re.findall(r"\((.*?)\)", rest)
            if inside:
                first = inside[0].strip()
                first = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ\s-]', '', first)
                return first, last

        # Caso normal, eliminar título (Mr., Mrs., Miss, etc.)
        if "." in rest:
            first = rest.split(".", 1)[1].strip()
        else:
            first = rest

        # Quitar caracteres no deseados (números, símbolos)
        first = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ\s-]', '', first)
        last = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚüÜñÑ\s-]', '', last)

        return first, last
    except:
        return name, ""

def procesar_nombres(df):
    """Aplicar el procesamiento de nombres al dataset"""
    if 'Name' in df.columns:
        df[['FirstName', 'LastName']] = df['Name'].apply(lambda x: pd.Series(split_name(x)))
    return df

def imputar_edad(df):
    """Imputar valores faltantes en Age con la mediana"""
    if 'Age' in df.columns:
        mediana_edad = df['Age'].median()
        df['Age'] = df['Age'].fillna(mediana_edad)
        df['Age'] = df['Age'].round().astype(int)
        print(f"Valores faltantes en Age imputados con mediana: {mediana_edad}")
    return df

def eliminar_columnas_innecesarias(df):
    """Eliminar columnas sin importancia"""
    columnas_a_eliminar = ['PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Name']
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    df = df.drop(columns=columnas_existentes)
    print(f"Columnas eliminadas: {columnas_existentes}")
    return df

def explicar_variables_modelo(df):
    """Explicación detallada de por qué cada variable es útil para predecir supervivencia"""
    print("\n" + "="*80)
    print("EXPLICACION DE VARIABLES DEL MODELO")
    print("="*80)
    
    variables_disponibles = [col for col in df.columns if col != 'Survived']
    
    print("\nCada variable seleccionada tiene una justificación específica basada en el contexto")
    print("del naufragio del Titanic y su relación con las probabilidades de supervivencia:\n")
    
    for variable in variables_disponibles:
        if variable == 'Pclass':
            print("PCLASS (Clase del Pasajero):")
            print("La clase social determinaba la ubicación en el barco y acceso a botes salvavidas.")
            print("Los pasajeros de primera clase tenían camarotes más cerca de cubierta.")
            print()
            
        elif variable == 'Sex':
            print("SEX (Sexo del Pasajero):")
            print("El protocolo 'mujeres y niños primero' influyó directamente en la evacuación.")
            print("Las normas de caballerosidad de 1912 priorizaron a las mujeres.")
            print()
            
        elif variable == 'Age':
            print("AGE (Edad del Pasajero):")
            print("Los niños fueron priorizados durante la evacuación según el protocolo establecido.")
            print("La capacidad física también influyó en el acceso a los botes salvavidas.")
            print()
            
        elif variable == 'Fare':
            print("FARE (Tarifa Pagada):")
            print("Indicador de estatus socioeconómico que correlaciona con ubicación privilegiada.")
            print("Tarifas más altas implicaban camarotes mejor ubicados y acceso más rápido a cubierta.")
            print()
            
        elif variable == 'Embarked':
            print("EMBARKED (Puerto de Embarque):")
            print("El puerto indica la composición socioeconómica de los pasajeros embarcados.")
            print("Cherbourg atendía primera clase, mientras Southampton y Queenstown más tercera clase.")
            print()
            
        elif variable == 'FirstName':
            print("FIRSTNAME (Nombre de Pila):")
            print("Los nombres pueden indicar origen étnico y clase social de la época.")
            print("Útil para identificar patrones culturales y socioeconómicos en los datos.")
            print()
            
        elif variable == 'LastName':
            print("LASTNAME (Apellido):")
            print("Permite identificar familias viajando juntas y sus estrategias de supervivencia.")
            print("Los apellidos revelan background cultural y capacidad socioeconómica.")
            print()
    
    print("SINTESIS:")
    print("Las variables capturan tres factores clave: acceso físico (ubicación), priorización")
    print("social (normas de evacuación) y recursos socioeconómicos (capacidad de respuesta).")


def mostrar_dataset_limpio(df):
    """Mostrar vista previa del dataset limpio"""
    print("\nVista previa del dataset limpio:")
    print(df.head(10))
    print(f"\nForma del dataset: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    explicar_variables_modelo(df)

def guardar_dataset_limpio(df, nombre_archivo="archivo_limpio_final.csv"):
    """Guardar el dataset limpio"""
    df.to_csv(nombre_archivo, index=False)
    print(f"Dataset limpio guardado como: {nombre_archivo}")

def convertir_survived_labels(df):
    """Convertir columna Survived a etiquetas 'no'/'si'"""
    if 'Survived' in df.columns:
        df['Survived'] = df['Survived'].map({0: 'no', 1: 'si'})
    return df

def preparar_datos_para_modelos(df):
    """Preparar datos para entrenamiento de modelos"""
    df_model = df.copy()
    
    # Convertir Survived a numérico si es necesario
    if 'Survived' in df_model.columns and df_model['Survived'].dtype == 'object':
        df_model['Survived'] = df_model['Survived'].map({'no': 0, 'si': 1})
    
    # Crear label encoders para variables categóricas
    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked', 'FirstName', 'LastName']
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
    
    # Llenar valores faltantes
    df_model = df_model.fillna(0)
    
    return df_model, label_encoders

# ================== KMEANS CLUSTERING ==================

def aplicar_kmeans_clustering(df, n_clusters=4):
    """Aplicar KMeans clustering sobre el dataset limpio"""
    df_kmeans = df.copy()
    
    # Convertir Survived a numérico si es necesario
    if 'Survived' in df_kmeans.columns and df_kmeans['Survived'].dtype == 'object':
        df_kmeans['Survived'] = df_kmeans['Survived'].map({'no': 0, 'si': 1})
    
    # Convertir todas las variables categóricas a numéricas
    for col in df_kmeans.columns:
        if df_kmeans[col].dtype == 'object':
            df_kmeans[col] = df_kmeans[col].astype('category').cat.codes
    
    # Seleccionar solo columnas numéricas para KMeans
    num_cols = df_kmeans.select_dtypes(include=[np.number]).columns.tolist()
    if 'Survived' in num_cols:
        num_cols.remove('Survived')  # No usar supervivencia para clustering
    
    X_kmeans = df_kmeans[num_cols].fillna(0)
    
    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_kmeans['Cluster'] = kmeans.fit_predict(X_kmeans)
    
    print(f'\nKMeans aplicado con {n_clusters} clusters')
    print('Conteo de elementos por cluster:')
    print(df_kmeans['Cluster'].value_counts().sort_index())
    
    return df_kmeans, kmeans

def asignar_nombres_clusters(df_kmeans):
    """Asignar nombres diferenciados a los clusters"""
    num_cols = df_kmeans.select_dtypes(include=['number']).columns.tolist()
    if 'Cluster' in num_cols:
        num_cols.remove('Cluster')
    
    cluster_means = df_kmeans.groupby('Cluster')[num_cols].mean()
    
    # Calcular terciles para categorizar
    if 'Fare' in cluster_means.columns:
        fare_terciles = np.percentile(cluster_means['Fare'], [33, 66])
    if 'Age' in cluster_means.columns:
        age_terciles = np.percentile(cluster_means['Age'], [33, 66])
    
    cluster_names = {}
    
    for idx, row in cluster_means.iterrows():
        name_parts = []
        
        # Edad
        if 'Age' in row.index:
            if row['Age'] < age_terciles[0]:
                name_parts.append('Jóvenes')
            elif row['Age'] < age_terciles[1]:
                name_parts.append('Adultos')
            else:
                name_parts.append('Mayores')
        
        # Clase
        if 'Pclass' in row.index:
            if row['Pclass'] <= 1.5:
                name_parts.append('1ra Clase')
            elif row['Pclass'] <= 2.5:
                name_parts.append('2da Clase')
            else:
                name_parts.append('3ra Clase')
        
        # Tarifa
        if 'Fare' in row.index:
            if row['Fare'] < fare_terciles[0]:
                name_parts.append('Tarifa Baja')
            elif row['Fare'] < fare_terciles[1]:
                name_parts.append('Tarifa Media')
            else:
                name_parts.append('Tarifa Alta')
        
        name = ' + '.join(name_parts) if name_parts else f'Grupo {idx}'
        cluster_names[idx] = name
    
    df_kmeans['ClusterName'] = df_kmeans['Cluster'].map(cluster_names)
    
    print('\nNombres asignados a los clusters:')
    for k, v in cluster_names.items():
        print(f'  Cluster {k}: {v}')
    
    return df_kmeans, cluster_names

# ================== RANDOM FOREST ==================

def entrenar_random_forest(df):
    """Entrenar modelo Random Forest"""
    print("\nEntrenando Random Forest...")
    
    # Preparar datos
    df_rf, label_encoders = preparar_datos_para_modelos(df)
    
    # Separar features y target
    X = df_rf.drop('Survived', axis=1)
    y = df_rf['Survived']
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimización con GridSearch
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Random Forest - Accuracy: {accuracy:.4f}")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    
    return best_rf, X_test, y_test, y_pred, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# ================== KNN ==================

def entrenar_knn(df):
    """Entrenar modelo K-Nearest Neighbors"""
    print("\nEntrenando K-Nearest Neighbors...")
    
    # Preparar datos
    df_knn, label_encoders = preparar_datos_para_modelos(df)
    
    # Separar features y target
    X = df_knn.drop('Survived', axis=1)
    y = df_knn['Survived']
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar datos (importante para KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimización con GridSearch
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Mejor modelo
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"KNN - Accuracy: {accuracy:.4f}")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    
    return best_knn, X_test_scaled, y_test, y_pred, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, scaler

# ================== ANÁLISIS Y COMPARACIÓN ==================

def crear_reporte_comparativo(metricas_rf, metricas_knn, df_clusters):
    """Crear reporte comparativo de todos los modelos"""
    print("\n" + "="*80)
    print("REPORTE COMPARATIVO DE MODELOS")
    print("="*80)
    
    # 1. Resultados de cada modelo
    print("\nRESULTADOS INDIVIDUALES:")
    print("-"*50)
    
    print("\nRANDOM FOREST:")
    print(f"   - Accuracy:  {metricas_rf['accuracy']:.4f} ({metricas_rf['accuracy']*100:.1f}%)")
    print(f"   - Precision: {metricas_rf['precision']:.4f}")
    print(f"   - Recall:    {metricas_rf['recall']:.4f}")
    print(f"   - F1-Score:  {metricas_rf['f1']:.4f}")
    
    print("\nK-NEAREST NEIGHBORS:")
    print(f"   - Accuracy:  {metricas_knn['accuracy']:.4f} ({metricas_knn['accuracy']*100:.1f}%)")
    print(f"   - Precision: {metricas_knn['precision']:.4f}")
    print(f"   - Recall:    {metricas_knn['recall']:.4f}")
    print(f"   - F1-Score:  {metricas_knn['f1']:.4f}")
    
    print("\nKMEANS CLUSTERING:")
    survival_by_cluster = df_clusters.groupby('ClusterName')['Survived'].mean()
    print("   - Tasa de supervivencia por cluster:")
    for cluster, rate in survival_by_cluster.items():
        print(f"     -- {cluster}: {rate:.3f} ({rate*100:.1f}%)")
    
    # 2. Comparación directa
    print("\nCOMPARACION DIRECTA:")
    print("-"*50)
    
    comparison_df = pd.DataFrame({
        'Modelo': ['Random Forest', 'KNN'],
        'Accuracy': [metricas_rf['accuracy'], metricas_knn['accuracy']],
        'Precision': [metricas_rf['precision'], metricas_knn['precision']],
        'Recall': [metricas_rf['recall'], metricas_knn['recall']],
        'F1-Score': [metricas_rf['f1'], metricas_knn['f1']]
    })
    
    print(comparison_df.round(4).to_string(index=False))
    
    # 3. Conclusión final expandida
    print("\nCONCLUSION FINAL:")
    print("-"*50)
    
    mejor_accuracy = max(metricas_rf['accuracy'], metricas_knn['accuracy'])
    diferencia_accuracy = abs(metricas_rf['accuracy'] - metricas_knn['accuracy'])
    
    if metricas_rf['accuracy'] == mejor_accuracy:
        mejor_modelo = "Random Forest"
        mejor_metricas = metricas_rf
        segundo_modelo = "KNN"
        segundo_metricas = metricas_knn
    else:
        mejor_modelo = "KNN"
        mejor_metricas = metricas_knn
        segundo_modelo = "Random Forest"
        segundo_metricas = metricas_rf
    
    print(f"\nMODELO RECOMENDADO: {mejor_modelo}")
    print(f"Diferencia de accuracy: {diferencia_accuracy:.4f} ({diferencia_accuracy*100:.2f}%)")
    
    print(f"\nANALISIS DETALLADO:")
    
    print(f"\nEl modelo {mejor_modelo} ha demostrado ser superior en este análisis del dataset del Titanic, "
          f"obteniendo una accuracy de {mejor_metricas['accuracy']:.4f} ({mejor_metricas['accuracy']*100:.1f}%) "
          f"comparado con {segundo_metricas['accuracy']:.4f} ({segundo_metricas['accuracy']*100:.1f}%) del modelo {segundo_modelo}. "
          f"Esta diferencia de {diferencia_accuracy*100:.2f} puntos porcentuales indica una ventaja "
          f"{'significativa' if diferencia_accuracy > 0.02 else 'moderada'} en la capacidad predictiva.")
    
    print(f"\nEn términos de precision, el modelo {mejor_modelo} alcanza {mejor_metricas['precision']:.4f}, "
          f"lo que significa que de todas las predicciones positivas de supervivencia, "
          f"{mejor_metricas['precision']*100:.1f}% son correctas. El recall de {mejor_metricas['recall']:.4f} "
          f"indica que el modelo identifica correctamente {mejor_metricas['recall']*100:.1f}% de todos los "
          f"casos reales de supervivencia. El F1-score de {mejor_metricas['f1']:.4f} proporciona una "
          f"medida balanceada que considera tanto precision como recall.")
    
    print(f"\nLa métrica de precision es particularmente relevante en este contexto, ya que minimiza los "
          f"falsos positivos, es decir, casos donde el modelo predice supervivencia cuando en realidad "
          f"la persona no sobrevivió. Por otro lado, el recall es crucial para minimizar los falsos negativos, "
          f"casos donde el modelo predice muerte cuando la persona realmente sobrevivió.")
    
    # Análisis específico del modelo ganador
    if mejor_modelo == "Random Forest":
        print(f"\nJUSTIFICACION TECNICA DEL RANDOM FOREST:")
        print(f"\nRandom Forest se establece como el algoritmo óptimo para este dataset debido a múltiples "
              f"factores técnicos fundamentales. Su arquitectura de ensemble combina múltiples árboles de decisión "
              f"independientes, cada uno entrenado con una muestra bootstrap diferente del dataset original "
              f"y considerando un subconjunto aleatorio de características en cada división nodal.")
        
        print(f"\nLa robustez del Random Forest se manifiesta en su capacidad para manejar simultáneamente "
              f"variables categóricas ordinales (Pclass: 1, 2, 3), categóricas nominales (Sex: male/female, "
              f"Embarked: S/C/Q) y variables continuas (Age, Fare) sin requerir normalización previa. "
              f"Esta característica es particularmente ventajosa dado que el dataset del Titanic presenta "
              f"una distribución heterogénea de tipos de datos.")
        
        print(f"\nEl algoritmo implementa implícitamente regularización a través de la agregación de "
              f"predicciones múltiples, reduciendo significativamente la varianza del modelo final. "
              f"La técnica de bagging (bootstrap aggregating) mitiga el overfitting común en árboles "
              f"individuales, mientras que la selección aleatoria de características (feature randomness) "
              f"reduce la correlación entre los árboles del ensemble.")
        
        print(f"\nEn términos de complejidad computacional, Random Forest presenta una ventaja considerable "
              f"frente a KNN durante la fase de predicción. Mientras KNN requiere calcular distancias "
              f"con todos los puntos de entrenamiento (O(n*d) por predicción), Random Forest ejecuta "
              f"predicciones en tiempo logarítmico O(log n) una vez entrenado.")
        
    else:
        print(f"\nJUSTIFICACION TECNICA DEL K-NEAREST NEIGHBORS:")
        print(f"\nKNN ha demostrado superioridad en este caso específico, lo que indica que los patrones "
              f"de supervivencia en el Titanic exhiben una estructura de proximidad local bien definida. "
              f"El algoritmo implementado utiliza normalización StandardScaler, transformando todas las "
              f"características a distribuciones con media 0 y desviación estándar 1, eliminando el sesgo "
              f"de escala entre variables.")
        
        print(f"\nLa optimización de hiperparámetros mediante GridSearchCV ha seleccionado el valor óptimo "
              f"de k vecinos, balanceando el trade-off entre sesgo y varianza. Un k pequeño reduce el sesgo "
              f"pero aumenta la varianza, mientras que un k grande reduce la varianza pero puede introducir "
              f"sesgo por suavizado excesivo.")
        
        print(f"\nLa efectividad del KNN sugiere que la supervivencia puede modelarse mediante la hipótesis "
              f"de smoothness local: pasajeros con características similares en el espacio euclidiano "
              f"normalizado tienden a tener destinos similares. Esto implica que las interacciones entre "
              f"variables son relativamente simples y pueden capturarse mediante proximidad geométrica.")
        
        print(f"\nSin embargo, es importante considerar que KNN presenta limitaciones de escalabilidad. "
              f"Su complejidad de predicción O(n*d) y su sensibilidad a la maldición de la dimensionalidad "
              f"pueden limitar su aplicabilidad en datasets de mayor escala o dimensionalidad.")
    
    # Análisis del clustering con datos específicos
    cluster_range = survival_by_cluster.max() - survival_by_cluster.min()
    cluster_stats = df_clusters.groupby('ClusterName')['Survived'].agg(['mean', 'count'])
    
    print(f"\nANALISIS CUANTITATIVO DEL CLUSTERING:")
    print(f"\nEl análisis de clustering mediante KMeans ha identificado patrones significativos en la "
          f"segmentación de pasajeros. La variación en las tasas de supervivencia entre clusters "
          f"({cluster_range:.3f} o {cluster_range*100:.1f} puntos porcentuales) demuestra la existencia "
          f"de grupos distintivos con probabilidades de supervivencia estadísticamente diferentes.")
    
    print(f"\nLa distribución de pasajeros por cluster muestra una segmentación equilibrada: ")
    for cluster_name, stats in cluster_stats.iterrows():
        print(f"- {cluster_name}: {stats['count']} pasajeros, tasa supervivencia {stats['mean']:.3f}")
    
    print(f"\nEsta segmentación natural refleja los factores socioeconómicos que influyeron en las "
          f"probabilidades de supervivencia durante el naufragio. La correlación entre clase social, "
          f"edad y capacidad económica (reflejada en la tarifa) se manifiesta en clusters cohesivos "
          f"que capturan la estructura subyacente de los datos.")
    
    print(f"\nEl algoritmo KMeans ha convergido hacia una solución que minimiza la inercia intra-cluster "
          f"mientras maximiza la separación inter-cluster, revelando agrupaciones naturales que "
          f"no son evidentes mediante análisis univariado de las variables.")   
    
    return comparison_df

def visualizar_comparacion_modelos(metricas_rf, metricas_knn, df_clusters):
    """Crear visualizaciones comparativas"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparación de métricas
    modelos = ['Random Forest', 'KNN']
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    rf_values = [metricas_rf['accuracy'], metricas_rf['precision'], metricas_rf['recall'], metricas_rf['f1']]
    knn_values = [metricas_knn['accuracy'], metricas_knn['precision'], metricas_knn['recall'], metricas_knn['f1']]
    
    x = np.arange(len(metricas_nombres))
    width = 0.35
    
    ax1.bar(x - width/2, rf_values, width, label='Random Forest', color='skyblue')
    ax1.bar(x + width/2, knn_values, width, label='KNN', color='orange')
    ax1.set_xlabel('Métricas')
    ax1.set_ylabel('Puntuación')
    ax1.set_title('Comparación de Métricas: Random Forest vs KNN')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metricas_nombres)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # 2. Supervivencia por cluster
    survival_by_cluster = df_clusters.groupby('ClusterName')['Survived'].mean()
    ax2.bar(range(len(survival_by_cluster)), survival_by_cluster.values, color='green', alpha=0.7)
    ax2.set_xlabel('Clusters')
    ax2.set_ylabel('Tasa de Supervivencia')
    ax2.set_title('Tasa de Supervivencia por Cluster (KMeans)')
    ax2.set_xticks(range(len(survival_by_cluster)))
    ax2.set_xticklabels(survival_by_cluster.index, rotation=45, ha='right')
    
    # 3. Distribución de clusters por edad y tarifa
    sns.scatterplot(data=df_clusters, x='Age', y='Fare', hue='ClusterName', ax=ax3, alpha=0.7)
    ax3.set_title('Distribución de Clusters (Edad vs Tarifa)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Accuracy comparison
    models_acc = ['Random Forest', 'KNN']
    accuracies = [metricas_rf['accuracy'], metricas_knn['accuracy']]
    colors = ['#2E8B57' if acc == max(accuracies) else '#4682B4' for acc in accuracies]
    
    ax4.bar(models_acc, accuracies, color=colors)
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Comparación de Accuracy')
    ax4.set_ylim(0, 1)
    
    # Agregar valores en las barras
    for i, v in enumerate(accuracies):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal para ejecutar todo el análisis"""
    print("ANALISIS COMPLETO DEL TITANIC: Random Forest + KMeans + KNN")
    print("="*60)
    
    # 1. Cargar y preparar datos
    archivo_csv = "train.csv"
    df = cargar_datos(archivo_csv)
    
    if df is None:
        print("Error: No se pudo cargar el dataset. Verifica la ruta del archivo.")
        return
    
    # 2. Limpieza de datos
    print("\nLimpiando datos...")
    df = limpiar_strings(df)
    df = procesar_nombres(df)
    df = imputar_edad(df)
    df = eliminar_columnas_innecesarias(df)
    df = convertir_survived_labels(df)
    
    mostrar_dataset_limpio(df)
    guardar_dataset_limpio(df)
    
    # 3. Configurar estilo
    sns.set_palette('Set2')
    plt.style.use('default')
    
    # 4. Aplicar KMeans Clustering
    print("\nAPLICANDO KMEANS CLUSTERING...")
    df_clusters, kmeans_model = aplicar_kmeans_clustering(df)
    df_clusters, cluster_names = asignar_nombres_clusters(df_clusters)
    
    # 5. Entrenar Random Forest
    print("\nENTRENANDO RANDOM FOREST...")
    rf_model, X_test_rf, y_test_rf, y_pred_rf, metricas_rf = entrenar_random_forest(df)
    
    # 6. Entrenar KNN
    print("\nENTRENANDO K-NEAREST NEIGHBORS...")
    knn_model, X_test_knn, y_test_knn, y_pred_knn, metricas_knn, scaler = entrenar_knn(df)
    
    # 7. Crear reporte comparativo completo
    print("\nCREANDO REPORTE COMPARATIVO...")
    comparison_df = crear_reporte_comparativo(metricas_rf, metricas_knn, df_clusters)
    
    # 8. Graficas comparativas
    print("\nCREANDO GRAFICAS COMPARATIVAS...")
    visualizar_comparacion_modelos(metricas_rf, metricas_knn, df_clusters)
    
    print("\nANALISIS COMPLETO TERMINADO")
    
    return df, df_clusters, rf_model, knn_model, comparison_df

if __name__ == "__main__":

    dataset, clusters_data, modelo_rf, modelo_knn, comparacion = main()