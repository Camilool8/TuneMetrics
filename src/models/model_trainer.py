# Entrenamiento de modelos para TuneMetrics

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import xgboost as xgb
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """
    Codificador JSON especial para manejar tipos de datos de NumPy que no son serializables.
    Convierte los tipos de NumPy a tipos nativos de Python (int, float).
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class TuneMetricsModelTrainer:
    """
    Entrenador de modelos para predicción de engagement musical
    """
    
    def __init__(self, config):
        """
        Inicializa el entrenador con configuración
        
        Args:
            config (dict): Configuración del proyecto
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.results = {}
        
        # Configurar random state para reproducibilidad
        self.random_state = config.get('random_state', 42)
        
    def load_gold_data(self, data_path: str):
        """
        Carga datos Gold procesados
        
        Args:
            data_path (str): Ruta al archivo de datos Gold
        """
        print("🔄 Cargando datos Gold para modelado...")
        
        try:
            if data_path.endswith('.parquet'):
                self.gold_data = pd.read_parquet(data_path)
            else:
                self.gold_data = pd.read_csv(data_path)
            
            print(f"✅ Datos cargados: {len(self.gold_data):,} canciones")
            print(f"  📊 Features disponibles: {len(self.gold_data.columns)}")
            
            # Verificar distribución de categorías
            category_dist = self.gold_data['engagement_category'].value_counts()
            print("  📈 Distribución de engagement:")
            for category, count in category_dist.items():
                percentage = (count / len(self.gold_data)) * 100
                print(f"    {category}: {count:,} ({percentage:.1f}%)")
            
            return self.gold_data
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise
    
    def prepare_features_and_splits(self):
        """
        Prepara features y splits temporales para modelado
        """
        print("\n🔄 Preparando features y splits temporales...")
        
        # Definir features para modelado
        feature_columns = [
            'completion_rate_score', 'skip_resistance_score', 'context_preference_score',
            'consistency_score', 'platform_appeal_score', 'total_plays', 'popularity_score',
            'weekend_listening_rate', 'hour_variability', 'ms_played_mean', 'ms_played_std'
        ]
        
        # Verificar features disponibles
        available_features = [col for col in feature_columns if col in self.gold_data.columns]
        missing_features = [col for col in feature_columns if col not in self.gold_data.columns]
        
        if missing_features:
            print(f"  ⚠️  Features faltantes: {missing_features}")
        
        self.feature_names = available_features
        print(f"  ✅ Features seleccionadas: {len(self.feature_names)}")
        
        # Extraer features y target
        X = self.gold_data[self.feature_names].copy()
        y = self.gold_data['engagement_category'].copy()
        
        # Manejar valores faltantes
        X = X.fillna(X.median())
        
        # Codificar target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Crear splits temporales
        train_mask = self.gold_data['data_split'] == 'train'
        val_mask = self.gold_data['data_split'] == 'validation'
        test_mask = self.gold_data['data_split'] == 'test'
        
        self.X_train = X[train_mask]
        self.y_train = y_encoded[train_mask]
        self.X_val = X[val_mask]
        self.y_val = y_encoded[val_mask]
        self.X_test = X[test_mask]
        self.y_test = y_encoded[test_mask]
        
        print(f"  📊 Train set: {len(self.X_train):,} canciones")
        print(f"  📊 Validation set: {len(self.X_val):,} canciones")
        print(f"  📊 Test set: {len(self.X_test):,} canciones")
        
        # Verificar distribución en cada split
        for split_name, y_split in [('Train', self.y_train), ('Val', self.y_val), ('Test', self.y_test)]:
            unique, counts = np.unique(y_split, return_counts=True)
            dist = {self.label_encoder.classes_[i]: count for i, count in zip(unique, counts)}
            print(f"  📈 {split_name} distribution: {dist}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def train_random_forest(self):
        """Entrena modelo Random Forest con GridSearch"""
        print("\n🌲 Entrenando Random Forest...")
        
        # Parámetros para GridSearch
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        # Crear modelo base
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # GridSearch con validación temporal
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=tscv, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )
        
        # Entrenar en train + validation
        X_train_val = pd.concat([self.X_train, self.X_val])
        y_train_val = np.concatenate([self.y_train, self.y_val])
        
        grid_search.fit(X_train_val, y_train_val)
        
        # Mejor modelo
        best_rf = grid_search.best_estimator_
        self.models['random_forest'] = best_rf
        
        print(f"  ✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"  ✅ Mejor score CV: {grid_search.best_score_:.4f}")
        
        return best_rf
    
    def train_xgboost(self):
        """Entrena modelo XGBoost con GridSearch"""
        print("\n🚀 Entrenando XGBoost...")
        
        # Parámetros para GridSearch
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Crear modelo base
        xgb_base = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        # GridSearch con validación temporal
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=tscv, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )
        
        # Entrenar en train + validation
        X_train_val = pd.concat([self.X_train, self.X_val])
        y_train_val = np.concatenate([self.y_train, self.y_val])
        
        grid_search.fit(X_train_val, y_train_val)
        
        # Mejor modelo
        best_xgb = grid_search.best_estimator_
        self.models['xgboost'] = best_xgb
        
        print(f"  ✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"  ✅ Mejor score CV: {grid_search.best_score_:.4f}")
        
        return best_xgb
    
    def train_logistic_regression(self):
        """Entrena modelo de Regresión Logística"""
        print("\n📊 Entrenando Regresión Logística...")
        
        # Escalar features para Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        
        self.scalers['logistic_regression'] = scaler
        
        # Parámetros para GridSearch
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None],
            'max_iter': [1000]
        }
        
        # Crear modelo base
        lr_base = LogisticRegression(random_state=self.random_state)
        
        # GridSearch
        grid_search = GridSearchCV(
            lr_base, param_grid, cv=3, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )
        
        # Entrenar en train + validation escalados
        X_train_val_scaled = np.vstack([X_train_scaled, X_val_scaled])
        y_train_val = np.concatenate([self.y_train, self.y_val])
        
        grid_search.fit(X_train_val_scaled, y_train_val)
        
        # Mejor modelo
        best_lr = grid_search.best_estimator_
        self.models['logistic_regression'] = best_lr
        
        print(f"  ✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"  ✅ Mejor score CV: {grid_search.best_score_:.4f}")
        
        return best_lr
    
    def train_mlp(self):
        """Entrena modelo MLP (Red Neuronal)"""
        print("\n🧠 Entrenando MLP (Red Neuronal)...")
        
        # Escalar features para MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        
        self.scalers['mlp'] = scaler
        
        # Parámetros para GridSearch
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500]
        }
        
        # Crear modelo base
        mlp_base = MLPClassifier(random_state=self.random_state)
        
        # GridSearch
        grid_search = GridSearchCV(
            mlp_base, param_grid, cv=3, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )
        
        # Entrenar en train + validation escalados
        X_train_val_scaled = np.vstack([X_train_scaled, X_val_scaled])
        y_train_val = np.concatenate([self.y_train, self.y_val])
        
        grid_search.fit(X_train_val_scaled, y_train_val)
        
        # Mejor modelo
        best_mlp = grid_search.best_estimator_
        self.models['mlp'] = best_mlp
        
        print(f"  ✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"  ✅ Mejor score CV: {grid_search.best_score_:.4f}")
        
        return best_mlp
    
    def evaluate_models(self):
        """Evalúa todos los modelos en el conjunto de test"""
        print("\n📊 Evaluando modelos en conjunto de test...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n  🔍 Evaluando {model_name}...")
            
            # Preparar datos de test
            if model_name in ['logistic_regression', 'mlp']:
                X_test_processed = self.scalers[model_name].transform(self.X_test)
            else:
                X_test_processed = self.X_test
            
            # Predicciones
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)
            
            # Calcular métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            
            # AUC para cada clase (one-vs-rest)
            try:
                auc_ovr = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
            except:
                auc_ovr = None
            
            # Métricas por clase
            class_report = classification_report(
                self.y_test, y_pred, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Matriz de confusión
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            # Almacenar resultados
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_weighted': f1,
                'f1_macro': f1_macro,
                'auc_ovr': auc_ovr,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist()
            }
            
            # Imprimir métricas principales
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    F1-Score (weighted): {f1:.4f}")
            print(f"    F1-Score (macro): {f1_macro:.4f}")
            if auc_ovr:
                print(f"    AUC (OvR): {auc_ovr:.4f}")
            
            # Feature importance (si está disponible)
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                results[model_name]['feature_importance'] = feature_importance
                
                # Top 5 features más importantes
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"    Top 5 features:")
                for feature, importance in top_features:
                    print(f"      {feature}: {importance:.4f}")
        
        self.results = results
        return results
    
    def select_best_model(self):
        """Selecciona el mejor modelo basado en criterios definidos"""
        print("\n🏆 Seleccionando mejor modelo...")
        
        # Criterios de selección (pueden ser personalizados)
        criteria_weights = {
            'accuracy': 0.4,
            'f1_macro': 0.4,
            'f1_weighted': 0.2
        }
        
        model_scores = {}
        
        for model_name, metrics in self.results.items():
            # Calcular score compuesto
            score = sum(
                metrics[criterion] * weight 
                for criterion, weight in criteria_weights.items()
            )
            model_scores[model_name] = score
            
            print(f"  {model_name}: {score:.4f}")
        
        # Mejor modelo
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = self.models[best_model_name]
        best_score = model_scores[best_model_name]
        
        print(f"\n  🥇 Mejor modelo: {best_model_name} (score: {best_score:.4f})")
        
        # Verificar criterios mínimos
        best_metrics = self.results[best_model_name]
        min_accuracy = self.config.get('target_metrics', {}).get('min_accuracy', 0.85)
        min_f1 = self.config.get('target_metrics', {}).get('min_f1_score', 0.80)
        
        meets_criteria = (
            best_metrics['accuracy'] >= min_accuracy and 
            best_metrics['f1_macro'] >= min_f1
        )
        
        if meets_criteria:
            print(f"  ✅ Cumple criterios mínimos (Acc≥{min_accuracy}, F1≥{min_f1})")
        else:
            print(f"  ⚠️  No cumple criterios mínimos (Acc≥{min_accuracy}, F1≥{min_f1})")
        
        return best_model_name, best_model, best_metrics
    
    def save_models_and_results(self, output_dir: str):
        """Guarda modelos entrenados y resultados"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Guardando modelos y resultados en {output_dir}...")
        
        # Guardar modelos
        models_dir = output_path / "trained"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = models_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"  ✅ Modelo guardado: {model_path}")
        
        # Guardar scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = models_dir / f"{scaler_name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            print(f"  ✅ Scaler guardado: {scaler_path}")
        
        # Guardar label encoder
        encoder_path = models_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        print(f"  ✅ Label encoder guardado: {encoder_path}")
        
        # Guardar resultados
        metrics_dir = output_path / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        # Preparar resultados para JSON (sin numpy arrays)
        results_for_json = {}
        for model_name, metrics in self.results.items():
            results_for_json[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_weighted': float(metrics['f1_weighted']),
                'f1_macro': float(metrics['f1_macro']),
                'auc_ovr': float(metrics['auc_ovr']) if metrics['auc_ovr'] else None,
                'confusion_matrix': metrics['confusion_matrix'],
                'feature_importance': metrics.get('feature_importance', {})
            }
        
        # Guardar métricas en JSON
        metrics_path = metrics_dir / "model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results_for_json, f, indent=2, cls=NumpyEncoder)
        print(f"  ✅ Métricas guardadas: {metrics_path}")
        
        # Guardar configuración
        config_path = output_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"  ✅ Configuración guardada: {config_path}")
    
    def generate_training_report(self):
        """Genera reporte completo del entrenamiento"""
        best_model_name, _, best_metrics = self.select_best_model()
        
        report = f"""
🎵 TUNEMETRICS - REPORTE DE ENTRENAMIENTO DE MODELOS 🎵
{'='*70}

📊 DATOS DE ENTRENAMIENTO:
├── Total de canciones: {len(self.gold_data):,}
├── Features utilizadas: {len(self.feature_names)}
├── Train set: {len(self.X_train):,} canciones
├── Validation set: {len(self.X_val):,} canciones
└── Test set: {len(self.X_test):,} canciones

🏆 MEJOR MODELO: {best_model_name.upper()}
├── Accuracy: {best_metrics['accuracy']:.4f}
├── F1-Score (macro): {best_metrics['f1_macro']:.4f}
├── F1-Score (weighted): {best_metrics['f1_weighted']:.4f}
└── AUC (OvR): {best_metrics.get('auc_ovr', 'N/A')}

📈 COMPARACIÓN DE MODELOS:
"""
        
        for model_name, metrics in self.results.items():
            report += f"""
{model_name.upper()}:
  ├── Accuracy: {metrics['accuracy']:.4f}
  ├── F1-macro: {metrics['f1_macro']:.4f}
  └── F1-weighted: {metrics['f1_weighted']:.4f}"""
        
        # Criterios de éxito
        min_accuracy = self.config.get('target_metrics', {}).get('min_accuracy', 0.85)
        min_f1 = self.config.get('target_metrics', {}).get('min_f1_score', 0.80)
        
        meets_criteria = (
            best_metrics['accuracy'] >= min_accuracy and 
            best_metrics['f1_macro'] >= min_f1
        )
        
        report += f"""

✅ CRITERIOS DE ÉXITO:
├── Accuracy mínima requerida: {min_accuracy}
├── F1-Score mínimo requerido: {min_f1}
└── ¿Cumple criterios?: {'SÍ' if meets_criteria else 'NO'}

🎯 FEATURES MÁS IMPORTANTES (Mejor modelo):
"""
        
        if 'feature_importance' in best_metrics:
            top_features = sorted(
                best_metrics['feature_importance'].items(), 
                key=lambda x: x[1], reverse=True
            )[:10]
            
            for i, (feature, importance) in enumerate(top_features, 1):
                report += f"\n  {i:2d}. {feature}: {importance:.4f}"
        
        report += f"""

🚀 MODELOS LISTOS PARA DEPLOYMENT:
├── Modelos serializados (.pkl) guardados
├── Escaladores guardados
├── Configuración de entrenamiento guardada
└── Métricas de evaluación guardadas

📝 PRÓXIMOS PASOS:
1. Validar modelos con datos reales de negocio
2. Implementar pipeline de inferencia
3. Crear dashboard de monitoreo
4. Establecer proceso de reentrenamiento
        """
        
        return report

def main():
    """Función principal para ejecutar el entrenamiento de modelos"""
    
    # Configuración
    config = {
        'random_state': 42,
        'target_metrics': {
            'min_accuracy': 0.85,
            'min_f1_score': 0.80
        }
    }
    
    # Rutas
    gold_data_path = "data/processed/gold_data.parquet"
    models_output_dir = "models"
    
    try:
        # Crear entrenador
        trainer = TuneMetricsModelTrainer(config)
        
        # Cargar datos y preparar
        trainer.load_gold_data(gold_data_path)
        trainer.prepare_features_and_splits()
        
        # Entrenar todos los modelos
        trainer.train_random_forest()
        trainer.train_xgboost()
        trainer.train_logistic_regression()
        trainer.train_mlp()
        
        # Evaluar modelos
        trainer.evaluate_models()
        
        # Seleccionar mejor modelo
        trainer.select_best_model()
        
        # Guardar resultados
        trainer.save_models_and_results(models_output_dir)
        
        # Generar reporte
        report = trainer.generate_training_report()
        print(report)
        
        # Guardar reporte
        with open("reports/training_report.txt", "w") as f:
            f.write(report)
        
        print("\n✅ Entrenamiento de modelos completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento de modelos: {e}")
        raise

if __name__ == "__main__":
    main()