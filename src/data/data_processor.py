# Pipeline de procesamiento de datos para TuneMetrics

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TuneMetricsDataProcessor:
    """
    Procesador de datos para TuneMetrics que implementa arquitectura Medallion:
    Bronze (raw) → Silver (cleaned) → Gold (aggregated & features)
    """
    
    def __init__(self, config):
        """
        Inicializa el procesador con configuración
        
        Args:
            config (dict): Configuración del proyecto
        """
        self.config = config
        self.bronze_data = None
        self.silver_data = None
        self.gold_data = None
        
        # Configuraciones clave
        self.estimated_duration_ms = config.get('estimated_song_duration_ms', 210000)  # 3.5 min
        self.engagement_thresholds = config.get('engagement_thresholds', {'high': 0.75, 'medium': 0.45})
        
    def load_bronze_data(self, data_path: str):
        """
        Carga datos raw en capa Bronze
        
        Args:
            data_path (str): Ruta al archivo CSV raw
        """
        print("🔄 Cargando datos en capa BRONZE...")
        
        try:
            self.bronze_data = pd.read_csv(data_path)
            print(f"✅ Bronze: {len(self.bronze_data):,} registros cargados")
            
            # Validaciones básicas
            required_columns = [
                'spotify_track_uri', 'ts', 'platform', 'ms_played', 
                'track_name', 'artist_name', 'album_name', 
                'reason_start', 'reason_end', 'shuffle', 'skipped'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.bronze_data.columns]
            if missing_columns:
                raise ValueError(f"Columnas faltantes: {missing_columns}")
            
            print("✅ Validación de estructura: PASADA")
            return self.bronze_data
            
        except Exception as e:
            print(f"❌ Error cargando Bronze data: {e}")
            raise
    
    def create_silver_data(self):
        """
        Procesa Bronze → Silver: Limpieza y validación de datos
        """
        print("\n🔄 Procesando Bronze → SILVER...")
        
        if self.bronze_data is None:
            raise ValueError("Debe cargar Bronze data primero")
        
        # Copiar datos para Silver
        silver = self.bronze_data.copy()
        initial_count = len(silver)
        
        # 1. Conversión de tipos de datos
        print("  📝 Convirtiendo tipos de datos...")
        silver['ts'] = pd.to_datetime(silver['ts'])
        silver['ms_played'] = pd.to_numeric(silver['ms_played'], errors='coerce')
        silver['shuffle'] = silver['shuffle'].astype(bool)
        silver['skipped'] = silver['skipped'].astype(bool)
        
        # 2. Limpieza de datos
        print("  🧹 Limpiando datos...")
        
        # Eliminar registros con valores críticos faltantes
        critical_nulls = silver[
            silver['spotify_track_uri'].isna() |
            silver['ts'].isna() |
            silver['ms_played'].isna() |
            silver['track_name'].isna() |
            silver['artist_name'].isna()
        ]
        print(f"    ⚠️  Registros con nulls críticos: {len(critical_nulls):,}")
        silver = silver.dropna(subset=['spotify_track_uri', 'ts', 'ms_played', 'track_name', 'artist_name'])
        
        # Filtrar ms_played válidos (0 a 30 minutos)
        invalid_duration = silver[(silver['ms_played'] < 0) | (silver['ms_played'] > 1800000)]
        print(f"    ⚠️  Registros con duración inválida: {len(invalid_duration):,}")
        silver = silver[(silver['ms_played'] >= 0) & (silver['ms_played'] <= 1800000)]
        
        # Limpiar nombres de artistas y canciones
        silver['artist_name'] = silver['artist_name'].str.strip()
        silver['track_name'] = silver['track_name'].str.strip()
        silver['album_name'] = silver['album_name'].str.strip()
        
        # Rellenar valores faltantes en variables categóricas
        silver['reason_start'] = silver['reason_start'].fillna('unknown')
        silver['reason_end'] = silver['reason_end'].fillna('unknown')
        silver['platform'] = silver['platform'].fillna('unknown')
        
        # 3. Crear variables derivadas temporales
        print("  📅 Creando variables temporales...")
        silver['year'] = silver['ts'].dt.year
        silver['month'] = silver['ts'].dt.month
        silver['day_of_week'] = silver['ts'].dt.dayofweek
        silver['hour'] = silver['ts'].dt.hour
        silver['is_weekend'] = silver['day_of_week'].isin([5, 6])
        
        # 4. Crear variables de contexto
        print("  🎯 Creando variables de contexto...")
        
        # Intencionalidad de escucha
        silver['is_intentional'] = ~silver['shuffle']
        
        # Categorizar razones de inicio
        intentional_starts = ['clickrow', 'search', 'playlist']
        silver['intentional_start'] = silver['reason_start'].isin(intentional_starts)
        
        # Categorizar razones de fin
        natural_ends = ['trackdone', 'endplay']
        silver['natural_end'] = silver['reason_end'].isin(natural_ends)
        
        # 5. Métricas individuales de engagement
        print("  📊 Calculando métricas de engagement individuales...")
        
        # Completion rate (ms_played / duración estimada)
        silver['completion_rate'] = np.clip(silver['ms_played'] / self.estimated_duration_ms, 0, 1)
        
        # Skip resistance (1 si no fue skipped, 0 si fue skipped)
        silver['skip_resistance'] = (~silver['skipped']).astype(int)
        
        # Context score (intencional vs shuffle)
        silver['context_score'] = silver['is_intentional'].astype(int)
        
        # Engagement score individual (combinación ponderada)
        silver['individual_engagement_score'] = (
            0.5 * silver['completion_rate'] +
            0.3 * silver['skip_resistance'] +
            0.2 * silver['context_score']
        )
        
        # 6. Validaciones finales
        print("  ✅ Validaciones finales...")
        final_count = len(silver)
        cleaned_count = initial_count - final_count
        print(f"    📉 Registros eliminados: {cleaned_count:,} ({cleaned_count/initial_count*100:.1f}%)")
        print(f"    📊 Registros finales: {final_count:,}")
        
        # Guardar Silver data
        self.silver_data = silver
        print("✅ Silver: Datos limpios y enriquecidos creados")
        
        return self.silver_data
    
    def create_gold_data(self):
        """
        Procesa Silver → Gold: Agregación por canción y métricas finales
        """
        print("\n🔄 Procesando Silver → GOLD...")
        
        if self.silver_data is None:
            raise ValueError("Debe crear Silver data primero")
        
        print("  📊 Agregando datos por canción...")
        
        # Agregar por canción (spotify_track_uri)
        agg_dict = {
            # Información básica de la canción
            'track_name': 'first',
            'artist_name': 'first',
            'album_name': 'first',
            
            # Métricas temporales
            'ts': ['min', 'max'],  # Primera y última reproducción
            'year': ['min', 'max'],  # Rango de años
            
            # Métricas de volumen
            'spotify_track_uri': 'count',  # Total de reproducciones
            
            # Métricas de duración
            'ms_played': ['mean', 'median', 'std', 'min', 'max'],
            
            # Métricas de engagement
            'completion_rate': ['mean', 'median', 'std'],
            'skip_resistance': 'mean',
            'context_score': 'mean',
            'individual_engagement_score': 'mean',
            
            # Métricas de comportamiento
            'skipped': 'mean',  # Tasa de skip
            'shuffle': 'mean',   # Tasa de shuffle
            'is_intentional': 'mean',  # Tasa de escucha intencional
            
            # Métricas de contexto
            'platform': 'nunique',  # Diversidad de plataformas
            'reason_start': 'nunique',  # Diversidad de razones de inicio
            'is_weekend': 'mean',  # Proporción de escucha en fin de semana
            'hour': lambda x: x.std()  # Variabilidad horaria
        }
        
        gold = self.silver_data.groupby('spotify_track_uri').agg(agg_dict).reset_index()
        
        # Aplanar columnas multinivel
        gold.columns = ['_'.join(col).strip() if col[1] else col[0] for col in gold.columns.values]
        gold = gold.rename(columns={'spotify_track_uri_': 'spotify_track_uri'})
        
        # Renombrar columnas principales
        column_mapping = {
            'spotify_track_uri_count': 'total_plays',
            'skip_resistance_mean': 'avg_skip_resistance',
            'context_score_mean': 'avg_context_score',
            'individual_engagement_score_mean': 'avg_individual_engagement',
            'skipped_mean': 'skip_rate',
            'shuffle_mean': 'shuffle_rate',
            'is_intentional_mean': 'intentional_rate',
            'platform_nunique': 'platform_diversity',
            'reason_start_nunique': 'start_reason_diversity',
            'is_weekend_mean': 'weekend_listening_rate',
            'hour_<lambda>': 'hour_variability'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in gold.columns:
                gold = gold.rename(columns={old_name: new_name})
        
        print(f"  📈 Canciones únicas agregadas: {len(gold):,}")
        
        # 7. Crear métricas compuestas de engagement
        print("  🎯 Calculando métricas compuestas de engagement...")
        
        # Completion Rate Score (normalizado)
        gold['completion_rate_score'] = gold['completion_rate_mean']
        
        # Skip Resistance Score
        gold['skip_resistance_score'] = gold['avg_skip_resistance']
        
        # Context Preference Score (preferencia por escucha intencional)
        gold['context_preference_score'] = gold['intentional_rate']
        
        # Consistency Score (basado en baja variabilidad)
        gold['consistency_score'] = 1 / (1 + gold['completion_rate_std'].fillna(0))
        
        # Popularity Score (basado en número de reproducciones, normalizado log)
        gold['popularity_score'] = np.log1p(gold['total_plays']) / np.log1p(gold['total_plays'].max())
        
        # Multi-platform Appeal (diversidad de plataformas normalizada)
        max_platforms = gold['platform_diversity'].max()
        gold['platform_appeal_score'] = gold['platform_diversity'] / max_platforms
        
        # 8. Engagement Score Final (métrica compuesta principal)
        print("  🏆 Calculando Engagement Score Final...")
        
        # Pesos para cada componente
        weights = {
            'completion_rate_score': 0.35,      # Más importante
            'skip_resistance_score': 0.25,      # Segundo más importante
            'context_preference_score': 0.20,   # Intencionalidad
            'consistency_score': 0.10,          # Consistencia
            'platform_appeal_score': 0.10       # Diversidad de plataformas
        }
        
        gold['final_engagement_score'] = sum(
            gold[feature] * weight for feature, weight in weights.items()
        )
        
        # Normalizar score final a [0, 1]
        gold['final_engagement_score'] = (
            gold['final_engagement_score'] - gold['final_engagement_score'].min()
        ) / (gold['final_engagement_score'].max() - gold['final_engagement_score'].min())
        
        # 9. Crear categorías de engagement objetivo
        print("  🎯 Creando categorías de engagement...")
        
        high_threshold = self.engagement_thresholds['high']
        medium_threshold = self.engagement_thresholds['medium']
        
        def categorize_engagement(score):
            if score >= high_threshold:
                return 'High'
            elif score >= medium_threshold:
                return 'Medium'
            else:
                return 'Low'
        
        gold['engagement_category'] = gold['final_engagement_score'].apply(categorize_engagement)
        
        # Estadísticas de categorías
        category_counts = gold['engagement_category'].value_counts()
        print("  📊 Distribución de categorías:")
        for category, count in category_counts.items():
            percentage = (count / len(gold)) * 100
            print(f"    {category}: {count:,} canciones ({percentage:.1f}%)")
        
        # 10. Crear split temporal
        print("  📅 Creando split temporal...")
        
        train_years = self.config.get('train_years', list(range(2013, 2023)))
        val_years = self.config.get('validation_years', [2023])
        test_years = self.config.get('test_years', [2024])
        
        gold['data_split'] = gold['year_min'].apply(lambda year: 
            'train' if year in train_years 
            else 'validation' if year in val_years 
            else 'test' if year in test_years 
            else 'unknown'
        )
        
        split_counts = gold['data_split'].value_counts()
        print("  📊 Distribución temporal:")
        for split, count in split_counts.items():
            percentage = (count / len(gold)) * 100
            print(f"    {split}: {count:,} canciones ({percentage:.1f}%)")
        
        # 11. Seleccionar features finales para modelado
        print("  🎯 Seleccionando features para modelado...")
        
        feature_columns = [
            # Identificadores
            'spotify_track_uri', 'track_name_first', 'artist_name_first', 'album_name_first',
            
            # Features de engagement
            'completion_rate_score', 'skip_resistance_score', 'context_preference_score',
            'consistency_score', 'platform_appeal_score',
            
            # Features de volumen y popularidad
            'total_plays', 'popularity_score',
            
            # Features temporales y contextuales
            'weekend_listening_rate', 'hour_variability',
            
            # Features estadísticas
            'ms_played_mean', 'ms_played_std', 'completion_rate_std',
            
            # Target y splits
            'final_engagement_score', 'engagement_category', 'data_split'
        ]
        
        # Verificar que todas las columnas existen
        available_features = [col for col in feature_columns if col in gold.columns]
        missing_features = [col for col in feature_columns if col not in gold.columns]
        
        if missing_features:
            print(f"    ⚠️  Features faltantes: {missing_features}")
        
        gold_final = gold[available_features].copy()
        
        # Guardar Gold data
        self.gold_data = gold_final
        print(f"✅ Gold: {len(gold_final):,} canciones con {len(available_features)} features listas para modelado")
        
        return self.gold_data
    
    def save_processed_data(self, output_dir: str):
        """
        Guarda los datos procesados en archivos
        
        Args:
            output_dir (str): Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n💾 Guardando datos procesados...")
        
        # Guardar Silver data
        if self.silver_data is not None:
            silver_path = output_path / "silver_data.parquet"
            self.silver_data.to_parquet(silver_path, index=False)
            print(f"  ✅ Silver data guardada: {silver_path}")
        
        # Guardar Gold data
        if self.gold_data is not None:
            gold_path = output_path / "gold_data.parquet"
            self.gold_data.to_parquet(gold_path, index=False)
            print(f"  ✅ Gold data guardada: {gold_path}")
            
            # Guardar también en CSV para compatibilidad
            gold_csv_path = output_path / "gold_data.csv"
            self.gold_data.to_csv(gold_csv_path, index=False)
            print(f"  ✅ Gold data CSV guardada: {gold_csv_path}")
    
    def generate_processing_report(self):
        """Genera reporte del procesamiento"""
        if self.gold_data is None:
            return "No hay datos Gold para reportar"
        
        report = f"""
🎵 TUNEMETRICS - REPORTE DE PROCESAMIENTO DE DATOS 🎵
{'='*60}

📊 RESUMEN DEL PIPELINE:
├── Bronze (Raw): {len(self.bronze_data):,} reproducciones individuales
├── Silver (Cleaned): {len(self.silver_data):,} reproducciones válidas
└── Gold (Aggregated): {len(self.gold_data):,} canciones únicas

🎯 MÉTRICAS DE ENGAGEMENT CREADAS:
├── Completion Rate Score: Duración vs estimación (3.5min)
├── Skip Resistance Score: Proporción de reproducciones completas
├── Context Preference Score: Escucha intencional vs shuffle
├── Consistency Score: Baja variabilidad en comportamiento
├── Platform Appeal Score: Diversidad de dispositivos
└── Final Engagement Score: Métrica compuesta principal

📈 DISTRIBUCIÓN DE CATEGORÍAS:
{self.gold_data['engagement_category'].value_counts().to_string()}

📅 DISTRIBUCIÓN TEMPORAL:
{self.gold_data['data_split'].value_counts().to_string()}

🔧 FEATURES DISPONIBLES PARA MODELADO: {len(self.gold_data.columns)}

✅ DATOS LISTOS PARA ENTRENAMIENTO DE MODELOS
        """
        
        return report

def main():
    """Función principal para ejecutar el pipeline de procesamiento"""
    
    # Configuración básica
    config = {
        'estimated_song_duration_ms': 210000,  # 3.5 minutos
        'engagement_thresholds': {'high': 0.75, 'medium': 0.45},
        'train_years': list(range(2013, 2023)),
        'validation_years': [2023],
        'test_years': [2024]
    }
    
    # Rutas (ajustar según tu estructura)
    input_path = "data/raw/spotify_history.csv"
    output_dir = "data/processed"
    
    try:
        # Crear procesador
        processor = TuneMetricsDataProcessor(config)
        
        # Ejecutar pipeline completo
        processor.load_bronze_data(input_path)
        processor.create_silver_data()
        processor.create_gold_data()
        
        # Guardar resultados
        processor.save_processed_data(output_dir)
        
        # Generar reporte
        report = processor.generate_processing_report()
        print(report)
        
        # Guardar reporte
        with open("reports/data_processing_report.txt", "w") as f:
            f.write(report)
        
        print("\n✅ Pipeline de procesamiento completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en pipeline de procesamiento: {e}")
        raise

if __name__ == "__main__":
    main()