# Análisis Exploratorio de Datos para TuneMetrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SpotifyDataExplorer:
    """
    Clase para análisis exploratorio de datos de Spotify
    """
    
    def __init__(self, data_path: str):
        """
        Inicializa el explorador con los datos de Spotify
        
        Args:
            data_path (str): Ruta al archivo CSV de datos
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Carga los datos desde el archivo CSV"""
        try:
            print("Cargando datos de Spotify...")
            self.df = pd.read_csv(self.data_path)
            print(f"Datos cargados exitosamente: {len(self.df):,} registros")
            
            # Convertir timestamp a datetime
            self.df['ts'] = pd.to_datetime(self.df['ts'])
            self.df['year'] = self.df['ts'].dt.year
            self.df['month'] = self.df['ts'].dt.month
            self.df['hour'] = self.df['ts'].dt.hour
            self.df['day_of_week'] = self.df['ts'].dt.dayofweek
            
            print("Variables temporales creadas")
            
        except Exception as e:
            print(f"Error cargando datos: {e}")
            raise
    
    def basic_info(self):
        """Muestra información básica del dataset"""
        print("=" * 60)
        print("INFORMACIÓN BÁSICA DEL DATASET")
        print("=" * 60)
        
        print(f"Dimensiones: {self.df.shape}")
        print(f"Período temporal: {self.df['ts'].min()} - {self.df['ts'].max()}")
        print(f"Canciones únicas: {self.df['spotify_track_uri'].nunique():,}")
        print(f"Artistas únicos: {self.df['artist_name'].nunique():,}")
        print(f"Álbumes únicos: {self.df['album_name'].nunique():,}")
        
        print("\nINFORMACIÓN DE COLUMNAS:")
        print(self.df.info())
        
        print("\nVALORES FALTANTES:")
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Percentage': missing_percentage
        })
        print(missing_df[missing_df['Missing Count'] > 0])
    
    def analyze_categorical_variables(self):
        """Analiza las variables categóricas"""
        print("\n" + "=" * 60)
        print("ANÁLISIS DE VARIABLES CATEGÓRICAS")
        print("=" * 60)
        
        # Plataformas
        platform_counts = self.df['platform'].value_counts()
        print(f"\nPLATAFORMAS ({platform_counts.sum():,} total):")
        for platform, count in platform_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {platform}: {count:,} ({percentage:.1f}%)")
        
        # Reason start/end
        print(f"\nRAZONES DE INICIO (top 10):")
        reason_start_counts = self.df['reason_start'].value_counts().head(10)
        for reason, count in reason_start_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {reason}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nRAZONES DE FIN (top 10):")
        reason_end_counts = self.df['reason_end'].value_counts().head(10)
        for reason, count in reason_end_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {reason}: {count:,} ({percentage:.1f}%)")
    
    def analyze_engagement_metrics(self):
        """Analiza métricas clave de engagement"""
        print("\n" + "=" * 60)
        print("ANÁLISIS DE MÉTRICAS DE ENGAGEMENT")
        print("=" * 60)
        
        # Estadísticas de ms_played
        ms_played_stats = self.df['ms_played'].describe()
        print("\nESTADÍSTICAS DE DURACIÓN DE REPRODUCCIÓN:")
        print(f"  Mínimo: {ms_played_stats['min']:,.0f} ms")
        print(f"  Q1: {ms_played_stats['25%']:,.0f} ms ({ms_played_stats['25%']/60000:.1f} min)")
        print(f"  Mediana: {ms_played_stats['50%']:,.0f} ms ({ms_played_stats['50%']/60000:.1f} min)")
        print(f"  Media: {ms_played_stats['mean']:,.0f} ms ({ms_played_stats['mean']/60000:.1f} min)")
        print(f"  Q3: {ms_played_stats['75%']:,.0f} ms ({ms_played_stats['75%']/60000:.1f} min)")
        print(f"  Máximo: {ms_played_stats['max']:,.0f} ms ({ms_played_stats['max']/60000:.1f} min)")
        
        # Análisis de comportamiento
        shuffle_rate = self.df['shuffle'].mean() * 100
        skip_rate = self.df['skipped'].mean() * 100
        
        print(f"\nCOMPORTAMIENTO DE ESCUCHA:")
        print(f"  Tasa de shuffle: {shuffle_rate:.1f}%")
        print(f"  Tasa de skip: {skip_rate:.1f}%")
        print(f"  Escucha intencional: {100-shuffle_rate:.1f}%")
        print(f"  Resistencia al skip: {100-skip_rate:.1f}%")
        
        # Engagement por plataforma
        print(f"\nENGAGEMENT POR PLATAFORMA:")
        platform_engagement = self.df.groupby('platform').agg({
            'ms_played': 'mean',
            'skipped': 'mean',
            'shuffle': 'mean'
        }).round(4)
        
        for platform in platform_engagement.index:
            avg_duration = platform_engagement.loc[platform, 'ms_played']
            skip_rate = platform_engagement.loc[platform, 'skipped'] * 100
            shuffle_rate = platform_engagement.loc[platform, 'shuffle'] * 100
            print(f"  {platform}:")
            print(f"    Duración promedio: {avg_duration/60000:.1f} min")
            print(f"    Tasa de skip: {skip_rate:.1f}%")
            print(f"    Tasa de shuffle: {shuffle_rate:.1f}%")
    
    def analyze_temporal_patterns(self):
        """Analiza patrones temporales"""
        print("\n" + "=" * 60)
        print("ANÁLISIS DE PATRONES TEMPORALES")
        print("=" * 60)
        
        # Distribución por año
        yearly_counts = self.df['year'].value_counts().sort_index()
        print("\nREPRODUCCIONES POR AÑO:")
        for year, count in yearly_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {year}: {count:,} ({percentage:.1f}%)")
        
        # Top artistas
        print(f"\nTOP 15 ARTISTAS POR REPRODUCCIONES:")
        top_artists = self.df['artist_name'].value_counts().head(15)
        for i, (artist, count) in enumerate(top_artists.items(), 1):
            avg_duration = self.df[self.df['artist_name'] == artist]['ms_played'].mean()
            print(f"  {i:2d}. {artist}: {count:,} plays ({avg_duration/60000:.1f}min avg)")
    
    def create_visualization_dashboard(self, save_path: str = None):
        """Crea un dashboard de visualizaciones"""
        print("\n" + "=" * 60)
        print("CREANDO VISUALIZACIONES")
        print("=" * 60)
        
        # Configurar el estilo
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('TuneMetrics: Dashboard de Análisis Exploratorio', fontsize=16, fontweight='bold')
        
        # 1. Distribución de duración de reproducción
        axes[0, 0].hist(self.df['ms_played'] / 60000, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribución de Duración de Reproducción')
        axes[0, 0].set_xlabel('Duración (minutos)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].axvline(self.df['ms_played'].mean() / 60000, color='red', linestyle='--', label='Media')
        axes[0, 0].legend()
        
        # 2. Engagement por plataforma
        platform_data = self.df.groupby('platform')['ms_played'].mean() / 60000
        axes[0, 1].bar(range(len(platform_data)), platform_data.values, color='lightcoral')
        axes[0, 1].set_title('Duración Promedio por Plataforma')
        axes[0, 1].set_xlabel('Plataforma')
        axes[0, 1].set_ylabel('Duración Promedio (min)')
        axes[0, 1].set_xticks(range(len(platform_data)))
        axes[0, 1].set_xticklabels(platform_data.index, rotation=45, ha='right')
        
        # 3. Tasa de skip por año
        yearly_skip = self.df.groupby('year')['skipped'].mean() * 100
        axes[0, 2].plot(yearly_skip.index, yearly_skip.values, marker='o', linewidth=2, color='green')
        axes[0, 2].set_title('Evolución de la Tasa de Skip')
        axes[0, 2].set_xlabel('Año')
        axes[0, 2].set_ylabel('Tasa de Skip (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Patrones por hora del día
        hourly_patterns = self.df.groupby('hour')['ms_played'].mean() / 60000
        axes[1, 0].plot(hourly_patterns.index, hourly_patterns.values, marker='s', linewidth=2, color='purple')
        axes[1, 0].set_title('Duración Promedio por Hora del Día')
        axes[1, 0].set_xlabel('Hora del Día')
        axes[1, 0].set_ylabel('Duración Promedio (min)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Shuffle vs No Shuffle
        shuffle_comparison = self.df.groupby('shuffle')['ms_played'].mean() / 60000
        labels = ['No Shuffle (Intencional)', 'Shuffle (Pasivo)']
        colors = ['gold', 'lightblue']
        axes[1, 1].bar(labels, shuffle_comparison.values, color=colors)
        axes[1, 1].set_title('Duración: Shuffle vs No Shuffle')
        axes[1, 1].set_ylabel('Duración Promedio (min)')
        
        # 6. Top 10 artistas
        top_artists = self.df['artist_name'].value_counts().head(10)
        axes[1, 2].barh(range(len(top_artists)), top_artists.values, color='orange')
        axes[1, 2].set_title('Top 10 Artistas por Reproducciones')
        axes[1, 2].set_xlabel('Número de Reproducciones')
        axes[1, 2].set_yticks(range(len(top_artists)))
        axes[1, 2].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                   for name in top_artists.index])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard guardado en: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Genera un reporte resumen del análisis"""
        print("\n" + "=" * 60)
        print("REPORTE RESUMEN - TUNEMETRICS EDA")
        print("=" * 60)
        
        # Métricas clave
        total_tracks = self.df['spotify_track_uri'].nunique()
        total_artists = self.df['artist_name'].nunique()
        avg_duration = self.df['ms_played'].mean() / 60000
        skip_rate = self.df['skipped'].mean() * 100
        shuffle_rate = self.df['shuffle'].mean() * 100
        
        # Completion rate estimado
        estimated_duration = 3.5  # minutos
        completion_rate = (avg_duration / estimated_duration) * 100
        
        report = f"""
MÉTRICAS CLAVE PARA MODELO DE ENGAGEMENT:
├── Dataset: {len(self.df):,} reproducciones de {total_tracks:,} canciones únicas
├── Período: {self.df['year'].min()}-{self.df['year'].max()} ({self.df['year'].nunique()} años)
├── Artistas: {total_artists:,} únicos
│
├── ENGAGEMENT BASELINE:
│   ├── Duración promedio: {avg_duration:.1f} min
│   ├── Completion rate estimado: {completion_rate:.1f}%
│   ├── Skip resistance: {100-skip_rate:.1f}%
│   └── Intentional listening: {100-shuffle_rate:.1f}%
│
├── DISTRIBUCIÓN TEMPORAL:
│   ├── Pico de actividad: {self.df.groupby('year').size().idxmax()}
│   └── Registros por año: {self.df.groupby('year').size().mean():.0f} promedio
│
└── PREPARACIÓN PARA MODELADO:
    ├── Variables target: Engagement categories (Alto/Medio/Bajo)
    ├── Features principales: completion_rate, skip_resistance, context_preference
    ├── Split temporal: 2013-2022 (train), 2023 (val), 2024 (test)
    └── Agregación objetivo: ~{total_tracks:,} canciones para modelado
        """
        
        print(report)
        return report

def main():
    """Función principal para ejecutar el análisis exploratorio"""
    # Configurar rutas (ajustar según tu estructura)
    data_path = "data/raw/spotify_history.csv"
    
    try:
        # Crear explorador
        explorer = SpotifyDataExplorer(data_path)
        
        # Ejecutar análisis completo
        explorer.basic_info()
        explorer.analyze_categorical_variables()
        explorer.analyze_engagement_metrics()
        explorer.analyze_temporal_patterns()
        
        # Crear visualizaciones
        explorer.create_visualization_dashboard("reports/figures/eda_dashboard.png")
        
        # Generar reporte
        report = explorer.generate_summary_report()
        
        # Guardar reporte
        with open("reports/eda_summary_report.txt", "w") as f:
            f.write(report)
        
        print("\nAnálisis exploratorio completado exitosamente!")
        
    except Exception as e:
        print(f"Error en análisis exploratorio: {e}")
        raise

if __name__ == "__main__":
    main()