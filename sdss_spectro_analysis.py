"""
SDSS Spectroscopic Data Analysis Module
Author: Cris Luna

Specialized analysis for optical spectroscopic catalogs (DR17 and DR19)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class SpectroscopicAnalyzer:
    """
    Analyzer for SDSS optical spectroscopic data
    """
    
    def __init__(self, data: pd.DataFrame, catalog_name: str = "Spectroscopic"):
        """
        Initialize analyzer
        
        Args:
            data: DataFrame with spectroscopic data
            catalog_name: Name of the catalog for labeling
        """
        self.data = data.copy()
        self.catalog_name = catalog_name
        self.stats = {}
        
        # Standardize column names
        self.data.columns = self.data.columns.str.lower()
        
        logger.info(f"Initialized SpectroscopicAnalyzer with {len(self.data)} objects")
        self._identify_columns()
    
    def _identify_columns(self):
        """Identify available data columns"""
        cols = self.data.columns
        
        # Position columns
        self.has_ra = 'ra' in cols
        self.has_dec = 'dec' in cols
        
        # Redshift columns
        self.z_col = None
        for z_name in ['z', 'z_pipe2d', 'z_noqso', 'z_best']:
            if z_name in cols:
                self.z_col = z_name
                break
        
        # Classification columns
        self.class_col = None
        for class_name in ['class', 'specclass', 'objtype']:
            if class_name in cols:
                self.class_col = class_name
                break
        
        # Signal-to-noise
        self.sn_col = None
        for sn_name in ['sn_median_all', 'snmedian', 'sn_median']:
            if sn_name in cols:
                self.sn_col = sn_name
                break
        
        # Magnitudes
        self.mag_cols = [col for col in cols if 'mag' in col]
        
        # Colors (pre-calculated or calculate them)
        self.color_gr = 'color_gr' in cols
        
        logger.info(f"Available features: z={self.z_col is not None}, "
                   f"class={self.class_col is not None}, "
                   f"sn={self.sn_col is not None}, "
                   f"colors={self.color_gr}")
    
    def calculate_colors(self) -> pd.DataFrame:
        """
        Calculate color indices from magnitudes
        
        Returns:
            DataFrame with added color columns
        """
        logger.info("Calculating color indices...")
        
        # Try to find SDSS ugriz magnitudes
        mag_names = ['mag_0', 'mag_1', 'mag_2', 'mag_3', 'mag_4']  # u, g, r, i, z
        
        if all(col in self.data.columns for col in mag_names):
            self.data['color_ug'] = self.data['mag_0'] - self.data['mag_1']
            self.data['color_gr'] = self.data['mag_1'] - self.data['mag_2']
            self.data['color_ri'] = self.data['mag_2'] - self.data['mag_3']
            self.data['color_iz'] = self.data['mag_3'] - self.data['mag_4']
            self.data['color_ur'] = self.data['mag_0'] - self.data['mag_2']
            
            logger.info("✓ Calculated 5 color indices")
            self.color_gr = True
            
        else:
            logger.warning("Could not find standard magnitude columns")
        
        return self.data
    
    def basic_statistics(self) -> Dict:
        """
        Generate basic statistics
        
        Returns:
            Dictionary with statistical summaries
        """
        logger.info("Computing basic statistics...")
        
        stats = {
            'total_objects': len(self.data),
            'catalog_name': self.catalog_name
        }
        
        # Redshift statistics
        if self.z_col:
            z_data = self.data[self.z_col].dropna()
            stats['redshift'] = {
                'count': len(z_data),
                'mean': z_data.mean(),
                'median': z_data.median(),
                'std': z_data.std(),
                'min': z_data.min(),
                'max': z_data.max(),
                'q25': z_data.quantile(0.25),
                'q75': z_data.quantile(0.75)
            }
        
        # Object classification
        if self.class_col:
            class_counts = self.data[self.class_col].value_counts()
            stats['classifications'] = class_counts.to_dict()
        
        # Signal-to-noise
        if self.sn_col:
            sn_data = self.data[self.sn_col].dropna()
            stats['signal_to_noise'] = {
                'mean': sn_data.mean(),
                'median': sn_data.median(),
                'std': sn_data.std()
            }
        
        # Sky coverage
        if self.has_ra and self.has_dec:
            stats['sky_coverage'] = {
                'ra_range': (self.data['ra'].min(), self.data['ra'].max()),
                'dec_range': (self.data['dec'].min(), self.data['dec'].max())
            }
        
        self.stats = stats
        return stats
    
    def print_summary(self):
        """Print formatted summary statistics"""
        if not self.stats:
            self.basic_statistics()
        
        print("\n" + "="*80)
        print(f"SPECTROSCOPIC DATA SUMMARY: {self.catalog_name}")
        print("="*80)
        
        print(f"\n📊 Dataset Size:")
        print(f"   Total objects: {self.stats['total_objects']:,}")
        
        if 'redshift' in self.stats:
            z = self.stats['redshift']
            print(f"\n🌌 Redshift Statistics:")
            print(f"   Valid measurements: {z['count']:,}")
            print(f"   Mean z: {z['mean']:.4f}")
            print(f"   Median z: {z['median']:.4f}")
            print(f"   Std dev: {z['std']:.4f}")
            print(f"   Range: [{z['min']:.4f}, {z['max']:.4f}]")
            print(f"   IQR: [{z['q25']:.4f}, {z['q75']:.4f}]")
        
        if 'classifications' in self.stats:
            print(f"\n🔍 Object Classifications:")
            for obj_type, count in sorted(self.stats['classifications'].items(), 
                                         key=lambda x: x[1], reverse=True):
                pct = 100 * count / self.stats['total_objects']
                print(f"   {obj_type:20s}: {count:8,} ({pct:5.2f}%)")
        
        if 'signal_to_noise' in self.stats:
            sn = self.stats['signal_to_noise']
            print(f"\n📡 Signal-to-Noise:")
            print(f"   Mean S/N: {sn['mean']:.2f}")
            print(f"   Median S/N: {sn['median']:.2f}")
        
        if 'sky_coverage' in self.stats:
            sky = self.stats['sky_coverage']
            print(f"\n🗺️  Sky Coverage:")
            print(f"   RA range: [{sky['ra_range'][0]:.2f}°, {sky['ra_range'][1]:.2f}°]")
            print(f"   Dec range: [{sky['dec_range'][0]:.2f}°, {sky['dec_range'][1]:.2f}°]")
    
    def plot_redshift_analysis(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive redshift analysis plots
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.z_col:
            logger.warning("No redshift column available")
            return
        
        logger.info("Creating redshift analysis plots...")
        
        z_data = self.data[self.z_col].dropna()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Redshift Analysis - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Full redshift distribution
        ax = axes[0, 0]
        ax.hist(z_data, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(z_data.median(), color='red', linestyle='--', 
                  label=f'Median: {z_data.median():.3f}')
        ax.axvline(z_data.mean(), color='orange', linestyle='--', 
                  label=f'Mean: {z_data.mean():.3f}')
        ax.set_xlabel('Redshift', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Full Redshift Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Low-z detail (z < 0.5)
        ax = axes[0, 1]
        z_low = z_data[z_data < 0.5]
        ax.hist(z_low, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
        ax.set_xlabel('Redshift', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Low Redshift Detail (z < 0.5)', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. Cumulative distribution
        ax = axes[0, 2]
        z_sorted = np.sort(z_data)
        cumulative = np.arange(1, len(z_sorted) + 1) / len(z_sorted)
        ax.plot(z_sorted, cumulative, linewidth=2, color='darkgreen')
        ax.set_xlabel('Redshift', fontsize=11)
        ax.set_ylabel('Cumulative Fraction', fontsize=11)
        ax.set_title('Cumulative Distribution', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 4. Log-scale histogram
        ax = axes[1, 0]
        ax.hist(z_data, bins=100, alpha=0.7, edgecolor='black', color='mediumpurple')
        ax.set_xlabel('Redshift', fontsize=11)
        ax.set_ylabel('Count (log scale)', fontsize=11)
        ax.set_title('Redshift Distribution (Log Scale)', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # 5. Redshift vs RA (if available)
        ax = axes[1, 1]
        if self.has_ra:
            sample = self.data.sample(min(10000, len(self.data)))
            scatter = ax.scatter(sample['ra'], sample[self.z_col], 
                               s=1, alpha=0.5, c=sample[self.z_col], 
                               cmap='viridis')
            ax.set_xlabel('Right Ascension [deg]', fontsize=11)
            ax.set_ylabel('Redshift', fontsize=11)
            ax.set_title('Redshift vs RA', fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='z')
        else:
            ax.text(0.5, 0.5, 'RA not available', 
                   ha='center', va='center', transform=ax.transAxes)
        ax.grid(alpha=0.3)
        
        # 6. Redshift bins
        ax = axes[1, 2]
        bins = [0, 0.1, 0.3, 0.6, 1.0, 2.0, z_data.max()]
        bin_labels = ['0-0.1', '0.1-0.3', '0.3-0.6', '0.6-1.0', '1.0-2.0', f'>{2.0:.1f}']
        z_binned = pd.cut(z_data, bins=bins, labels=bin_labels)
        bin_counts = z_binned.value_counts().sort_index()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(bin_counts)))
        ax.bar(range(len(bin_counts)), bin_counts.values, color=colors, 
              edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(bin_counts)))
        ax.set_xticklabels(bin_counts.index, rotation=45)
        ax.set_xlabel('Redshift Bin', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Objects per Redshift Bin', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Redshift analysis complete")
    
    def plot_classification_analysis(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Analyze object classifications
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.class_col:
            logger.warning("No classification column available")
            return
        
        logger.info("Creating classification analysis plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Object Classification Analysis - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Classification pie chart
        ax = axes[0]
        class_counts = self.data[self.class_col].value_counts()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        wedges, texts, autotexts = ax.pie(class_counts.values, 
                                           labels=class_counts.index,
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Object Type Distribution', fontweight='bold')
        
        # 2. Classification vs redshift (if available)
        ax = axes[1]
        if self.z_col:
            for obj_type in class_counts.index[:5]:  # Top 5 classes
                mask = self.data[self.class_col] == obj_type
                z_values = self.data.loc[mask, self.z_col].dropna()
                
                if len(z_values) > 0:
                    ax.hist(z_values, bins=50, alpha=0.5, label=obj_type)
            
            ax.set_xlabel('Redshift', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Redshift Distribution by Object Type', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.bar(range(len(class_counts)), class_counts.values, 
                  color=colors, edgecolor='black')
            ax.set_xticks(range(len(class_counts)))
            ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Object Type Counts', fontweight='bold')
            ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Classification analysis complete")
    
    def plot_color_magnitude_diagram(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Create color-magnitude diagrams
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.color_gr:
            logger.info("Calculating colors first...")
            self.calculate_colors()
        
        if 'color_gr' not in self.data.columns:
            logger.warning("Could not create colors for CMD")
            return
        
        logger.info("Creating color-magnitude diagram...")
        
        # Find magnitude column
        mag_col = None
        for col in ['mag_2', 'mag_r', 'modelmag_r', 'psfmag_r']:
            if col in self.data.columns:
                mag_col = col
                break
        
        if not mag_col:
            logger.warning("No magnitude column found for CMD")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Color-Magnitude Diagrams - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # Prepare data
        valid = (self.data['color_gr'].notna() & 
                self.data[mag_col].notna())
        
        if self.z_col:
            valid &= self.data[self.z_col].notna()
        
        sample_data = self.data[valid].sample(min(20000, valid.sum()))
        
        # 1. CMD colored by redshift
        ax = axes[0]
        if self.z_col:
            scatter = ax.scatter(sample_data['color_gr'], 
                               sample_data[mag_col],
                               c=sample_data[self.z_col],
                               cmap='viridis', 
                               s=5, 
                               alpha=0.6,
                               vmin=0,
                               vmax=sample_data[self.z_col].quantile(0.95))
            plt.colorbar(scatter, ax=ax, label='Redshift')
            ax.set_title('CMD (colored by redshift)', fontweight='bold')
        else:
            ax.scatter(sample_data['color_gr'], 
                      sample_data[mag_col],
                      s=5, 
                      alpha=0.5,
                      color='steelblue')
            ax.set_title('Color-Magnitude Diagram', fontweight='bold')
        
        ax.set_xlabel('(g - r) [mag]', fontsize=11)
        ax.set_ylabel(f'{mag_col} [mag]', fontsize=11)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.5, 2.5)
        
        # 2. CMD colored by object type
        ax = axes[1]
        if self.class_col:
            classes = sample_data[self.class_col].unique()
            colors_map = plt.cm.Set1(np.linspace(0, 1, len(classes)))
            
            for i, obj_type in enumerate(classes[:5]):  # Top 5 classes
                mask = sample_data[self.class_col] == obj_type
                ax.scatter(sample_data.loc[mask, 'color_gr'],
                          sample_data.loc[mask, mag_col],
                          s=10,
                          alpha=0.6,
                          label=obj_type,
                          color=colors_map[i])
            
            ax.legend(markerscale=2, fontsize=9)
            ax.set_title('CMD (colored by object type)', fontweight='bold')
        else:
            # Density plot as alternative
            ax.hexbin(sample_data['color_gr'], 
                     sample_data[mag_col],
                     gridsize=50, 
                     cmap='YlOrRd',
                     mincnt=1)
            ax.set_title('CMD (density)', fontweight='bold')
        
        ax.set_xlabel('(g - r) [mag]', fontsize=11)
        ax.set_ylabel(f'{mag_col} [mag]', fontsize=11)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.5, 2.5)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Color-magnitude diagram complete")
    
    def plot_sky_distribution(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Plot sky distribution of objects
        
        Args:
            figsize: Figure size (width, height)
        """
        if not (self.has_ra and self.has_dec):
            logger.warning("RA/Dec columns not available")
            return
        
        logger.info("Creating sky distribution plots...")
        
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Sky Distribution - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # Sample for faster plotting
        sample = self.data.sample(min(50000, len(self.data)))
        
        # 1. Mollweide projection
        ax1 = plt.subplot(121, projection='mollweide')
        
        ra_rad = np.radians(sample['ra'] - 180)
        dec_rad = np.radians(sample['dec'])
        
        if self.z_col and self.z_col in sample.columns:
            scatter = ax1.scatter(ra_rad, dec_rad, 
                                 c=sample[self.z_col],
                                 s=1, 
                                 alpha=0.5,
                                 cmap='plasma',
                                 vmin=0,
                                 vmax=sample[self.z_col].quantile(0.95))
            plt.colorbar(scatter, ax=ax1, label='Redshift', fraction=0.046)
        else:
            ax1.scatter(ra_rad, dec_rad, s=1, alpha=0.3, color='steelblue')
        
        ax1.set_title('Mollweide Projection', fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. RA/Dec scatter
        ax2 = plt.subplot(122)
        
        if self.z_col and self.z_col in sample.columns:
            scatter = ax2.scatter(sample['ra'], 
                                 sample['dec'],
                                 c=sample[self.z_col],
                                 s=5,
                                 alpha=0.5,
                                 cmap='plasma',
                                 vmin=0,
                                 vmax=sample[self.z_col].quantile(0.95))
            plt.colorbar(scatter, ax=ax2, label='Redshift')
        else:
            ax2.scatter(sample['ra'], 
                       sample['dec'],
                       s=5,
                       alpha=0.3,
                       color='steelblue')
        
        ax2.set_xlabel('Right Ascension [deg]', fontsize=11)
        ax2.set_ylabel('Declination [deg]', fontsize=11)
        ax2.set_title('RA-Dec Distribution', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Sky distribution plots complete")
    
    def full_analysis(self):
        """Run complete analysis pipeline"""
        logger.info(f"Running full analysis for {self.catalog_name}...")
        
        # Print summary
        self.print_summary()
        
        # Create all plots
        if self.z_col:
            self.plot_redshift_analysis()
        
        if self.class_col:
            self.plot_classification_analysis()
        
        if self.has_ra and self.has_dec:
            self.plot_sky_distribution()
        
        self.plot_color_magnitude_diagram()
        
        logger.info("✓ Full analysis complete")


def load_and_analyze(catalog_id: str = 'spectro_dr19', 
                     data_dir: str = './sdss_data'):
    """
    Convenience function to load and analyze spectroscopic data
    
    Args:
        catalog_id: ID of catalog to load
        data_dir: Path to data directory
        
    Returns:
        SpectroscopicAnalyzer instance
    """
    data_path = Path(data_dir) / 'processed' / f'{catalog_id}_processed.parquet'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    
    logger.info(f"Loading {catalog_id} from {data_path}")
    data = pd.read_parquet(data_path)
    
    analyzer = SpectroscopicAnalyzer(data, catalog_name=catalog_id)
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*80)
    print("🔬 SDSS SPECTROSCOPIC DATA ANALYZER")
    print("="*80)
    
    # Try to load spectro_dr19
    try:
        analyzer = load_and_analyze('spectro_dr19')
        analyzer.full_analysis()
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nPlease run sdss_pipeline.py first to download and process the data.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()