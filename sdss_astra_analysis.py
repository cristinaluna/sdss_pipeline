"""
SDSS ASTRA Analysis Module
Author: Cris Luna

Specialized analysis for ASTRA stellar parameter pipeline results
ASTRA analyzes both BOSS and APOGEE spectra for Milky Way Mapper targets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class ASTRAAnalyzer:
    """
    Analyzer for ASTRA stellar analysis pipeline results
    """
    
    def __init__(self, data: pd.DataFrame, catalog_name: str = "ASTRA"):
        """
        Initialize analyzer
        
        Args:
            data: DataFrame with ASTRA results
            catalog_name: Name for labeling
        """
        self.data = data.copy()
        self.catalog_name = catalog_name
        self.stats = {}
        
        # Standardize column names
        self.data.columns = self.data.columns.str.lower()
        
        logger.info(f"Initialized ASTRAAnalyzer with {len(self.data)} objects")
        self._identify_columns()
        self._identify_pipelines()
    
    def _identify_columns(self):
        """Identify available data columns"""
        cols = self.data.columns
        
        # Stellar parameters
        self.has_teff = 'teff' in cols
        self.has_logg = 'logg' in cols
        self.has_feh = 'fe_h' in cols or 'feh' in cols
        self.has_m_h = 'm_h' in cols
        self.has_alpha_m = 'alpha_m' in cols
        self.has_v_rad = 'v_rad' in cols or 'vrad' in cols
        
        # Position
        self.has_ra = 'ra' in cols
        self.has_dec = 'dec' in cols
        
        # Source information
        self.has_source = 'source_pk' in cols or 'source' in cols
        self.has_pipeline = 'pipeline' in cols or 'result_flags' in cols
        
        # Quality flags
        flag_cols = [col for col in cols if 'flag' in col.lower()]
        self.flag_columns = flag_cols
        
        logger.info(f"Available features: Teff={self.has_teff}, logg={self.has_logg}, "
                   f"[Fe/H]={self.has_feh}, [α/M]={self.has_alpha_m}")
        logger.info(f"Quality flag columns: {len(flag_cols)}")
    
    def _identify_pipelines(self):
        """Identify which analysis pipelines were used"""
        self.pipelines = []
        
        # Check for pipeline-specific columns
        pipeline_indicators = {
            'FERRE': ['ferre', 'aspcap'],
            'The Payne': ['payne'],
            'SLAM': ['slam'],
            'APOGEENet': ['apogeenet'],
            'The Cannon': ['cannon']
        }
        
        cols_lower = [c.lower() for c in self.data.columns]
        
        for pipeline, indicators in pipeline_indicators.items():
            if any(any(ind in col for ind in indicators) for col in cols_lower):
                self.pipelines.append(pipeline)
        
        if self.pipelines:
            logger.info(f"Detected pipelines: {', '.join(self.pipelines)}")
        else:
            logger.info("No specific pipeline identifiers found")
    
    def basic_statistics(self) -> Dict:
        """
        Generate basic statistics
        
        Returns:
            Dictionary with statistical summaries
        """
        logger.info("Computing basic statistics...")
        
        stats = {
            'catalog_name': self.catalog_name,
            'total_objects': len(self.data),
            'pipelines_detected': self.pipelines
        }
        
        # Temperature statistics
        if self.has_teff:
            teff_data = self.data['teff'].dropna()
            teff_valid = teff_data[(teff_data > 2500) & (teff_data < 10000)]
            stats['teff'] = {
                'count': len(teff_valid),
                'mean': teff_valid.mean(),
                'median': teff_valid.median(),
                'std': teff_valid.std(),
                'min': teff_valid.min(),
                'max': teff_valid.max()
            }
        
        # Surface gravity statistics
        if self.has_logg:
            logg_data = self.data['logg'].dropna()
            logg_valid = logg_data[(logg_data > -1) & (logg_data < 6)]
            stats['logg'] = {
                'count': len(logg_valid),
                'mean': logg_valid.mean(),
                'median': logg_valid.median(),
                'std': logg_valid.std()
            }
        
        # Metallicity statistics
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            feh_data = self.data[feh_col].dropna()
            feh_valid = feh_data[(feh_data > -4) & (feh_data < 1)]
            stats['feh'] = {
                'count': len(feh_valid),
                'mean': feh_valid.mean(),
                'median': feh_valid.median(),
                'std': feh_valid.std(),
                'metal_poor': (feh_valid < -1).sum(),
                'metal_rich': (feh_valid > 0).sum()
            }
        
        # Alpha enhancement
        if self.has_alpha_m:
            alpha_data = self.data['alpha_m'].dropna()
            alpha_valid = alpha_data[(alpha_data > -0.5) & (alpha_data < 0.8)]
            stats['alpha_m'] = {
                'count': len(alpha_valid),
                'mean': alpha_valid.mean(),
                'median': alpha_valid.median(),
                'std': alpha_valid.std(),
                'alpha_enhanced': (alpha_valid > 0.2).sum()
            }
        
        # Radial velocity
        if self.has_v_rad:
            vrad_col = 'v_rad' if 'v_rad' in self.data.columns else 'vrad'
            vrad_data = self.data[vrad_col].dropna()
            vrad_valid = vrad_data[(vrad_data > -1000) & (vrad_data < 1000)]
            stats['v_rad'] = {
                'count': len(vrad_valid),
                'mean': vrad_valid.mean(),
                'median': vrad_valid.median(),
                'std': vrad_valid.std()
            }
        
        self.stats = stats
        return stats
    
    def print_summary(self):
        """Print formatted summary statistics"""
        if not self.stats:
            self.basic_statistics()
        
        print("\n" + "="*80)
        print(f"ASTRA STELLAR ANALYSIS SUMMARY: {self.catalog_name}")
        print("="*80)
        
        print(f"\n📊 Dataset Size:")
        print(f"   Total objects: {self.stats['total_objects']:,}")
        
        if self.stats['pipelines_detected']:
            print(f"\n🔬 Analysis Pipelines Detected:")
            for pipeline in self.stats['pipelines_detected']:
                print(f"   • {pipeline}")
        
        if 'teff' in self.stats:
            t = self.stats['teff']
            print(f"\n🌡️  Effective Temperature:")
            print(f"   Valid measurements: {t['count']:,}")
            print(f"   Mean: {t['mean']:.0f} K")
            print(f"   Median: {t['median']:.0f} K")
            print(f"   Range: [{t['min']:.0f}, {t['max']:.0f}] K")
        
        if 'logg' in self.stats:
            g = self.stats['logg']
            print(f"\n🪐 Surface Gravity (log g):")
            print(f"   Valid measurements: {g['count']:,}")
            print(f"   Mean: {g['mean']:.2f}")
            print(f"   Median: {g['median']:.2f}")
        
        if 'feh' in self.stats:
            m = self.stats['feh']
            print(f"\n⚛️  Metallicity [Fe/H]:")
            print(f"   Valid measurements: {m['count']:,}")
            print(f"   Mean: {m['mean']:.3f} dex")
            print(f"   Median: {m['median']:.3f} dex")
            print(f"   Metal-poor ([Fe/H] < -1): {m['metal_poor']:,} ({100*m['metal_poor']/m['count']:.1f}%)")
            print(f"   Metal-rich ([Fe/H] > 0): {m['metal_rich']:,} ({100*m['metal_rich']/m['count']:.1f}%)")
        
        if 'alpha_m' in self.stats:
            a = self.stats['alpha_m']
            print(f"\n🔵 Alpha Enhancement [α/M]:")
            print(f"   Valid measurements: {a['count']:,}")
            print(f"   Mean: {a['mean']:.3f} dex")
            print(f"   Median: {a['median']:.3f} dex")
            print(f"   α-enhanced ([α/M] > 0.2): {a['alpha_enhanced']:,} ({100*a['alpha_enhanced']/a['count']:.1f}%)")
        
        if 'v_rad' in self.stats:
            v = self.stats['v_rad']
            print(f"\n🚀 Radial Velocity:")
            print(f"   Valid measurements: {v['count']:,}")
            print(f"   Mean: {v['mean']:.1f} km/s")
            print(f"   Median: {v['median']:.1f} km/s")
            print(f"   Std dev: {v['std']:.1f} km/s")
    
    def plot_hr_diagram(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Create Hertzsprung-Russell diagrams
        
        Args:
            figsize: Figure size (width, height)
        """
        if not (self.has_teff and self.has_logg):
            logger.warning("Teff and logg required for HR diagram")
            return
        
        logger.info("Creating HR diagram...")
        
        # Filter valid data
        valid = ((self.data['teff'] > 2500) & 
                (self.data['teff'] < 10000) &
                (self.data['logg'] > -1) &
                (self.data['logg'] < 6))
        
        data = self.data[valid].copy()
        
        # Sample if too large
        if len(data) > 30000:
            data = data.sample(30000)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Hertzsprung-Russell Diagram - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. HR diagram colored by metallicity
        ax = axes[0]
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in data.columns else 'feh'
            feh_valid = data[feh_col].notna() & (data[feh_col] > -4) & (data[feh_col] < 1)
            plot_data = data[feh_valid]
            
            scatter = ax.scatter(plot_data['teff'], 
                               plot_data['logg'],
                               c=plot_data[feh_col],
                               s=5,
                               alpha=0.6,
                               cmap='RdYlBu_r',
                               vmin=-2,
                               vmax=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('[Fe/H] [dex]', fontsize=11)
            ax.set_title('HR Diagram (colored by [Fe/H])', fontweight='bold')
        else:
            ax.scatter(data['teff'], 
                      data['logg'],
                      s=5,
                      alpha=0.5,
                      color='steelblue')
            ax.set_title('Hertzsprung-Russell Diagram', fontweight='bold')
        
        ax.set_xlabel('Effective Temperature [K]', fontsize=11)
        ax.set_ylabel('log g [cgs]', fontsize=11)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        
        # Add evolutionary sequence labels
        ax.axhline(y=3.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(4000, 3.3, 'Giants', fontsize=9, color='gray', style='italic')
        ax.text(5500, 4.7, 'Main Sequence', fontsize=9, color='gray', style='italic')
        
        # 2. HR diagram colored by alpha enhancement
        ax = axes[1]
        if self.has_alpha_m:
            alpha_valid = data['alpha_m'].notna() & (data['alpha_m'] > -0.5) & (data['alpha_m'] < 0.8)
            plot_data = data[alpha_valid]
            
            scatter = ax.scatter(plot_data['teff'], 
                               plot_data['logg'],
                               c=plot_data['alpha_m'],
                               s=5,
                               alpha=0.6,
                               cmap='RdYlGn',
                               vmin=-0.2,
                               vmax=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('[α/M] [dex]', fontsize=11)
            ax.set_title('HR Diagram (colored by [α/M])', fontweight='bold')
        else:
            # Density plot as alternative
            ax.hexbin(data['teff'], 
                     data['logg'],
                     gridsize=50,
                     cmap='YlOrRd',
                     mincnt=1)
            ax.set_title('HR Diagram (density)', fontweight='bold')
        
        ax.set_xlabel('Effective Temperature [K]', fontsize=11)
        ax.set_ylabel('log g [cgs]', fontsize=11)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ HR diagram complete")
    
    def plot_chemical_abundances(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot chemical abundance distributions and relations
        
        Args:
            figsize: Figure size (width, height)
        """
        logger.info("Creating chemical abundance plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Chemical Abundance Analysis - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Metallicity distribution
        ax = axes[0, 0]
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            feh_data = self.data[feh_col]
            feh_valid = feh_data[(feh_data > -4) & (feh_data < 1)]
            
            ax.hist(feh_valid, bins=60, alpha=0.7, edgecolor='black', color='steelblue')
            ax.axvline(feh_valid.median(), color='red', linestyle='--',
                      label=f'Median: {feh_valid.median():.3f}')
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Solar')
            ax.set_xlabel('[Fe/H] [dex]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Metallicity Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 2. Alpha enhancement distribution
        ax = axes[0, 1]
        if self.has_alpha_m:
            alpha_data = self.data['alpha_m']
            alpha_valid = alpha_data[(alpha_data > -0.5) & (alpha_data < 0.8)]
            
            ax.hist(alpha_valid, bins=50, alpha=0.7, edgecolor='black', color='forestgreen')
            ax.axvline(alpha_valid.median(), color='red', linestyle='--',
                      label=f'Median: {alpha_valid.median():.3f}')
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Solar')
            ax.set_xlabel('[α/M] [dex]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Alpha Enhancement Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 3. [α/M] vs [Fe/H] - Chemical evolution diagram
        ax = axes[0, 2]
        if self.has_feh and self.has_alpha_m:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            valid = ((self.data[feh_col] > -4) & 
                    (self.data[feh_col] < 1) &
                    (self.data['alpha_m'] > -0.5) &
                    (self.data['alpha_m'] < 0.8))
            
            sample = self.data[valid].sample(min(20000, valid.sum()))
            
            hexbin = ax.hexbin(sample[feh_col], 
                              sample['alpha_m'],
                              gridsize=50,
                              cmap='YlOrRd',
                              mincnt=1)
            plt.colorbar(hexbin, ax=ax, label='Count')
            
            # Add chemical evolution tracks
            ax.axhline(y=0.2, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(-2.5, 0.25, 'α-enhanced', fontsize=9, color='blue')
            ax.text(-2.5, 0.05, 'α-normal', fontsize=9, color='blue')
            
            ax.set_xlabel('[Fe/H] [dex]', fontsize=11)
            ax.set_ylabel('[α/M] [dex]', fontsize=11)
            ax.set_title('Chemical Evolution ([α/M] vs [Fe/H])', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # 4. Metallicity vs Temperature
        ax = axes[1, 0]
        if self.has_teff and self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            valid = ((self.data['teff'] > 2500) & 
                    (self.data['teff'] < 10000) &
                    (self.data[feh_col] > -4) &
                    (self.data[feh_col] < 1))
            
            sample = self.data[valid].sample(min(15000, valid.sum()))
            
            scatter = ax.scatter(sample['teff'], 
                               sample[feh_col],
                               c=sample['logg'] if self.has_logg else 'steelblue',
                               s=5,
                               alpha=0.5,
                               cmap='viridis')
            
            if self.has_logg:
                plt.colorbar(scatter, ax=ax, label='log g')
            
            ax.set_xlabel('Effective Temperature [K]', fontsize=11)
            ax.set_ylabel('[Fe/H] [dex]', fontsize=11)
            ax.set_title('Metallicity vs Temperature', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # 5. Population identification
        ax = axes[1, 1]
        if self.has_feh and self.has_alpha_m:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            valid = ((self.data[feh_col] > -4) & 
                    (self.data[feh_col] < 1) &
                    (self.data['alpha_m'] > -0.5) &
                    (self.data['alpha_m'] < 0.8))
            
            sample = self.data[valid]
            
            # Define populations
            thin_disk = sample[(sample[feh_col] > -0.5) & (sample['alpha_m'] < 0.15)]
            thick_disk = sample[(sample[feh_col] > -1) & (sample['alpha_m'] > 0.2)]
            halo = sample[sample[feh_col] < -1]
            
            ax.scatter(thin_disk[feh_col], thin_disk['alpha_m'], 
                      s=5, alpha=0.5, label=f'Thin Disk ({len(thin_disk)})', color='blue')
            ax.scatter(thick_disk[feh_col], thick_disk['alpha_m'], 
                      s=5, alpha=0.5, label=f'Thick Disk ({len(thick_disk)})', color='orange')
            ax.scatter(halo[feh_col], halo['alpha_m'], 
                      s=5, alpha=0.5, label=f'Halo ({len(halo)})', color='red')
            
            ax.set_xlabel('[Fe/H] [dex]', fontsize=11)
            ax.set_ylabel('[α/M] [dex]', fontsize=11)
            ax.set_title('Galactic Population Identification', fontweight='bold')
            ax.legend(markerscale=3)
            ax.grid(alpha=0.3)
        
        # 6. Metallicity cumulative distribution
        ax = axes[1, 2]
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            feh_data = self.data[feh_col]
            feh_valid = feh_data[(feh_data > -4) & (feh_data < 1)].sort_values()
            
            cumulative = np.arange(1, len(feh_valid) + 1) / len(feh_valid)
            ax.plot(feh_valid, cumulative, linewidth=2, color='darkgreen')
            ax.axvline(feh_valid.median(), color='red', linestyle='--',
                      label=f'Median: {feh_valid.median():.3f}')
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Solar')
            
            ax.set_xlabel('[Fe/H] [dex]', fontsize=11)
            ax.set_ylabel('Cumulative Fraction', fontsize=11)
            ax.set_title('Metallicity CDF', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Chemical abundance plots complete")
    
    def plot_kinematics(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Plot kinematic properties (radial velocities)
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.has_v_rad:
            logger.warning("Radial velocity data not available")
            return
        
        logger.info("Creating kinematic plots...")
        
        vrad_col = 'v_rad' if 'v_rad' in self.data.columns else 'vrad'
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Kinematic Analysis - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Radial velocity distribution
        ax = axes[0, 0]
        vrad_data = self.data[vrad_col].dropna()
        vrad_valid = vrad_data[(vrad_data > -1000) & (vrad_data < 1000)]
        
        ax.hist(vrad_valid, bins=60, alpha=0.7, edgecolor='black', color='mediumseagreen')
        ax.axvline(vrad_valid.median(), color='red', linestyle='--',
                  label=f'Median: {vrad_valid.median():.1f} km/s')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Zero velocity')
        ax.set_xlabel('Radial Velocity [km/s]', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Radial Velocity Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Velocity vs metallicity
        ax = axes[0, 1]
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            valid = ((self.data[vrad_col] > -1000) & 
                    (self.data[vrad_col] < 1000) &
                    (self.data[feh_col] > -4) &
                    (self.data[feh_col] < 1))
            
            sample = self.data[valid].sample(min(15000, valid.sum()))
            
            hexbin = ax.hexbin(sample[feh_col], 
                              sample[vrad_col],
                              gridsize=40,
                              cmap='viridis',
                              mincnt=1)
            plt.colorbar(hexbin, ax=ax, label='Count')
            
            ax.set_xlabel('[Fe/H] [dex]', fontsize=11)
            ax.set_ylabel('Radial Velocity [km/s]', fontsize=11)
            ax.set_title('Velocity vs Metallicity', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # 3. Velocity dispersion by metallicity bins
        ax = axes[1, 0]
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
            valid = ((self.data[vrad_col] > -1000) & 
                    (self.data[vrad_col] < 1000) &
                    (self.data[feh_col] > -4) &
                    (self.data[feh_col] < 1))
            
            data_valid = self.data[valid].copy()
            
            # Create metallicity bins
            feh_bins = np.linspace(-3, 0.5, 15)
            data_valid['feh_bin'] = pd.cut(data_valid[feh_col], bins=feh_bins)
            
            # Calculate dispersion per bin
            bin_stats = data_valid.groupby('feh_bin')[vrad_col].agg(['mean', 'std', 'count'])
            bin_centers = [(interval.left + interval.right) / 2 for interval in bin_stats.index]
            
            # Filter bins with enough stars
            mask = bin_stats['count'] > 10
            
            ax.errorbar(np.array(bin_centers)[mask], 
                       bin_stats.loc[mask, 'mean'], 
                       yerr=bin_stats.loc[mask, 'std'],
                       fmt='o-', 
                       capsize=5, 
                       capthick=2,
                       color='steelblue',
                       label='Velocity dispersion')
            
            ax.set_xlabel('[Fe/H] [dex]', fontsize=11)
            ax.set_ylabel('Radial Velocity [km/s]', fontsize=11)
            ax.set_title('Velocity Dispersion by Metallicity', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. High velocity stars
        ax = axes[1, 1]
        vrad_high = vrad_valid[np.abs(vrad_valid) > 200]
        
        ax.hist(vrad_high, bins=30, alpha=0.7, edgecolor='black', color='crimson')
        ax.set_xlabel('Radial Velocity [km/s]', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'High Velocity Stars (|v| > 200 km/s)\nN = {len(vrad_high)}', 
                    fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Kinematic plots complete")
    
    def plot_sky_distribution(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Plot sky distribution of ASTRA targets
        
        Args:
            figsize: Figure size (width, height)
        """
        if not (self.has_ra and self.has_dec):
            logger.warning("RA/Dec not available")
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
        
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in sample.columns else 'feh'
            feh_valid = sample[feh_col]
            feh_valid = feh_valid[(feh_valid > -4) & (feh_valid < 1)]
            valid_mask = sample[feh_col].isin(feh_valid)
            
            scatter = ax1.scatter(ra_rad[valid_mask], 
                                 dec_rad[valid_mask],
                                 c=sample.loc[valid_mask, feh_col],
                                 s=1,
                                 alpha=0.5,
                                 cmap='RdYlBu_r',
                                 vmin=-2,
                                 vmax=0.5)
            plt.colorbar(scatter, ax=ax1, label='[Fe/H]', fraction=0.046)
        else:
            ax1.scatter(ra_rad, dec_rad, s=1, alpha=0.3, color='steelblue')
        
        ax1.set_title('Mollweide Projection', fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. RA/Dec scatter
        ax2 = plt.subplot(122)
        
        if self.has_alpha_m:
            alpha_valid = sample['alpha_m']
            alpha_valid = alpha_valid[(alpha_valid > -0.5) & (alpha_valid < 0.8)]
            valid_mask = sample['alpha_m'].isin(alpha_valid)
            
            scatter = ax2.scatter(sample.loc[valid_mask, 'ra'],
                                 sample.loc[valid_mask, 'dec'],
                                 c=sample.loc[valid_mask, 'alpha_m'],
                                 s=5,
                                 alpha=0.5,
                                 cmap='RdYlGn',
                                 vmin=-0.2,
                                 vmax=0.5)
            plt.colorbar(scatter, ax=ax2, label='[α/M]')
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
    
    def identify_stellar_populations(self) -> pd.DataFrame:
        """
        Classify stars into Galactic populations based on chemistry
        
        Returns:
            DataFrame with population labels
        """
        if not (self.has_feh and self.has_alpha_m):
            logger.warning("Metallicity and alpha enhancement required")
            return None
        
        logger.info("Identifying stellar populations...")
        
        feh_col = 'fe_h' if 'fe_h' in self.data.columns else 'feh'
        data_with_pop = self.data.copy()
        
        # Initialize population column
        data_with_pop['population'] = 'Unknown'
        
        # Define populations based on chemistry
        # Thin disk: metal-rich, low alpha
        thin_disk = ((data_with_pop[feh_col] > -0.5) & 
                    (data_with_pop['alpha_m'] < 0.15))
        data_with_pop.loc[thin_disk, 'population'] = 'Thin Disk'
        
        # Thick disk: intermediate metallicity, alpha-enhanced
        thick_disk = ((data_with_pop[feh_col] > -1.5) & 
                     (data_with_pop[feh_col] <= -0.5) &
                     (data_with_pop['alpha_m'] > 0.2))
        data_with_pop.loc[thick_disk, 'population'] = 'Thick Disk'
        
        # Halo: metal-poor
        halo = (data_with_pop[feh_col] < -1.5)
        data_with_pop.loc[halo, 'population'] = 'Halo'
        
        # Count populations
        pop_counts = data_with_pop['population'].value_counts()
        
        print("\n" + "="*70)
        print("Galactic Population Classification")
        print("="*70)
        for pop, count in pop_counts.items():
            pct = 100 * count / len(data_with_pop)
            print(f"   {pop:15s}: {count:8,} ({pct:5.2f}%)")
        
        logger.info("✓ Population classification complete")
        
        return data_with_pop
    
    def compare_pipelines(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Compare results from different analysis pipelines if available
        
        Args:
            figsize: Figure size (width, height)
        """
        # Look for pipeline-specific columns
        pipeline_cols = {}
        
        # Search for Teff from different pipelines
        for col in self.data.columns:
            col_lower = col.lower()
            if 'teff' in col_lower:
                # Extract pipeline name
                for pipeline in ['ferre', 'payne', 'cannon', 'slam', 'apogeenet']:
                    if pipeline in col_lower:
                        if pipeline not in pipeline_cols:
                            pipeline_cols[pipeline] = {}
                        pipeline_cols[pipeline]['teff'] = col
        
        if len(pipeline_cols) < 2:
            logger.info("Not enough pipeline-specific columns for comparison")
            return
        
        logger.info(f"Comparing {len(pipeline_cols)} pipelines...")
        
        fig, axes = plt.subplots(1, len(pipeline_cols)-1, figsize=figsize)
        if len(pipeline_cols) == 2:
            axes = [axes]
        
        fig.suptitle(f'Pipeline Comparison - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        pipelines = list(pipeline_cols.keys())
        
        for i in range(len(pipelines)-1):
            ax = axes[i]
            pipe1 = pipelines[0]
            pipe2 = pipelines[i+1]
            
            if 'teff' in pipeline_cols[pipe1] and 'teff' in pipeline_cols[pipe2]:
                col1 = pipeline_cols[pipe1]['teff']
                col2 = pipeline_cols[pipe2]['teff']
                
                valid = (self.data[col1].notna() & 
                        self.data[col2].notna() &
                        (self.data[col1] > 2500) & 
                        (self.data[col1] < 10000) &
                        (self.data[col2] > 2500) & 
                        (self.data[col2] < 10000))
                
                sample = self.data[valid].sample(min(10000, valid.sum()))
                
                hexbin = ax.hexbin(sample[col1], 
                                  sample[col2],
                                  gridsize=40,
                                  cmap='YlOrRd',
                                  mincnt=1)
                
                # Add 1:1 line
                lims = [2500, 10000]
                ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='1:1')
                
                ax.set_xlabel(f'{pipe1.upper()} Teff [K]', fontsize=11)
                ax.set_ylabel(f'{pipe2.upper()} Teff [K]', fontsize=11)
                ax.set_title(f'{pipe1.upper()} vs {pipe2.upper()}', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                ax.set_aspect('equal')
                
                plt.colorbar(hexbin, ax=ax, label='Count')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Pipeline comparison complete")
    
    def full_analysis(self):
        """Run complete ASTRA analysis"""
        logger.info(f"Running full ASTRA analysis...")
        
        # Print summary
        self.print_summary()
        
        # Create all plots
        if self.has_teff and self.has_logg:
            self.plot_hr_diagram()
        
        if self.has_feh or self.has_alpha_m:
            self.plot_chemical_abundances()
        
        if self.has_v_rad:
            self.plot_kinematics()
        
        if self.has_ra and self.has_dec:
            self.plot_sky_distribution()
        
        # Population analysis
        if self.has_feh and self.has_alpha_m:
            pop_data = self.identify_stellar_populations()
            
            if pop_data is not None:
                # Save with population labels
                output_path = Path('./sdss_data/processed/astra_with_populations.parquet')
                pop_data.to_parquet(output_path, index=False)
                logger.info(f"💾 Saved population-labeled data to: {output_path}")
        
        # Pipeline comparison
        self.compare_pipelines()
        
        logger.info("✓ Full ASTRA analysis complete")


def load_and_analyze(catalog_id: str = 'astra_dr19', 
                     data_dir: str = './sdss_data'):
    """
    Convenience function to load and analyze ASTRA data
    
    Args:
        catalog_id: ID of catalog to load
        data_dir: Path to data directory
        
    Returns:
        ASTRAAnalyzer instance
    """
    data_path = Path(data_dir) / 'processed' / f'{catalog_id}_processed.parquet'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")
    
    logger.info(f"Loading {catalog_id} from {data_path}")
    data = pd.read_parquet(data_path)
    
    analyzer = ASTRAAnalyzer(data, catalog_name=catalog_id)
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*80)
    print("🔬 SDSS ASTRA STELLAR ANALYSIS ANALYZER")
    print("="*80)
    
    try:
        analyzer = load_and_analyze('astra_dr19')
        analyzer.full_analysis()
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nPlease run sdss_pipeline.py first to download and process ASTRA data.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()