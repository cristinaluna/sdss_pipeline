"""
SDSS APOGEE Infrared Data Analysis Module
Author: Cris Luna

Specialized analysis for APOGEE infrared spectroscopic data
Combines allStar and allVisit catalogs for comprehensive stellar analysis
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


class APOGEEAnalyzer:
    """
    Analyzer for APOGEE infrared spectroscopic data
    """
    
    def __init__(self, 
                 allstar_data: pd.DataFrame = None,
                 allvisit_data: pd.DataFrame = None,
                 catalog_name: str = "APOGEE"):
        """
        Initialize analyzer
        
        Args:
            allstar_data: DataFrame with allStar catalog (stellar parameters)
            allvisit_data: DataFrame with allVisit catalog (individual visits)
            catalog_name: Name for labeling
        """
        self.allstar = allstar_data.copy() if allstar_data is not None else None
        self.allvisit = allvisit_data.copy() if allvisit_data is not None else None
        self.catalog_name = catalog_name
        self.stats = {}
        
        if self.allstar is not None:
            self.allstar.columns = self.allstar.columns.str.lower()
            logger.info(f"Loaded allStar: {len(self.allstar)} unique stars")
        
        if self.allvisit is not None:
            self.allvisit.columns = self.allvisit.columns.str.lower()
            logger.info(f"Loaded allVisit: {len(self.allvisit)} individual visits")
        
        if self.allstar is None and self.allvisit is None:
            raise ValueError("Must provide at least one of allstar_data or allvisit_data")
        
        self._identify_columns()
    
    def _identify_columns(self):
        """Identify available columns"""
        if self.allstar is not None:
            cols = self.allstar.columns
            
            # Stellar parameters
            self.has_teff = 'teff' in cols
            self.has_logg = 'logg' in cols
            self.has_feh = 'fe_h' in cols or 'feh' in cols
            self.has_alpha = 'm_h' in cols or 'alpha_m' in cols
            
            # Position
            self.has_ra = 'ra' in cols
            self.has_dec = 'dec' in cols
            
            # Quality metrics
            self.has_snr = 'snr' in cols
            self.has_vhelio = 'vhelio_avg' in cols or 'vhelio' in cols
            
            logger.info(f"AllStar features: Teff={self.has_teff}, logg={self.has_logg}, "
                       f"[Fe/H]={self.has_feh}, position={self.has_ra}")
        
        if self.allvisit is not None:
            visit_cols = self.allvisit.columns
            self.has_visit_snr = 'snr' in visit_cols
            self.has_visit_vhelio = 'vhelio' in visit_cols
            logger.info(f"AllVisit loaded with {len(visit_cols)} columns")
    
    def basic_statistics(self) -> Dict:
        """
        Generate basic statistics
        
        Returns:
            Dictionary with statistical summaries
        """
        logger.info("Computing basic statistics...")
        
        stats = {'catalog_name': self.catalog_name}
        
        # AllStar statistics
        if self.allstar is not None:
            stats['allstar'] = {
                'total_stars': len(self.allstar)
            }
            
            # Temperature statistics
            if self.has_teff:
                teff_data = self.allstar['teff'].dropna()
                teff_valid = teff_data[(teff_data > 3000) & (teff_data < 8000)]
                stats['allstar']['teff'] = {
                    'count': len(teff_valid),
                    'mean': teff_valid.mean(),
                    'median': teff_valid.median(),
                    'std': teff_valid.std(),
                    'min': teff_valid.min(),
                    'max': teff_valid.max()
                }
            
            # Surface gravity statistics
            if self.has_logg:
                logg_data = self.allstar['logg'].dropna()
                logg_valid = logg_data[(logg_data > -1) & (logg_data < 6)]
                stats['allstar']['logg'] = {
                    'count': len(logg_valid),
                    'mean': logg_valid.mean(),
                    'median': logg_valid.median(),
                    'std': logg_valid.std()
                }
            
            # Metallicity statistics
            if self.has_feh:
                feh_col = 'fe_h' if 'fe_h' in self.allstar.columns else 'feh'
                feh_data = self.allstar[feh_col].dropna()
                feh_valid = feh_data[(feh_data > -3) & (feh_data < 1)]
                stats['allstar']['feh'] = {
                    'count': len(feh_valid),
                    'mean': feh_valid.mean(),
                    'median': feh_valid.median(),
                    'std': feh_valid.std()
                }
            
            # SNR statistics
            if self.has_snr:
                snr_data = self.allstar['snr'].dropna()
                stats['allstar']['snr'] = {
                    'mean': snr_data.mean(),
                    'median': snr_data.median(),
                    'q25': snr_data.quantile(0.25),
                    'q75': snr_data.quantile(0.75)
                }
        
        # AllVisit statistics
        if self.allvisit is not None:
            stats['allvisit'] = {
                'total_visits': len(self.allvisit)
            }
            
            # Count unique stars in visits
            if 'apogee_id' in self.allvisit.columns:
                stats['allvisit']['unique_stars'] = self.allvisit['apogee_id'].nunique()
                visits_per_star = self.allvisit.groupby('apogee_id').size()
                stats['allvisit']['visits_per_star'] = {
                    'mean': visits_per_star.mean(),
                    'median': visits_per_star.median(),
                    'max': visits_per_star.max()
                }
            
            # Visit SNR
            if self.has_visit_snr:
                visit_snr = self.allvisit['snr'].dropna()
                stats['allvisit']['snr'] = {
                    'mean': visit_snr.mean(),
                    'median': visit_snr.median()
                }
        
        self.stats = stats
        return stats
    
    def print_summary(self):
        """Print formatted summary statistics"""
        if not self.stats:
            self.basic_statistics()
        
        print("\n" + "="*80)
        print(f"APOGEE INFRARED DATA SUMMARY: {self.catalog_name}")
        print("="*80)
        
        # AllStar summary
        if 'allstar' in self.stats:
            st = self.stats['allstar']
            print(f"\n⭐ AllStar Catalog:")
            print(f"   Total stars: {st['total_stars']:,}")
            
            if 'teff' in st:
                t = st['teff']
                print(f"\n   🌡️  Effective Temperature:")
                print(f"      Valid measurements: {t['count']:,}")
                print(f"      Mean: {t['mean']:.0f} K")
                print(f"      Median: {t['median']:.0f} K")
                print(f"      Range: [{t['min']:.0f}, {t['max']:.0f}] K")
            
            if 'logg' in st:
                g = st['logg']
                print(f"\n   🪐 Surface Gravity (log g):")
                print(f"      Valid measurements: {g['count']:,}")
                print(f"      Mean: {g['mean']:.2f}")
                print(f"      Median: {g['median']:.2f}")
            
            if 'feh' in st:
                m = st['feh']
                print(f"\n   ⚛️  Metallicity [Fe/H]:")
                print(f"      Valid measurements: {m['count']:,}")
                print(f"      Mean: {m['mean']:.3f} dex")
                print(f"      Median: {m['median']:.3f} dex")
            
            if 'snr' in st:
                s = st['snr']
                print(f"\n   📡 Signal-to-Noise Ratio:")
                print(f"      Mean: {s['mean']:.1f}")
                print(f"      Median: {s['median']:.1f}")
        
        # AllVisit summary
        if 'allvisit' in self.stats:
            vt = self.stats['allvisit']
            print(f"\n🔭 AllVisit Catalog:")
            print(f"   Total visits: {vt['total_visits']:,}")
            
            if 'unique_stars' in vt:
                print(f"   Unique stars observed: {vt['unique_stars']:,}")
                
                if 'visits_per_star' in vt:
                    v = vt['visits_per_star']
                    print(f"\n   📊 Visits per star:")
                    print(f"      Mean: {v['mean']:.1f}")
                    print(f"      Median: {v['median']:.0f}")
                    print(f"      Maximum: {v['max']:.0f}")
            
            if 'snr' in vt:
                s = vt['snr']
                print(f"\n   📡 Visit SNR:")
                print(f"      Mean: {s['mean']:.1f}")
                print(f"      Median: {s['median']:.1f}")
    
    def plot_hr_diagram(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Create Hertzsprung-Russell diagram
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.allstar is None:
            logger.warning("AllStar data not available")
            return
        
        if not (self.has_teff and self.has_logg):
            logger.warning("Teff and logg required for HR diagram")
            return
        
        logger.info("Creating HR diagram...")
        
        # Filter valid data
        valid = ((self.allstar['teff'] > 3000) & 
                (self.allstar['teff'] < 8000) &
                (self.allstar['logg'] > -1) &
                (self.allstar['logg'] < 6))
        
        data = self.allstar[valid].copy()
        
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
            feh_valid = data[feh_col].notna() & (data[feh_col] > -3) & (data[feh_col] < 1)
            plot_data = data[feh_valid]
            
            scatter = ax.scatter(plot_data['teff'], 
                               plot_data['logg'],
                               c=plot_data[feh_col],
                               s=5,
                               alpha=0.6,
                               cmap='RdYlBu_r',
                               vmin=-1.5,
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
        
        # 2. Density HR diagram
        ax = axes[1]
        hexbin = ax.hexbin(data['teff'], 
                          data['logg'],
                          gridsize=50,
                          cmap='YlOrRd',
                          mincnt=1)
        cbar = plt.colorbar(hexbin, ax=ax)
        cbar.set_label('Number of stars', fontsize=11)
        
        ax.set_xlabel('Effective Temperature [K]', fontsize=11)
        ax.set_ylabel('log g [cgs]', fontsize=11)
        ax.set_title('HR Diagram (density)', fontweight='bold')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ HR diagram complete")
    
    def plot_stellar_parameters(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot distributions of stellar parameters
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.allstar is None:
            logger.warning("AllStar data not available")
            return
        
        logger.info("Creating stellar parameter plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Stellar Parameter Distributions - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Temperature distribution
        ax = axes[0, 0]
        if self.has_teff:
            teff_data = self.allstar['teff']
            teff_valid = teff_data[(teff_data > 3000) & (teff_data < 8000)]
            
            ax.hist(teff_valid, bins=50, alpha=0.7, edgecolor='black', color='orangered')
            ax.axvline(teff_valid.median(), color='blue', linestyle='--', 
                      label=f'Median: {teff_valid.median():.0f} K')
            ax.set_xlabel('Effective Temperature [K]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Temperature Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 2. Surface gravity distribution
        ax = axes[0, 1]
        if self.has_logg:
            logg_data = self.allstar['logg']
            logg_valid = logg_data[(logg_data > -1) & (logg_data < 6)]
            
            ax.hist(logg_valid, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            ax.axvline(logg_valid.median(), color='red', linestyle='--',
                      label=f'Median: {logg_valid.median():.2f}')
            ax.set_xlabel('log g [cgs]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Surface Gravity Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 3. Metallicity distribution
        ax = axes[0, 2]
        if self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.allstar.columns else 'feh'
            feh_data = self.allstar[feh_col]
            feh_valid = feh_data[(feh_data > -3) & (feh_data < 1)]
            
            ax.hist(feh_valid, bins=50, alpha=0.7, edgecolor='black', color='forestgreen')
            ax.axvline(feh_valid.median(), color='red', linestyle='--',
                      label=f'Median: {feh_valid.median():.3f}')
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Solar')
            ax.set_xlabel('[Fe/H] [dex]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Metallicity Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. Teff vs [Fe/H]
        ax = axes[1, 0]
        if self.has_teff and self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.allstar.columns else 'feh'
            valid = ((self.allstar['teff'] > 3000) & 
                    (self.allstar['teff'] < 8000) &
                    (self.allstar[feh_col] > -3) &
                    (self.allstar[feh_col] < 1))
            
            sample = self.allstar[valid].sample(min(10000, valid.sum()))
            
            ax.hexbin(sample['teff'], sample[feh_col],
                     gridsize=40, cmap='YlOrRd', mincnt=1)
            ax.set_xlabel('Effective Temperature [K]', fontsize=11)
            ax.set_ylabel('[Fe/H] [dex]', fontsize=11)
            ax.set_title('Teff vs Metallicity', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # 5. SNR distribution
        ax = axes[1, 1]
        if self.has_snr:
            snr_data = self.allstar['snr'].dropna()
            snr_valid = snr_data[snr_data < 500]  # Remove outliers
            
            ax.hist(snr_valid, bins=50, alpha=0.7, edgecolor='black', color='mediumpurple')
            ax.axvline(snr_valid.median(), color='red', linestyle='--',
                      label=f'Median: {snr_valid.median():.1f}')
            ax.set_xlabel('Signal-to-Noise Ratio', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('SNR Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 6. logg vs [Fe/H] (Kiel diagram)
        ax = axes[1, 2]
        if self.has_logg and self.has_feh:
            feh_col = 'fe_h' if 'fe_h' in self.allstar.columns else 'feh'
            valid = ((self.allstar['logg'] > -1) & 
                    (self.allstar['logg'] < 6) &
                    (self.allstar[feh_col] > -3) &
                    (self.allstar[feh_col] < 1))
            
            sample = self.allstar[valid].sample(min(10000, valid.sum()))
            
            ax.hexbin(sample['logg'], sample[feh_col],
                     gridsize=40, cmap='viridis', mincnt=1)
            ax.set_xlabel('log g [cgs]', fontsize=11)
            ax.set_ylabel('[Fe/H] [dex]', fontsize=11)
            ax.set_title('Kiel Diagram (log g vs [Fe/H])', fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Stellar parameter plots complete")
    
    def plot_visit_analysis(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Analyze visit statistics from allVisit
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.allvisit is None:
            logger.warning("AllVisit data not available")
            return
        
        logger.info("Creating visit analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Visit Analysis - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Visits per star
        ax = axes[0, 0]
        if 'apogee_id' in self.allvisit.columns:
            visits_per_star = self.allvisit.groupby('apogee_id').size()
            
            ax.hist(visits_per_star, bins=range(1, min(50, visits_per_star.max())), 
                   alpha=0.7, edgecolor='black', color='steelblue')
            ax.set_xlabel('Number of Visits', fontsize=11)
            ax.set_ylabel('Number of Stars', fontsize=11)
            ax.set_title('Visits per Star Distribution', fontweight='bold')
            ax.set_yscale('log')
            ax.grid(alpha=0.3)
        
        # 2. Visit SNR distribution
        ax = axes[0, 1]
        if self.has_visit_snr:
            snr_data = self.allvisit['snr'].dropna()
            snr_valid = snr_data[snr_data < 500]
            
            ax.hist(snr_valid, bins=50, alpha=0.7, edgecolor='black', color='coral')
            ax.axvline(snr_valid.median(), color='blue', linestyle='--',
                      label=f'Median: {snr_valid.median():.1f}')
            ax.set_xlabel('Visit SNR', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Visit SNR Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 3. Radial velocity distribution
        ax = axes[1, 0]
        if self.has_visit_vhelio:
            vhelio_data = self.allvisit['vhelio'].dropna()
            vhelio_valid = vhelio_data[(vhelio_data > -500) & (vhelio_data < 500)]
            
            ax.hist(vhelio_valid, bins=50, alpha=0.7, edgecolor='black', color='mediumseagreen')
            ax.axvline(vhelio_valid.median(), color='red', linestyle='--',
                      label=f'Median: {vhelio_valid.median():.1f} km/s')
            ax.set_xlabel('Heliocentric Radial Velocity [km/s]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Radial Velocity Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. Combining visits: SNR improvement
        ax = axes[1, 1]
        if self.has_visit_snr and 'apogee_id' in self.allvisit.columns:
            # Calculate combined SNR for stars with multiple visits
            multi_visit_stars = self.allvisit.groupby('apogee_id').filter(lambda x: len(x) >= 2)
            
            if len(multi_visit_stars) > 0:
                star_stats = multi_visit_stars.groupby('apogee_id').agg({
                    'snr': ['mean', 'count']
                }).reset_index()
                star_stats.columns = ['apogee_id', 'mean_snr', 'n_visits']
                
                # SNR should increase with sqrt(N) visits
                scatter = ax.scatter(star_stats['n_visits'], 
                                    star_stats['mean_snr'],
                                    c=star_stats['mean_snr'],
                                    s=20,
                                    alpha=0.5,
                                    cmap='viridis')
                
                ax.set_xlabel('Number of Visits', fontsize=11)
                ax.set_ylabel('Mean Visit SNR', fontsize=11)
                ax.set_title('SNR vs Number of Visits', fontweight='bold')
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Mean SNR')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Visit analysis complete")
    
    def combine_allstar_allvisit(self) -> pd.DataFrame:
        """
        Combine allStar and allVisit data
        
        Returns:
            Merged DataFrame with star parameters and visit statistics
        """
        if self.allstar is None or self.allvisit is None:
            logger.error("Both allStar and allVisit required for merging")
            return None
        
        logger.info("Combining allStar and allVisit catalogs...")
        
        # Calculate visit statistics per star
        visit_stats = self.allvisit.groupby('apogee_id').agg({
            'snr': ['mean', 'std', 'count'],
            'vhelio': ['mean', 'std'] if 'vhelio' in self.allvisit.columns else []
        })
        
        visit_stats.columns = ['_'.join(col).strip() for col in visit_stats.columns.values]
        visit_stats = visit_stats.reset_index()
        
        # Merge with allStar
        combined = self.allstar.merge(visit_stats, 
                                     on='apogee_id', 
                                     how='left',
                                     suffixes=('_star', '_visit'))
        
        logger.info(f"✓ Combined dataset: {len(combined)} stars with visit data")
        
        return combined
    
    def plot_sky_distribution(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Plot sky distribution of APOGEE targets
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.allstar is None:
            logger.warning("AllStar data not available")
            return
        
        if not (self.has_ra and self.has_dec):
            logger.warning("RA/Dec not available")
            return
        
        logger.info("Creating sky distribution plots...")
        
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Sky Distribution - {self.catalog_name}', 
                    fontsize=16, fontweight='bold')
        
        # Sample for faster plotting
        sample = self.allstar.sample(min(30000, len(self.allstar)))
        
        # 1. Mollweide projection
        ax1 = plt.subplot(121, projection='mollweide')
        
        ra_rad = np.radians(sample['ra'] - 180)
        dec_rad = np.radians(sample['dec'])
        
        if self.has_teff:
            teff_valid = sample['teff']
            teff_valid = teff_valid[(teff_valid > 3000) & (teff_valid < 8000)]
            valid_mask = sample['teff'].isin(teff_valid)
            
            scatter = ax1.scatter(ra_rad[valid_mask], 
                                 dec_rad[valid_mask],
                                 c=sample.loc[valid_mask, 'teff'],
                                 s=1,
                                 alpha=0.5,
                                 cmap='RdYlBu_r',
                                 vmin=3500,
                                 vmax=6500)
            plt.colorbar(scatter, ax=ax1, label='Teff [K]', fraction=0.046)
        else:
            ax1.scatter(ra_rad, dec_rad, s=1, alpha=0.3, color='steelblue')
        
        ax1.set_title('Mollweide Projection', fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. Galactic coordinates if available
        ax2 = plt.subplot(122)
        
        if 'glon' in sample.columns and 'glat' in sample.columns:
            if self.has_feh:
                feh_col = 'fe_h' if 'fe_h' in sample.columns else 'feh'
                feh_valid = sample[feh_col]
                feh_valid = feh_valid[(feh_valid > -3) & (feh_valid < 1)]
                valid_mask = sample[feh_col].isin(feh_valid)
                
                scatter = ax2.scatter(sample.loc[valid_mask, 'glon'],
                                     sample.loc[valid_mask, 'glat'],
                                     c=sample.loc[valid_mask, feh_col],
                                     s=5,
                                     alpha=0.5,
                                     cmap='RdYlBu_r',
                                     vmin=-1.5,
                                     vmax=0.5)
                plt.colorbar(scatter, ax=ax2, label='[Fe/H]')
            else:
                ax2.scatter(sample['glon'], sample['glat'], 
                           s=5, alpha=0.3, color='steelblue')
            
            ax2.set_xlabel('Galactic Longitude [deg]', fontsize=11)
            ax2.set_ylabel('Galactic Latitude [deg]', fontsize=11)
            ax2.set_title('Galactic Coordinates', fontweight='bold')
        else:
            # Fallback to RA/Dec
            ax2.scatter(sample['ra'], sample['dec'],
                       s=5, alpha=0.3, color='steelblue')
            ax2.set_xlabel('Right Ascension [deg]', fontsize=11)
            ax2.set_ylabel('Declination [deg]', fontsize=11)
            ax2.set_title('RA-Dec Distribution', fontweight='bold')
        
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Sky distribution complete")
    
    def full_analysis(self):
        """Run complete APOGEE analysis"""
        logger.info(f"Running full APOGEE analysis...")
        
        # Print summary
        self.print_summary()
        
        # AllStar analyses
        if self.allstar is not None:
            self.plot_hr_diagram()
            self.plot_stellar_parameters()
            self.plot_sky_distribution()
        
        # AllVisit analyses
        if self.allvisit is not None:
            self.plot_visit_analysis()
        
        # Combined analysis
        if self.allstar is not None and self.allvisit is not None:
            print("\n" + "="*80)
            print("🔗 Combined AllStar + AllVisit Analysis")
            print("="*80)
            combined = self.combine_allstar_allvisit()
            
            if combined is not None:
                print(f"\n✓ Successfully merged {len(combined)} stars with visit statistics")
                print(f"   Available columns: {len(combined.columns)}")
                
                # Save combined data
                output_path = Path('./sdss_data/processed/apogee_combined.parquet')
                combined.to_parquet(output_path, index=False)
                print(f"   💾 Saved to: {output_path}")
        
        logger.info("✓ Full APOGEE analysis complete")


def load_and_analyze(allstar_id: str = 'apogee_allstar',
                     allvisit_id: str = 'apogee_allvisit',
                     data_dir: str = './sdss_data'):
    """
    Convenience function to load and analyze APOGEE data
    
    Args:
        allstar_id: ID of allStar catalog
        allvisit_id: ID of allVisit catalog
        data_dir: Path to data directory
        
    Returns:
        APOGEEAnalyzer instance
    """
    data_dir = Path(data_dir)
    
    # Load allStar
    allstar_path = data_dir / 'processed' / f'{allstar_id}_processed.parquet'
    allstar = None
    if allstar_path.exists():
        logger.info(f"Loading {allstar_id}")
        allstar = pd.read_parquet(allstar_path)
    else:
        logger.warning(f"AllStar not found: {allstar_path}")
    
    # Load allVisit
    allvisit_path = data_dir / 'processed' / f'{allvisit_id}_processed.parquet'
    allvisit = None
    if allvisit_path.exists():
        logger.info(f"Loading {allvisit_id}")
        allvisit = pd.read_parquet(allvisit_path)
    else:
        logger.warning(f"AllVisit not found: {allvisit_path}")
    
    if allstar is None and allvisit is None:
        raise FileNotFoundError("Neither allStar nor allVisit data found")
    
    analyzer = APOGEEAnalyzer(allstar, allvisit, catalog_name="APOGEE")
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*80)
    print("🔴 SDSS APOGEE INFRARED DATA ANALYZER")
    print("="*80)
    
    try:
        analyzer = load_and_analyze()
        analyzer.full_analysis()
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nPlease run sdss_pipeline.py first to download and process APOGEE data.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()