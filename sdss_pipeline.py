"""
SDSS Complete Data Pipeline - Multi-catalog Download and Analysis
Author: Cris Luna

Downloads multiple SDSS catalogs (spectroscopic, photometric, infrared, ASTRA)
and prepares them for future analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import warnings
from typing import Optional, Dict, List
import logging
from pathlib import Path
import gzip
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Disable warnings
warnings.filterwarnings('ignore')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")


class SDSSDataManager:
    """
    Manager for downloading and organizing multiple SDSS data products
    """
    
    # Define all available SDSS catalogs
    CATALOGS = {
        'spectro_dr19': {
            'url': 'https://dr19.sdss.org/sas/dr19/spectro/boss/redux/v6_1_3/spAll-v6_1_3.fits.gz',
            'filename': 'spAll-v6_1_3.fits.gz',
            'description': 'SDSS-V Optical Spectroscopic Catalog (DR19)',
            'size': '985 MB',
            'type': 'spectroscopic'
        },
        'spectro_dr17': {
            'url': 'https://dr17.sdss.org/sas/dr17/sdss/spectro/redux/specObj-dr17.fits',
            'filename': 'specObj-dr17.fits',
            'description': 'SDSS I-IV Optical Spectroscopic Catalog (DR17)',
            'size': '6.7 GB',
            'type': 'spectroscopic'
        },
        'astra_dr19': {
            'url': 'https://dr19.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraMWMLite-0.6.0.fits.gz',
            'filename': 'astraMWMLite-0.6.0.fits.gz',
            'description': 'ASTRA Stellar Analysis Results (DR19)',
            'size': '854 MB',
            'type': 'analysis'
        },
        'apogee_allstar': {
            'url': 'https://dr19.sdss.org/sas/dr19/spectro/apogee/redux/1.3/summary/allStar-1.3-apo25m.fits',
            'filename': 'allStar-1.3-apo25m.fits',
            'description': 'APOGEE Infrared Stellar Parameters',
            'size': '368 MB',
            'type': 'infrared'
        },
        'apogee_allvisit': {
            'url': 'https://dr19.sdss.org/sas/dr19/spectro/apogee/redux/1.3/summary/allVisit-1.3-apo25m.fits',
            'filename': 'allVisit-1.3-apo25m.fits',
            'description': 'APOGEE Infrared Individual Visits',
            'size': '909 MB',
            'type': 'infrared'
        },
        'photo_dr17': {
            'url': 'https://dr17.sdss.org/sas/dr17/sdss/spectro/redux/photoPosPlate-dr17.fits',
            'filename': 'photoPosPlate-dr17.fits',
            'description': 'SDSS Photometric Position Matches',
            'size': '16 GB',
            'type': 'photometric'
        }
    }
    
    def __init__(self, data_dir: str = './sdss_data'):
        """Initialize data manager"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / 'raw').mkdir(exist_ok=True)
        (self.data_dir / 'processed').mkdir(exist_ok=True)
        
        # Session for downloads
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info(f"Initialized SDSS Data Manager. Data directory: {self.data_dir}")
    
    def list_available_catalogs(self):
        """Display all available catalogs"""
        print("\n" + "="*80)
        print("Available SDSS Catalogs")
        print("="*80)
        
        for cat_id, cat_info in self.CATALOGS.items():
            status = "✓ Downloaded" if (self.data_dir / 'raw' / cat_info['filename']).exists() else "○ Not downloaded"
            print(f"\n[{cat_id}] {status}")
            print(f"  Type: {cat_info['type']}")
            print(f"  Description: {cat_info['description']}")
            print(f"  Size: {cat_info['size']}")
    
    def download_catalog(self, catalog_id: str, force: bool = False) -> Optional[Path]:
        """
        Download a specific catalog
        
        Args:
            catalog_id: ID of catalog to download
            force: Force re-download if file exists
            
        Returns:
            Path to downloaded file
        """
        if catalog_id not in self.CATALOGS:
            logger.error(f"Unknown catalog: {catalog_id}")
            logger.info(f"Available catalogs: {', '.join(self.CATALOGS.keys())}")
            return None
        
        cat_info = self.CATALOGS[catalog_id]
        local_path = self.data_dir / 'raw' / cat_info['filename']
        
        # Check if already exists
        if local_path.exists() and not force:
            logger.info(f"File already exists: {local_path}")
            return local_path
        
        logger.info(f"Downloading: {cat_info['description']} ({cat_info['size']})")
        logger.info(f"URL: {cat_info['url']}")
        
        try:
            with self.session.get(cat_info['url'], stream=True, timeout=120) as r:
                r.raise_for_status()
                
                total_size = int(r.headers.get('content-length', 0))
                
                with open(local_path, 'wb') as f:
                    downloaded = 0
                    chunk_size = 8192 * 128  # 1MB chunks
                    
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                if int(percent) % 5 == 0:
                                    logger.info(f"Progress: {percent:.0f}%")
            
            logger.info(f"✓ Download complete: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading {catalog_id}: {e}")
            if local_path.exists():
                local_path.unlink()  # Remove partial download
            return None


class SDSSPipeline:
    """
    SDSS data processing pipeline
    """
    
    def __init__(self, data_dir: str = './sdss_data'):
        """Initialize pipeline"""
        self.data_manager = SDSSDataManager(data_dir)
        self.data_dir = self.data_manager.data_dir
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        
        logger.info("Initialized SDSS Pipeline")
    
    def load_fits_to_dataframe(self, fits_path: Path, max_rows: int = None) -> Optional[pd.DataFrame]:
        """
        Load FITS file into pandas DataFrame with proper handling
        
        Args:
            fits_path: Path to FITS file
            max_rows: Maximum rows to load (None = all)
            
        Returns:
            DataFrame with catalog data
        """
        try:
            from astropy.table import Table
            
            logger.info(f"Loading FITS: {fits_path.name}")
            
            # Load table
            table = Table.read(str(fits_path))
            logger.info(f"Loaded {len(table)} rows, {len(table.columns)} columns")
            
            # Limit rows if requested
            if max_rows and len(table) > max_rows:
                logger.info(f"Limiting to first {max_rows} rows")
                table = table[:max_rows]
            
            # Separate column types
            simple_cols = []
            multi_cols = []
            
            for name in table.colnames:
                if len(table[name].shape) <= 1:
                    simple_cols.append(name)
                else:
                    multi_cols.append(name)
            
            logger.info(f"Simple columns: {len(simple_cols)}, Multi-dimensional: {len(multi_cols)}")
            
            # Convert simple columns first
            df = table[simple_cols].to_pandas()
            
            # Fix endianness issues (convert to native byte order)
            for col in df.columns:
                # Check if dtype has byteorder attribute (numpy dtypes do, pandas extension dtypes don't)
                if hasattr(df[col].dtype, 'byteorder'):
                    if df[col].dtype.byteorder not in ('=', '|'):
                        df[col] = df[col].astype(df[col].dtype.newbyteorder('='))
            
            # Expand multidimensional columns selectively
            if multi_cols:
                logger.info("Expanding multidimensional columns...")
                
                for col_name in multi_cols:
                    try:
                        col_data = table[col_name].data
                        
                        # Fix endianness (check if dtype has byteorder attribute)
                        if hasattr(col_data.dtype, 'byteorder'):
                            if col_data.dtype.byteorder not in ('=', '|'):
                                col_data = col_data.astype(col_data.dtype.newbyteorder('='))
                        
                        col_shape = col_data.shape
                        
                        # Expand small arrays
                        if len(col_shape) == 2 and col_shape[1] <= 10:
                            logger.info(f"  Expanding {col_name} ({col_shape[1]} dims)")
                            for i in range(col_shape[1]):
                                df[f"{col_name}_{i}"] = col_data[:, i]
                        
                        # Store medium arrays as strings
                        elif len(col_shape) == 2 and col_shape[1] <= 32:
                            logger.info(f"  String-ifying {col_name} ({col_shape[1]} dims)")
                            df[col_name] = [','.join(map(str, row)) for row in col_data]
                        
                        else:
                            logger.info(f"  Skipping {col_name} (shape: {col_shape})")
                            
                    except Exception as e:
                        logger.warning(f"  Error processing {col_name}: {e}")
                        continue
            
            logger.info(f"✓ Final DataFrame: {df.shape}")
            return df
            
        except ImportError:
            logger.error("astropy not installed. Install: pip install astropy")
            return None
        except Exception as e:
            logger.error(f"Error loading FITS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def prepare_data(self, df: pd.DataFrame, catalog_type: str) -> pd.DataFrame:
        """
        Clean and prepare data based on catalog type
        
        Args:
            df: Raw DataFrame
            catalog_type: Type of catalog
            
        Returns:
            Prepared DataFrame
        """
        logger.info(f"Preparing {catalog_type} data...")
        
        # Convert to lowercase
        df.columns = df.columns.str.lower()
        
        # Handle different catalog types
        if catalog_type == 'spectroscopic':
            # Calculate colors if magnitudes exist
            if 'mag_0' in df.columns and 'mag_2' in df.columns:
                # Assuming ugriz order: u=0, g=1, r=2, i=3, z=4
                df['color_gr'] = df['mag_1'] - df['mag_2']  # g-r
                df['color_ur'] = df['mag_0'] - df['mag_2']  # u-r
                df['color_iz'] = df['mag_3'] - df['mag_4']  # i-z
                logger.info("Calculated colors from MAG array")
        
        elif catalog_type == 'infrared':
            # APOGEE specific processing
            if 'teff' in df.columns:
                # Filter valid stellar parameters
                valid = (df['teff'] > 3000) & (df['teff'] < 8000)
                logger.info(f"Valid Teff range: {valid.sum()} / {len(df)}")
        
        # Remove infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Data prepared: {df.shape}")
        return df
    
    def load_catalog(self, catalog_id: str, max_rows: int = 10000) -> Optional[pd.DataFrame]:
        """
        Load a downloaded catalog into memory
        
        Args:
            catalog_id: Catalog to load
            max_rows: Maximum rows to load
            
        Returns:
            Loaded DataFrame
        """
        if catalog_id not in self.data_manager.CATALOGS:
            logger.error(f"Unknown catalog: {catalog_id}")
            return None
        
        cat_info = self.data_manager.CATALOGS[catalog_id]
        fits_path = self.data_dir / 'raw' / cat_info['filename']
        
        if not fits_path.exists():
            logger.error(f"File not found: {fits_path}")
            logger.info("Download it first using data_manager.download_catalog()")
            return None
        
        # Load FITS
        df = self.load_fits_to_dataframe(fits_path, max_rows=max_rows)
        
        if df is not None:
            # Prepare data
            df = self.prepare_data(df, cat_info['type'])
            
            # Store in memory
            self.loaded_data[catalog_id] = df
            
            # Save processed version
            output_path = self.data_dir / 'processed' / f"{catalog_id}_processed.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"✓ Saved processed data: {output_path}")
        
        return df
    
    def quick_analysis(self, catalog_id: str = None, data: pd.DataFrame = None):
        """
        Perform quick visual analysis
        
        Args:
            catalog_id: ID of loaded catalog (or provide data directly)
            data: DataFrame to analyze
        """
        if data is None:
            if catalog_id and catalog_id in self.loaded_data:
                data = self.loaded_data[catalog_id]
            else:
                logger.error("No data provided")
                return
        
        logger.info(f"Creating visualizations for {len(data)} objects...")
        
        # Check available columns
        has_colors = 'color_gr' in data.columns
        has_z = 'z' in data.columns
        has_ra_dec = 'ra' in data.columns and 'dec' in data.columns
        has_teff = 'teff' in data.columns
        has_logg = 'logg' in data.columns
        
        # Find magnitude column
        mag_col = None
        for col in ['mag_2', 'mag_r', 'modelmag_r', 'psfmag_r', 'r']:
            if col in data.columns:
                mag_col = col
                break
        
        # Count plots
        plots = []
        if has_colors and mag_col:
            plots.append('cmd')
        if has_ra_dec:
            plots.append('sky')
        if has_z:
            plots.append('redshift')
        if has_teff and has_logg:
            plots.append('hr')
        if mag_col:
            plots.append('mag_hist')
        
        if not plots:
            logger.warning("No plottable columns found")
            return
        
        # Create figure
        n_plots = len(plots)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        plot_idx = 1
        
        # Color-Magnitude Diagram
        if 'cmd' in plots:
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            plot_idx += 1
            
            valid = data['color_gr'].notna() & data[mag_col].notna()
            
            scatter = ax.scatter(data.loc[valid, 'color_gr'], 
                               data.loc[valid, mag_col],
                               c=data.loc[valid, 'z'] if has_z else range(valid.sum()),
                               cmap='viridis', s=5, alpha=0.5)
            
            if has_z:
                plt.colorbar(scatter, ax=ax, label='Redshift')
            
            ax.set_xlabel('(g - r) [mag]', fontsize=11)
            ax.set_ylabel(f'{mag_col} [mag]', fontsize=11)
            ax.set_title('Color-Magnitude Diagram', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(alpha=0.3)
        
        # Sky Distribution
        if 'sky' in plots:
            ax = plt.subplot(n_rows, n_cols, plot_idx, projection='aitoff')
            plot_idx += 1
            
            valid = data['ra'].notna() & data['dec'].notna()
            ra_rad = np.radians(data.loc[valid, 'ra'] - 180)
            dec_rad = np.radians(data.loc[valid, 'dec'])
            
            ax.scatter(ra_rad, dec_rad, s=1, alpha=0.3)
            ax.set_title('Sky Distribution', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Redshift Distribution
        if 'redshift' in plots:
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            plot_idx += 1
            
            z_valid = data['z'].dropna()
            ax.hist(z_valid, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(z_valid.median(), color='red', linestyle='--', 
                      label=f'Median: {z_valid.median():.3f}')
            ax.set_xlabel('Redshift', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Redshift Distribution', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # HR Diagram
        if 'hr' in plots:
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            plot_idx += 1
            
            valid = data['teff'].notna() & data['logg'].notna()
            scatter = ax.scatter(data.loc[valid, 'teff'], 
                               data.loc[valid, 'logg'],
                               s=5, alpha=0.5)
            
            ax.set_xlabel('Teff [K]', fontsize=11)
            ax.set_ylabel('log(g)', fontsize=11)
            ax.set_title('HR Diagram', fontweight='bold')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.grid(alpha=0.3)
        
        # Magnitude Histogram
        if 'mag_hist' in plots:
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            plot_idx += 1
            
            mag_valid = data[mag_col].dropna()
            ax.hist(mag_valid, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'{mag_col} [mag]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Magnitude Distribution', fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'SDSS Quick Analysis ({len(data)} objects)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        logger.info("✓ Visualizations complete")
    
    def summary_statistics(self, catalog_id: str = None, data: pd.DataFrame = None):
        """Generate summary statistics"""
        if data is None:
            if catalog_id and catalog_id in self.loaded_data:
                data = self.loaded_data[catalog_id]
            else:
                logger.error("No data provided")
                return
        
        print("\n" + "="*70)
        print(f"Summary Statistics")
        print("="*70)
        print(f"Total objects: {len(data):,}")
        print(f"Total columns: {len(data.columns)}")
        
        # Numeric summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns: {len(numeric_cols)}")
            print("\nKey Statistics:")
            print(data[numeric_cols].describe().T[['mean', 'std', 'min', 'max']])


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🔭 SDSS COMPLETE DATA PIPELINE")
    print("="*80)
    
    # Initialize
    pipeline = SDSSPipeline(data_dir='./sdss_data')
    
    # Show available catalogs
    pipeline.data_manager.list_available_catalogs()
    
    # Check which catalogs are downloaded
    print("\n" + "="*80)
    print("Checking downloaded catalogs...")
    print("="*80)
    
    downloaded_catalogs = []
    missing_catalogs = []
    
    for cat_id, cat_info in pipeline.data_manager.CATALOGS.items():
        fits_path = pipeline.data_dir / 'raw' / cat_info['filename']
        if fits_path.exists():
            downloaded_catalogs.append(cat_id)
            print(f"✓ {cat_id}: Already downloaded")
        else:
            missing_catalogs.append(cat_id)
            print(f"○ {cat_id}: Not downloaded")
    
    # Ask to download missing catalogs
    if missing_catalogs:
        print("\n" + "="*80)
        print(f"Found {len(missing_catalogs)} catalog(s) not downloaded")
        print("="*80)
        
        for cat_id in missing_catalogs:
            cat_info = pipeline.data_manager.CATALOGS[cat_id]
            print(f"\n📦 {cat_id}")
            print(f"   Description: {cat_info['description']}")
            print(f"   Size: {cat_info['size']}")
            print(f"   Type: {cat_info['type']}")
            
            # Ask user
            response = input(f"\n   Download {cat_id}? (y/n): ").strip().lower()
            
            if response == 'y':
                print(f"\n   Downloading {cat_id}...")
                fits_path = pipeline.data_manager.download_catalog(cat_id)
                
                if fits_path:
                    downloaded_catalogs.append(cat_id)
                    print(f"   ✓ Downloaded successfully")
                else:
                    print(f"   ✗ Download failed")
            else:
                print(f"   ⊘ Skipped")
    
    # Process all downloaded catalogs
    if not downloaded_catalogs:
        print("\n⚠️  No catalogs to process. Exiting.")
        return pipeline, {}
    
    print("\n" + "="*80)
    print(f"Processing {len(downloaded_catalogs)} downloaded catalog(s)...")
    print("="*80)
    
    processed_data = {}
    
    for i, cat_id in enumerate(downloaded_catalogs, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(downloaded_catalogs)}] Processing: {cat_id}")
        print("="*80)
        
        try:
            # Load and process
            data = pipeline.load_catalog(cat_id, max_rows=10000)
            
            if data is not None:
                processed_data[cat_id] = data
                
                # Show summary
                print(f"\n📊 Summary for {cat_id}:")
                print(f"   Objects loaded: {len(data):,}")
                print(f"   Total columns: {len(data.columns)}")
                
                # Quick stats
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                print(f"   Numeric columns: {len(numeric_cols)}")
                
                # Check for key columns
                key_cols = ['ra', 'dec', 'z', 'teff', 'logg', 'color_gr']
                found_cols = [col for col in key_cols if col in data.columns]
                if found_cols:
                    print(f"   Key columns found: {', '.join(found_cols)}")
                
                print(f"   ✓ Saved to: sdss_data/processed/{cat_id}_processed.parquet")
        
        except Exception as e:
            logger.error(f"Failed to process {cat_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Create visualizations for all processed catalogs
    if processed_data:
        print("\n" + "="*80)
        print("Creating visualizations...")
        print("="*80)
        
        for cat_id, data in processed_data.items():
            print(f"\n📈 Visualizing {cat_id}...")
            try:
                pipeline.quick_analysis(data=data)
            except Exception as e:
                logger.warning(f"Could not create visualizations for {cat_id}: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   Catalogs downloaded: {len(downloaded_catalogs)}")
    print(f"   Catalogs processed: {len(processed_data)}")
    
    if processed_data:
        print(f"\n✓ Successfully processed:")
        for cat_id in processed_data.keys():
            obj_count = len(processed_data[cat_id])
            col_count = len(processed_data[cat_id].columns)
            print(f"   • {cat_id}: {obj_count:,} objects, {col_count} columns")
    
    print(f"\n📁 Data locations:")
    print(f"   Raw FITS files: {pipeline.data_dir / 'raw'}")
    print(f"   Processed files: {pipeline.data_dir / 'processed'}")
    
    print(f"\n💡 Quick reload example:")
    print(f"   import pandas as pd")
    for cat_id in processed_data.keys():
        print(f"   {cat_id} = pd.read_parquet('sdss_data/processed/{cat_id}_processed.parquet')")
    
    return pipeline, processed_data


if __name__ == "__main__":
    pipeline, data = main()