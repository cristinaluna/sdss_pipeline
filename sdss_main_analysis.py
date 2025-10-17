"""
SDSS Master Analysis Runner
Author: Cris Luna

Unified interface to run all SDSS data analyses:
- Spectroscopic (optical spectra)
- APOGEE (infrared spectra)
- ASTRA (stellar parameters)
"""

import sys
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sdss_spectro_analysis import SpectroscopicAnalyzer, load_and_analyze as load_spectro
from sdss_apogee_analysis import APOGEEAnalyzer, load_and_analyze as load_apogee
from sdss_astra_analysis import ASTRAAnalyzer, load_and_analyze as load_astra

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SDSSMasterAnalyzer:
    """
    Master controller for all SDSS data analyses
    """
    
    def __init__(self, data_dir: str = './sdss_data'):
        """
        Initialize master analyzer
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        
        self.analyzers = {}
        self.available_catalogs = self._scan_available_catalogs()
        
        logger.info(f"Initialized SDSS Master Analyzer")
        logger.info(f"Found {len(self.available_catalogs)} processed catalogs")
    
    def _scan_available_catalogs(self):
        """Scan for available processed catalogs"""
        if not self.processed_dir.exists():
            logger.warning(f"Processed directory not found: {self.processed_dir}")
            return []
        
        parquet_files = list(self.processed_dir.glob('*.parquet'))
        catalogs = []
        
        for file in parquet_files:
            # Extract catalog ID from filename
            catalog_id = file.stem.replace('_processed', '')
            catalogs.append(catalog_id)
        
        return catalogs
    
    def list_available_catalogs(self):
        """Display available catalogs grouped by type"""
        print("\n" + "="*80)
        print("AVAILABLE PROCESSED CATALOGS")
        print("="*80)
        
        # Group by type
        spectro = [c for c in self.available_catalogs if 'spectro' in c]
        apogee = [c for c in self.available_catalogs if 'apogee' in c]
        astra = [c for c in self.available_catalogs if 'astra' in c]
        photo = [c for c in self.available_catalogs if 'photo' in c]
        other = [c for c in self.available_catalogs 
                if c not in spectro and c not in apogee and c not in astra and c not in photo]
        
        if spectro:
            print("\n🔭 Spectroscopic Catalogs (Optical):")
            for cat in spectro:
                print(f"   • {cat}")
        
        if apogee:
            print("\n🔴 APOGEE Catalogs (Infrared):")
            for cat in apogee:
                print(f"   • {cat}")
        
        if astra:
            print("\n🔬 ASTRA Catalogs (Stellar Analysis):")
            for cat in astra:
                print(f"   • {cat}")
        
        if photo:
            print("\n📷 Photometric Catalogs:")
            for cat in photo:
                print(f"   • {cat}")
        
        if other:
            print("\n📦 Other Catalogs:")
            for cat in other:
                print(f"   • {cat}")
        
        if not self.available_catalogs:
            print("\n⚠️  No processed catalogs found.")
            print("   Run sdss_pipeline.py first to download and process data.")
    
    def analyze_spectroscopic(self, catalog_id: str = 'spectro_dr19'):
        """
        Run spectroscopic analysis
        
        Args:
            catalog_id: Spectroscopic catalog to analyze
        """
        if catalog_id not in self.available_catalogs:
            logger.error(f"Catalog not found: {catalog_id}")
            return None
        
        print("\n" + "="*80)
        print(f"RUNNING SPECTROSCOPIC ANALYSIS: {catalog_id}")
        print("="*80)
        
        try:
            analyzer = load_spectro(catalog_id, str(self.data_dir))
            self.analyzers[catalog_id] = analyzer
            
            analyzer.full_analysis()
            
            return analyzer
            
        except Exception as e:
            logger.error(f"Failed to analyze {catalog_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_apogee(self, 
                       allstar_id: str = 'apogee_allstar',
                       allvisit_id: str = 'apogee_allvisit'):
        """
        Run APOGEE analysis
        
        Args:
            allstar_id: AllStar catalog ID
            allvisit_id: AllVisit catalog ID
        """
        print("\n" + "="*80)
        print("RUNNING APOGEE INFRARED ANALYSIS")
        print("="*80)
        
        try:
            analyzer = load_apogee(allstar_id, allvisit_id, str(self.data_dir))
            self.analyzers['apogee'] = analyzer
            
            analyzer.full_analysis()
            
            return analyzer
            
        except Exception as e:
            logger.error(f"Failed to analyze APOGEE: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_astra(self, catalog_id: str = 'astra_dr19'):
        """
        Run ASTRA analysis
        
        Args:
            catalog_id: ASTRA catalog to analyze
        """
        if catalog_id not in self.available_catalogs:
            logger.error(f"Catalog not found: {catalog_id}")
            return None
        
        print("\n" + "="*80)
        print(f"RUNNING ASTRA STELLAR ANALYSIS: {catalog_id}")
        print("="*80)
        
        try:
            analyzer = load_astra(catalog_id, str(self.data_dir))
            self.analyzers[catalog_id] = analyzer
            
            analyzer.full_analysis()
            
            return analyzer
            
        except Exception as e:
            logger.error(f"Failed to analyze {catalog_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_all(self):
        """Run all available analyses"""
        print("\n" + "="*80)
        print("🚀 RUNNING ALL SDSS ANALYSES")
        print("="*80)
        
        results = {}
        
        # Analyze spectroscopic catalogs
        spectro_cats = [c for c in self.available_catalogs if 'spectro' in c]
        for cat in spectro_cats:
            print(f"\n{'='*80}")
            print(f"Analyzing: {cat}")
            print("="*80)
            results[cat] = self.analyze_spectroscopic(cat)
        
        # Analyze APOGEE if both catalogs available
        apogee_allstar = 'apogee_allstar' in self.available_catalogs
        apogee_allvisit = 'apogee_allvisit' in self.available_catalogs
        
        if apogee_allstar or apogee_allvisit:
            print(f"\n{'='*80}")
            print("Analyzing: APOGEE")
            print("="*80)
            results['apogee'] = self.analyze_apogee()
        
        # Analyze ASTRA catalogs
        astra_cats = [c for c in self.available_catalogs if 'astra' in c]
        for cat in astra_cats:
            print(f"\n{'='*80}")
            print(f"Analyzing: {cat}")
            print("="*80)
            results[cat] = self.analyze_astra(cat)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ ALL ANALYSES COMPLETE!")
        print("="*80)
        
        successful = sum(1 for v in results.values() if v is not None)
        print(f"\nSuccessfully analyzed: {successful}/{len(results)} catalogs")
        
        for cat, analyzer in results.items():
            status = "✓" if analyzer is not None else "✗"
            print(f"   {status} {cat}")
        
        return results
    
    def interactive_menu(self):
        """Interactive menu for selecting analyses"""
        while True:
            print("\n" + "="*80)
            print("SDSS MASTER ANALYSIS - INTERACTIVE MENU")
            print("="*80)
            
            print("\n1. List available catalogs")
            print("2. Analyze spectroscopic data")
            print("3. Analyze APOGEE data")
            print("4. Analyze ASTRA data")
            print("5. Run all analyses")
            print("6. Show loaded analyzers")
            print("0. Exit")
            
            try:
                choice = input("\nSelect option: ").strip()
                
                if choice == '0':
                    print("\n👋 Exiting...")
                    break
                
                elif choice == '1':
                    self.list_available_catalogs()
                
                elif choice == '2':
                    spectro_cats = [c for c in self.available_catalogs if 'spectro' in c]
                    if not spectro_cats:
                        print("\n⚠️  No spectroscopic catalogs found")
                        continue
                    
                    print("\nAvailable spectroscopic catalogs:")
                    for i, cat in enumerate(spectro_cats, 1):
                        print(f"   {i}. {cat}")
                    
                    cat_choice = input("\nSelect catalog (number or name): ").strip()
                    
                    if cat_choice.isdigit():
                        idx = int(cat_choice) - 1
                        if 0 <= idx < len(spectro_cats):
                            self.analyze_spectroscopic(spectro_cats[idx])
                    else:
                        if cat_choice in spectro_cats:
                            self.analyze_spectroscopic(cat_choice)
                        else:
                            print(f"❌ Unknown catalog: {cat_choice}")
                
                elif choice == '3':
                    self.analyze_apogee()
                
                elif choice == '4':
                    astra_cats = [c for c in self.available_catalogs if 'astra' in c]
                    if not astra_cats:
                        print("\n⚠️  No ASTRA catalogs found")
                        continue
                    
                    print("\nAvailable ASTRA catalogs:")
                    for i, cat in enumerate(astra_cats, 1):
                        print(f"   {i}. {cat}")
                    
                    cat_choice = input("\nSelect catalog (number or name): ").strip()
                    
                    if cat_choice.isdigit():
                        idx = int(cat_choice) - 1
                        if 0 <= idx < len(astra_cats):
                            self.analyze_astra(astra_cats[idx])
                    else:
                        if cat_choice in astra_cats:
                            self.analyze_astra(cat_choice)
                        else:
                            print(f"❌ Unknown catalog: {cat_choice}")
                
                elif choice == '5':
                    confirm = input("\n⚠️  This will run ALL analyses. Continue? (y/n): ").strip().lower()
                    if confirm == 'y':
                        self.analyze_all()
                
                elif choice == '6':
                    if not self.analyzers:
                        print("\n📭 No analyzers loaded yet")
                    else:
                        print("\n📊 Loaded analyzers:")
                        for name, analyzer in self.analyzers.items():
                            print(f"   • {name}: {type(analyzer).__name__}")
                
                else:
                    print("\n❌ Invalid option")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("🌌 SDSS MASTER ANALYSIS SYSTEM")
    print("="*80)
    print("\nThis tool provides unified access to all SDSS data analyses:")
    print("  • Spectroscopic data (optical spectra)")
    print("  • APOGEE data (infrared spectra)")
    print("  • ASTRA data (stellar parameters)")
    
    # Initialize master analyzer
    master = SDSSMasterAnalyzer()
    
    # Check if any catalogs available
    if not master.available_catalogs:
        print("\n" + "="*80)
        print("⚠️  NO PROCESSED CATALOGS FOUND")
        print("="*80)
        print("\nPlease run sdss_pipeline.py first to download and process SDSS data.")
        print("\nExample:")
        print("  python sdss_pipeline.py")
        return
    
    # Show available catalogs
    master.list_available_catalogs()
    
    # Ask user what to do
    print("\n" + "="*80)
    print("ANALYSIS OPTIONS")
    print("="*80)
    print("\n1. Run all analyses automatically")
    print("2. Interactive menu (choose what to analyze)")
    print("3. Exit")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        master.analyze_all()
    elif choice == '2':
        master.interactive_menu()
    else:
        print("\n👋 Exiting...")
    
    print("\n" + "="*80)
    print("✨ SDSS MASTER ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()