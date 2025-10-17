# 🔭 SDSS Data Pipeline

A comprehensive Python pipeline for downloading, processing, and visualizing data from the Sloan Digital Sky Survey (SDSS) Data Release 19.

## Features

- **Direct Downloads**: Downloads SDSS catalogs directly from the official servers
- **Automatic Processing**: Converts FITS files to pandas DataFrames with cleaning
- **Rich Visualizations**: Creates publication-quality plots including:
  - Sky distribution maps
  - Redshift distributions
  - Color-magnitude diagrams
  - Data quality metrics
- **Memory Efficient**: Handles large datasets with configurable row limits
- **Multiple Catalogs**: Support for optical spectra, APOGEE, and ASTRA data

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```python
python sdss_pipeline.py
```

This will:
1. Download the optical spectroscopic catalog (~985 MB compressed)
2. Process 10,000 objects (configurable)
3. Create comprehensive visualizations
4. Save processed data as Parquet file

### Advanced Usage

```python
from sdss_pipeline import SDSSPipeline

# Initialize pipeline
pipeline = SDSSPipeline(data_dir="sdss_data")

# List available catalogs
pipeline.list_available_catalogs()

# Run pipeline for specific catalog
df = pipeline.run(
    catalog='optical_spectra',  # or 'apogee_allstar', 'astra_summary'
    max_rows=50000  # None for all rows (memory intensive!)
)

# Access the data
print(df.head())
print(df.describe())
```

## Available Catalogs

- **optical_spectra**: SDSS-V optical spectroscopic data (v6_1_3)
- **apogee_allstar**: APOGEE stellar parameters (1.3)
- **apogee_allvisit**: APOGEE visit-level data (1.3)
- **astra_summary**: ASTRA pipeline results (0.6.0)

## Project Structure

```
sdss_pipeline/
├── sdss_pipeline.py          # Main pipeline code
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── sdss_data/               # Data directory (ignored by git)
│   ├── raw/                 # Downloaded FITS files
│   └── processed/           # Processed Parquet files
└── output_data_examples/    # Visualization outputs
    └── *.png                # Analysis plots
```

## Output Files

### Processed Data
- Location: `sdss_data/processed/`
- Format: Parquet (efficient columnar storage)
- Files: `{catalog_name}_processed.parquet`

### Visualizations
- Location: `output_data_examples/`
- Format: PNG (150 DPI)
- Includes: Sky maps, redshift distributions, color diagrams, etc.

## Memory Management

Large SDSS catalogs can be memory-intensive. Use the `max_rows` parameter to limit data:

```python
# Load only 10,000 rows (good for testing)
df = pipeline.run(catalog='optical_spectra', max_rows=10000)

# Load all rows (requires significant RAM)
df = pipeline.run(catalog='optical_spectra', max_rows=None)
```

## Troubleshooting

### Download Issues

If downloads fail:
1. Check your internet connection
2. Verify SDSS servers are accessible: https://dr19.sdss.org
3. Try downloading manually from the URLs in the code

### Memory Errors

If you encounter memory errors:
1. Reduce `max_rows` parameter
2. Close other applications
3. Process catalogs individually

### Missing Dependencies

If imports fail:
```bash
pip install --upgrade -r requirements.txt
```

## Data Sources

All data downloaded from:
- **SDSS DR19**: https://dr19.sdss.org
- **Documentation**: https://www.sdss.org/dr19/

## 📖 Citation

### Citing This Pipeline

If you use this pipeline in your research or projects, please cite:

**Plain text:**
```
C. Luna (2025), SDSS Data Pipeline: Automated Download and Analysis of Sloan Digital Sky Survey Catalogs, DOI: 10.5281/zenodo.17375241
```

**BibTeX:**
```bibtex
@software{luna2025sdss,
  author       = {Luna, Cristina},
  title        = {{SDSS Data Pipeline: Automated Download and 
                   Analysis of Sloan Digital Sky Survey Catalogs}},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.17375241},
  url          = {https://doi.org/10.5281/zenodo.17375241}
}
```

**APA:**
```
Luna, C. (2025). SDSS Data Pipeline: Automated Download and Analysis of Sloan Digital Sky Survey Catalogs (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17375241
```

### Citing SDSS Data

If you use SDSS data accessed through this pipeline in publications, you **must also cite** the appropriate SDSS data release:

**SDSS DR19:**
```bibtex
@article{sdss_dr19,
  author       = {{Almeida}, A. and {Anderson}, S.~F. and others},
  title        = {{The Nineteenth Data Release of the Sloan Digital Sky Surveys}},
  journal      = {ApJS},
  year         = 2024,
  note         = {In preparation}
}
```

**General SDSS Citation:**
```
Funding for the Sloan Digital Sky Survey (SDSS) has been provided by the Alfred P. Sloan 
Foundation, the Participating Institutions, the National Aeronautics and Space Administration, 
the National Science Foundation, the U.S. Department of Energy, the Japanese Monbukagakusho, 
and the Max Planck Society.
```

For detailed citation guidelines, visit: [https://www.sdss.org/collaboration/citing-sdss/](https://www.sdss.org/collaboration/citing-sdss/)

### Acknowledgment Example

If you use this pipeline in a paper, consider adding:

> "Data analysis was performed using the SDSS Data Pipeline (Luna 2025, DOI: 10.5281/zenodo.17375241), 
> which facilitated the download and processing of SDSS DR19 spectroscopic catalogs."

## 📄 License

### Pipeline Code License

This pipeline code is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

```
SDSS Data Pipeline - Automated Download and Analysis Tool
Copyright (C) 2025 Cris Luna

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```

**What this means:**
- ✅ **Free to use** for any purpose (research, education, commercial)
- ✅ **Free to modify** and adapt to your needs
- ✅ **Free to distribute** copies and modifications
- ⚠️ **Share-alike**: If you modify and distribute this code, you must also release your modifications under AGPL-3.0
- ⚠️ **Network use**: If you run a modified version on a server that users can interact with, you must provide access to your source code
- ⚠️ **Attribution required**: You must preserve copyright notices and provide attribution

For the full license text, see the [LICENSE](LICENSE) file or visit [https://www.gnu.org/licenses/agpl-3.0.html](https://www.gnu.org/licenses/agpl-3.0.html)

### SDSS Data License

**The astronomical data accessed through this pipeline belongs to the SDSS Collaboration** and is subject to their data access policy:

> All SDSS data are publicly available without restrictions. However, users are requested to:
> 1. **Acknowledge SDSS** in publications using the data
> 2. **Cite the appropriate data release** (DR17, DR19, etc.)
> 3. **Follow the SDSS publication policy** for collaborative science

**SDSS Data Rights:**
- 🌍 **Public Domain**: SDSS data are freely available to the scientific community and the public
- 📚 **No proprietary period**: All data become public immediately upon release
- 🤝 **Attribution requested**: While not legally required, scientific courtesy requires proper citation

For complete details, see:
- [SDSS Data Access Policy](https://www.sdss.org/collaboration/data-access-policy/)
- [SDSS Collaboration Policy](https://www.sdss.org/collaboration/the-collaboration-policy/)
- [SDSS Citation Guidelines](https://www.sdss.org/collaboration/citing-sdss/)

### Third-Party Libraries

This pipeline uses the following open-source libraries, each with their own licenses:

| Library | License | Purpose |
|---------|---------|---------|
| [NumPy](https://numpy.org) | BSD-3-Clause | Numerical computing |
| [Pandas](https://pandas.pydata.org) | BSD-3-Clause | Data manipulation |
| [Matplotlib](https://matplotlib.org) | PSF-based | Visualization |
| [Seaborn](https://seaborn.pydata.org) | BSD-3-Clause | Statistical plots |
| [Astropy](https://www.astropy.org) | BSD-3-Clause | Astronomy tools |
| [PyArrow](https://arrow.apache.org/docs/python/) | Apache-2.0 | Parquet format |

All dependencies are compatible with the AGPL-3.0 license.

### Disclaimer

```
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

This pipeline is **not officially endorsed** by the SDSS Collaboration. It is an independent tool created to facilitate access to publicly available SDSS data.

### Contact

For questions about the pipeline code license: [Open an issue](https://github.com/yourusername/sdss-pipeline/issues)

For questions about SDSS data usage: Contact [SDSS Helpdesk](https://www.sdss.org/help/)

## Author

Cris Luna

## Acknowledgments

- SDSS Collaboration for providing public data access
- Astropy community for excellent astronomy tools