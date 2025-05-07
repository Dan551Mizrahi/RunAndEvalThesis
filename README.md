# Hypergraph Analysis and Minimal Hitting Set Enumeration Evaluation

This project provides a framework for analyzing hypergraph properties and evaluating the performance of Minimal Hitting Set (MHS) enumeration algorithm of `PyMELib` library, particularly focusing on preprocessing time and enumeration delays using rooted disjoint branch-nice tree decompositions. It includes tools for running experiments on datasets, collecting performance metrics, and analyzing the results.

## Features

* **Hypergraph Feature Extraction**: Calculates various structural features of hypergraphs, including:
    * Number of vertices and hyperedges.
    * Hyperedge size statistics (min, max, average).
    * Overall size and density.
    * Number of connected components and diameter.
    * Treewidth based on rooted disjoint branch-nice tree decomposition.
    * Root-specific measures like join node counts/sizes and branching characteristics.
* **MHS Enumeration & Evaluation**:
    * Implements preprocessing and enumeration phases for MHS algorithms based on tree decomposition (using `PyMELib`).
    * Measures preprocessing runtime and individual MHS enumeration delays.
* **Dataset Processing**:
    * Script (`RunningOnDataset.py`) to run the analysis pipeline on entire folders of hypergraph files (`.dat`, `.graph`). The format of the hypergraph files is expected to be compatible with `PyMELib`.
    * Utilizes multiprocessing (`pebble`) for efficient execution on multiple files.
    * Includes timeout and memory limit handling (`memoryManagement.py`).
* **Results Analysis & Visualization**:
    * Saves detailed results (features, runtimes, delays) to JSON files (`WritingResultsToJSON.py`).
    * Scripts within the `figures_analysis/` directory for processing these results:
        * Reading and aggregating data from JSON files (`readJSONtoDict.py`).
        * Filtering data based on specific criteria (`FilterData.py`).
        * Calculating aggregate statistics and correlations (`FindAggFunctionOnData.py`).
        * Generating plots using Matplotlib to visualize relationships between features and performance (`ProfessionalGraphsPLT.py`).
        * XGboost and linear regression analysis for deeper insights into the data (`XGboostTry.py`).

## Dependencies

* Python 3.x
* **PyMELib**
* `networkx`: For graph operations.
* `tqdm`: For progress bars.
* `pebble`: For multiprocessing with timeouts.
* `matplotlib`: For generating plots.
* `numpy`: For numerical operations, especially in analysis.
* `scipy`: Potentially used by dependencies.
* `sklearn`: For linear regression in analysis plots.
* `xgboost`: For XGBoost analysis.
* `pandas`: For data manipulation and analysis.

## File Structure
```
RunAndEvalThesis/
├── PreprocessEnumRunEval.py    - Core logic for feature extraction, preprocessing, and MHS enumeration for a single hypergraph.
├── RootMeasures.py             - Functions to calculate specific measures from tree decompositions.
├── RunningOnDataset.py         - Script to run the analysis on folders of hypergraphs, handles multiprocessing and errors.
├── WritingResultsToJSON.py     - Utility to save experiment results dictionaries to JSON format.
├── memoryManagement.py         - Utility for setting process memory limits.
├── figures_analysis/           - Directory containing scripts for analyzing and plotting results.
│   ├── FilterData.py           - Filters results data based on criteria.
│   ├── FindAggFunctionOnData.py- Calculates aggregate statistics (mean, correlation etc.) on results.
│   ├── ProfessionalGraphsPLT.py- Generates various plots from the analysis results.
│   ├── XGboostTry.py           - Implements XGBoost and linear regression analysis on the results.
│   └── readJSONtoDict.py       - Reads and aggregates results from JSON files.
└── README.md                   - This file.
```
## Usage

###Analyzing a Single Hypergraph
The core logic is in PreprocessEnumRunEval.py. 
You can import and use the running_times_in_dict function to analyze a single hypergraph file and get a dictionary of results. 
See the if __name__ == '__main__': block in PreprocessEnumRunEval.py for an example.

### Running on Datasets (RunningOnDataset.py)
Use the RunningOnDataset.py script to process entire folders of hypergraph files (can be used with CLI).

Arguments:
folder_path: Path to the folder containing the hypergraph files (.dat or .graph).
-d or --folder_of_datasets: If set, treats folder_path as a directory containing multiple dataset folders, processing each one.

## Analyzing and Plotting Results
Navigate to the figures_analysis/ directory and run the scripts there (e.g., ProfessionalGraphsPLT.py) to analyze the generated JSON files and create plots. 
You might need to adjust paths or parameters within these scripts depending on where your results are stored and what figures do you want to create.

## Output Format (JSON)
The analysis for each hypergraph is saved as a JSON file. The structure includes:

Calculated hypergraph features (e.g., "Num of Vertices", "Treewidth", "Number of Join Nodes").
Performance metrics:
"Preprocessing Runtime" (seconds).
"Number of Minimal Hitting Sets" found.
"Delays": A list containing the cumulative time (seconds) to enumerate each MHS.
"Average delay" (total enumeration time / number of MHS).
Error indicators (e.g., "Timeout", "Memory", "Recursion") if the analysis failed for a specific reason.
```JSON
{
    "Num of Vertices": 150,
    "Num of Hyperedges": 50,
    "n + m": 200,
    "Max Hyperedge Size": 7,
    "Min Hyperedge Size": 2,
    "Avg Hyperedge Size": 4.5,
    "Size of Hypergraph": 375,
    "Number of Connected Components": 1,
    "Diameter": 8,
    "Sparsity": 0.03,
    "Treewidth": 5,
    "(m + n) * tw": 1000,
    "Number of Join Nodes": 15,
    "Size of Join Nodes": 60,
    "Special Join Measure": 3125.0,
    "Max Join Node Size": 5,
    "Min Join Node Size": 3,
    "Number of Branching": 45,
    "Max Branching": 4,
    "NDBTD Width": 6,
    "Preprocessing Runtime": 15.234,
    "Number of Minimal Hitting Sets": 5230,
    "Delays": [0.001, 0.002, 0.003 ],
    "Average delay": 0.02303174
}
```

## AI Usage Acknowledgment

Parts of this project were developed with the assistance of AI tools. Specifically:
* **Google Gemini:** Utilized for generating code structures, explaining algorithms, debugging, and drafting documentation (including portions of this README).
* **GitHub Copilot:** Employed for code completion and suggestions during the development process.
