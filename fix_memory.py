"""
Regenerate 02_features.ipynb with memory-efficient data loading.
"""
import json

nb_path = r"d:\storage-anomaly-detection\notebooks\02_features.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

# We need to replace cell index 2 (the Step 1 code cell)
# and cell index 1 (the Step 1 markdown) with updated versions.

# First, let's find the Step 1 code cell — it's the first code cell
code_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code_cell_idx = i
        break

print(f"First code cell is at index {code_cell_idx}")

# Also find the markdown cell right before it (Step 1 explanation)
md_idx = code_cell_idx - 1

# Update the Step 1 markdown to explain the memory-efficient approach
nb["cells"][md_idx]["source"] = [
    "# PART A -- DATA CLEANING\n",
    "\n",
    "---\n",
    "\n",
    "## Step 1 -- Load the Dataset (Memory-Efficient Approach)\n",
    "\n",
    "**What are we doing?**\n",
    "\n",
    "In Notebook 01, we explored *one day* of data to understand the structure. Now we need to load **all 92 days** into a single DataFrame. But there's a problem -- loading everything at once can exceed your computer's RAM.\n",
    "\n",
    "**Why does loading data cause memory issues?**\n",
    "\n",
    "Each daily CSV has ~330,000 rows. With 92 files, that's ~30 million rows. Even with only 19 columns, pandas stores each number as a 64-bit float (8 bytes). Quick math:\n",
    "- 30M rows x 19 columns x 8 bytes = ~4.5 GB just for numbers\n",
    "- Plus string columns (serial_number, model) which use even more\n",
    "- Plus the intermediate list of DataFrames doubles the peak usage\n",
    "\n",
    "That can easily exceed 8-12 GB of available RAM.\n",
    "\n",
    "**Our memory-efficient strategy (3 tricks):**\n",
    "\n",
    "1. **Identify the top 3 drive models FIRST** -- we scan only the `model` column (super lightweight), then filter per-file during loading. This cuts the row count dramatically before data even enters memory.\n",
    "\n",
    "2. **Use `float32` instead of `float64`** for all S.M.A.R.T. columns -- halves memory usage with negligible precision loss (we don't need 15 decimal places for sensor readings).\n",
    "\n",
    "3. **Concatenate incrementally** -- we process each file, filter it, and only keep the rows that pass our model filter.\n",
    "\n",
    "Think of it like moving apartments. Instead of loading EVERYTHING into the truck and then sorting at the new place, you sort first and only load what you need. Much more efficient.\n",
    "\n",
    "**How does this connect to the ML pipeline?**\n",
    "\n",
    "This is the *foundation*. Every future step depends on this DataFrame. By being memory-smart here, we avoid crashes and can work on machines with limited RAM."
]

# Update the Step 1 code cell with memory-efficient version
nb["cells"][code_cell_idx]["source"] = [
    "# ============================================================\n",
    "# Step 1 -- Load All 92 CSV Files (Memory-Efficient)\n",
    "# ============================================================\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import glob\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# --- Define which columns to load ---\n",
    "meta_cols = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']\n",
    "\n",
    "smart_cols = [\n",
    "    'smart_1_raw',    # Read Error Rate\n",
    "    'smart_3_raw',    # Spin-Up Time\n",
    "    'smart_4_raw',    # Start/Stop Count\n",
    "    'smart_5_raw',    # Reallocated Sectors      <- HIGH failure relevance\n",
    "    'smart_7_raw',    # Seek Error Rate\n",
    "    'smart_9_raw',    # Power-On Hours\n",
    "    'smart_10_raw',   # Spin Retry Count\n",
    "    'smart_12_raw',   # Power Cycle Count\n",
    "    'smart_187_raw',  # Reported Uncorrectable   (low coverage ~34%)\n",
    "    'smart_188_raw',  # Command Timeout           (low coverage ~34%)\n",
    "    'smart_190_raw',  # Airflow Temperature        (low coverage ~34%)\n",
    "    'smart_192_raw',  # Power-Off Retract Count\n",
    "    'smart_193_raw',  # Load/Unload Cycle Count\n",
    "    'smart_194_raw',  # Temperature (Celsius)\n",
    "    'smart_197_raw',  # Current Pending Sectors   <- HIGH failure relevance\n",
    "    'smart_198_raw',  # Uncorrectable Sectors     <- HIGH failure relevance\n",
    "    'smart_199_raw',  # UltraDMA CRC Error Count\n",
    "]\n",
    "\n",
    "use_cols = meta_cols + smart_cols\n",
    "\n",
    "# Define dtypes to reduce memory: float32 instead of float64\n",
    "# This saves ~50% memory for numeric columns\n",
    "smart_dtypes = {col: 'float32' for col in smart_cols}\n",
    "smart_dtypes['capacity_bytes'] = 'float64'  # needs full precision (large number)\n",
    "smart_dtypes['failure'] = 'int8'             # only 0 or 1, so 1 byte is enough\n",
    "\n",
    "# --- Find all CSV files ---\n",
    "csv_folder = '../data/raw/data_Q4_2025/'\n",
    "csv_files = sorted(glob.glob(os.path.join(csv_folder, '*.csv')))\n",
    "print(f'Found {len(csv_files)} CSV files')\n",
    "print(f'First: {os.path.basename(csv_files[0])}  |  Last: {os.path.basename(csv_files[-1])}')\n",
    "print()\n",
    "\n",
    "# -------------------------------------------------------\n",
    "# TRICK 1: Identify the top 3 models BEFORE full loading\n",
    "# We scan ONLY the 'model' column from each file (very fast, ~100MB total)\n",
    "# -------------------------------------------------------\n",
    "print('Phase 1: Scanning for top 3 drive models (lightweight)...')\n",
    "model_counts = pd.Series(dtype='int64')\n",
    "\n",
    "for filepath in csv_files:\n",
    "    # Read ONLY the 'model' column -- uses almost no memory\n",
    "    models = pd.read_csv(filepath, usecols=['model'])['model']\n",
    "    day_counts = models.value_counts()\n",
    "    model_counts = model_counts.add(day_counts, fill_value=0)\n",
    "\n",
    "model_counts = model_counts.sort_values(ascending=False).astype(int)\n",
    "top_3_models = model_counts.head(3).index.tolist()\n",
    "\n",
    "print(f'Top 3 models identified: {top_3_models}')\n",
    "for m in top_3_models:\n",
    "    print(f'  {m}: {model_counts[m]:,} rows across all files')\n",
    "print()\n",
    "\n",
    "# -------------------------------------------------------\n",
    "# TRICK 2 & 3: Load with dtype optimization + filter per file\n",
    "# -------------------------------------------------------\n",
    "print('Phase 2: Loading data (filtered to top 3 models, float32)...')\n",
    "dfs = []\n",
    "\n",
    "for i, filepath in enumerate(csv_files):\n",
    "    # Load with optimized dtypes\n",
    "    day_df = pd.read_csv(filepath, usecols=use_cols, dtype=smart_dtypes)\n",
    "    \n",
    "    # Filter to top 3 models IMMEDIATELY -- drops ~40-60% of rows per file\n",
    "    day_df = day_df[day_df['model'].isin(top_3_models)]\n",
    "    \n",
    "    dfs.append(day_df)\n",
    "    \n",
    "    if (i + 1) % 10 == 0 or (i + 1) == len(csv_files):\n",
    "        print(f'  Loaded {i+1}/{len(csv_files)}: {os.path.basename(filepath)}  ({len(day_df):,} rows kept)')\n",
    "\n",
    "# Concatenate all days\n",
    "df = pd.concat(dfs, ignore_index=True)\n",
    "del dfs  # free memory immediately\n",
    "\n",
    "print()\n",
    "print(f'Combined DataFrame: {df.shape[0]:,} rows x {df.shape[1]} columns')\n",
    "print(f'Memory usage: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB')\n",
    "print()\n",
    "df.head()"
]

# Now update the Step 1 post-markdown (cell after the code cell)
post_md_idx = code_cell_idx + 1
nb["cells"][post_md_idx]["source"] = [
    "### What just happened?\n",
    "\n",
    "We loaded all 92 CSV files using a **two-phase approach**:\n",
    "\n",
    "1. **Phase 1 (lightweight scan):** Read only the `model` column from each file to find the top 3 models. This used almost no memory.\n",
    "2. **Phase 2 (filtered load):** Read all 19 columns but immediately filtered each file to only the top 3 models. This means ~40-60% fewer rows enter memory.\n",
    "\n",
    "**Memory-saving tricks used:**\n",
    "\n",
    "| Trick | What It Does | Savings |\n",
    "|---|---|---|\n",
    "| `dtype='float32'` | Uses 4 bytes per number instead of 8 | ~50% less memory for SMART columns |\n",
    "| `dtype='int8'` for failure | Uses 1 byte instead of 8 | ~87% less for that column |\n",
    "| Filter per file | Drops non-top-3 rows before appending | ~40-60% fewer rows |\n",
    "| `del dfs` | Frees the intermediate list after concat | Reclaims GB of memory |\n",
    "\n",
    "The total memory should now be well under 4 GB -- manageable on most machines.\n",
    "\n",
    "**NOTE:** Since we already filtered to top 3 models here, Step 4 will be quick -- it just confirms the filtering.\n",
    "\n",
    "---"
]

# Also need to update Step 4 since we already filtered in Step 1
# Find Step 4 cells -- search for "Step 4" in markdown cells
step4_md_idx = None
step4_code_idx = None
step4_post_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "Step 4" in src and "Filter" in src and cell["cell_type"] == "markdown" and step4_md_idx is None:
        step4_md_idx = i
    elif step4_md_idx is not None and cell["cell_type"] == "code" and step4_code_idx is None:
        step4_code_idx = i
    elif step4_code_idx is not None and cell["cell_type"] == "markdown" and step4_post_idx is None:
        step4_post_idx = i
        break

print(f"Step 4 cells: md={step4_md_idx}, code={step4_code_idx}, post={step4_post_idx}")

# Update Step 4 markdown
nb["cells"][step4_md_idx]["source"] = [
    "## Step 4 -- Verify the Drive Model Filter\n",
    "\n",
    "**What are we doing?**\n",
    "\n",
    "In Step 1, we already filtered to the top 3 models during loading (to save memory). Now let's verify that worked correctly and see the distribution.\n",
    "\n",
    "**Why does filtering by model matter?**\n",
    "\n",
    "Different drive models have different hardware, firmware, and S.M.A.R.T. behavior. A temperature of 40C might be normal for one model but dangerously hot for another. Mixing ALL models would be like training a doctor on humans, cats, and fish -- the \"normal\" vital signs are completely different.\n",
    "\n",
    "By focusing on the top 3 (most data = strongest patterns), the ML model learns consistent behavior."
]

# Update Step 4 code
nb["cells"][step4_code_idx]["source"] = [
    "# ============================================================\n",
    "# Step 4 -- Verify the Drive Model Filter\n",
    "# ============================================================\n",
    "\n",
    "# We already filtered in Step 1, so let's just verify\n",
    "model_counts = df['model'].value_counts()\n",
    "\n",
    "print(f'Unique drive models: {len(model_counts)}')\n",
    "print()\n",
    "print('Drive model distribution:')\n",
    "for model, count in model_counts.items():\n",
    "    pct = count / len(df) * 100\n",
    "    print(f'  {model:<30s}  {count:>10,} rows  ({pct:.1f}%)')\n",
    "\n",
    "print()\n",
    "print(f'Total rows: {len(df):,}')\n",
    "print(f'All rows belong to top 3 models: {len(model_counts) == 3}')"
]

# Update Step 4 post-markdown
nb["cells"][step4_post_idx]["source"] = [
    "### What just happened?\n",
    "\n",
    "We confirmed that the DataFrame contains exactly 3 drive models. The filtering was already done in Step 1 during loading (our memory optimization trick).\n",
    "\n",
    "---"
]

# Save
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4, ensure_ascii=True)

print("Notebook updated with memory-efficient loading!")
