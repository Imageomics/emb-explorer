# emb-explorer

**emb-explorer** is a Streamlit-based visual exploration and clustering tool for image datasets.

![Embedding Clusters](docs/images/app_screenshot_1.png)

![Cluster Summary](docs/images/app_screenshot_2.png)


## Features

* **Batch Image Embedding:**
  Efficiently embed large collections of images using the pretrained model (e.g., CLIP, BioCLIP) on CPU or GPU (preferably), with customizable batch size and parallelism. 
* **Clustering:**
  Reduces embedding vectors to 2D using PCA, T-SNE, and UMAP. Performs K-Means clustering and display result using a scatter plot. Explore clusters via interactive scatter plots. Click on data points to preview images and details.
* **Cluster-Based Repartitioning:**
  Copy/repartition images into cluster-specific folders with a single click. Generates a summary CSV for downstream use.
* **Clustering Summary:**
  Displays cluster sizes, variances, and representative images for each cluster, helping you evaluate clustering quality.

## Installation

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver. Install `uv` first if you haven't already:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```bash
# Clone the repository
git clone https://github.com/Imageomics/emb-explorer.git
cd emb-explorer

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/Imageomics/emb-explorer.git
cd emb-explorer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Option 3: From requirements.txt (Legacy)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from requirements.txt
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration with CUDA, install the additional GPU dependencies:

```bash
# With uv
uv pip install -e ".[gpu]"

# With pip
pip install -e ".[gpu]"
```

## Usage

### Running the Application

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the Streamlit app
streamlit run app.py
```

### Command Line Tools

The project also provides command-line utilities:

```bash
# List all available models
python list_models.py --format table

# List models in JSON format
python list_models.py --format json --pretty

# List models as names only
python list_models.py --format names

# Get help for the list models command
python list_models.py --help
```

### Quick Setup Script

For convenience, you can use the provided setup script:

```bash
# Make script executable and run
chmod +x setup.sh

# Install dependencies and run the app
./setup.sh

# Or run specific commands
./setup.sh install  # Just install dependencies
./setup.sh models   # List available models
./setup.sh run      # Just run the app
./setup.sh help     # Show help
```

### Development Testing

To test your installation:

```bash
# Run development tests
python test_installation.py
```

### Running on Remote Compute Nodes

If running the app on a remote compute node (e.g., HPC cluster), you'll need to set up port forwarding to access the Streamlit interface from your local machine.

1. **Start the app on the compute node:**
   ```bash
   # On the remote compute node
   streamlit run app.py
   ```
   Note the port number (default is 8501) and the compute node hostname.

2. **Set up SSH port forwarding from your local machine:**
   ```bash
   # From your local machine
   ssh -N -L 8501:<COMPUTE_NODE>:8501 <USERNAME>@<LOGIN_NODE>
   ```
   
   **Example:**
   ```bash
   ssh -N -L 8501:c0828.ten.osc.edu:8501 username@cardinal.osc.edu
   ```
   
   Replace:
   - `<COMPUTE_NODE>` with the actual compute node hostname (e.g., `c0828.ten.osc.edu`)
   - `<USERNAME>` with your username
   - `<LOGIN_NODE>` with the login node address (e.g., `cardinal.osc.edu`)

3. **Access the app:**
   Open your web browser and navigate to `http://localhost:8501`

The `-N` flag prevents SSH from executing remote commands, and `-L` sets up the local port forwarding.



## Acknowledgements

* [CLIP](https://github.com/openai/CLIP)
* [Streamlit](https://streamlit.io/)
* [Altair](https://altair-viz.github.io/)

---