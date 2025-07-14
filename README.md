# emb-explorer

**emb-explorer** is a Streamlit-based visual exploration and clustering tool for image datasets.

![Embedding Clusters](docs/images/app_screenshot_1.png)

![Cluster Summary](docs/images/app_screenshot_2.png)


## Features

* **Batch Image Embedding:**
  Efficiently embed large collections of images using the pretrained model (e.g., CLIP, BioCLIP) on CPU or GPU (preferably), with customizable batch size and parallelism. Check all available at [`data/available_models.csv`](data/available_models.csv).
* **Clustering:**
  Reduces embedding vectors to 2D using PCA, T-SNE, and UMAP. Performs K-Means clustering and display result using a scatter plot. Explore clusters via interactive scatter plots. Click on data points to preview images and details.
* **Cluster-Based Repartitioning:**
  Copy/repartition images into cluster-specific folders with a single click. Generates a summary CSV for downstream use.
* **Clustering Summary:**
  Displays cluster sizes, variances, and representative images for each cluster, helping you evaluate clustering quality.

## Installation

Create a virtual environment with `uv` and install dependencies: 
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Activate virtual environment
source .venv/bin/activate 
# Host app
streamlit run app.py
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