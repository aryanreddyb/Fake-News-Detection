========================================================================
FAKE NEWS DETECTION (FND) - HOW TO RUN GUIDE
========================================================================

This guide walks you through setting up the environment, placing the dataset,
and running the end-to-end pipeline (ModernBERT baseline + Graph Neural Networks).

------------------------------------------------------------------------
STEP 1: CLONE THE REPO & OPEN DIRECTORY
------------------------------------------------------------------------
Open a terminal / command prompt and run:

    git clone https://github.com/aryanreddyb/Fake-News-Detection.git
    cd Fake-News-Detection


------------------------------------------------------------------------
STEP 2: ENVIRONMENT SETUP
------------------------------------------------------------------------

Option A: Fast Automatic Setup (Windows)
----------------------------------------
Simply run the setup batch file:

    setup.bat

(This creates a virtual environment '.venv', installs core packages,
and configures PyTorch + PyTorch Geometric automatically.)


Option B: Manual Setup (Windows / macOS / Linux)
------------------------------------------------
1. Create and activate a Python virtual environment:
   - On Windows:
       python -m venv .venv
       .venv\Scripts\activate
   - On macOS / Linux:
       python3 -m venv .venv
       source .venv/bin/activate

2. Install core dependencies:
       pip install -r requirements.txt

3. Install PyTorch:
   - For NVIDIA GPU (CUDA 12.1+):
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   - For CPU only:
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

4. Install PyTorch Geometric:
       pip install torch_geometric


------------------------------------------------------------------------
STEP 3: PLACE THE DATASET FILES
------------------------------------------------------------------------
Because the raw dataset files are large (>180MB), they are not stored on GitHub.

1. Ensure a 'data' folder exists in the project root:
       mkdir data

2. Place the following three Fakeddit TSV files inside the 'data/' directory:
       data/multimodal_train.tsv
       data/multimodal_validate.tsv
       data/multimodal_test_public.tsv

3. (Optional check) Verify your files:
       python inspect_data.py


------------------------------------------------------------------------
STEP 4: EXECUTION PIPELINE (RUN IN THIS ORDER)
------------------------------------------------------------------------
Make sure your virtual environment is active (.venv), then run:

1. Preprocessing:
   Cleans text, handles missing metadata, and creates CSV files in data/processed/.
       python src/preprocessing.py

2. (Recommended for Fast Testing / Laptops) Create Pilot Dataset:
   The full dataset has ~500k rows. To test quickly on a laptop or CPU,
   sample down to a 10,000-row pilot set:
       python src/create_pilot_data.py
   (Note: To return to the full dataset anytime, simply re-run python src/preprocessing.py)

3. Train Baseline Language Model (ModernBERT):
   Fine-tunes the transformer model on text titles:
       python src/train_baseline.py
   (Weights will be saved to outputs/models/modernbert_baseline/)

4. Build the Social / Community Graph:
   Constructs network graph edges connecting submissions by author, subreddit, etc.
       python src/graph_builder.py
   (Outputs saved to outputs/graphs/)

5. Extract Node Embeddings:
   Passes the graph nodes through the fine-tuned model to generate 768-dim embeddings:
       python src/embedding_extraction.py
   (Embeddings saved to outputs/embeddings/node_embeddings.pt)

6. Train Graph Neural Networks (GNN):
   Trains and evaluates node classification on the graph.

   - To train GraphSAGE:
       python src/gnn_train.py --model graphsage

   - To train GATv2 (Graph Attention Network v2):
       python src/gnn_train.py --model gatv2

   (Model weights and metrics are saved to outputs/models/)
========================================================================
