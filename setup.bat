@echo off
echo ========================================================
echo Fake News Detection - Capstone Project Setup
echo ========================================================

echo 1. Creating Python Virtual Environment...
python -m venv .venv
call .venv\Scripts\activate

echo 2. Installing Core Dependencies...
pip install -r requirements.txt

echo 3. Installing PyTorch (CPU Version by default for compatibility)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo 4. Installing PyTorch Geometric...
pip install torch_geometric

echo ========================================================
echo Setup Complete! 
echo To activate the environment in the future, run: .venv\Scripts\activate
echo ========================================================
pause
