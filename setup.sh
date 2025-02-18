# Clone the repository
git clone https://github.com/abjadai/catt.git

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies (if any)
pip install -r requirements.txt

cd catt
mkdir models/
wget -P models/ https://github.com/abjadai/catt/releases/download/v2/best_ed_mlm_ns_epoch_178.pt