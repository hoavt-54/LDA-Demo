virtualenv -p python3 env
pip install -r requirements.txt
mkdir -p lda_vizs
python main.py
open lda_vizs/lda_visualization_30.html