# pip3 install -r requirements.txt
conda env create -f packages.yaml
export PYTHONPATH="$PYTHONPATH:`pwd`/models"