conda env create -f tf-apple.yml -n ml
conda activate ml
pip install tsaug
python -m pip install -U matplotlib
pip install imblearn
yes | conda install seaborn
pip install tqdm
pip install typeguard
pip install fastdtw
pip install pyts
yes | conda install pandas
yes | conda install -c conda-forge statsmodels
pip install pycatch22
