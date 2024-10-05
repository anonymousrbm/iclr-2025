mkdir data

# HGD
wget https://github.com/nbereux/dataset/raw/main/1kg_xtrain.d.zip -O data/1kg_xtrain.d.zip
unzip -o data/1kg_xtrain.d.zip -d data/
rm data/1kg_xtrain.d.zip

# MICKEY
wget https://github.com/nbereux/dataset/raw/main/mickey.npy.zip -O data/mickey.npy.zip
unzip -o data/mickey.npy.zip -d data/
rm data/mickey.npy.zip

# MNIST
wget https://figshare.com/ndownloader/files/25635053 -O data/mnist.pkl.gz

# 2d3c
wget https://github.com/nbereux/dataset/raw/main/2d3c.zip -O data/2d3c.zip
unzip -o data/2d3c.zip -d data/data_2d3c_balanced_seed18_N1000.npy
rm data/2d3c.zip

# CelebA_32_bw

# CelebA_64_bw

# BKACE
wget https://github.com/nbereux/dataset/raw/main/BKACE.fasta -O data/BKACE.fasta

# PF00014
wget https://github.com/pagnani/ArDCAData/raw/master/data/PF00014/PF00014_mgap6.fasta.gz -O data/PF00014.fasta.gz
gzip -d data/PF00014.fasta.gz

# PF00072
wget https://github.com/nbereux/dataset/raw/main/PF00072.fasta -O data/PF00072.fasta

# PF00076
wget https://github.com/pagnani/ArDCAData/raw/master/data/PF00076/PF00076_mgap6.fasta.gz -O data/PF00076.fasta.gz
gzip -d data/PF00076.fasta.gz

# PF00595
wget https://github.com/pagnani/ArDCAData/raw/master/data/PF00595/PF00595_mgap6.fasta.gz -O data/PF00595.fasta.gz
gzip -d data/PF00595.fasta.gz

# PF13354
wget https://github.com/pagnani/ArDCAData/raw/master/data/PF13354/PF13354_wo_ref_seqs.fasta.gz -O data/PF13354.fasta.gz
gzip -d data/PF13354.fasta.gz



python scripts/preprocess_data/MNIST.py

python scripts/preprocess_data/HGD.py

python scripts/preprocess_data/MICKEY.py

python scripts/preprocess_data/2d3c.py
