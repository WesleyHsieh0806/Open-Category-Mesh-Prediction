# create a new environment
# conda create --name Open-Category python=3.9 -y
# conda activate Open-Category


pip install -r requirements.txt
# pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# install pytorch and torchvision
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# install pre-built pytorch3d and its dependency
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# For Demo and Tests/Linting
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python
pip install black usort flake8 flake8-bugbear flake8-comprehensions

# install pytorch3d
conda install pytorch3d -c pytorch3d

# install hydra to read configuration
pip install hydra-core --upgrade


# install open3d
pip install open3d==0.17.0

