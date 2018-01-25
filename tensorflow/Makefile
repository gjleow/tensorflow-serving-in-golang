## Install conda
install_conda:
	pyenv install miniconda3-latest && pyenv local miniconda3-latest

## Set up conda environment
create_env:
	conda env create --file environment.yaml

## activate conda environment
activate_env:
	pyenv local miniconda3-latest/envs/tensorflow

## Set up and activate environment
env: install_conda create_env activate_env

## remove conda environment
remove_env:
	pyenv local miniconda3-latest && conda env remove -n tensorflow && pyenv uninstall -f miniconda3-latest
