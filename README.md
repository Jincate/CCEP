# Centralized Cooperative Exploration Policy for Continuous Control Tasks

PyTorch implementation of Centralized Cooperative Exploration Policy(CCEP). This method is tested on [MuJoCo](https://mujoco.org) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 

This code is developed with python 3.7 and the networks are trained using [PyTorch 1.10.2](https://github.com/pytorch/pytorch). The version of [MuJoCo](https://mujoco.org) is 2.1.2.
 
## Installation
To get the environment installed correctly, you will need to [download](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) mujoco files and have the path added to your PYTHONPATH environment variable.
```Shell

# Download MuJoCo files
sudo mkdir /home/.mujoco
cp mujoco210-linux-x86_64.tar.gz ~/.mujoco
cd .mujoco
tar -zxvf mujoco210-linux-x86_64.tar.gz

# Add Environment Dependencies
sudo vim ~/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Update Environment Dependencies
source ~/.bashrc

# Install mujoco.
pip3 install -U 'mujoco-py<2.2,>=2.1'
```
A more explicit installation of mujoco can be found [here](https://github.com/openai/mujoco-py#install-mujoco):

## Usage

Experiments on single environments can be run by calling:

```Python
python main.py --env HalfCheetah-v3
```
