## 1. Setup
### Create a Conda Environment
```
# install the python
conda create -n DL python==3.8.0
conda activate DL
# install the pytorch and torchvision
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge
```
### Install Other Dependencies
```
pip install -r requirements.txt
```

## 2. Dataset Download and Divided
When you run the experiment, the dataset will be automatically downloaded and divided according to the configuration file. But it will only be done once for the same configuration file.


## 3. Run the Experiment
You can find some configuration files in folder `config` and they correspond to different attack, defense and recover methods.  
The support attack algorithms includes **Semantic Backdoor Attack, Distributed Backdoor Attack(DBA).**  
The support defend algorithms includes **Krum, Multi-Krum, FoolsGold, Multi-metrics, Flame.**  
The support recovery algorithms includes **Retrain, FedRecover, FedEraser, Crab.**  
For exemple, to execute **DBA+Flame+FedEraser**, you can run the following commands:
```bash
python main_fed.py --config dba_flame_federaser
```
```bash
python main_recover.py --config dba_flame_federaser
```
For more detailed parameters setting, you can check the configuration files.