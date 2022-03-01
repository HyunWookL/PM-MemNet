# Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting
This is a PyTorch implementation of the paper: Hyunwook Lee, Seungmin Jin, Hyeshin Chu, Hongkyu Lim, Sungahn Ko, Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting, ICLR 2022.
## Requirements
python3  
scipy>=0.19.0  
numpy  
pandas  
pyyaml  
torch>=1.9.0  

## Data Preparation

### Download Datasets
The traffic data files for NAVER-Seoul is posted on [Google Drive](https://drive.google.com/drive/folders/1HjpDu7EjBKvA2e7WrsyPZGFEJL5PhzeG).
The other datasets, including METR-LA, can be found in [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [Li et al.](https://github.com/liyaguang/DCRNN).

### Process Datasets
In the data processing stage, We have same process as Li et al.
```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY,NAVER-Seoul}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_fiilename=data/metr-la.h5 --seq_length_x INPUT_SEQ_LENGTH --seq_length_y PRED_SEQ_LENGTH

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_fiilename=data/pems-bay.h5 --seq_length_x INPUT_SEQ_LENGTH --seq_length_y PRED_SEQ_LENGTH

# NAVER-Seoul
python generate_training_data.py --output_dir=data/NAVER-Seoul --traffic_df_fiilename=data/naver-seoul.csv --seq_length_x INPUT_SEQ_LENGTH --seq_length_y PRED_SEQ_LENGTH
```

## Model Training
Code and detailed instruction will be updated soon.
