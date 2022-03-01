# Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting
This is a PyTorch implementation of the paper: Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting.
## Requirements
python3
The detailed requirements are posted in requirements.txt 
## Data Preparation

### Download Datasets
The traffic data files for NAVER-Seoul is posted on Google Drive.
The other datasets, including METR-LA, can be found in Google Drive or Baidu Yun links provided by Li et al.

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
