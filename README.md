# Forgetting_TracIn

See result in results filefolder

## Usage of this repository: 

- RunCode by python file

1. create filefolder: cpts1, cpts2, datasets
2. pip install -r requirements.txt
3. main package:
   - torch               ==1.12.1
   - torchaudio          ==0.12.1
   - torchvision         ==0.13.1
   - Pillow              ==9.2.0
   - matplotlib          ==3.5.2
   - matplotlib-inline   ==0.1.6
   - numpy               ==1.23.3
   - cffi                ==1.15.1
4. change tool.CONSTANTS.py's acc_detail_per_epoch_file_path, forgetting_score_results, TracIn_results, TracIn_original_train_indexes_original_test_indexes to YourPath
5. python main.py

- Run code by Colab:
  https://colab.research.google.com/drive/16WzYUcuedznhxsoXLSqRK4aaFnWwM9ow?usp=sharing

  - create cpts1, cpts2 folder