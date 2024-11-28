# MH-MoE
The implementation of "Leveraging Heterogeneous Experts with Advantageous Pattern Memory Learning for Traffic Prediction".

### Training Instructions
To train the model, modify the parameters in `run_moe.sh`. The framework will automatically perform training and testing.

```bash
cd model
bash run_moe.sh
```

### Parameters in `run_moe.sh`
- **`MODE`**: Specifies the mode of operation (`train` or `test`).
- **`MODEL_LIST`**: A list of expert models to use, separated by commas (e.g., `STID,GWN,STWave`).
- **`load_train_paths`**: Paths to trained weights for the models in `MODEL_LIST`, listed in the same order.
- **`dataset_use`**: The dataset used for training or testing (e.g., `PEMS04`).
- **`batch_size`**: The number of samples processed at once during training or testing (default is 32).

### Acknowledgement
Our work is built upon the code base of [FlashST](https://github.com/HKUDS/FlashST) and [STID](https://github.com/GestaltCogTeam/STID), which we have adapted to meet the requirements of our research. We sincerely appreciate the authors for making their implementations and associated resources publicly available.