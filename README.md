# TAFT Setup, Training, and Evaluation Guide

This README provides a detailed guide on how to configure, set up, train, and evaluate the TAFT models. The models of TAFT were trained and evaluated using 8 A100 Nvidia GPUs. 

## Prerequisites

- **Hardware**: 8 A100 Nvidia GPUs (or alternative GPUs, though adjustments may be necessary)
- **Software**: Ensure all required packages are installed as specified in `requirements.txt`

## Initial Setup

1. **Clone the Repository**: Clone the TAFT repository to your local machine.
2. **Install Dependencies**: Install all required packages using the following command:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Main Script

The main script for starting the TAFT process is `main.py`. It comes with several flags to control different aspects of the training and evaluation process.

### Available Flags

- `--quick`: Generates data with a small size and trains models with few runs and epochs. This is crucial for the initial setup to verify if everything is working correctly and all packages are installed. This helps avoid wasting time and computational effort on a full training or evaluation process if there are issues.
   ```sh
   python3 main.py --quick
   ```

- `--batchSize`: Adjust the batch sizes for the process if you do not have 8 A100 Nvidia GPUs available. The default batch sizes are `32, 32, 16, 1, 8`. You can modify them to fit your hardware, for example, `8, 8, 4, 2, 2`. Different batch sizes correspond to different models used during the TAFT workflow.
   ```sh
   python3 main.py --batchSize "8,8,4,2,2"
   ```

- `--data`: Skip the data generation process and use the data provided within the paper.
   ```sh
   python3 main.py --data
   ```

- `--model`: Apply only the last evaluation stage on the correction model FLAN-T5 used and trained within the paper.
   ```sh
   python3 main.py --model
   ```

### Important Notes

- We have not uploaded all 60 models used within the detection stage. These can be easily trained using `main.py`.
- Ensure that you run the following command to verify the setup:
  ```sh
  python3 main.py --quick
  ```

### Execution Time
Please note that running the entire process can take several hours due to the computation-intensive training steps of the 60 models involved. Ensure that you have sufficient computational resources and time before initiating the full training process.

## FLAN-T5 Model Download

Due to file size limitations of git and LFS, the FLAN-T5 model is hosted on Figshare. You can download it using the following command:

```sh
wget -P /your/current/folder/models/Correction/paperModel/ https://figshare.com/ndownloader/files/47153303
```

Replace `/your/current/folder` with your actual directory path.

## UDATA and FlashFill Code

Within the `UDATA` folder, the code used to apply UDATA and FlashFill is provided.

- **UDATA**: The code for UDATA comes from the repository [ieee-bigdata2019-transformation](https://github.com/minhptx/ieee-bigdata2019-transformation).
- **FlashFill**: We used the built-in FlashFill feature from Excel to evaluate FlashFill.

## Summary

By following this guide, you should be able to set up, train, and evaluate the TAFT models successfully. Ensure all dependencies are installed and test the setup with the `--quick` flag before proceeding with full-scale training or evaluation.
