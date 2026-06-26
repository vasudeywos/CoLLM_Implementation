## Pretrained Weights & Training Logs

The pretrained checkpoints, training logs, and intermediate experiment outputs for this project are hosted on Google Drive due to GitHub's file size limitations.

📂 **Google Drive (Weights & Logs):**  
https://drive.google.com/drive/folders/1dndI8ub09-Iyyx5a9YEYCPdKVxQxhToi?usp=sharing

### Contents

The drive contains:

```
checkpoints_stage1_*/
checkpoints_stage2_*/
checkpoints_stage3_*/

logs_stage1_*/
logs_stage2_*/
logs_stage3_*/
logs_stage3_run3_resume/
```

These folders include:

- Model checkpoints from all training stages
- Optimizer and scheduler states (where available)
- TensorBoard/event logs
- Training and validation logs
- Resumed training checkpoints
- Other experiment artifacts required for reproducing the reported results

### Usage

1. Download the required checkpoint directory from the Google Drive link.
2. Place the downloaded folder in the project root.

Example:

```text
CoLLM_Implementation/
├── checkpoints_stage3_run3/
├── logs_stage3_run3/
├── logs_stage3_run3_resume/
├── train_stage3.py
└── ...
```

Training can then be resumed by pointing the appropriate script to the downloaded checkpoint directory.

> **Note:** The weights are hosted externally because GitHub does not support storing large model checkpoints efficiently.
