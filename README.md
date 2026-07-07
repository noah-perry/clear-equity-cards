# CLEAR Equity Cards

Measuring language-related inequities in AI systems

## Repository Structure

```
├───00_data_raw
├───01_data_processed
├───02_output_model_experiments
├───03_output_analysis
└───scripts
    ├───01_process_data
    ├───02_model_experiments
    └───03_analysis
```

This repo is structured as a sequential pipeline and contains only code files. 

Empty `data_*` and `output_*` folders are included to provide the directory structure used in the scripts.

In the `scripts` folder, the scripts in each subfolder `0N_*` follow these conventions: 
- Read from data/output folder of preceding number: `0(N-1)_data_*`/`0(N-1)_output_*`
- Write to data/output folder of the same number: `0N_data_*`/`0N_output_*`

Analytical results can be reproduced as follows:
1. Create .env file containing `PROJECT_ROOT = [your filepath here]`
2. Download the data files listed in `00_data_raw/README.md`
3. Place the data files in the `00_data_raw` folder
4. Run the scripts in order