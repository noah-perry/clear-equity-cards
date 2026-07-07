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

Empty `data_*` and `output-*` folders are included to provide the directory structure used in the scripts.

In the `scripts` folder, the scripts in each subfolder follow these conventions: 
- Read from `data_*`/`output_*` folders of lower number
- Write to the `data_*`/`output_*` folder of the same number

Analytical results can be reproduced as follows:
1. Download the data files listed in `00_data_raw/README.md`
2. Place the data files in the `00_data_raw` folder
3. Run the scripts in order