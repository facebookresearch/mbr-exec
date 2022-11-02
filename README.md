# Natural Language to Code Translation with Execution
[Freda Shi](mailto:freda@ttic.edu), Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, [Sida I. Wang](mailto:sida@fb.com)

## Setup
1. Download the [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), [Spider](https://yale-lily.github.io/spider), and [NL2Bash](https://github.com/TellinaTool/nl2bash) datasets to `data/` and follow their instructions for necessary preprocessing steps. 
2. Download our [collected Codex data](https://dl.fbaipublicfiles.com/mbr-exec/mbr-exec-release.zip). We have included the pre-executed result with the data; see also `execution.py` if you'd like to execute automatically collected code locally. 
3. Install the `conda` environment by 
```bash
conda env create -f env.yml
```
--- 
## Run the Selector
Suppose that the collected Codex data is located at `data/mbr-exec/`, the following code returns the execution accuracy of selected MBPP `test` split code among `5` samples which are collected with temperature `0.3` (from Codex), using the `mbr_exec`(ours) method and random seed `0`: 

```python
from sample_selectors import select_mbpp
select_mbpp(('test', 0.3, 'mbr_exec', 'data/mbr-exec/mbpp/', 5, 0))
```

See also the code for more details. 



--- 
## License
MIT
