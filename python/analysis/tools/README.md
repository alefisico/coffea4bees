# Some useful tools

## Merge friend tree metafiles

```console
python -m analysis.tools.merge_friend_meta [-h] -o OUTPUT -i INPUT [INPUT ...] [--cleanup]

options:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        input metafiles
  -o OUTPUT, --output OUTPUT
                        output metafile
  --cleanup             remove input metafiles after merging
```
