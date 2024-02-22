# php-plots ![Python version](https://img.shields.io/badge/Python-%E2%89%A53.6-blue)

PHP based plot browser for EOS (sub)directories via web.cern.ch.

For detailed setup and usage instructions, see the [CAT documentation](https://cms-analysis.docs.cern.ch/guidelines/other/plot_browser).

The current project supersedes a previous version of the plot browser `index.php` script.
It can still be accessed through the [`old_version` branch](https://gitlab.cern.ch/cms-analysis/general/php-plots/-/tree/old_version), however, please mind the potential outdated instructions.

## Settings of the main `index.php` file

The main `index.php` file contains a few settings at the top of the file that can be configured according to your needs.

- `$main_extension`: Extension of plot files to show in cards. Defaults to `"png"`.
- `$additional_extensions`: Additional extensions to link in card footer if existing. Defaults to `("png", "pdf", "jpg", "jpeg", "gif", "eps", "svg", "root", "cxx", "txt", "rtf", "log", "csv")`.
- `$search_mode`: The search mode in case one or multiple search patterns are provided. Defaults to `"any"`.
    - `"any"`: Any search pattern must match.
    - `"all"`: All search patterns must match.
    - `"exact"`: The search pattern must match as is.

## Additional scripts

A handful of scripts (prefixed with `pb` for plot browser) are provided to help you with the deployment of files.

### `bin/pb_copy_index.py`

The `index.php` file is meant to be copied into every subdirectory that should have plot browsing capabilities when visited.
You can use the `pb_copy_index.py` script to (recursively) copy the file into specific directories.

```shell
> pb_copy_index.py --help

usage: pb_copy_index.py [-h] [--recursive] directories [directories ...]

Copies the index.php file of the plot browser to various directories.

positional arguments:
  directories      the directories to copy the index.php file to

optional arguments:
  -h, --help       show this help message and exit
  --recursive, -r  copy the index.php file recursively into all subdirectories
```

### `bin/pb_pdf_to_png.py`

Many plotting pipelines produce only pdf files, however, it can often be helpful to also have accompanying png files stored next to them.
This is also true for the plot browser, which is way faster at showing simple png files compared to rendering many pdf files inside your web browser.
You can use the `pb_pdf_to_png.py` script to convert multiple pdf files at once, optionally recursively in all subdirectories of a given path.

```shell
> pb_pdf_to_png.py --help

usage: pb_pdf_to_png.py [-h] [--recursive] [--cores CORES] paths [paths ...]

Converts one or multiple pdf files to png using "pdftocairo".

positional arguments:
  paths                 files to convert or directories to check for pdf files

optional arguments:
  -h, --help            show this help message and exit
  --recursive, -r       convert pdf files recursively in all subdirectories
  --cores CORES, -j CORES
                        number of cores to use for parallel conversion
```

### `bin/pb_deploy_plots.py`

If you produce your plots at a location that is not within a directory accessible through a public website (e.g. `www`), the typical workflow is to copy multiple files somewhere into your `www` directory while potentially preserving directory structures.
This can be achieved with the `pb_deploy_plots.py` script which, in addition, also copies the `index.php` file into any newly created subdirectory and optionally also converts pdf into png files.

```shell
> pb_deploy_plots.py --help

usage: pb_deploy_plots.py [-h] [--extensions EXTENSIONS] [--pdf-to-png] [--recursive] [--cores CORES]
                          sources [sources ...] destination

Copies images recursively to a target directory, adds plot browser index files to all newly created directories, and optionally
converts pdf files to png.

positional arguments:
  sources               source files or directories to check for plots
  destination           target directory to copy files to

optional arguments:
  -h, --help            show this help message and exit
  --extensions EXTENSIONS, -e EXTENSIONS
                        comma-separated extensions of files to copy; default: ('png', 'pdf', 'jpg', 'jpeg', 'gif', 'eps',
                        'svg', 'root', 'cxx', 'txt', 'rtf', 'log')
  --pdf-to-png, -c      convert pdf files to png
  --recursive, -r       convert pdf files recursively in all subdirectories
  --cores CORES, -j CORES
                        number of cores to use for parallel conversion of pdf files
```
