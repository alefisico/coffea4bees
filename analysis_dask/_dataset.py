from typing import TypedDict


class DatasetMetadata(TypedDict):
    """
    A metadata representing all (process, year) combinations.
    To specify data eras, the year can be a dictionary like {UL17: [B, C, D, E, F], UL18: [A, B, C, D]}
    """

    processes: str | list[str]
    years: str | list[str] | dict[str, list[str]]


def _ensure_iter(str_or_any):
    if isinstance(str_or_any, str):
        return [str_or_any]
    return str_or_any


def _fetch_mc(db):
    return {k: db[k] for k in ["sumw", "sumw2"]}


def _fetch_data(db):
    return {k: db[k] for k in []}  # placeholder


def picoAOD(
    metadata: dict[str],
    datasets: list[DatasetMetadata],
    tree: str = "Events",
):
    """
    Create a fileset for :func:`coffea.dataset_tools.preprocess` from ``"metadata/datasets_*.yml".``
    """
    fileset = {}
    for dataset in datasets:
        processes = _ensure_iter(dataset["processes"])
        years = _ensure_iter(dataset["years"])
        for process in processes:
            if isinstance(years, list):
                for year in years:
                    key = f"{process}_{year}"
                    if key in fileset:
                        raise ValueError(f"Dataset {key} already exists. ")
                    db = metadata[process][year]["picoAOD"]
                    fileset[key] = {
                        "files": {file: tree for file in db["files"]},
                        "metadata": dict(
                            process=process,
                            year=year,
                            **_fetch_mc(db),
                        ),
                    }
            elif isinstance(years, dict):
                for year, eras in years.items():
                    for era in eras:
                        key = f"{process}_{year}_{era}"
                        if key in fileset:
                            raise ValueError(f"Dataset {key} already exists. ")
                        db = metadata[process][year]["picoAOD"][era]
                        fileset[key] = {
                            "files": {file: tree for file in db["files"]},
                            "metadata": dict(
                                process=process,
                                year=year,
                                era=era,
                                **_fetch_data(db),
                            ),
                        }
    return fileset
