#!/bin/bash
_pyml_task_autocomplete() {
    local cur cmd opts
    cur="${COMP_WORDS[COMP_CWORD]}"
    opts=$(python -m classifier.task.autocomplete._bind "${COMP_WORDS[@]:0:COMP_CWORD+1}" 2>&1)
    exit_code=$?
    while [ $exit_code -eq 254 ]; do
        (python -m classifier.task.autocomplete._core &)
        opts=$(python -m classifier.task.autocomplete._bind wait "${COMP_WORDS[@]:0:COMP_CWORD+1}" 2>&1)
        exit_code=$?
    done
    if [ $exit_code -eq 0 ]; then
        mapfile -t COMPREPLY < <( compgen -W "${opts}" -- ${cur})
        return 0
    elif [ $exit_code -eq 1 ]; then
        echo "${opts}"
    elif [ $exit_code -eq 255 ]; then
        if command -v _filedir &>/dev/null; then
            _filedir
            return 0
        fi
    fi
    COMPREPLY=()
    return 0
}

complete -F _pyml_task_autocomplete "./pyml.py"