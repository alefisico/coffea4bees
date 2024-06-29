#!/bin/bash
_script_completion() {
    local cur opts
    cur="${COMP_WORDS[COMP_CWORD]}"
    opts=$(python -m classifier.task.autocomplete._core "${COMP_WORDS[@]}" 2>&1)
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    elif [ $exit_code -eq 1 ]; then
        _filedir
    else
        COMPREPLY=()
        return 0
    fi
}

complete -F _script_completion "./run_classifier.py"