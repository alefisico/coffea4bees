#!/bin/bash
_script_completion() {
    local cur
    cur="${COMP_WORDS[COMP_CWORD]}"
    COMPREPLY=( $(compgen -W "$(python -m classifier.autocomplete._core "${COMP_WORDS[@]}")" -- ${cur}) )
    return 0
}

complete -F _script_completion "./run_classifier.py"