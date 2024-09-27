import subprocess
import os

def is_git_directory(path = '.'):
    return subprocess.call(['git', '-C', path, 'status'], stderr=subprocess.STDOUT, stdout = open(os.devnull, 'w')) == 0

def get_git_revision_short_hash() -> str:
    if is_git_directory():
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    else: return 'Not run locally.'

def get_git_revision_hash() -> str:
    if is_git_directory(): return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    else: return 'Not run locally.'


def get_git_diff_master() -> str:
    if is_git_directory(): return subprocess.check_output(['git', 'diff', 'origin/master', 'HEAD']).decode('ascii')
    else: return 'Not run locally.'

def get_git_diff() -> str:
    if is_git_directory():
        # Get the list of files not ignored
        files = subprocess.check_output(['git', 'ls-files']).decode('ascii').strip().split('\n')
        if files:
            # Pass these files to git diff
            return subprocess.check_output(['git', 'diff', 'HEAD'] + files).decode('ascii')
        else:
            return 'No changes to show.'
    else:
        return 'Not run locally'
