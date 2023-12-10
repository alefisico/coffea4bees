import subprocess

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_diff_master() -> str:
    return subprocess.check_output(['git', 'diff', 'origin/master', 'HEAD']).decode('ascii')

def get_git_diff() -> str:
    return subprocess.check_output(['git', 'diff', 'HEAD']).decode('ascii')
