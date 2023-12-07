import subprocess

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_git_diff_master() -> str:
    ps = subprocess.Popen(['git', 'diff', 'origin/master', 'HEAD'], stdout=subprocess.PIPE)
    return subprocess.check_output(('base64', '-w', '0'), stdin=ps.stdout).decode("ascii")
