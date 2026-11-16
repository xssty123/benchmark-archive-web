#!/usr/bin/env python3
"""Collect one environment TXT in the actual training host/container (stdlib only)."""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


PTA_CODE = r'''
from importlib.metadata import version, PackageNotFoundError
try:
    print('torch_npu ' + version('torch_npu'))
except PackageNotFoundError:
    print('torch_npu: not installed')
'''


def redact(text):
    # Avoid persisting embedded URL credentials/query tokens from package provenance.
    text = re.sub(r'((?:https?|git\+https?)://)[^/\s@]+@', r'\1[REDACTED]@', text)
    text = re.sub(r'((?:https?|git\+https?)://[^\s"<>?]+)\?[^\s"<>]+',
                  r'\1?[REDACTED_QUERY]', text)
    return text


def command(argv, timeout=90, cwd=None):
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors='replace', timeout=timeout, check=False, cwd=cwd,
                           env={**os.environ, 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
                                'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0'})
        return f'$ {shlex.join(argv)}\nexit={p.returncode}\n{p.stdout.rstrip()}\n'
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'$ {shlex.join(argv)}\nunavailable: {e}\n'


def latest_commit(repo):
    return command(['git', '--no-pager', 'log', '-1', '--no-color', '--no-decorate',
                    '--no-show-signature', '--date=default',
                    '--format=commit %H%nAuthor: %an <%ae>%nDate:   %ad'],
                   cwd=str(repo.resolve()), timeout=15)


def find_repositories(root, warnings):
    """Inspect only the root and its direct children, without project-specific names."""
    repositories = set()
    if (root / '.git').exists():
        repositories.add(root)
    try:
        for child in root.iterdir():
            if child.is_dir() and not child.is_symlink() and (child / '.git').exists():
                repositories.add(child.resolve())
    except OSError as e:
        warnings.append(str(e))
    return sorted(repositories)


def select_repositories(root, explicit, warnings):
    # Explicit paths replace automatic discovery, rather than extending its scope.
    if explicit:
        repositories = set()
        for path in explicit:
            path = path.expanduser()
            repo = (path if path.is_absolute() else root / path).resolve()
            if repo.is_dir() and (repo / '.git').exists():
                repositories.add(repo)
            else:
                warnings.append(f'Skipped {repo}: not a repository root (missing .git); '
                                'parent repositories are not searched.')
        return sorted(repositories)
    return find_repositories(root, warnings)


CANN_ENV_KEYS = ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH')


def normalize_cann_root(path):
    """Normalize installation, architecture and OPP paths without searching parents."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f'Not an existing directory: {path}')
    if path.name == 'opp':
        path = path.parent
    if re.fullmatch(r'(?:aarch64|x86_64)-linux', path.name):
        path = path.parent
    return path


def select_cann_root(explicit):
    notes = []
    if explicit:
        entries = [('--cann-root', path) for path in explicit]
        notes.append('Explicit --cann-root overrides environment selection.')
    else:
        entries = [(key, Path(os.environ[key].strip())) for key in CANN_ENV_KEYS
                   if os.environ.get(key, '').strip()]
    if not entries:
        return None, ['CANN not detected: no CANN environment path is set. '
                      'Activate the training environment or specify --cann-root. '
                      'Default installation directories are not scanned.']
    roots = set()
    invalid = False
    for source, path in entries:
        try:
            root = normalize_cann_root(path)
            notes.append(f'{source}: {path} -> {root}')
            roots.add(root)
        except (OSError, RuntimeError, ValueError) as error:
            notes.append(f'Invalid CANN path ({source}): {error}')
            invalid = True
    if len(roots) > 1:
        notes.append('CANN path conflict: paths refer to different installations. '
                     'No CANN version queried; fix the environment or specify one --cann-root.')
        return None, notes
    if invalid:
        notes.append('No CANN version queried because a specified path is invalid; no fallback used.')
        return None, notes
    return next(iter(roots)), notes


def within_installation(path, root):
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def cann_candidates(root, extra_files, notes):
    """Read only the selected installation; never discover other installations."""
    files, queries = {}, {}
    query = root / 'query_pkg_version.sh'
    if query.is_file():
        if within_installation(query, root):
            queries[query.resolve()] = query
        else:
            notes.append(f'Skipped query outside selected CANN installation: {query}')
    for pattern in ('version.info', 'version.cfg', '*install.info',
                    '*-linux/*install.info', '*-linux/ascend_*/*install.info',
                    '*-linux/ascend_*/version.info', '*-linux/ascend_*/version.cfg',
                    'opp/version.info', 'opp/version.cfg', '*-linux/opp/version.info',
                    '*-linux/opp/version.cfg'):
        for file in root.glob(pattern):
            if file.is_file():
                if within_installation(file, root):
                    files.setdefault(file.resolve(), file)
                else:
                    notes.append(f'Skipped version file outside selected CANN installation: {file}')
    for file in extra_files:
        file = file.expanduser()
        if not within_installation(file, root):
            notes.append(f'Skipped --cann-info outside selected CANN installation: {file}')
        elif not file.is_file():
            notes.append(f'Missing --cann-info file: {file}')
        else:
            files.setdefault(file.resolve(), file)
    return list(files.values()), list(queries.values())


def collect_cann(explicit, extra_files):
    parts = ['\n===== CANN version files =====']
    root, notes = select_cann_root(explicit)
    if root is None:
        parts.extend(notes)
        if extra_files:
            parts.append('--cann-info skipped: select a CANN installation first.')
        return parts
    candidates, queries = cann_candidates(root, extra_files, notes)
    parts.extend(notes)
    parts.append(f'Selected CANN installation: {root}')
    for query in queries:
        parts.append(f'Official CANN query script: {query} -> {query.resolve()}')
        parts.append(command(['bash', str(query.resolve())], cwd=str(query.resolve().parent)))
    for file in candidates:
        try:
            parts += [f'File: {file} -> {file.resolve()}', file.read_text(errors='replace')]
        except OSError as error:
            parts.append(f'Unable to read {file}: {error}')
    if not candidates and not queries:
        parts.append('No readable CANN version files or query script found in the selected '
                     'installation; check its path or supply --cann-info within that installation.')
    return parts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('environment.txt'))
    p.add_argument('--python', default=sys.executable, help='Python used by this training run')
    p.add_argument('--mode', choices=('pip', 'uv'), default='pip')
    p.add_argument('--root', type=Path, default=Path.cwd(), help='Inspect this directory and direct children only (no recursion)')
    p.add_argument('--repo', type=Path, action='append', default=[], help='Only inspect these repositories; repeatable, relative to --root')
    p.add_argument('--cann-info', type=Path, action='append', default=[], help='Additional version file inside the selected CANN installation')
    p.add_argument('--cann-root', type=Path, action='append', default=[], help='Select one CANN installation instead of environment paths; no automatic fallback')
    a = p.parse_args()
    if a.output.exists():
        p.error('output exists; choose another path (no overwrite)')
    a.root = a.root.expanduser().resolve()
    if not a.root.is_dir():
        p.error('--root must be an existing directory')
    if a.mode == 'uv' and not shutil.which('uv'):
        p.error('uv not found; run in the training environment with uv available')
    package_cmd = [a.python, '-m', 'pip'] if a.mode == 'pip' else ['uv', '--offline', '--no-cache', 'pip']
    python_args = [] if a.mode == 'pip' else ['--python', a.python]
    parts = collect_cann(a.cann_root, a.cann_info)
    parts.extend(['\n===== PTA version =====', command([a.python, '-c', PTA_CODE])])
    parts.append('\n===== Source repositories (root/direct children or explicit paths) =====')
    warnings = []
    repositories = select_repositories(a.root, a.repo, warnings)
    parts.append('Explicit --repo paths only; automatic discovery disabled.' if a.repo else
                 'Inspect root and direct children only; no recursion, no directory symlinks, '
                 'no parent-repository lookup. Directory names are not hard-coded. '
                 'Use --repo to restrict the selection or specify deeper/external repositories.')
    parts.extend(f'Scan warning: {warning}' for warning in warnings)
    if not repositories:
        parts.append('No Git repositories found; no source commit collected.')
    for repo in repositories:
        parts.append(f'\nRepository: {repo.resolve()}')
        parts.append(latest_commit(repo))
    parts.extend([f'\n===== {a.mode} list =====',
             command([*package_cmd, 'list', *python_args, '--format=columns'])])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        f.write(redact('\n'.join(parts)) + '\n')
    print(f'Environment snapshot written: {a.output}')


if __name__ == '__main__':
    main()
