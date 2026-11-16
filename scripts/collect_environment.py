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


def cann_candidates(extra):
    roots = [Path(os.environ[k]).expanduser() for k in
             ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH') if os.environ.get(k)]
    for base in (Path('/usr/local/Ascend'), Path.home() / 'Ascend'):
        roots.extend([base / 'cann', base / 'cann/latest', base / 'ascend-toolkit/latest',
                      base / 'nnrt/latest', base / 'nnae/latest', base / 'driver',
                      base / 'nnal/atb/latest', base / 'nnal/asdsip/latest'])
    roots.extend(extra)
    files, queries = {}, {}
    for root in roots:
        if not root.is_dir():
            continue
        query = root / 'query_pkg_version.sh'
        if query.is_file():
            queries.setdefault(query.resolve(), query)
        for pattern in ('version.info', 'version.cfg', '*install.info',
                        '*-linux/*install.info', '*-linux/ascend_*/*install.info',
                        '*-linux/ascend_*/version.info', '*-linux/ascend_*/version.cfg'):
            for file in root.glob(pattern):
                if file.is_file():
                    files.setdefault(file.resolve(), file)
    return list(files.values()), list(queries.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('environment.txt'))
    p.add_argument('--python', default=sys.executable, help='Python used by this training run')
    p.add_argument('--mode', choices=('pip', 'uv'), default='pip')
    p.add_argument('--root', type=Path, default=Path.cwd(), help='Inspect this directory and direct children only (no recursion)')
    p.add_argument('--repo', type=Path, action='append', default=[], help='Only inspect these repositories; repeatable, relative to --root')
    p.add_argument('--cann-info', type=Path, action='append', default=[])
    p.add_argument('--cann-root', type=Path, action='append', default=[], help='Additional CANN installation directory')
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
    parts = []
    parts.append('\n===== CANN version files =====')
    candidates, queries = cann_candidates(a.cann_root)
    candidates += a.cann_info
    parts.append('Auto-detected candidates only; confirm which installation was loaded by the run.')
    for query in queries:
        parts.append(f'Official CANN query script: {query} -> {query.resolve()}')
        parts.append(command(['bash', str(query.resolve())], cwd=str(query.resolve().parent)))
    found = False
    for file in dict.fromkeys(candidates):
        if file.is_file():
            found = True
            try:
                parts += [f'File: {file} -> {file.resolve()}', file.read_text(errors='replace')]
            except OSError as e:
                parts.append(f'Unable to read {file}: {e}')
        elif a.cann_info:
            parts.append(f'Missing: {file}')
    if not found:
        parts.append('No readable CANN version files found; see query script output above, '
                     'or supply --cann-info / --cann-root.')
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
