"""Synthetic environment fixtures only; no training environment is sampled."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'public/scripts/collect_environment.py'
spec = importlib.util.spec_from_file_location('collector', SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def make_repo(self, root):
        repo = root / 'repo'
        repo.mkdir()
        def git(*args):
            return subprocess.run(['git', '-c', 'commit.gpgsign=false', '-c', 'user.name=Fixture',
                                   '-c', 'user.email=fixture@example.invalid', '-C', str(repo), *args],
                                  check=True, capture_output=True, text=True)
        git('init')
        git('commit', '--allow-empty', '-m', 'Do not include this commit message')
        return repo, git

    def test_commit_ignores_custom_git_output(self):
        with tempfile.TemporaryDirectory() as folder:
            repo, git = self.make_repo(Path(folder))
            git('config', 'format.pretty', 'oneline')
            git('config', 'log.date', 'relative')
            git('config', 'color.ui', 'always')
            result = collector.latest_commit(repo)
            self.assertIn('commit ' + git('rev-parse', 'HEAD').stdout.strip(), result)
            self.assertIn('Author: Fixture <fixture@example.invalid>', result)
            self.assertRegex(result, r'Date:   [A-Z][a-z]{2} [A-Z][a-z]{2} ')
            self.assertNotIn('Do not include', result)
            self.assertNotIn('\x1b', result)
            self.assertEqual(len(result.splitlines()), 5)

    def test_explicit_repo_rejects_subdirectory_without_parent_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            repo, git = self.make_repo(root)
            child = repo / 'ordinary'
            child.mkdir()
            warnings = []
            selected = collector.select_repositories(root, [child, root / 'missing'], warnings)
            self.assertEqual(selected, [])
            self.assertEqual(len(warnings), 2)
            self.assertIn(str(child), warnings[0])
            worktree = root / 'worktree'
            git('worktree', 'add', '--detach', str(worktree))
            self.assertEqual(collector.select_repositories(root, [Path('worktree')], []), [worktree])

    def test_direct_children_worktrees_without_nested_scan(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            repo = root / 'source with spaces'
            repo.mkdir()
            def git(*args):
                return subprocess.run(['git', '-c', 'commit.gpgsign=false', '-c', 'user.name=Fixture',
                                       '-c', 'user.email=fixture@example.invalid', '-C', str(repo), *args],
                                      check=True, capture_output=True, text=True)
            git('init')
            (repo / 'source.txt').write_text('fixture')
            git('add', 'source.txt')
            git('commit', '-m', 'fixture')
            commit = collector.latest_commit(repo)
            self.assertTrue(commit.startswith('$ git --no-pager log -1 '))
            self.assertIn('\nexit=0\ncommit ', commit)
            self.assertIn('Author: Fixture', commit)
            self.assertIn('Date:', commit)
            self.assertNotIn('    fixture', commit)
            self.assertEqual(len(commit.splitlines()), 5)
            worktree = root / 'worktree'
            git('worktree', 'add', '--detach', str(worktree))
            (repo / 'nested/.git').mkdir(parents=True)
            (root / '.venv/ignored/.git').mkdir(parents=True)
            (root / 'exp/deep/source/.git').mkdir(parents=True)
            (root / 'loop').symlink_to(root, target_is_directory=True)
            found = collector.find_repositories(root, [])
            self.assertEqual(set(found), {repo, worktree})
            self.assertEqual(collector.find_repositories(repo, []), [repo, repo / 'nested'])
            self.assertEqual(collector.find_repositories(repo / '.git', []), [])
            self.assertEqual(collector.select_repositories(root, [Path('worktree')], []), [worktree])
            self.assertEqual(collector.select_repositories(root, [repo / 'nested'], []), [repo / 'nested'])
            (repo / 'source.txt').write_text('modified')
            self.assertIn(' M source.txt', collector.command(['git', '-C', str(repo), 'status', '--porcelain']))

    def test_cann_new_old_and_symlink_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'aarch64-linux').mkdir()
            version = root / 'aarch64-linux/ascend_toolkit_install.info'
            version.write_text('version=fixture')
            query = root / 'query_pkg_version.sh'
            query.write_text('# fixture only')
            alias = root / 'alias'
            alias.symlink_to(root, target_is_directory=True)
            with patch.dict(os.environ, {'ASCEND_HOME_PATH': str(alias)}):
                files, queries = collector.cann_candidates([root])
            self.assertEqual(sum(p.resolve() == version.resolve() for p in files), 1)
            self.assertEqual(sum(p.resolve() == query.resolve() for p in queries), 1)

    def test_both_modes_target_same_python_without_package_commits(self):
        for mode in ('pip', 'uv'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                output = root / 'environment.txt'
                calls = []
                def command(argv, **kwargs):
                    calls.append(argv)
                    if collector.PTA_CODE in argv:
                        return 'torch_npu fixture'
                    if 'list' in argv:
                        return 'Package Version\nuvloop 0.23.0\nvcs-versioning 2.2.2'
                    if 'show' in argv:
                        self.fail('Package details must not be collected')
                    return 'synthetic output'
                args = ['collector', '--mode', mode, '--python', '/training env/bin/python',
                        '--root', str(root), '--output', str(output)]
                with patch('sys.argv', args), patch.object(collector, 'command', command), \
                     patch.object(collector, 'find_repositories', return_value=[]), \
                     patch.object(collector, 'cann_candidates', return_value=([], [])), \
                     patch.object(collector.shutil, 'which', return_value='/bin/uv'), contextlib.redirect_stdout(io.StringIO()):
                    collector.main()
                    with self.assertRaises(SystemExit):
                        with contextlib.redirect_stderr(io.StringIO()):
                            collector.main()  # Existing output must not be overwritten.
                report = output.read_text()
                self.assertNotIn('abc123', report)
                self.assertNotIn('Installed package provenance', report)
                self.assertNotIn('Collector host:', report)
                self.assertNotIn('Platform:', report)
                self.assertFalse(any(cmd[0] == 'npu-smi' for cmd in calls))
                self.assertNotIn('===== Python =====', report)
                self.assertIn('uvloop 0.23.0', report)
                self.assertIn('vcs-versioning 2.2.2', report)
                self.assertIn('torch_npu fixture', report)
                self.assertNotIn('Name:', report)
                self.assertNotIn('Summary:', report)
                headings = ['===== CANN version files', '===== PTA version',
                            '===== Source repositories', f'===== {mode} list']
                positions = [report.index(title) for title in headings]
                self.assertEqual(positions, sorted(positions))
                for action in ('list',):
                    call = next(cmd for cmd in calls if action in cmd)
                    if mode == 'pip':
                        self.assertEqual(call[:3], ['/training env/bin/python', '-m', 'pip'])
                    else:
                        self.assertEqual(call[call.index('--python') + 1], '/training env/bin/python')
                self.assertTrue(any(collector.PTA_CODE in cmd and cmd[0] == '/training env/bin/python' for cmd in calls))

    def test_credentials_redacted_without_hiding_file_source(self):
        value = collector.redact('https://user:secret@example.org/repo?token=secret file:///source')
        self.assertNotIn('secret', value)
        self.assertIn('file:///source', value)


if __name__ == '__main__':
    unittest.main()
