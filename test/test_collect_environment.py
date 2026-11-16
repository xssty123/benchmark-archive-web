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

    def make_cann(self, path, build='B211'):
        path.mkdir(parents=True)
        (path / 'aarch64-linux').mkdir()
        (path / 'opp').mkdir()
        (path / 'aarch64-linux/ascend_toolkit_install.info').write_text('version=9.2.0\ninnerversion=' + build)
        (path / 'opp/version.info').write_text('Version=9.2.0\n')
        (path / 'query_pkg_version.sh').write_text('# synthetic fixture, never execute')
        return path.resolve()

    def test_cann_environment_alias_arch_and_opp_select_one_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_cann(Path(folder) / 'cann-9.2.0')
            alias = Path(folder) / 'latest'
            alias.symlink_to(root, target_is_directory=True)
            env = {'ASCEND_HOME_PATH': str(alias), 'ASCEND_TOOLKIT_HOME': str(root / 'aarch64-linux'),
                   'ASCEND_OPP_PATH': str(alias / 'opp')}
            with patch.dict(os.environ, env, clear=True):
                selected, notes = collector.select_cann_root([])
            self.assertEqual(selected, root)
            files, queries = collector.cann_candidates(selected, [alias / 'opp/version.info'], notes)
            self.assertEqual(len(files), 2)
            self.assertEqual(queries, [root / 'query_pkg_version.sh'])
            self.assertEqual(sum(p.resolve() == root / 'opp/version.info' for p in files), 1)

    def test_cann_conflicting_paths_query_neither_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.make_cann(Path(folder) / 'first', 'B211')
            second = self.make_cann(Path(folder) / 'second', 'B236')
            with patch.dict(os.environ, {'ASCEND_HOME_PATH': str(first), 'ASCEND_OPP_PATH': str(second / 'opp')}, clear=True), \
                 patch.object(collector, 'command') as command:
                report = '\n'.join(collector.collect_cann([], []))
            command.assert_not_called()
            self.assertIn('CANN path conflict', report)
            self.assertIn(str(first), report)
            self.assertIn(str(second), report)
            self.assertNotIn('innerversion=', report)

    def test_cann_unset_never_scans_default_installations(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(collector, 'cann_candidates') as candidates, \
             patch.object(collector, 'command') as command:
            report = '\n'.join(collector.collect_cann([], []))
        candidates.assert_not_called()
        command.assert_not_called()
        self.assertIn('CANN not detected', report)
        self.assertNotIn('/usr/local/Ascend', report)

    def test_cann_override_ignores_environment_and_queries_only_explicit_root(self):
        with tempfile.TemporaryDirectory() as folder:
            selected = self.make_cann(Path(folder) / 'selected', 'B211')
            other = self.make_cann(Path(folder) / 'other', 'B236')
            env = {'ASCEND_HOME_PATH': str(other), 'ASCEND_OPP_PATH': str(Path(folder) / 'missing')}
            with patch.dict(os.environ, env, clear=True), patch.object(collector, 'command', return_value='fixture query') as command:
                report = '\n'.join(collector.collect_cann([selected], []))
            command.assert_called_once_with(['bash', str(selected / 'query_pkg_version.sh')], cwd=str(selected))
            self.assertIn('innerversion=B211', report)
            self.assertNotIn('B236', report)
            self.assertNotIn(str(other), report)
            self.assertIn('overrides environment', report)

    def test_cann_invalid_environment_does_not_fall_back_to_valid_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_cann(Path(folder) / 'valid')
            with patch.dict(os.environ, {'ASCEND_HOME_PATH': str(root), 'ASCEND_TOOLKIT_HOME': str(root / 'missing')}, clear=True), \
                 patch.object(collector, 'command') as command:
                report = '\n'.join(collector.collect_cann([], []))
            command.assert_not_called()
            self.assertIn('Invalid CANN path', report)
            self.assertIn('no fallback', report)
            with patch.dict(os.environ, {'ASCEND_HOME_PATH': str(root)}, clear=True):
                selected, _ = collector.select_cann_root([root / 'missing'])
            self.assertIsNone(selected)

    def test_cann_repeated_explicit_roots_must_resolve_to_one_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.make_cann(Path(folder) / 'first')
            second = self.make_cann(Path(folder) / 'second')
            selected, _ = collector.select_cann_root([first, first / 'aarch64-linux'])
            self.assertEqual(selected, first)
            selected, notes = collector.select_cann_root([first, second])
            self.assertIsNone(selected)
            self.assertIn('conflict', '\n'.join(notes))

    def test_cann_opp_only_old_layout_and_explicit_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'aarch64-linux/opp').mkdir(parents=True)
            info = root / 'aarch64-linux/ascend_toolkit_install.info'
            info.write_text('version=old-fixture')
            custom = root / 'custom.txt'
            custom.write_text('build=custom-fixture')
            with patch.dict(os.environ, {'ASCEND_OPP_PATH': str(root / 'aarch64-linux/opp')}, clear=True), \
                 patch.object(collector, 'command') as command:
                report = '\n'.join(collector.collect_cann([], [custom]))
            command.assert_not_called()
            self.assertIn('version=old-fixture', report)
            self.assertIn('build=custom-fixture', report)

    def test_cann_does_not_follow_external_metadata_or_query_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_cann(Path(folder) / 'selected')
            other = self.make_cann(Path(folder) / 'other', 'B236')
            query = root / 'query_pkg_version.sh'
            query.unlink()
            query.symlink_to(other / 'query_pkg_version.sh')
            link = root / 'external_install.info'
            link.symlink_to(other / 'aarch64-linux/ascend_toolkit_install.info')
            with patch.object(collector, 'command') as command:
                report = '\n'.join(collector.collect_cann([root], [other / 'opp/version.info', root / 'missing.txt']))
            command.assert_not_called()
            self.assertNotIn('B236', report)
            self.assertIn('Skipped query outside', report)
            self.assertIn('Skipped version file outside', report)
            self.assertIn('Skipped --cann-info outside', report)
            self.assertIn('Missing --cann-info', report)

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
                     patch.dict(os.environ, {}, clear=True), \
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
