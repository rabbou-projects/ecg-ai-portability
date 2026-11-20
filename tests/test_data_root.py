"""The data folder is data/ of the main checkout, from any worktree, unless ECS_DATA moves it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ecs import config


def test_the_default_is_the_data_folder_of_a_checkout_of_this_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ECS_DATA", raising=False)
    root = config.repo_root()
    assert config.data_root() == root / "data"
    assert (root / "pyproject.toml").is_file()
    assert (root / ".git").exists()


def test_a_linked_worktree_resolves_to_the_main_checkout(tmp_path: Path) -> None:
    main, linked = tmp_path / "main", tmp_path / "main/.claude/worktrees/w"
    (main / ".git/worktrees/w").mkdir(parents=True)
    linked.mkdir(parents=True)
    (linked / ".git").write_text(f"gitdir: {main / '.git/worktrees/w'}\n")
    assert config.repo_root(linked) == main
    assert config.repo_root(main) == main


def test_a_relative_gitdir_is_read_from_the_worktree(tmp_path: Path) -> None:
    """git worktree.useRelativePaths writes the gitdir relative to the worktree."""
    main = tmp_path / "main"
    linked = main / ".claude/worktrees/w"
    (main / ".git/worktrees/w").mkdir(parents=True)
    linked.mkdir(parents=True)
    (linked / ".git").write_text("gitdir: ../../../.git/worktrees/w\n")
    assert config.repo_root(linked) == main


def test_a_git_file_that_is_not_a_worktree_keeps_its_own_folder(tmp_path: Path) -> None:
    """A submodule's .git file points into .git/modules, not .git/worktrees."""
    (tmp_path / ".git").write_text("gitdir: ../.git/modules/sub\n")
    assert config.repo_root(tmp_path) == tmp_path


def test_ecs_data_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ECS_DATA", str(tmp_path / "elsewhere"))
    assert config.data_root() == tmp_path / "elsewhere"
    monkeypatch.setenv("ECS_DATA", "~/somewhere")
    assert config.data_root() == Path.home() / "somewhere"


def test_the_corpora_kept_in_this_repository_default_under_the_data_folder(tmp_path: Path) -> None:
    """The defaults are read at import, so each one is read in a fresh interpreter."""
    code = "from ecs import config; print(config.SPH_DIR, config.ACS_DIR, sep='\\n')"
    env = {"ECS_DATA": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == [str(tmp_path / "sph"), str(tmp_path / "acs")]
