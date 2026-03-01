"""Tests for workspace model and YAML loader."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tether.core.workspace import Workspace, load_workspaces


class TestWorkspaceModel:
    def test_primary_directory(self):
        ws = Workspace(
            name="test",
            directories=[Path("/a"), Path("/b"), Path("/c")],
        )
        assert ws.primary_directory == Path("/a")

    def test_frozen(self):
        ws = Workspace(name="test", directories=[Path("/a")])
        with pytest.raises(ValidationError, match="frozen"):
            ws.name = "other"

    def test_description_default(self):
        ws = Workspace(name="test", directories=[Path("/a")])
        assert ws.description == ""


class TestLoadWorkspaces:
    def test_no_file_returns_empty(self, tmp_path):
        result = load_workspaces(tmp_path, [tmp_path])
        assert result == {}

    def test_valid_yaml(self, tmp_path):
        dir_a = tmp_path / "repo-a"
        dir_b = tmp_path / "repo-b"
        dir_a.mkdir()
        dir_b.mkdir()

        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        ws_file = tether_dir / "workspaces.yaml"
        ws_file.write_text(
            yaml.dump(
                {
                    "workspaces": {
                        "myws": {
                            "description": "My workspace",
                            "directories": [str(dir_a), str(dir_b)],
                        }
                    }
                }
            )
        )

        result = load_workspaces(tmp_path, [dir_a, dir_b])
        assert "myws" in result
        ws = result["myws"]
        assert ws.name == "myws"
        assert ws.description == "My workspace"
        assert ws.directories == [dir_a.resolve(), dir_b.resolve()]
        assert ws.primary_directory == dir_a.resolve()

    def test_yml_extension(self, tmp_path):
        dir_a = tmp_path / "repo"
        dir_a.mkdir()
        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        ws_file = tether_dir / "workspaces.yml"
        ws_file.write_text(
            yaml.dump({"workspaces": {"ws1": {"directories": [str(dir_a)]}}})
        )

        result = load_workspaces(tmp_path, [dir_a])
        assert "ws1" in result

    def test_dir_not_in_approved_is_skipped(self, tmp_path):
        dir_a = tmp_path / "repo-a"
        dir_b = tmp_path / "repo-b"
        dir_a.mkdir()
        dir_b.mkdir()

        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        (tether_dir / "workspaces.yaml").write_text(
            yaml.dump(
                {
                    "workspaces": {
                        "myws": {
                            "directories": [str(dir_a), str(dir_b)],
                        }
                    }
                }
            )
        )

        # Only dir_a is approved
        result = load_workspaces(tmp_path, [dir_a])
        ws = result["myws"]
        assert len(ws.directories) == 1
        assert ws.directories[0] == dir_a.resolve()

    def test_dir_not_exists_is_skipped(self, tmp_path):
        dir_a = tmp_path / "repo-a"
        dir_a.mkdir()
        nonexistent = tmp_path / "ghost"

        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        (tether_dir / "workspaces.yaml").write_text(
            yaml.dump(
                {
                    "workspaces": {
                        "myws": {
                            "directories": [str(dir_a), str(nonexistent)],
                        }
                    }
                }
            )
        )

        result = load_workspaces(tmp_path, [dir_a])
        ws = result["myws"]
        assert len(ws.directories) == 1

    def test_empty_directories_list_skipped(self, tmp_path):
        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        (tether_dir / "workspaces.yaml").write_text(
            yaml.dump({"workspaces": {"empty": {"directories": []}}})
        )

        result = load_workspaces(tmp_path, [tmp_path])
        assert result == {}

    def test_all_dirs_invalid_skips_workspace(self, tmp_path):
        nonexistent = tmp_path / "ghost"
        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        (tether_dir / "workspaces.yaml").write_text(
            yaml.dump({"workspaces": {"bad": {"directories": [str(nonexistent)]}}})
        )

        result = load_workspaces(tmp_path, [tmp_path])
        assert result == {}

    def test_multiple_workspaces(self, tmp_path):
        dir_a = tmp_path / "frontend"
        dir_b = tmp_path / "backend"
        dir_a.mkdir()
        dir_b.mkdir()

        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        (tether_dir / "workspaces.yaml").write_text(
            yaml.dump(
                {
                    "workspaces": {
                        "fe": {"directories": [str(dir_a)]},
                        "be": {"directories": [str(dir_b)]},
                    }
                }
            )
        )

        result = load_workspaces(tmp_path, [dir_a, dir_b])
        assert len(result) == 2
        assert "fe" in result
        assert "be" in result

    def test_invalid_yaml_returns_empty(self, tmp_path):
        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        (tether_dir / "workspaces.yaml").write_text(":::bad yaml{{{")

        result = load_workspaces(tmp_path, [tmp_path])
        assert result == {}

    def test_tilde_expansion(self, tmp_path):
        dir_a = tmp_path / "repo"
        dir_a.mkdir()

        tether_dir = tmp_path / ".tether"
        tether_dir.mkdir()
        (tether_dir / "workspaces.yaml").write_text(
            yaml.dump({"workspaces": {"ws": {"directories": [str(dir_a)]}}})
        )

        result = load_workspaces(tmp_path, [dir_a])
        assert "ws" in result
