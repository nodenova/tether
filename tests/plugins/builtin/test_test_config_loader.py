"""Tests for project test config loader."""

import pytest

from tether.plugins.builtin.test_config_loader import (
    ProjectTestConfig,
    load_project_test_config,
)
from tether.plugins.builtin.test_runner import (
    TestConfig,
    _merge_project_config,
)


class TestLoadProjectTestConfig:
    @pytest.fixture
    def tether_dir(self, tmp_path):
        d = tmp_path / ".tether"
        d.mkdir()
        return d

    def test_load_valid_yaml(self, tmp_path, tether_dir):
        config_file = tether_dir / "test.yaml"
        config_file.write_text(
            "url: http://localhost:3000\n"
            "server: npm run dev\n"
            "framework: next.js\n"
            "credentials:\n"
            "  api_token: abc123\n"
            "preconditions:\n"
            "  - Backend must be running\n"
            "focus_areas:\n"
            "  - SKU replacement\n"
        )
        result = load_project_test_config(str(tmp_path))
        assert result is not None
        assert result.url == "http://localhost:3000"
        assert result.server == "npm run dev"
        assert result.framework == "next.js"
        assert result.credentials == {"api_token": "abc123"}
        assert result.preconditions == ["Backend must be running"]
        assert result.focus_areas == ["SKU replacement"]

    def test_load_yml_extension(self, tmp_path, tether_dir):
        config_file = tether_dir / "test.yml"
        config_file.write_text("url: http://localhost:8080\n")
        result = load_project_test_config(str(tmp_path))
        assert result is not None
        assert result.url == "http://localhost:8080"

    def test_load_missing_file(self, tmp_path):
        result = load_project_test_config(str(tmp_path))
        assert result is None

    def test_load_invalid_yaml(self, tmp_path, tether_dir):
        config_file = tether_dir / "test.yaml"
        config_file.write_text("url: [invalid: yaml: {{")
        result = load_project_test_config(str(tmp_path))
        assert result is None

    def test_load_empty_file(self, tmp_path, tether_dir):
        config_file = tether_dir / "test.yaml"
        config_file.write_text("")
        result = load_project_test_config(str(tmp_path))
        assert result is not None
        assert result.url is None

    def test_load_partial_config(self, tmp_path, tether_dir):
        config_file = tether_dir / "test.yaml"
        config_file.write_text("framework: django\n")
        result = load_project_test_config(str(tmp_path))
        assert result is not None
        assert result.framework == "django"
        assert result.url is None
        assert result.credentials == {}

    def test_yaml_takes_precedence_over_yml(self, tmp_path, tether_dir):
        (tether_dir / "test.yaml").write_text("url: http://yaml\n")
        (tether_dir / "test.yml").write_text("url: http://yml\n")
        result = load_project_test_config(str(tmp_path))
        assert result is not None
        assert result.url == "http://yaml"


class TestProjectTestConfigModel:
    def test_defaults(self):
        c = ProjectTestConfig()
        assert c.url is None
        assert c.server is None
        assert c.framework is None
        assert c.directory is None
        assert c.credentials == {}
        assert c.preconditions == []
        assert c.focus_areas == []
        assert c.environment == {}

    def test_frozen(self):
        from pydantic import ValidationError

        c = ProjectTestConfig(url="http://localhost")
        with pytest.raises(ValidationError, match="frozen"):
            c.url = "http://other"  # type: ignore[misc]


class TestMergeProjectConfig:
    def test_cli_overrides_project(self):
        cli = TestConfig(app_url="http://cli")
        project = ProjectTestConfig(url="http://project")
        merged = _merge_project_config(cli, project)
        assert merged.app_url == "http://cli"

    def test_project_fills_gaps(self):
        cli = TestConfig()
        project = ProjectTestConfig(
            url="http://project",
            server="npm run dev",
            framework="next.js",
            directory="tests/e2e",
        )
        merged = _merge_project_config(cli, project)
        assert merged.app_url == "http://project"
        assert merged.dev_server_command == "npm run dev"
        assert merged.framework == "next.js"
        assert merged.test_directory == "tests/e2e"

    def test_partial_merge(self):
        cli = TestConfig(framework="react")
        project = ProjectTestConfig(url="http://project", framework="next.js")
        merged = _merge_project_config(cli, project)
        assert merged.app_url == "http://project"
        assert merged.framework == "react"  # CLI wins

    def test_no_updates_returns_same(self):
        cli = TestConfig(
            app_url="http://cli",
            dev_server_command="npm start",
            framework="react",
            test_directory="tests/",
        )
        project = ProjectTestConfig()
        merged = _merge_project_config(cli, project)
        assert merged is cli  # No copy needed
