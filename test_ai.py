#!/usr/bin/env python3
"""
Tests for ai.py - Interactive AI Inference Launcher

Run with: pytest test_ai.py -v
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Import the module under test
import ai


class TestCheckPort:
    """Tests for check_port function."""

    def test_port_open(self):
        """Should return True when lsof finds a process."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert ai.check_port(8000) is True
            mock_run.assert_called_once()

    def test_port_closed(self):
        """Should return False when lsof finds nothing."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert ai.check_port(8000) is False

    def test_port_check_timeout(self):
        """Should return False on timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="lsof", timeout=2)
            assert ai.check_port(8000) is False

    def test_port_check_exception(self):
        """Should return False on any exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Command not found")
            assert ai.check_port(8000) is False


class TestFetchUrlJson:
    """Tests for fetch_url_json function."""

    def test_successful_fetch(self):
        """Should parse JSON response correctly."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"models": [{"name": "test"}]}'
            )
            result = ai.fetch_url_json("http://localhost:8000/v1/models")
            assert result == {"models": [{"name": "test"}]}

    def test_failed_fetch(self):
        """Should return None on curl failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = ai.fetch_url_json("http://localhost:8000/v1/models")
            assert result is None

    def test_invalid_json(self):
        """Should return None on invalid JSON."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not json")
            result = ai.fetch_url_json("http://localhost:8000/v1/models")
            assert result is None

    def test_fetch_exception(self):
        """Should return None on exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Network error")
            result = ai.fetch_url_json("http://localhost:8000/v1/models")
            assert result is None


class TestGetOllamaModels:
    """Tests for get_ollama_models function."""

    def test_models_from_api(self):
        """Should fetch models from Ollama API."""
        api_response = {
            "models": [
                {"name": "qwen3:30b", "size": 16106127360},
                {"name": "llama3:8b", "size": 4661224448},
            ]
        }
        with patch.object(ai, "fetch_url_json", return_value=api_response):
            models = ai.get_ollama_models()
            assert len(models) == 2
            assert models[0]["id"] == "qwen3:30b"
            assert models[0]["name"] == "qwen3:30b"
            assert "15.0GB" in models[0]["size"]
            assert models[1]["id"] == "llama3:8b"

    def test_models_from_cli_fallback(self):
        """Should fallback to CLI when API fails."""
        cli_output = """NAME              ID              SIZE      MODIFIED
qwen3:30b         abc123          15 GB     2 days ago
llama3:8b         def456          4.3 GB    1 week ago"""

        with patch.object(ai, "fetch_url_json", return_value=None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=cli_output
                )
                models = ai.get_ollama_models()
                assert len(models) == 2
                assert models[0]["id"] == "qwen3:30b"
                assert models[1]["id"] == "llama3:8b"

    def test_no_models_available(self):
        """Should return empty list when no models."""
        with patch.object(ai, "fetch_url_json", return_value=None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="")
                models = ai.get_ollama_models()
                assert models == []


class TestGetVllmModels:
    """Tests for get_vllm_models function."""

    def test_models_from_api(self):
        """Should fetch models from vLLM /v1/models endpoint."""
        api_response = {
            "data": [
                {"id": "Qwen/Qwen3-8B"},
                {"id": "cyankiwi/GLM-4.7-Flash-AWQ-4bit"},
            ]
        }
        with patch.object(ai, "fetch_url_json", return_value=api_response):
            models = ai.get_vllm_models()
            assert len(models) == 2
            assert models[0]["id"] == "Qwen/Qwen3-8B"
            assert models[1]["id"] == "cyankiwi/GLM-4.7-Flash-AWQ-4bit"

    def test_no_models(self):
        """Should return empty list when API fails."""
        with patch.object(ai, "fetch_url_json", return_value=None):
            models = ai.get_vllm_models()
            assert models == []

    def test_empty_data(self):
        """Should return empty list when data is empty."""
        with patch.object(ai, "fetch_url_json", return_value={"data": []}):
            models = ai.get_vllm_models()
            assert models == []


class TestGetLlamacppModels:
    """Tests for get_llamacpp_models function."""

    def test_models_from_api_single_mode(self):
        """Should fetch models from llama.cpp /v1/models endpoint (single mode - no status)."""
        # Single mode: no status field means model is loaded
        api_response = {
            "data": [
                {"id": "unsloth/GLM-4.7-Flash", "object": "model"},
            ]
        }
        with patch.object(ai, "fetch_url_json", return_value=api_response):
            with patch.object(Path, "exists", return_value=False):
                models = ai.get_llamacpp_models()
                assert len(models) == 1
                assert models[0]["id"] == "unsloth/GLM-4.7-Flash"
                assert models[0]["loaded"] is True
                assert "[loaded]" in models[0]["name"]

    def test_models_from_api_router_mode_loaded(self):
        """Should detect loaded models in router mode via status field."""
        # Router mode: status.value = "loaded"
        api_response = {
            "data": [
                {"id": "GLM-4.7-Flash", "status": {"value": "loaded"}},
            ]
        }
        with patch.object(ai, "fetch_url_json", return_value=api_response):
            with patch.object(Path, "exists", return_value=False):
                models = ai.get_llamacpp_models()
                assert len(models) == 1
                assert models[0]["id"] == "GLM-4.7-Flash"
                assert models[0]["loaded"] is True
                assert "[loaded]" in models[0]["name"]

    def test_models_from_api_router_mode_unloaded(self):
        """Should detect unloaded models in router mode via status field."""
        # Router mode: status.value = "unloaded"
        api_response = {
            "data": [
                {"id": "GLM-4.7-Flash", "status": {"value": "unloaded"}},
            ]
        }
        with patch.object(ai, "fetch_url_json", return_value=api_response):
            with patch.object(Path, "exists", return_value=False):
                models = ai.get_llamacpp_models()
                assert len(models) == 1
                assert models[0]["id"] == "GLM-4.7-Flash"
                assert models[0]["loaded"] is False
                assert "(available)" in models[0]["name"]

    def test_models_from_local_directory(self, tmp_path):
        """Should scan models/ directory for GGUF files."""
        # Create a mock GGUF file
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        gguf_dir = models_dir / "GLM-4.7-Flash"
        gguf_dir.mkdir()
        gguf_file = gguf_dir / "model-Q4_K.gguf"
        gguf_file.write_bytes(b"x" * (5 * 1024**3))  # 5GB fake file

        # No API response means no loaded models
        def mock_fetch(url):
            if "/models" in url:
                return None
            return None

        with patch.object(ai, "fetch_url_json", side_effect=mock_fetch):
            with patch.object(ai, "SCRIPT_DIR", tmp_path):
                models = ai.get_llamacpp_models()
                assert len(models) == 1
                assert "GLM-4.7-Flash" in models[0]["name"]
                assert "(available)" in models[0]["name"]
                assert models[0]["loaded"] is False

    def test_router_mode_loaded_and_available(self, tmp_path):
        """In router mode, should show both loaded and available models."""
        # Create a mock GGUF file that's NOT in the API response
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        gguf_dir = models_dir / "Other-Model"
        gguf_dir.mkdir()
        gguf_file = gguf_dir / "other.gguf"
        gguf_file.write_bytes(b"x" * (2 * 1024**3))  # 2GB fake file

        # API returns a loaded model different from the local file (router mode)
        def mock_fetch(url):
            if "/v1/models" in url:
                return {"data": [{"id": "unsloth/GLM-4.7-Flash", "status": {"value": "loaded"}}]}
            return None

        with patch.object(ai, "fetch_url_json", side_effect=mock_fetch):
            with patch.object(ai, "SCRIPT_DIR", tmp_path):
                models = ai.get_llamacpp_models()
                # Should have both the loaded model and the available one
                assert len(models) == 2
                loaded = [m for m in models if m["loaded"]]
                available = [m for m in models if not m["loaded"]]
                assert len(loaded) == 1
                assert len(available) == 1
                assert loaded[0]["id"] == "unsloth/GLM-4.7-Flash"
                assert "Other-Model" in available[0]["name"]

    def test_skip_already_loaded_local_model(self, tmp_path):
        """Should not duplicate a model that's both loaded and in local dir."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        gguf_dir = models_dir / "GLM-4.7-Flash-GGUF"
        gguf_dir.mkdir()
        gguf_file = gguf_dir / "GLM-4.7-Flash-UD-Q4_K_XL.gguf"
        gguf_file.write_bytes(b"x" * (1024**3))

        # API returns the same model as loaded (using alias that matches filename) - router mode
        def mock_fetch(url):
            if "/v1/models" in url:
                return {"data": [{"id": "unsloth/GLM-4.7-Flash", "status": {"value": "loaded"}}]}
            return None

        with patch.object(ai, "fetch_url_json", side_effect=mock_fetch):
            with patch.object(ai, "SCRIPT_DIR", tmp_path):
                models = ai.get_llamacpp_models()
                # Should only show the loaded version, not duplicate
                # The alias "unsloth/GLM-4.7-Flash" should match the local file
                # "GLM-4.7-Flash-GGUF/GLM-4.7-Flash-UD-Q4_K_XL.gguf"
                assert len(models) == 1
                assert models[0]["loaded"] is True


class TestRouterModeFunctions:
    """Tests for router mode load/unload functions."""

    def test_load_model_success(self):
        """Should return True when model loads successfully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="{}")
            result = ai.load_llamacpp_model("models/test.gguf")
            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "curl" in call_args
            assert "models/load" in str(call_args)

    def test_load_model_failure(self):
        """Should return False when model load fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = ai.load_llamacpp_model("models/nonexistent.gguf")
            assert result is False

    def test_load_model_timeout(self):
        """Should return False on timeout (model loading can be slow)."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="curl", timeout=120)
            result = ai.load_llamacpp_model("models/large.gguf")
            assert result is False

    def test_unload_model_success(self):
        """Should return True when model unloads successfully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="{}")
            result = ai.unload_llamacpp_model("GLM-4.7-Flash")
            assert result is True
            call_args = mock_run.call_args[0][0]
            assert "models/unload" in str(call_args)

    def test_unload_model_failure(self):
        """Should return False when unload fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = ai.unload_llamacpp_model("not-loaded")
            assert result is False


class TestRouterModeDetection:
    """Tests for router mode detection."""

    def test_router_mode_detected(self):
        """Should detect router mode when status field is present."""
        api_response = {
            "data": [{"id": "model1", "status": {"value": "loaded"}}]
        }
        with patch.object(ai, "fetch_url_json", return_value=api_response):
            assert ai.is_llamacpp_router_mode() is True

    def test_single_mode_detected(self):
        """Should detect single mode when no status field."""
        api_response = {
            "data": [{"id": "model1", "object": "model"}]
        }
        with patch.object(ai, "fetch_url_json", return_value=api_response):
            assert ai.is_llamacpp_router_mode() is False

    def test_server_not_running(self):
        """Should return False when server not running."""
        with patch.object(ai, "fetch_url_json", return_value=None):
            assert ai.is_llamacpp_router_mode() is False


class TestLlamacppModelSelection:
    """Tests for llama.cpp model selection with actions."""

    def test_select_loaded_model(self):
        """Should return 'use' action for loaded model selection."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
            {"id": "model2", "name": "Model2 (available)", "size": "3GB", "loaded": False},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ("use", "model1")
            model_id, action = ai.select_llamacpp_model_with_actions(models)
            assert model_id == "model1"
            assert action == "use"

    def test_select_available_model_to_load(self):
        """Should return 'load' action for available model selection."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
            {"id": "/path/to/model2.gguf", "name": "Model2 (available)", "size": "3GB", "loaded": False},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ("load", "/path/to/model2.gguf")
            model_id, action = ai.select_llamacpp_model_with_actions(models)
            assert model_id == "/path/to/model2.gguf"
            assert action == "load"

    def test_select_manage_option(self):
        """Should return 'manage' action for model management."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ("manage", None)
            model_id, action = ai.select_llamacpp_model_with_actions(models)
            assert model_id is None
            assert action == "manage"

    def test_select_back(self):
        """Should return 'back' action when going back."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = ("back", None)
            model_id, action = ai.select_llamacpp_model_with_actions(models)
            assert model_id is None
            assert action == "back"

    def test_select_cancelled(self):
        """Should handle cancelled selection (None return)."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = None
            model_id, action = ai.select_llamacpp_model_with_actions(models)
            assert model_id is None
            assert action == "back"


class TestManageLlamacppModels:
    """Tests for llama.cpp model management (unloading)."""

    def test_unload_model_success(self):
        """Should successfully unload a model."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "model1"
            with patch.object(ai, "unload_llamacpp_model", return_value=True):
                with patch.object(ai.console, "print"):
                    result = ai.manage_llamacpp_models(models)
                    assert result is True

    def test_unload_model_failure(self):
        """Should handle unload failure gracefully."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "model1"
            with patch.object(ai, "unload_llamacpp_model", return_value=False):
                with patch.object(ai.console, "print"):
                    result = ai.manage_llamacpp_models(models)
                    assert result is True  # Still returns True to continue

    def test_no_loaded_models(self):
        """Should handle case with no loaded models."""
        models = [
            {"id": "model1", "name": "Model1 (available)", "size": "5GB", "loaded": False},
        ]
        with patch.object(ai.console, "print"):
            result = ai.manage_llamacpp_models(models)
            assert result is True

    def test_back_from_management(self):
        """Should return True when user selects back."""
        models = [
            {"id": "model1", "name": "Model1 [loaded]", "size": "5GB", "loaded": True},
        ]
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = None  # Back
            result = ai.manage_llamacpp_models(models)
            assert result is True


class TestLoadLlamacppModelInteractive:
    """Tests for interactive model loading."""

    def test_load_success(self):
        """Should return True on successful load."""
        with patch.object(ai, "load_llamacpp_model", return_value=True):
            with patch.object(ai.console, "print"):
                result = ai.load_llamacpp_model_interactive("/path/to/model.gguf")
                assert result is True

    def test_load_failure(self):
        """Should return False on load failure."""
        with patch.object(ai, "load_llamacpp_model", return_value=False):
            with patch.object(ai.console, "print"):
                result = ai.load_llamacpp_model_interactive("/path/to/model.gguf")
                assert result is False

    def test_extracts_model_name_from_path(self):
        """Should extract friendly name from path."""
        with patch.object(ai, "load_llamacpp_model", return_value=True) as mock_load:
            with patch.object(ai.console, "print"):
                ai.load_llamacpp_model_interactive("/home/user/models/GLM-4.7-Flash/model.gguf")
                # Verify the model was loaded (exact args depend on path processing)
                mock_load.assert_called_once()


class TestGetBackendStatus:
    """Tests for get_backend_status function."""

    def test_all_backends_running(self):
        """Should report all backends as running with models."""
        with patch.object(ai, "check_port", return_value=True):
            with patch.object(ai, "get_ollama_models", return_value=[{"id": "m1", "name": "m1", "size": ""}]):
                with patch.object(ai, "get_vllm_models", return_value=[{"id": "m2", "name": "m2", "size": ""}]):
                    with patch.object(ai, "get_llamacpp_models", return_value=[{"id": "m3", "name": "m3", "size": ""}]):
                        status = ai.get_backend_status()

                        assert status["ollama"]["running"] is True
                        assert len(status["ollama"]["models"]) == 1
                        assert status["vllm"]["running"] is True
                        assert len(status["vllm"]["models"]) == 1
                        assert status["llamacpp"]["running"] is True
                        assert len(status["llamacpp"]["models"]) == 1

    def test_no_backends_running(self):
        """Should report all backends as stopped."""
        with patch.object(ai, "check_port", return_value=False):
            with patch.object(ai, "get_ollama_models", return_value=[]):
                status = ai.get_backend_status()

                assert status["ollama"]["running"] is False
                assert status["vllm"]["running"] is False
                assert status["llamacpp"]["running"] is False

    def test_ollama_autostart(self):
        """Ollama should be marked running if CLI returns models even when port check fails."""
        def mock_check_port(port):
            return False  # Port not listening

        with patch.object(ai, "check_port", side_effect=mock_check_port):
            # Ollama CLI works even without explicit server
            with patch.object(ai, "get_ollama_models", return_value=[{"id": "m1", "name": "m1", "size": ""}]):
                status = ai.get_backend_status()
                # Ollama auto-starts, so if models exist, it's "running"
                assert status["ollama"]["running"] is True
                assert len(status["ollama"]["models"]) == 1


class TestLaunchCommandConstruction:
    """Tests for launch command construction logic."""

    def test_ollama_crush_command(self):
        """Should construct correct command for Ollama + Crush."""
        with patch("os.execvp") as mock_exec:
            ai.launch("ollama", "qwen3:30b", "crush")
            mock_exec.assert_called_once()
            cmd = mock_exec.call_args[0][1]
            assert "run_crush.sh" in cmd[0]
            assert "--backend" in cmd
            assert "ollama" in cmd
            assert "--model" in cmd
            assert "qwen3:30b" in cmd

    def test_ollama_opencode_command(self):
        """Should construct correct command for Ollama + OpenCode."""
        with patch("os.execvp") as mock_exec:
            ai.launch("ollama", "qwen3:30b", "opencode")
            mock_exec.assert_called_once()
            cmd = mock_exec.call_args[0][1]
            assert "run_claude.sh" in cmd[0]
            assert "--model" in cmd
            assert "qwen3:30b" in cmd

    def test_vllm_crush_command(self):
        """Should construct correct command for vLLM + Crush."""
        with patch("os.execvp") as mock_exec:
            ai.launch("vllm", "Qwen/Qwen3-8B", "crush")
            mock_exec.assert_called_once()
            cmd = mock_exec.call_args[0][1]
            assert "run_vllm.sh" in cmd[0]
            assert "crush" in cmd
            assert "--model" in cmd
            assert "Qwen/Qwen3-8B" in cmd

    def test_vllm_opencode_command(self):
        """Should construct correct command for vLLM + OpenCode."""
        with patch("os.execvp") as mock_exec:
            ai.launch("vllm", "Qwen/Qwen3-8B", "opencode")
            mock_exec.assert_called_once()
            cmd = mock_exec.call_args[0][1]
            assert "run_vllm.sh" in cmd[0]
            assert "opencode" in cmd

    def test_llamacpp_opencode_command(self):
        """Should construct correct command for llama.cpp + OpenCode."""
        with patch("os.execvp") as mock_exec:
            ai.launch("llamacpp", "unsloth/GLM-4.7-Flash", "opencode")
            mock_exec.assert_called_once()
            cmd = mock_exec.call_args[0][1]
            assert "run_llamacpp.sh" in cmd[0]
            assert "opencode" in cmd

    def test_llamacpp_crush_command(self):
        """Should construct correct command for llama.cpp + Crush."""
        with patch("os.execvp") as mock_exec:
            ai.launch("llamacpp", "unsloth/GLM-4.7-Flash", "crush")
            mock_exec.assert_called_once()
            cmd = mock_exec.call_args[0][1]
            assert "run_crush.sh" in cmd[0]
            assert "--vllm-host" in cmd
            assert "http://localhost:10000" in cmd


class TestBackendConfiguration:
    """Tests for backend configuration constants."""

    def test_backend_ports(self):
        """Should have correct default ports for each backend."""
        assert ai.BACKENDS["ollama"]["port"] == 11434
        assert ai.BACKENDS["vllm"]["port"] == 8000
        assert ai.BACKENDS["llamacpp"]["port"] == 10000

    def test_backend_check_urls(self):
        """Should have valid check URLs."""
        assert "11434" in ai.BACKENDS["ollama"]["check_url"]
        assert "8000" in ai.BACKENDS["vllm"]["check_url"]
        assert "10000" in ai.BACKENDS["llamacpp"]["check_url"]

    def test_all_backends_have_required_fields(self):
        """Each backend should have name, port, check_url."""
        for key, cfg in ai.BACKENDS.items():
            assert "name" in cfg, f"{key} missing 'name'"
            assert "port" in cfg, f"{key} missing 'port'"
            assert "check_url" in cfg, f"{key} missing 'check_url'"


class TestFrontendConfiguration:
    """Tests for frontend configuration constants."""

    def test_frontends_exist(self):
        """Should have opencode and crush frontends."""
        assert "opencode" in ai.FRONTENDS
        assert "crush" in ai.FRONTENDS

    def test_frontends_have_required_fields(self):
        """Each frontend should have name and desc."""
        for key, cfg in ai.FRONTENDS.items():
            assert "name" in cfg, f"{key} missing 'name'"
            assert "desc" in cfg, f"{key} missing 'desc'"


class TestInteractiveSelectionMocking:
    """Tests for interactive selection functions with mocked questionary."""

    def test_select_backend_returns_choice(self):
        """Should return selected backend key."""
        status = {
            "ollama": {"running": True, "models": [{"id": "m1"}], "port": 11434, "name": "Ollama"},
            "vllm": {"running": False, "models": [], "port": 8000, "name": "vLLM"},
            "llamacpp": {"running": False, "models": [], "port": 10000, "name": "llama.cpp"},
        }

        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "ollama"
            result = ai.select_backend(status)
            assert result == "ollama"

    def test_select_backend_exit(self):
        """Should return None when user selects Exit."""
        status = {
            "ollama": {"running": True, "models": [{"id": "m1"}], "port": 11434, "name": "Ollama"},
            "vllm": {"running": False, "models": [], "port": 8000, "name": "vLLM"},
            "llamacpp": {"running": False, "models": [], "port": 10000, "name": "llama.cpp"},
        }

        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = None
            result = ai.select_backend(status)
            assert result is None

    def test_select_model_returns_choice(self):
        """Should return selected model id."""
        models = [
            {"id": "qwen3:30b", "name": "qwen3:30b", "size": "15GB"},
            {"id": "llama3:8b", "name": "llama3:8b", "size": "4GB"},
        ]

        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "qwen3:30b"
            result = ai.select_model(models)
            assert result == "qwen3:30b"

    def test_select_model_back(self):
        """Should return None when user selects Back."""
        models = [{"id": "m1", "name": "m1", "size": ""}]

        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = None
            result = ai.select_model(models)
            assert result is None

    def test_select_frontend_returns_choice(self):
        """Should return selected frontend key."""
        with patch("questionary.select") as mock_select:
            mock_select.return_value.ask.return_value = "crush"
            result = ai.select_frontend()
            assert result == "crush"


class TestEndToEndFlow:
    """Integration tests for the full selection flow."""

    def test_full_flow_ollama_crush(self):
        """Test complete flow: Ollama backend -> model -> Crush frontend."""
        status = {
            "ollama": {"running": True, "models": [{"id": "qwen3:30b", "name": "qwen3:30b", "size": "15GB"}], "port": 11434, "name": "Ollama"},
            "vllm": {"running": False, "models": [], "port": 8000, "name": "vLLM"},
            "llamacpp": {"running": False, "models": [], "port": 10000, "name": "llama.cpp"},
        }

        with patch.object(ai, "get_backend_status", return_value=status):
            with patch.object(ai, "is_llamacpp_router_mode", return_value=False):
                with patch("questionary.select") as mock_select:
                    # Simulate user selections
                    mock_select.return_value.ask.side_effect = ["ollama", "qwen3:30b", "crush"]

                    with patch("os.execvp") as mock_exec:
                        with patch.object(ai, "print_header"):
                            with patch.object(ai, "print_status_table"):
                                with patch.object(ai.console, "print"):
                                    # Run main but catch the exec
                                    ai.main()

                                    # Verify launch was called correctly
                                    mock_exec.assert_called_once()
                                    cmd = mock_exec.call_args[0][1]
                                    assert "run_crush.sh" in cmd[0]
                                    assert "qwen3:30b" in cmd

    def test_full_flow_llamacpp_router_mode_loaded_model(self):
        """Test flow: llama.cpp router mode -> use loaded model -> OpenCode."""
        status = {
            "ollama": {"running": False, "models": [], "port": 11434, "name": "Ollama"},
            "vllm": {"running": False, "models": [], "port": 8000, "name": "vLLM"},
            "llamacpp": {
                "running": True,
                "models": [
                    {"id": "GLM-4.7-Flash", "name": "GLM-4.7-Flash [loaded]", "size": "", "loaded": True},
                ],
                "port": 10000,
                "name": "llama.cpp"
            },
        }

        with patch.object(ai, "get_backend_status", return_value=status):
            with patch.object(ai, "is_llamacpp_router_mode", return_value=True):
                with patch.object(ai, "get_llamacpp_models", return_value=status["llamacpp"]["models"]):
                    with patch("questionary.select") as mock_select:
                        # User selects: llamacpp -> use loaded model -> opencode
                        mock_select.return_value.ask.side_effect = [
                            "llamacpp",
                            ("use", "GLM-4.7-Flash"),
                            "opencode"
                        ]

                        with patch("os.execvp") as mock_exec:
                            with patch.object(ai, "print_header"):
                                with patch.object(ai, "print_status_table"):
                                    with patch.object(ai.console, "print"):
                                        ai.main()

                                        mock_exec.assert_called_once()
                                        cmd = mock_exec.call_args[0][1]
                                        assert "run_llamacpp.sh" in cmd[0]
                                        assert "opencode" in cmd

    def test_full_flow_llamacpp_router_mode_load_model(self):
        """Test flow: llama.cpp router mode -> load available model -> OpenCode."""
        initial_models = [
            {"id": "/path/model.gguf", "name": "Model (available)", "size": "5GB", "loaded": False},
        ]
        loaded_models = [
            {"id": "Model", "name": "Model [loaded]", "size": "", "loaded": True},
        ]

        status = {
            "ollama": {"running": False, "models": [], "port": 11434, "name": "Ollama"},
            "vllm": {"running": False, "models": [], "port": 8000, "name": "vLLM"},
            "llamacpp": {"running": True, "models": initial_models, "port": 10000, "name": "llama.cpp"},
        }

        # Track calls to get_llamacpp_models to return different values
        call_count = [0]
        def mock_get_models():
            call_count[0] += 1
            return loaded_models if call_count[0] > 1 else initial_models

        with patch.object(ai, "get_backend_status", return_value=status):
            with patch.object(ai, "is_llamacpp_router_mode", return_value=True):
                with patch.object(ai, "get_llamacpp_models", side_effect=mock_get_models):
                    with patch.object(ai, "load_llamacpp_model_interactive", return_value=True):
                        with patch("questionary.select") as mock_select:
                            # User selects: llamacpp -> load model -> opencode
                            mock_select.return_value.ask.side_effect = [
                                "llamacpp",
                                ("load", "/path/model.gguf"),
                                "opencode"
                            ]

                            with patch("os.execvp") as mock_exec:
                                with patch.object(ai, "print_header"):
                                    with patch.object(ai, "print_status_table"):
                                        with patch.object(ai.console, "print"):
                                            ai.main()

                                            mock_exec.assert_called_once()
                                            cmd = mock_exec.call_args[0][1]
                                            assert "run_llamacpp.sh" in cmd[0]

    def test_flow_with_back_navigation(self):
        """Test flow where user goes back from model selection."""
        status = {
            "ollama": {"running": True, "models": [{"id": "m1", "name": "m1", "size": ""}], "port": 11434, "name": "Ollama"},
            "vllm": {"running": True, "models": [{"id": "m2", "name": "m2", "size": ""}], "port": 8000, "name": "vLLM"},
            "llamacpp": {"running": False, "models": [], "port": 10000, "name": "llama.cpp"},
        }

        with patch.object(ai, "get_backend_status", return_value=status):
            with patch("questionary.select") as mock_select:
                # User selects ollama, then goes back (None), then selects vllm
                mock_select.return_value.ask.side_effect = [
                    "ollama",  # First backend selection
                    None,      # Back from model selection
                    "vllm",    # Second backend selection
                    "m2",      # Model selection
                    "opencode" # Frontend selection
                ]

                with patch("os.execvp") as mock_exec:
                    with patch.object(ai, "print_header"):
                        with patch.object(ai, "print_status_table"):
                            with patch.object(ai.console, "print"):
                                ai.main()

                                # Should have launched with vllm
                                cmd = mock_exec.call_args[0][1]
                                assert "run_vllm.sh" in cmd[0]

    def test_no_backends_available(self):
        """Test behavior when no backends are running."""
        status = {
            "ollama": {"running": False, "models": [], "port": 11434, "name": "Ollama"},
            "vllm": {"running": False, "models": [], "port": 8000, "name": "vLLM"},
            "llamacpp": {"running": False, "models": [], "port": 10000, "name": "llama.cpp"},
        }

        with patch.object(ai, "get_backend_status", return_value=status):
            with patch.object(ai, "print_header"):
                with patch.object(ai, "print_status_table"):
                    with patch.object(ai.console, "print") as mock_print:
                        ai.main()

                        # Should print message about no backends
                        calls = [str(c) for c in mock_print.call_args_list]
                        assert any("No backends available" in str(c) for c in calls)


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    def test_ctrl_c_during_status_check(self):
        """KeyboardInterrupt during status check should propagate."""
        with patch.object(ai, "get_backend_status", side_effect=KeyboardInterrupt):
            with patch.object(ai, "print_header"):
                with patch.object(ai.console, "print"):
                    # KeyboardInterrupt propagates from main()
                    # The if __name__ == "__main__" block catches it
                    with pytest.raises(KeyboardInterrupt):
                        ai.main()

    def test_ctrl_c_during_selection(self):
        """KeyboardInterrupt during selection should propagate."""
        status = {
            "ollama": {"running": True, "models": [{"id": "m1", "name": "m1", "size": ""}], "port": 11434, "name": "Ollama"},
            "vllm": {"running": False, "models": [], "port": 8000, "name": "vLLM"},
            "llamacpp": {"running": False, "models": [], "port": 10000, "name": "llama.cpp"},
        }

        with patch.object(ai, "get_backend_status", return_value=status):
            with patch("questionary.select") as mock_select:
                mock_select.return_value.ask.side_effect = KeyboardInterrupt
                with patch.object(ai, "print_header"):
                    with patch.object(ai, "print_status_table"):
                        with patch.object(ai.console, "print"):
                            with pytest.raises(KeyboardInterrupt):
                                ai.main()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
