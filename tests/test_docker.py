"""Opt-in real Docker lifecycle test; no LLM/API keys. Set RUN_DOCKER_TESTS=1."""
import os
import subprocess
import unittest

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH
from sandbox import download, open_sandbox, upload
from test_lab import fixture
import research


@unittest.skipUnless(os.getenv("RUN_DOCKER_TESTS") == "1", "set RUN_DOCKER_TESTS=1 for real Docker integration")
class DockerTests(unittest.TestCase):
    def test_sandbox_validation_download_and_cleanup_on_error(self):
        container_id = None
        with self.assertRaisesRegex(RuntimeError, "simulated run failure"):
            with open_sandbox("docker") as backend:
                container_id = backend.id
                report, sources = fixture()
                upload(backend, {REPORT_PATH: report, SOURCES_PATH: sources,
                                 VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                                 FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes()})
                self.assertEqual(backend.execute(f"python3 {FINALIZER_PATH}").exit_code, 0)
                result = backend.execute(f"python3 {VALIDATOR_PATH}")
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue(result.output.startswith("OK:"), result.output)
                self.assertEqual(download(backend, [REPORT_PATH])[REPORT_PATH], report)
                network = subprocess.run(["docker", "inspect", "--format", "{{.HostConfig.NetworkMode}}", container_id], capture_output=True, text=True)
                self.assertEqual(network.stdout.strip(), "none")
                raise RuntimeError("simulated run failure")
        result = subprocess.run(["docker", "inspect", container_id], capture_output=True)
        self.assertNotEqual(result.returncode, 0, "sandbox should be removed even after failure")
