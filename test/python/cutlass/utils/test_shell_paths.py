# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only regression checks for paths at subprocess and file-copy boundaries.

Load only the affected AST nodes so these checks need no CUDA imports, native
compilation, generated kernels, or profiler process. Shell APIs are replaced by
stubs that fail before launching anything, including when testing the old code.
Run with: python3 test/python/cutlass/utils/test_shell_paths.py
"""

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
PROFILER_PATH = Path("python/cutlass_cppgen/utils/profiler.py")
GENERATOR_PATH = Path("examples/44_multi_gemm_ir_and_codegen/ir_gen/gen_all_code.py")
WRAPPER_PATH = GENERATOR_PATH.with_name("generate.sh")


def shell_forbidden(*args, **kwargs):
    raise AssertionError("A shell must not be invoked for a path or profiler argument")


def source_tree(relative_path):
    return ast.parse((REPO_ROOT / relative_path).read_text(), filename=str(relative_path))


def profiler_method(namespace):
    cls = next(node for node in source_tree(PROFILER_PATH).body
               if isinstance(node, ast.ClassDef) and node.name == "CUDAEventProfiler")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "run_cutlass_profiler")
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(PROFILER_PATH), "exec"), namespace)
    return namespace["run_cutlass_profiler"]


class ShellPathTests(unittest.TestCase):
    def run_profiler(self, install_path, kernel_name="gemm", iterations=500, returncode=0,
                     cutlass3x=False):
        fake_run = mock.Mock(return_value=SimpleNamespace(
            stdout="diagnostic from stderr\nRuntime: 1.25\nBytes: 64\nFLOPs: 128\n",
            returncode=returncode,
        ))
        process_api = SimpleNamespace(
            run=fake_run, getoutput=shell_forbidden,
            PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
        )
        class Dimension(int):
            def __call__(self):
                return int(self)

        def coords(m, n, k):
            return SimpleNamespace(m=Dimension(m), n=Dimension(n), k=Dimension(k))

        namespace = {"CUTLASS_PATH": install_path, "subprocess": process_api, "re": re,
                     "GemmCoord": coords}
        method = profiler_method(namespace)
        problem = coords(2, 4, 8)
        profiler = SimpleNamespace(
            operation=SimpleNamespace(procedural_name=lambda: kernel_name,
                                      layout_name_3x=lambda: "nt"),
            arguments=SimpleNamespace(problem_size=problem, batch_count=3),
            warmup_iterations=7, iterations=iterations,
            bytes=lambda *args: 64, flops=lambda *args: 128,
        )
        self.assertEqual(method(profiler), 1.25)
        expected_kernel = kernel_name.replace("nt", "nn") if cutlass3x else kernel_name
        m, n = (4, 2) if cutlass3x else (2, 4)
        fake_run.assert_called_once_with(
            [install_path + "/build/tools/profiler/cutlass_profiler",
             "--kernels=" + expected_kernel, "--verification-providers=device",
             "--providers=cutlass", "--m=" + str(m), "--n=" + str(n),
             "--k=8", "--batch_count=3",
             "--alpha=1.0", "--beta=1.0", "--warmup-iterations=7",
             "--profiling-iterations=" + str(iterations)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )

    def test_profiler_normal_path_and_metrics(self):
        self.run_profiler("/opt/cutlass")

    def test_profiler_path_is_one_literal_argument(self):
        for path in ["/tmp/cutlass checkout", "/tmp/cutlass;literal",
                     "/tmp/$(literal)`value`", "/tmp/quotes'\" and [glob]*"]:
            with self.subTest(path=path):
                self.run_profiler(path)

    def test_profiler_fields_do_not_become_shell_source(self):
        self.run_profiler("/opt/cutlass", kernel_name="gemm;$(literal)",
                          iterations="1;`literal`")

    def test_profiler_retains_parseable_output_on_nonzero_exit(self):
        self.run_profiler("/opt/cutlass", returncode=1)

    def test_profiler_preserves_cutlass3x_layout_adjustment(self):
        self.run_profiler("/opt/cutlass", kernel_name="cutlass3x_nt", cutlass3x=True)

    def copy_headers(self, output_dir):
        # Evaluate only the two header-copy expressions, never generator imports
        # or its top-level code generation. Old os.system calls fail in the stub.
        calls = []
        for node in source_tree(GENERATOR_PATH).body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            constants = [item.value for item in ast.walk(node)
                         if isinstance(item, ast.Constant) and isinstance(item.value, str)]
            if any("../leaky_bias.h" in value or "../utils.h" in value
                   for value in constants):
                calls.append(node)
        self.assertEqual(len(calls), 2)
        module = ast.Module(body=calls, type_ignores=[])
        exec(compile(module, str(GENERATOR_PATH), "exec"), {
            "output_dir": str(output_dir), "shutil": shutil,
            "os": SimpleNamespace(system=shell_forbidden),
        })

    def test_header_copies_preserve_literal_paths_contents_and_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ir_gen = root / "ir_gen"
            ir_gen.mkdir()
            originals = {"leaky_bias.h": b"leaky header\n", "utils.h": b"utilities\n"}
            for name, content in originals.items():
                (root / name).write_bytes(content)
            previous_cwd = Path.cwd()
            try:
                os.chdir(ir_gen)
                for name in ["generated", "path with spaces", "semi;colon",
                             "$(literal)`value`", "quotes'\" and [glob]*"]:
                    with self.subTest(name=name):
                        output = root / name
                        sample = output / "sample"
                        sample.mkdir(parents=True)
                        for header in originals:
                            (sample / header).write_bytes(b"stale content")
                            (sample / header).chmod(0o600)
                        self.copy_headers(output)
                        self.assertEqual({p.name: p.read_bytes() for p in sample.iterdir()},
                                         originals)
                        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600
                                            for p in sample.iterdir()))
            finally:
                os.chdir(previous_cwd)

    def test_wrapper_passes_exact_paths_to_python(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "argv.json"
            stub = root / "python3"
            stub.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['CUTLASS_TEST_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
            )
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ.get("PATH", ""),
                       CUTLASS_TEST_ARGV=str(capture))
            for paths in [("config.json", "generated", "/opt/cutlass"),
                          ("config file;$(literal).json", "output `literal` [glob]*",
                           "cutlass path 'quoted'\";literal")]:
                with self.subTest(paths=paths):
                    subprocess.run(["bash", str(REPO_ROOT / WRAPPER_PATH), *paths],
                                   env=env, cwd=root, check=True, capture_output=True, text=True)
                    self.assertEqual(json.loads(capture.read_text()), [
                        "gen_all_code.py", "--config-file", paths[0],
                        "--gen-name", "FusedMultiGemmForward", "--output-dir", paths[1],
                        "--cutlass-dir", paths[2],
                    ])

    def test_header_copy_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp)
                with self.assertRaises(FileNotFoundError):
                    self.copy_headers(Path(temp) / "missing")
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
