from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, amd
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return True

    def hash(self) -> str:
        """Returns a unique identifier for this backend"""
        raise NotImplementedError

    def parse_options(self, options: dict) -> object:
        """
        Converts an `options` dictionary into an arbitrary object and returns it.
        This function may contain target-specific heuristics and check the legality of the provided options
        """
        raise NotImplementedError

    def add_stages(self, stages: dict, options: object) -> None:
        """
        Populates `stages` dictionary with entries of the form:
        ir_name [str] => Function[(src: str, metadata: dict) -> str|bytes]
        The value of each entry may populate a `metadata` dictionary.
        Stages will be run sequentially (in inseriton order) and can communicate using `metadata`.
        All stages are expected to return a `str` object, except for the last stage which returns
        a `bytes` object for execution by the launcher.
        """
        raise NotImplementedError

    def load_dialects(self, context):
        """
        Load additional MLIR dialects into the provided `context`
        """
        raise NotImplementedError

    def get_module_map(self) -> Dict[str, ModuleType]:
        """
        Return a map of interface modules to their device-specific implementations.
        """
        raise NotImplementedError
