from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, metal
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


@dataclass(frozen=True)
class MetalOptions:
    debug: bool = False
    num_warps: int = 4
    num_ctas: int = 1
    cluster_dims: tuple = (1, 1, 1)

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return True

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def hash(self) -> str:
        return str(id(self))

    def parse_options(self, options):
        return MetalOptions()

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["bin1"] = lambda src, metadata: self.make_bin1(src, metadata, options)
        stages["bin2"] = lambda src, metadata: self.make_bin2(src, metadata, options)

    def load_dialects(self, ctx):
        metal.load_dialects(ctx)

    def get_codegen_implementation(self):
        return {}

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.hip import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        return mod

    @staticmethod
    def make_llir(src, metadata, options):
        mod = src
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        return str(llvm_mod)

    @staticmethod
    def make_bin1(src, metadata, options):
        import pdb; pdb.set_trace()
        metadata["name"] = 'todo'

    @staticmethod
    def make_bin2(src, metadata, options):
        import pdb; pdb.set_trace()
