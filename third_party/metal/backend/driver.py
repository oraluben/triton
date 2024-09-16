from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver


class MetalDriver(GPUDriver):

    @staticmethod
    def is_active():
        return True

    def get_current_target(self):
        return GPUTarget("mps", None, 32)
