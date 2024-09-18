from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class MetalDriver(DriverBase):

    @staticmethod
    def is_active():
        return True

    def get_current_target(self):
        return GPUTarget("mps", None, 32)

    @staticmethod
    def get_current_device():
        return 0

    @staticmethod
    def get_current_stream(device):
        return None
