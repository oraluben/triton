# Proton - A Profiler for Triton

## Introduction

Proton is a lightweight profiler for Triton, designed to be used for code written in Python and to invoke underlying GPU kernels. Proton provides insightful information about the program context, metadata, and hardware performance metrics of the GPU kernels invoked.

## Installation

The following command installs the latest version of Proton.

```bash
git clone https://github.com/triton-lang/triton
cd triton/python
pip install .
```

To **not build** Proton, you can set the `TRITON_BUILD_PROTON` environment variable to `OFF`:

```bash
TRITON_BUILD_PROTON=OFF pip install .
```

## Usage

### Basic usage

More examples can be found in the [tutorials](tutorials) directory.

Proton can be used to profile *functions* and *regions* in Python code.

- The following examples demonstrate how to use Proton to profile a simple Python function.

```python
import triton.profiler as proton

# name: The path to the profile data
# context: The method used to annotate the context of each GPU kernel. Currently, "shadow" and "python" are supported.
session_id = proton.profile(func, name="profile_name", context="python")(args)
```

- The following examples demonstrate how to use Proton to profile a region in Python code.

```python
session_id = proton.start(name="profile_name", context="python")
...
# Skip a region
proton.deactivate(session_id)
...
# Restart profiling
proton.activate(session_id)
...
# Write out the profile data and finalize the profiler
proton.finalize()
```

### Scope

Unlike the *python* context that provide users with files, functions, and lines where the GPU kernels are invoked, the *shadow* context provides users with the annotated regions in the code. The following example demonstrates how to use the *shadow* context.

```python
import triton.profiler as proton


session_id = proton.start(name="profile_name", context="shadow")

with proton.scope("test0"):
    with proton.scope("test1"):
        foo[1,](x, y)
with proton.scope("test2"):
    foo[1,](x, y)

...
proton.finalize()
```

The *scope* utility also accepts flexible metrics, provided with a dictionary that maps from a string (metric name) to a value (int or float).
Proton will aggregate the metrics for each scope and write them to the profile data.
It is useful for users to understand the performance of the model at a high level.

```python
with proton.scope("test0", {"bytes": 1000}):
    with proton.scope("test1", {"bytes": 2000}):
        foo[1,](x, y)
with proton.scope("test2", {"bytes": 3000}):
    foo[1,](x, y)
```

### Hook

```python
import triton.profiler as proton
from typing import NamedTuple

# hook: When hook="triton", it enables proton to invoke launch_metadata function before launching the GPU kernel
proton.start("profile_name", hook="triton")

def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict
):
    return {"name": "<kernel_name>", "flops8": 1.0}

@triton.jit(launch_metadata=metadata_fn)
def foo(x, y):
    tl.store(y, tl.load(x))
```

The `metadata_fn` function is called before launching the GPU kernel to provide metadata for the GPU kernel, which returns a dictionary that maps from a string (metadata name) to a value (int or float).

Currently, **only the triton hook is supported**. In the dictionary returned by the `metadata_fn` function, we can supply the following keys:

```python
name: str  # The name of the kernel
flops8: float  # The number of 8-bit floating-point operations
flops16: float  # The number of 16-bit floating-point operations
flops32: float  # The number of 32-bit floating-point operations
flops64: float  # The number of 64-bit floating-point operations
bytes: int  # The number of bytes expected to be transferred
```

### Command line

Proton can be used as a command-line tool to profile Python scripts and Pytest tests.
The following examples demonstrate how to use Proton command-line.

```bash
proton [options] script.py [script_args] [script_options]
proton [options] pytest [pytest_args] [script_options]
python -m triton.profiler.proton [options] script.py [script_args] [script_options]
proton --instrument=[instrumentation pass] script.py
```

When profiling in the command line mode, the `proton.start` and `proton.finalize` functions are automatically called before and after the script execution. Any `proton.start` and `proton.finalize` functions in the script are ignored. Also, in the command line mode, only a single *session* is supported. Therefore, `proton.deactivate(session_id=1)` is invalid, while `proton.deactivate(session_id=0)` is valid.

### Visualizing the profile data

By default, proton profiles are in the *json* format and can be read by *Hatchet*. The following command visualizes the profile data on terminal.

```bash
pip install llnl-hatchet
proton-viewer -m time/s <profile.hatchet>
```
NOTE: `pip install hatchet` does not work because the API is slightly different.

### Visualizing sorted profile data
In addition visualizing the profile data on terminal through Hatchet. A sorted list of the kernels by the first metric can be done using the --print-sorted flag with proton-viewer

```bash
proton-viewer -m time/ns,time/% <profile.hatchet> --print-sorted
```
prints the sorted kernels by the time/ns since it is the first listed.

More options can be found by running the following command.

```bash
proton-viewer -h
```

### Advanced features
In addition to profiling, Proton also incorporates MLIR/LLVM based compiler instrumentation passes to get Triton level analysis
and optimization information. This feature is under active development and the list of available passes is expected to grow.

#### Available passes
print-mem-spaces: this pass prints the load and store address spaces (e.g. global, flat, shared) chosen by the compiler and attributes back to Triton source information.

Example usage with the Proton matmul tutorial:
```bash
$ proton --instrument=print-mem-spaces matmul.py
0     matmul_kernel     matmul.py:180:20     SHARED     STORE
1     matmul_kernel     matmul.py:181:20     SHARED     STORE
2     matmul_kernel     matmul.py:180:20     SHARED     LOAD
3     matmul_kernel     matmul.py:181:20     SHARED     LOAD
```
Notes: The instrument functionality is currently only available from the command line. Additionally the instrument and profile command line arguments can not be use simulantously.

### Instruction sampling (experimental)

Proton supports instruction sampling on NVIDIA GPUs.
Please note that this is an experimental feature and may not work on all GPUs.
You may experience ~20x end-to-end overhead when using instruction sampling, although the overhead for each individual GPU kernel is negligible.
The overhead is mostly caused by data transfer and processing on the CPU.
Additionally, the proton-viewer options `-i <regex> -d <depth> -t <threshold>` can be helpful for filtering out GPU kernels that are not of interest.
The following example demonstrates how to use instruction sampling:

```python
import triton.profiler as proton

proton.start(name="profile_name", context="shadow", backend="cupti_pcsampling")
```

## Proton *vs* nsys

- Runtime overhead (up to 1.5x)

Proton has a lower profiling overhead than nsys. Even for workload with a large number of small GPU kernels, proton triggers less than ~1.5x overhead.

For GPU-bound workload, both proton and nsys has similar overhead, with little impact on the workload.

The lower overhead of proton is due to its less profiling metrics and callbacks compared to nsys.

- Profile size (significantly smaller than nsys)

nsys traces and records every GPU kernel, while proton aggregates the metrics of GPU kernels under the same calling context.

As a result, proton's profile size can be up to thousands of times smaller than nsys's profile size, depending on the running time.

- Portability (support different GPUs)

Proton is designed to be portable and can be used on AMD GPUs. nsys only supports NVIDIA GPUs.

- Insights (more insightful than nsys on triton kernels)

Proton can register hooks to analyze the metadata of triton kernels, while nsys cannot. **Note** that the hooks do add additional overhead to proton.

## Proton *vs* ncu

Similar to the comparison between Proton and Nsight Systems (Nsys), Proton has a lower profiling overhead than Nsight Compute (NCU). We also plan to support instruction sampling on AMD GPUs.
However, Nsight Compute supports the collection of more detailed metrics than Proton, such as memory access patterns, memory transactions, and other instruction-level metrics.
In contrast, Proton only supports instruction sampling and is designed to be lightweight and portable.

## Known issues

- CUDA graph

`hooks` cannot be used to accurately accumulate the number of FLOPs in CUDA graph mode profiling because kernels are captured and launched separately; metrics are not accumulated when kernels are launched in graph mode. This issue can be circumvented by using `scope` to supply FLOPs.

If profiling is initiated after CUDA graph capturing, there may be minor memory leak issues.
This is because the number of kernels in a graph instance (i.e., `cuGraphExec`) is unknown, preventing the deletion of mappings between the kernel ID and the graph ID.

- Instruction sampling

If you encounter permission related problems when using instruction sampling, you can lookup this [page](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) for help.

The overhead of instruction sampling on NVIDIA GPUs is about 20x using Proton because we haven't enabled continuous sampling yet.
Continuous sampling can allow for more runtime optimizations, but it makes it more challenging to attribute performance data back to the GPU kernels because: (1) it enables profiling of concurrent kernels, (2) it doesn't allow profiling of time and instruction samples simultaneously, and (3) it works best if we have a separate thread dedicated to attributing instruction samples to the GPU kernels

- Visible devices on AMD GPUs

Environment variables such as `HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES` are not supported on AMD GPUs. Once it's set, we cannot find a valid mapping between the device ID returned by RocTracer and the physical device ID. Instead, `ROCR_VISIBLE_DEVICES` is recommended to be used.
