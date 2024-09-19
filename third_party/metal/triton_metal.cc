#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Pass/PassManager.h"

#include "IR/MetalDialect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;


namespace {
  using namespace mlir;
  using namespace mlir::LLVM;
  using namespace mlir::metal;

  void registerMetalDialectTranslation(DialectRegistry &registry) {
    registry.insert<MetalDialect>();
    // registry.addExtension(+[](MLIRContext *ctx, MetalDialect *dialect) {
    //   dialect->addInterfaces<MetalDialectLLVMIRTranslationInterface>();
    // });
  };

  void registerMetalDialectTranslation(MLIRContext &context) {
    DialectRegistry registry;
    registerMetalDialectTranslation(registry);
    context.appendDialectRegistry(registry);
  }

}

void init_triton_metal(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    registerMetalDialectTranslation(context);
    context.loadAllAvailableDialects();
  });
}
