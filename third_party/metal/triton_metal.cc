
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Pass/PassManager.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;


namespace {
  using namespace mlir;
  using namespace mlir::LLVM;
  using mlir::LLVM::detail::createIntrinsicCall;

  class MetalDialect : public ::mlir::Dialect
   {
    MetalDialect(::mlir::MLIRContext *context);

    void initialize();
    friend class ::mlir::MLIRContext;
  public:
    static constexpr ::llvm::StringLiteral getDialectNamespace() {
      return ::llvm::StringLiteral("metal");
    }
  };

  class MetalDialectLLVMIRTranslationInterface
      : public LLVMTranslationDialectInterface {
    public:
      using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;
  };

  void registerMetalDialectTranslation(DialectRegistry &registry) {
    registry.insert<MetalDialect>();
    registry.addExtension(+[](MLIRContext *ctx, MetalDialect *dialect) {
      dialect->addInterfaces<MetalDialectLLVMIRTranslationInterface>();
    });
  };

  void registerMetalDialectTranslation(MLIRContext &context) {
    DialectRegistry registry;
    registerMetalDialectTranslation(registry);
    context.appendDialectRegistry(registry);
  }

}

MLIR_DECLARE_EXPLICIT_TYPE_ID(::MetalDialect)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::MetalDialect)

MetalDialect::MetalDialect(::mlir::MLIRContext *context) : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MetalDialect>()) {}

void init_triton_metal(py::module &&m) {
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    registerMetalDialectTranslation(context);
    context.loadAllAvailableDialects();
  });
}