//===--- ConvertMetalToLLVM.cpp--------------------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "Conversion/MetalPasses.h"
#include "Conversion/MetalToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::metal {

#define GEN_PASS_DEF_CONVERTMETALTOLLVM
#include "Conversion/MetalPasses.h.inc"

namespace {
struct ConvertMetalToLLVM
    : public impl::ConvertMetalToLLVMBase<ConvertMetalToLLVM> {

  using impl::ConvertMetalToLLVMBase<
      ConvertMetalToLLVM>::ConvertMetalToLLVMBase;

  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter typeConverter(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::metal::populateMetalToLLVMConversionPatterns(patterns);

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyFullConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

class MetalToLLVMDialectInterface : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<MetalDialect, LLVM::LLVMDialect>();
  }
  void
  populateConvertToLLVMConversionPatterns(ConversionTarget &target,
                                          LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) const {
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::metal::populateMetalToLLVMConversionPatterns(patterns);
  }
};
} // end namespace

void registerMetalDialectTranslation(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::metal::MetalDialect *dialect) {
        dialect->addInterfaces<MetalToLLVMDialectInterface>();
      });
}

} // namespace mlir::metal
