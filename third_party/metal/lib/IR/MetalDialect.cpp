//===--- MetalDialect.cpp - Metal dialect ---------------------------------===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "IR/MetalDialect.h"
#include "IR/MetalOps.h"
#include "IR/MetalTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::metal;

#include "IR/MetalOpsDialect.cpp.inc"

void MetalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/MetalOps.cpp.inc"
      >();
  registerTypes();
}

mlir::Operation *MetalDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<mlir::metal::ConstantOp>(loc, cast<TypedAttr>(value));
}
