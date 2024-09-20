//===--- MetalPasses.h - Metal passes ---------------------------*- C++ -*-===//
//
// This source file is part of the metal-dialect open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef METAL_METALPASSES_H
#define METAL_METALPASSES_H

#include "IR/MetalDialect.h"
#include "IR/MetalOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace metal {
#define GEN_PASS_DECL
#include "Conversion/MetalPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/MetalPasses.h.inc"
} // namespace metal
} // namespace mlir

#endif // METAL_METALPASSES_H
