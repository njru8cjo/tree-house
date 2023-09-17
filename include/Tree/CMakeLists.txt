add_mlir_dialect(TreeOps tree)
add_mlir_doc(TreeDialect TreeDialect Tree/ -gen-dialect-doc)
add_mlir_doc(TreeOps TreeOps Tree/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TreePasses.td)
mlir_tablegen(TreePasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTreePassesIncGen)
