add_mlir_dialect_library(MLIRTree
        TreeTypes.cpp
        TreeDialect.cpp
        TreeOps.cpp
        TreePasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tree

        DEPENDS
        MLIRTreeOpsIncGen
        MLIRTreePassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
