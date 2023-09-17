add_llvm_library(TreePlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        tree-plugin.cpp

        DEPENDS
        MLIRTree
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRTree
        )
target_include_directories(
        TreePlugin
        PRIVATE
        "${TREE_BINARY_DIR}/include"
)
