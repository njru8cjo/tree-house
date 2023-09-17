# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own tree dialect.
add_mlir_aggregate(TreeCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  TreeCAPI
)

add_llvm_executable(tree-capi-test
  tree-capi-test.c
)
llvm_update_compile_flags(tree-capi-test)
target_link_libraries(tree-capi-test
  PRIVATE TreeCAPITestLib)
