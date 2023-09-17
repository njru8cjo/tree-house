add_mlir_public_c_api_library(TreeCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRTree
)
