include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_tree`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_tree.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TreePythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TreePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_tree"
  TD_FILE dialects/TreeOps.td
  SOURCES
    dialects/tree.py
  DIALECT_NAME tree)

declare_mlir_python_extension(TreePythonSources.Extension
  MODULE_NAME _treeDialects
  ADD_TO_PARENT TreePythonSources
  SOURCES
    TreeExtension.cpp
  EMBED_CAPI_LINK_LIBS
    TreeCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(TreePythonCAPI
  INSTALL_COMPONENT TreePythonModules
  INSTALL_DESTINATION python_packages/tree/mlir_tree/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/tree/mlir_tree/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    TreePythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(TreePythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/tree/mlir_tree"
  INSTALL_PREFIX "python_packages/tree/mlir_tree"
  DECLARED_SOURCES
    TreePythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    TreePythonCAPI
  )
