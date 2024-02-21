get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRTree
        )

add_llvm_executable(frontend main.cpp)
llvm_update_compile_flags(frontend)
target_link_libraries(frontend PRIVATE ${LIBS})

mlir_check_all_link_libraries(frontend)
