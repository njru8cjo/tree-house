get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRTree
        )
add_llvm_executable(main_example main_example.cpp)

llvm_update_compile_flags(main_example)
target_link_libraries(main_example PRIVATE ${LIBS})

mlir_check_all_link_libraries(main_example)
