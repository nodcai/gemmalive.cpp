# Each test_*.cpp is one executable and one ctest. A test returns 77 when
# what it needs (a model file, named by an environment variable) is missing;
# ctest reports it as skipped.

function(gemmalive_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE gemmalive)
    target_compile_options(${name} PRIVATE ${GEMMALIVE_WARNINGS})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

# Without a model.
gemmalive_add_test(test_calls)
gemmalive_add_test(test_spoken)
gemmalive_add_test(test_device)
gemmalive_add_test(test_sampler)
gemmalive_add_test(test_mel)
gemmalive_add_test(test_turn_head)

# With GEMMALIVE_MODEL, and for the conversations GEMMALIVE_MMPROJ.
gemmalive_add_test(test_tokenizer)
gemmalive_add_test(test_rollback)
gemmalive_add_test(test_hold)
