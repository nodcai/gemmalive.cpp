# Downloads the pinned model files into models/.
#
#   cmake -P scripts/download_models.cmake [-DMODELS_DIR=<dir>] [-DGEMMA=E2B|E4B] [-DPARTS=ON]
#
# Files that already exist with the expected hash are skipped. GEMMA picks
# the Gemma 4 (default E2B, 2.3 GB; E4B is 3.5 GB), which the program and
# tools then take with --gemma. PARTS also gets the model and mmproj that
# file was made of, which the heads are trained on (training/).

cmake_minimum_required(VERSION 3.24)

if(NOT MODELS_DIR)
    get_filename_component(MODELS_DIR "${CMAKE_CURRENT_LIST_DIR}/../models" ABSOLUTE)
endif()
if(NOT GEMMA)
    set(GEMMA E2B)
endif()

set(HF https://huggingface.co)

# repo@revision  path  sha256  [local name, if not the path]
if(GEMMA STREQUAL E2B)
    set(MODELS
        "nodcai/gemma-4-E2B-live-gguf@5f7d5c28bb91924cd2a706a30900176dadf87d0b|gemma-4-E2B-live-XS.gguf|d8341a7a552c269b90d72d2fe1c24078e3fb5efce92d8bedba6d696cfe22905f"
    )
    set(PARTS_OF_GEMMA
        "TheStageAI/gemma-4-E2B-it-GGUF@57cbf0912db499cff5cc9cf0d800c2247c49e376|gemma-4-E2B-it-XS-TS-Q3_K_S.gguf|d1e358e0a9f945084e8757090684ef698f9e99f41693dcd465f5aeaa3564c3b5"
        "ggml-org/gemma-4-E2B-it-GGUF@63f6c6f23b3e9e5f9737708daec2f88c9ac2e921|mmproj-gemma-4-E2B-it-Q8_0.gguf|042cc2561806cd3b5cb5efb25cab76911e3100c43cc09d854904f921924b1429"
    )
elseif(GEMMA STREQUAL E4B)
    set(MODELS
        "nodcai/gemma-4-E4B-live-gguf@5ccbbfdd5ff29d92ba805780a1725b08a876e1b0|gemma-4-E4B-live-XS.gguf|9e7a53a3e3a7170bde2820d2ee5598b3c31c9d4b1fa216cf1a957aa156884e55"
    )
    set(PARTS_OF_GEMMA
        "TheStageAI/gemma-4-E4B-it-GGUF@1f3817e38c6729a174bb002c37dbec5183620bd1|gemma-4-E4B-it-XS-TS-Q3_K_S.gguf|788e53407470a95f5dfd5d82690898bd641b499624eec7e2c09da2a23db0405c"
        "ggml-org/gemma-4-E4B-it-GGUF@27cd6836f8e2a6e5e6b57ea0e77fa72490706985|mmproj-gemma-4-E4B-it-Q8_0.gguf|9257f21a0f5358204b4a612cff83fad2e56412f4f607f0049331ce60c0cf9d34"
    )
else()
    message(FATAL_ERROR "GEMMA must be E2B or E4B, not ${GEMMA}")
endif()
if(PARTS)
    list(APPEND MODELS ${PARTS_OF_GEMMA})
endif()
list(APPEND MODELS
    "ggml-org/whisper-vad@9ffd54a1e1ee413ddf265af9913beaf518d1639b|ggml-silero-v5.1.2.bin|29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf"
    "nodcai/livevoice-0.5b-gguf@cb3cac5d2daf7b380588432fa5778927f656a0b0|livevoice-0.5b-COMPACT.gguf|d0f1847626959ab88a1e1e05b4d42d828dd91ec4e269cbf007732ebfdf5d6a95"
    "nodcai/livevoice-0.5b-gguf@cb3cac5d2daf7b380588432fa5778927f656a0b0|voices/en-Clarissa_female.gguf|355fd8f804153c3e443f858308bd8ba199bdf4b67b3bfcc9a5febe98c4bba96b"
)

foreach(entry IN LISTS MODELS)
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 repo_rev)
    list(GET fields 1 path)
    list(GET fields 2 sha256)
    set(name "${path}")
    list(LENGTH fields n_fields)
    if(n_fields GREATER 3)
        list(GET fields 3 name)
    endif()
    string(REPLACE "@" ";" repo_rev "${repo_rev}")
    list(GET repo_rev 0 repo)
    list(GET repo_rev 1 rev)

    set(dst "${MODELS_DIR}/${name}")

    if(EXISTS "${dst}")
        file(SHA256 "${dst}" actual)
        if(actual STREQUAL sha256)
            message(STATUS "ok        ${name}")
            continue()
        endif()
        message(STATUS "mismatch  ${name}, downloading again")
    endif()

    message(STATUS "download  ${repo}/${path}")
    file(DOWNLOAD "${HF}/${repo}/resolve/${rev}/${path}" "${dst}.part"
        SHOW_PROGRESS
        STATUS status
    )
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        file(REMOVE "${dst}.part")
        list(GET status 1 error)
        message(FATAL_ERROR "download of ${path} failed: ${error}")
    endif()

    file(SHA256 "${dst}.part" actual)
    if(NOT actual STREQUAL sha256)
        file(REMOVE "${dst}.part")
        message(FATAL_ERROR "sha256 mismatch for ${name}: got ${actual}, expected ${sha256}")
    endif()
    file(RENAME "${dst}.part" "${dst}")
endforeach()

message(STATUS "models in ${MODELS_DIR}")
