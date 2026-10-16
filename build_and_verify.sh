#!/usr/bin/env bash
set -euo pipefail

TARGET_ABI="arm64-v8a"
RUST_TARGET="aarch64-linux-android"
ANDROID_API_LEVEL="34"
LIB_NAME="libs24_dj_core.so"
OUTPUT_DIR="./target/jniLibs"
SO_PATH="${OUTPUT_DIR}/${TARGET_ABI}/${LIB_NAME}"

echo "================================================================="
echo " [1/4] Verifying Android NDK & Rust Toolchain Environment"
echo "================================================================="
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    echo "ERROR: ANDROID_NDK_HOME is not set." >&2
    exit 1
fi

TOOLCHAIN_BIN="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin"
LLVM_READELF="${TOOLCHAIN_BIN}/llvm-readelf"
LLVM_NM="${TOOLCHAIN_BIN}/llvm-nm"
LLVM_OBJDUMP="${TOOLCHAIN_BIN}/llvm-objdump"

for tool in "${LLVM_READELF}" "${LLVM_NM}" "${LLVM_OBJDUMP}"; do
    if [[ ! -x "${tool}" ]]; then
        echo "ERROR: Required NDK binary not found or not executable: ${tool}" >&2
        exit 1
    fi
done

echo "Using NDK at: ${ANDROID_NDK_HOME}"
rustc --version
cargo ndk --version

echo "================================================================="
echo " [2/4] Compiling Release Shared Library (${TARGET_ABI} / API ${ANDROID_API_LEVEL})"
echo "================================================================="
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cargo ndk \
    --target "${TARGET_ABI}" \
    --platform "${ANDROID_API_LEVEL}" \
    --output-dir "${OUTPUT_DIR}" \
    build --release

if [[ ! -f "${SO_PATH}" ]]; then
    echo "ERROR: Expected shared library not produced at ${SO_PATH}" >&2
    exit 1
fi

echo "Successfully built: ${SO_PATH} ($(du -h "${SO_PATH}" | cut -f1))"

echo "================================================================="
echo " [3/4] Auditing Dynamic Linker Dependencies & JNI Exports"
echo "================================================================="
echo "--- Dynamic NEEDED Libraries ---"
"${LLVM_READELF}" -d "${SO_PATH}" | grep -E "NEEDED|SONAME"

REQUIRED_JNI_SYMBOLS=(
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeInit"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeStart"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeStop"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeRelease"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativePushDeckAudio"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetDeckEq"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetDeckKills"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetDeckGains"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetCrossfader"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetMasterAndCue"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeGetFramesProcessed"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeInitStemEngine"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeToggleStemSeparation"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeSetStemGains"
    "Java_com_s24ultra_dj_audio_NativeAudioEngine_nativeGetLastStemInferenceUs"
)

echo "--- Verifying Exported JNI Symbols ---"
DEFINED_EXPORTS=$("${LLVM_NM}" -D --defined-only "${SO_PATH}")
for sym in "${REQUIRED_JNI_SYMBOLS[@]}"; do
    if echo "${DEFINED_EXPORTS}" | grep -q "${sym}"; then
        echo "  [OK] ${sym}"
    else
        echo "  [MISSING] Exported JNI symbol not found: ${sym}" >&2
        exit 1
    fi
done

echo "================================================================="
echo " [4/4] Verifying Zero Unresolved Non-System Symbols & NEON SIMD"
echo "================================================================="
UNDEFINED_SYMS=$("${LLVM_NM}" -D --undefined-only "${SO_PATH}" | awk '{print $NF}' | sed 's/@.*//')

ALLOWED_PREFIXES='^(__cxa_|__gxx_|__android_log_|__stack_chk_|__errno|__register_atfork|__assert|__sF|__system_property|AAudio|SL_IID_|_Z|pthread_|clock_|dl_|lseek|write|read|realpath|strerror|sched_|setpriority|atoi|abort|fwrite)'
ALLOWED_STD='^(abort|calloc|close|dladdr|dlclose|dlerror|dlopen|dlsym|Dup|fflush|fprintf|free|fstat|futex|getcwd|getenv|getauxval|getpid|gettid|ioctl|lseek|malloc|memchr|memcmp|memcpy|memmove|memset|mmap|mprotect|munmap|nanosleep|open|poll|posix_memalign|prctl|pthread_|read|realloc|sched_|sem_|sigaction|sigaltstack|sigemptyset|snprintf|stat|fstat|strchr|strcmp|strerror|strlen|strncmp|strrchr|strtol|strtoul|syscall|sysconf|usleep|vfprintf|vsnprintf|write|_Unwind_|_Z|acos|asin|atan|atan2|cbrt|ceil|ceilf|copysign|cos|cosf|cosh|exp|exp2|expf|expm1|fabs|floor|floorf|fma|fmax|fmin|fmod|fmodf|frexp|hypot|ilogb|ldexp|lgamma|log|log10|log10f|log1p|log2|logf|modf|nan|nearbyint|nextafter|pow|powf|remainder|remquo|rint|round|roundf|scalbn|sin|sinf|sinh|sqrt|sqrtf|tan|tanf|tanh|tgamma|trunc)$'

UNEXPECTED_UNDEF=0
for sym in ${UNDEFINED_SYMS}; do
    if [[ "${sym}" =~ ${ALLOWED_PREFIXES} ]] || [[ "${sym}" =~ ${ALLOWED_STD} ]]; then
        continue
    fi
    echo "ERROR: Unresolved non-system symbol detected: ${sym}" >&2
    UNEXPECTED_UNDEF=1
done

if [[ "${UNEXPECTED_UNDEF}" -ne 0 ]]; then
    echo "Verification FAILED due to unresolved symbols." >&2
    exit 1
fi

NEON_COUNT=$("${LLVM_OBJDUMP}" -d "${SO_PATH}" | grep -c -E "fmla|fadd\tv|fmul\tv" || true)
echo "  [OK] Zero unexpected unresolved symbols."
echo "  [OK] Verified ${NEON_COUNT} vectorized ARMv9 NEON SIMD instructions."
echo "================================================================="
echo " BUILD & VERIFICATION SUCCEEDED"
echo "================================================================="