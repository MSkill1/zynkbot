# OpenCL headers and ICD loader for the Android build

Used only when compiling the local-model engine (llama-cpp-sys-2, feature `opencl`) for
`aarch64-linux-android`. The NDK ships no OpenCL SDK, so the engine's build needs a header
directory and an import library to link against; at run time the phone's own
`/vendor/lib64/libOpenCL.so` is loaded instead (Qualcomm Adreno on Snapdragon phones,
PowerVR on the Pixel 10). `.cargo/config.toml` points `OPENCL_INCLUDE_DIR` and
`OPENCL_LIBRARY` here.

- `include/CL/` — KhronosGroup/OpenCL-Headers, main branch, 2026-10-23 (Apache-2.0).
- `lib/libOpenCL.so` — KhronosGroup/OpenCL-ICD-Loader, main branch, built 2026-10-23 with
  NDK 30.0.15729638 for arm64-v8a, ANDROID_PLATFORM 26, c++_shared (Apache-2.0). Link-time
  only; it is not packaged into the APK.

Rebuild: clone both repositories, `cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DOPENCL_ICD_LOADER_HEADERS_DIR=<headers> -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=26 -DANDROID_STL=c++_shared` in the loader, build, copy `libOpenCL.so` here.

A phone without a public `libOpenCL.so` cannot load an app linked this way; before this
build ships to anyone but the two test phones, the backend has to become a dynamically
loaded one (llama-cpp-sys-2 feature `dynamic-backends`) or the link made optional.
