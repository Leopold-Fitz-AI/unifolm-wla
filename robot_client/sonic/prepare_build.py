"""Prepare the isolated SONIC build on pi0.5 (TensorRT 10 runtime).

Only edits this task's external SONIC snapshot, not system headers or libraries.
The native locomotion planner requires TensorRT 10. Runtime libraries and headers
must be unpacked in this task's deps directory; system packages are untouched.
"""

import argparse
from pathlib import Path
import subprocess


def patch_tensorrt(sonic_root):
    path = Path(sonic_root) / "gear_sonic_deploy/src/TRTInference/InferenceEngine.cpp"
    source = path.read_text()
    if "UNIFOLM_TRT11_COMPAT" in source:
        return
    source = "// UNIFOLM_TRT11_COMPAT: static FP32 SONIC simulation build.\n" + source
    old = "const uint32_t explicitBatch = 1U << static_cast<uint32_t>( nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH );"
    if old not in source:
        raise ValueError("unrecognized upstream TensorRT wrapper; review source before patching")
    source = source.replace(old, "const uint32_t explicitBatch = 0; // TRT >= 10 is always explicit-batch")
    needle = "auto profile = builder->createOptimizationProfile();"
    source = source.replace(needle, needle + """
    auto setShapeValuesCompat = [&](char const* name, nvinfer1::OptProfileSelector selector, auto const& values) {
#if NV_TENSORRT_MAJOR >= 11
        std::vector<int64_t> values64(values.begin(), values.end());
        return profile->setShapeValuesV2(name, selector, values64.data(), values64.size());
#else
        return profile->setShapeValues(name, selector, values.data(), values.size());
#endif
    };
""")
    for index, selector in enumerate(("kMIN", "kOPT", "kMAX")):
        import re
        pattern = (r"profile->setShapeValues\(\s*name,\s*nvinfer1::OptProfileSelector::" + selector
                   + r",\s*std::get<" + str(index) + r">\(\s*sizes\s*\)\.data\(\),\s*std::get<"
                   + str(index) + r">\(\s*sizes\s*\)\.size\(\)\s*\)")
        source, count = re.subn(pattern, f"setShapeValuesCompat(name, nvinfer1::OptProfileSelector::{selector}, std::get<{index}>(sizes))", source)
        if count != 1:
            raise ValueError("unrecognized shape-profile API")
    start = source.index("    // Set the precision level")
    end = source.index("    // CUDA stream used for profiling", start)
    block = source[start:end]
    source = source[:start] + """
#if NV_TENSORRT_MAJOR >= 11
    if (options.precision == Precision::FP16) {
        LOG_ERROR("TRT11 requires an explicitly typed FP16 ONNX graph; use FP32 for this simulation build.");
        return false;
    }
#else
""" + block + "#endif\n" + source[end:]
    path.write_text(source)


def patch_domain(sonic_root):
    path = Path(sonic_root) / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
    source = path.read_text()
    old = "ChannelFactory::Instance()->Init(0, networkInterface);"
    if old in source:
        source = source.replace(old, 'ChannelFactory::Instance()->Init(std::getenv("SONIC_DDS_DOMAIN") ? '
                                'std::stoi(std::getenv("SONIC_DDS_DOMAIN")) : 0, networkInterface);')
        path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    sonic = root / "GR00T-WholeBodyControl"
    patch_tensorrt(sonic)
    patch_domain(sonic)
    sysroot = root / "deps/sysroot/usr"
    headers = root / "deps/headers"
    trt = root / "deps/trt10/tensorrt_libs"
    if not (trt / "libnvinfer.so.10").is_file():
        raise FileNotFoundError(f"unpack the TensorRT 10 runtime in {trt}")
    trt_headers = headers / "tensorrt10"
    ort = root / "venv-sim/lib/python3.10/site-packages/onnxruntime/capi"
    ort_lib = next(ort.glob("libonnxruntime.so.*"))
    soname = ort / "libonnxruntime.so.1"
    if not soname.exists():
        soname.symlink_to(ort_lib.name)
    boost = root / "venv-sim/lib/python3.10/site-packages/cmeel.prefix/include"
    flags = " ".join(f"-I{p}" for p in (trt_headers, headers / "zmq", sysroot / "include",
                                         boost, Path("/usr/include/eigen3")))
    command = ["cmake", "-S", str(sonic / "gear_sonic_deploy"), "-B", str(root / "build-controller"),
               "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DCMAKE_BUILD_TYPE=Release",
               f"-DCMAKE_PREFIX_PATH={sysroot}", f"-DCMAKE_CXX_FLAGS={flags}",
               "-DTensorRT_FIND_COMPONENTS=nvinfer;nvonnxparser",
               f"-DTensorRT_INCLUDE_DIR={trt_headers}",
               f"-DTensorRT_nvinfer_LIBRARY={trt / 'libnvinfer.so.10'}",
               f"-DTensorRT_nvonnxparser_LIBRARY={trt / 'libnvonnxparser.so.10'}",
               f"-Donnxruntime_INCLUDE_DIR={headers / 'onnxruntime'}",
               f"-Donnxruntime_LIBRARY={ort_lib}",
               f"-DZMQ_INCLUDE_DIR={sysroot / 'include'}",
               "-DZMQ_LIBRARY=/lib/x86_64-linux-gnu/libzmq.so.5",
               f"-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST={sysroot / 'src/googletest'}",
               f"-DCMAKE_BUILD_RPATH={trt};{ort};{sysroot / 'lib/x86_64-linux-gnu'}"]
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(root / "build-controller"), "--target", "g1_deploy_onnx_ref", "-j2"], check=True)


if __name__ == "__main__":
    main()
