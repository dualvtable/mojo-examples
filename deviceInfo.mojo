# Prints the system information

from IO import print
from String import String
from LLCL import num_cores
from TargetInfo import
     os_is_linux,
     os_is_windows,
     os_is_macos,
     has_sse4,
     has_avx,
     has_avx2,
     has_avx512f,
     has_avx512_vnni,
     has_neon,
     is_apple_m1,
     has_intel_amx

@always_inline("nodebug")
fn _current_target() -> __mlir_type.`!kgen.target`:
    return __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`

fn _current_cpu() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        _current_target(),
        `, "cpu" : !kgen.string`,
        `> : !kgen.string`,
    ]


def main() -> None:
    var os = ""
    if (os_is_linux()):
        os = "linux"
    elif (os_is_macos()):
        os = "macOS"
    else:
        os = "windows"
    let cpu = String(_current_cpu())
    let cpu_features = String(" ")
    if (has_sse4()):
        cpu_features + " sse4"
    if (has_avx()):
        cpu_features + " avx"
    if (has_avx2()):
        cpu_features + " avx2"
    if (has_avx512f()):
        cpu_features + " avx512f"
    if (has_avx512_vnni()):
        cpu_features + " avx512_vnni"
    if (has_intel_amx()):
        cpu_features + " intel_amx"
    if (has_neon()):
        cpu_features + " neon"
    if (is_apple_m1()):
        cpu_features + " Apple M1"
    print("System information: ")
    print("    OS          : ", os)
    print("    CPU         : ", cpu)
    print("    Num Cores   : ", num_cores())
    print("    CPU Features: ", cpu_features)
