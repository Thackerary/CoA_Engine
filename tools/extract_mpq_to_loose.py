#!/usr/bin/env python3
import argparse
import ctypes
from ctypes import wintypes
import fnmatch
import os
import struct
import sys
from pathlib import Path

# ==================== 1. 绑定 StormLib.dll C 接口 ====================
MPQ_OPEN_READ_ONLY = 0x00000100
SFILE_INVALID_SIZE = 0xFFFFFFFF

def load_stormlib(dll_path: Path):
    if not dll_path.exists():
        print(f"[Error] 找不到 {dll_path}，请先运行: cmake --build build --config Release -j")
        sys.exit(1)

    storm = ctypes.WinDLL(str(dll_path))

    # bool WINAPI SFileOpenArchive(const wchar_t* szMpqName, DWORD dwPriority, DWORD dwFlags, HANDLE* phMpq)
    storm.SFileOpenArchive.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    storm.SFileOpenArchive.restype  = ctypes.c_bool

    storm.SFileCloseArchive.argtypes = [wintypes.HANDLE]
    storm.SFileCloseArchive.restype  = ctypes.c_bool

    storm.SFileHasFile.argtypes = [wintypes.HANDLE, wintypes.LPCSTR]
    storm.SFileHasFile.restype  = ctypes.c_bool

    storm.SFileOpenFileEx.argtypes = [wintypes.HANDLE, wintypes.LPCSTR, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    storm.SFileOpenFileEx.restype  = ctypes.c_bool

    storm.SFileGetFileSize.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    storm.SFileGetFileSize.restype  = wintypes.DWORD

    storm.SFileReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p
    ]
    storm.SFileReadFile.restype = ctypes.c_bool

    storm.SFileCloseFile.argtypes = [wintypes.HANDLE]
    storm.SFileCloseFile.restype  = ctypes.c_bool

    return storm

def read_file_from_mpq(storm, h_mpq, internal_path: str) -> bytes | None:
    norm_mpq_path = internal_path.replace('/', '\\').encode('ascii', errors='ignore')
    if not storm.SFileHasFile(h_mpq, norm_mpq_path):
        return None

    h_file = wintypes.HANDLE()
    if not storm.SFileOpenFileEx(h_mpq, norm_mpq_path, 0, ctypes.byref(h_file)):
        return None

    file_size = storm.SFileGetFileSize(h_file, None)
    if file_size == 0 or file_size == SFILE_INVALID_SIZE:
        storm.SFileCloseFile(h_file)
        return None

    buf = ctypes.create_string_buffer(file_size)
    bytes_read = wintypes.DWORD(0)
    ok = storm.SFileReadFile(h_file, buf, file_size, ctypes.byref(bytes_read), None)
    storm.SFileCloseFile(h_file)

    if not ok or bytes_read.value != file_size:
        return None
    return buf.raw

# ==================== 2. MPQ 分类与优先级排序 ====================
BASE_MPQ_NAMES = {
    'common.mpq', 'common-2.mpq', 'expansion.mpq', 'lichking.mpq',
    'patch.mpq', 'patch-2.mpq', 'patch-3.mpq',
    'backup-enus.mpq', 'base-enus.mpq',
    'locale-enus.mpq', 'locale-zhcn.mpq', 'speech-enus.mpq', 'speech-zhcn.mpq',
    'expansion-locale-enus.mpq', 'lichking-locale-enus.mpq',
    'expansion-speech-enus.mpq', 'lichking-speech-enus.mpq',
    'patch-enus.mpq', 'patch-enus-2.mpq', 'patch-enus-3.mpq',
    'patch-zhcn.mpq', 'patch-zhcn-2.mpq', 'patch-zhcn-3.mpq',
}

def is_coa_patch_mpq(filename: str) -> bool:
    return filename.lower() not in BASE_MPQ_NAMES

def mpq_priority_key(path: Path) -> tuple[int, str]:
    stem = path.stem.lower() # 去掉 .mpq 后缀，避免 '.' (ASCII 46) 干扰比较
    if stem == 'patch':
        return (1000, '')
    if stem.startswith('patch-enus') or stem.startswith('patch-zhcn'):
        return (1100, stem)
    if stem.startswith('patch-'):
        tag = stem[6:] # 提取 'patch-' 之后的标识，如 '2', 'a', 'ca', 'wc3', 'z', 'zg'
        # 数字补丁 (patch-2..9) 排在字母补丁 (patch-a..z) 之前，字母补丁严格按标准字典序比较
        tier = 1200 if tag.isdigit() else 1300
        return (tier, tag)
    if 'lichking' in stem:  return (400, stem)
    if 'expansion' in stem: return (300, stem)
    if 'common-2' in stem:  return (200, stem)
    if 'common' in stem:    return (100, stem)
    return (50, stem)

# ==================== 3. 自动解析 M2 模型内部引用的 BLP 贴图与 .skin ====================
def parse_m2_dependencies(m2_rel_path: str, m2_bytes: bytes) -> list[str]:
    deps = []
    if m2_rel_path.endswith('.m2'):
        deps.append(m2_rel_path[:-3] + '00.skin')

    # 兼容 CoA 从高版本移植的 MD21 8字节封装头
    if len(m2_bytes) >= 0x60 and m2_bytes[:4] == b'MD21':
        m2_bytes = m2_bytes[8:]

    # 检查 MD20 头 (至少 0x58 字节以读取 offset 0x50 处的 textures M2Array)
    if len(m2_bytes) < 0x58 or m2_bytes[:4] != b'MD20':
        return deps

    tex_count, tex_offset = struct.unpack_from('<II', m2_bytes, 0x50)
    if tex_count == 0 or tex_count > 128 or tex_offset + tex_count * 16 > len(m2_bytes):
        return deps

    for i in range(tex_count):
        t_type, _flags, fn_len, fn_off = struct.unpack_from('<IIII', m2_bytes, tex_offset + i * 16)
        if t_type == 0 and fn_len > 1 and fn_off + fn_len <= len(m2_bytes):
            raw_str = m2_bytes[fn_off : fn_off + fn_len - 1].decode('ascii', errors='ignore')
            clean_path = raw_str.replace('\\', '/').lower().lstrip('/')
            if clean_path:
                deps.append(clean_path)
    return deps
def parse_wmo_dependencies(wmo_rel_path: str, wmo_bytes: bytes) -> list[str]:
    deps = []
    pos = 0
    n_groups = 0
    while pos + 8 <= len(wmo_bytes):
        magic = wmo_bytes[pos:pos+4]
        size = struct.unpack_from('<I', wmo_bytes, pos + 4)[0]
        data_start = pos + 8
        if data_start + size > len(wmo_bytes):
            break
        chunk = wmo_bytes[data_start : data_start + size]
        if magic == b'DHOM' and size >= 8: # Reversed 'MOHD'
            n_groups = struct.unpack_from('<I', chunk, 4)[0]
        elif magic == b'XTOM': # Reversed 'MOTX'
            for raw_s in chunk.split(b'\x00'):
                if len(raw_s) > 4:
                    p = raw_s.decode('ascii', errors='ignore').replace('\\', '/').lower().lstrip('/')
                    if p.endswith('.blp'):
                        deps.append(p)
        elif magic == b'NDOM': # Reversed 'MODN' (WMO 内部 M2 家具路径池)
            for raw_s in chunk.split(b'\x00'):
                if len(raw_s) > 4:
                    p = raw_s.decode('ascii', errors='ignore').replace('\\', '/').lower().lstrip('/')
                    if p.endswith('.mdx') or p.endswith('.mdl'):
                        p = p[:-4] + '.m2'
                    if p.endswith('.m2'):
                        deps.append(p)
        pos = data_start + size

    if n_groups > 0 and n_groups <= 512 and wmo_rel_path.endswith('.wmo'):
        base = wmo_rel_path[:-4]
        for g in range(n_groups):
            deps.append(f"{base}_{g:03d}.wmo")
    return deps
def parse_adt_dependencies(adt_bytes: bytes) -> list[str]:
    """扫描 .ADT 内的 MTEX(地表贴图)、MMDX(全部户外M2，含跨资料片瀑布/藤蔓)、MWMO(全部建筑)"""
    deps = []
    pos = 0
    while pos + 8 <= len(adt_bytes):
        magic = adt_bytes[pos:pos+4]
        size = struct.unpack_from('<I', adt_bytes, pos + 4)[0]
        data_start = pos + 8
        if data_start + size > len(adt_bytes):
            break
        chunk = adt_bytes[data_start : data_start + size]
        if magic in (b'XETM', b'XDMM', b'OMWM'): # Reversed 'MTEX', 'MMDX', 'MWMO'
            for raw_s in chunk.split(b'\x00'):
                if len(raw_s) > 4:
                    p = raw_s.decode('ascii', errors='ignore').replace('\\', '/').lower().lstrip('/')
                    if p.endswith('.mdx') or p.endswith('.mdl'):
                        p = p[:-4] + '.m2'
                    if p.endswith(('.blp', '.m2', '.wmo')):
                        deps.append(p)
        pos = data_start + size
    return deps
# ==================== 4. 批量提取主流程 ====================
TEST_PATTERNS = [
    'dbfilesclient/light*.dbc',
    'tileset/*.blp',
    'world/azeroth/elwynn/*',
    'world/generic/*',
    'world/wmo/*',
    'world/maps/azeroth/azeroth_32_48.adt',
    'world/maps/azeroth/azeroth_32_49.adt',
    'world/maps/azeroth/azeroth_31_48.adt',
    'world/maps/azeroth/azeroth_33_48.adt',
    'world/maps/azeroth/azeroth_31_49.adt',
    'world/maps/azeroth/azeroth_33_49.adt',
]

def extract_group(storm, mpq_paths: list[Path], out_dir: Path, mode: str, custom_filter: str,
                  fallback_mpq_paths: list[Path] | None = None):
    # 按修正后的字典序优先级从高到低排序
    mpq_paths = sorted(mpq_paths, key=mpq_priority_key, reverse=True)
    handles = []

    for p in mpq_paths:
        h = wintypes.HANDLE()
        if storm.SFileOpenArchive(str(p), 0, MPQ_OPEN_READ_ONLY, ctypes.byref(h)):
            handles.append((p.name, h))
            print(f"  [Mounted] {p.name}")

    if not handles:
        return

    # 挂载兜底 MPQ 集合（供 coa_patches 的 ADT/WMO 引用原版底包模型时跨包查找）
    fallback_handles = []
    if fallback_mpq_paths:
        for p in sorted(fallback_mpq_paths, key=mpq_priority_key, reverse=True):
            h = wintypes.HANDLE()
            if storm.SFileOpenArchive(str(p), 0, MPQ_OPEN_READ_ONLY, ctypes.byref(h)):
                fallback_handles.append((p.name, h))

    all_search_handles = handles + fallback_handles
    extracted_set = set()
    pending_deps = set()

    def save_virtual_file(rel_lower: str, data: bytes, source_mpq: str = ""):
        dest = out_dir / rel_lower
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        extracted_set.add(rel_lower)
        if rel_lower.endswith('.adt'):
            print(f"  [ADT Extracted] {rel_lower}  <-- from {source_mpq} ({len(data)} bytes)")
            for dep in parse_adt_dependencies(data):
                if dep not in extracted_set:
                    pending_deps.add(dep)
        elif rel_lower.endswith('.m2'):
            for dep in parse_m2_dependencies(rel_lower, data):
                if dep not in extracted_set:
                    pending_deps.add(dep)
        elif rel_lower.endswith('.wmo'):
            for dep in parse_wmo_dependencies(rel_lower, data):
                if dep not in extracted_set:
                    pending_deps.add(dep)

    # 1. 遍历各 MPQ 的 (listfile) 清单提取匹配文件
    for mpq_name, h_mpq in handles:
        listfile_bytes = read_file_from_mpq(storm, h_mpq, "(listfile)")
        if not listfile_bytes:
            continue

        lines = listfile_bytes.decode('utf-8', errors='ignore').splitlines()
        for raw_line in lines:
            rel_lower = raw_line.strip().replace('\\', '/').lower().lstrip('/')
            if not rel_lower or rel_lower in extracted_set:
                continue

            matched = False
            if mode == 'all':
                matched = True
            elif mode == 'test':
                matched = any(fnmatch.fnmatch(rel_lower, pat) for pat in TEST_PATTERNS)
            elif custom_filter:
                matched = fnmatch.fnmatch(rel_lower, custom_filter.lower())

            if matched:
                data = read_file_from_mpq(storm, h_mpq, raw_line.strip())
                if data:
                    save_virtual_file(rel_lower, data, mpq_name)

    # 2. 递归补齐 ADT / WMO / M2 引用的所有跨资料片模型、子房间、.skin 与 .blp 贴图
    while pending_deps:
        dep_path = pending_deps.pop()
        if dep_path in extracted_set:
            continue
        for mpq_name, h_mpq in all_search_handles:
            data = read_file_from_mpq(storm, h_mpq, dep_path)
            if data:
                save_virtual_file(dep_path, data, mpq_name)
                break

    for _name, h in handles:
        storm.SFileCloseArchive(h)
    for _name, h in fallback_handles:
        storm.SFileCloseArchive(h)

    print(f"  -> 已提取 {len(extracted_set)} 个文件至 {out_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="提取飞升服/3.3.5a MPQ 至全小写散文件目录")
    parser.add_argument("--data-dir", required=True, help="飞升服客户端 Data 目录路径 (例如 D:/WOW/CoA Client/Data)")
    parser.add_argument("--out-dir", default="build/Release/CoA_Assets", help="输出根目录 (默认项目内 CoA_Assets)")
    parser.add_argument("--dll", default="build/_deps/stormlib-build/Release/StormLib.dll", help="StormLib.dll 路径")
    parser.add_argument("--mode", choices=['test', 'all', 'filter'], default='test',
                        help="test=仅提取艾尔文森林测试集(秒级完成), all=提取全部文件, filter=自定义通配符")
    parser.add_argument("--filter", default="", help="当 --mode filter 时的通配符, 如 'creature/illidan/*'")
    args = parser.parse_args()

    storm = load_stormlib(Path(args.dll))
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[Error] Data 目录不存在: {data_dir}")
        sys.exit(1)

    # 扫描 Data 目录及子目录 (含 enUS, area-52, Content 等) 中的所有 .MPQ
    all_mpqs = list(data_dir.glob("*.[mM][pP][qQ]"))
    for sub in ("enUS", "zhCN", "enGB", "zhTW", "area-52", "Content"):
        sub_dir = data_dir / sub
        if sub_dir.exists():
            all_mpqs.extend(sub_dir.glob("*.[mM][pP][qQ]"))

    base_mpqs  = [p for p in all_mpqs if not is_coa_patch_mpq(p.name)]
    patch_mpqs = [p for p in all_mpqs if is_coa_patch_mpq(p.name)]

    out_root = Path(args.out_dir)
    print(f"=== 开始提取 [模式: {args.mode}] ===")
    print(f"[1/2] 处理原版底包 ({len(base_mpqs)} 个 MPQ) -> {out_root / 'wotlk_base'}")
    extract_group(storm, base_mpqs, out_root / "wotlk_base", args.mode, args.filter)

    print(f"[2/2] 处理飞升服魔改补丁 ({len(patch_mpqs)} 个 MPQ) -> {out_root / 'coa_patches'}")
    extract_group(storm, patch_mpqs, out_root / "coa_patches", args.mode, args.filter, fallback_mpq_paths=base_mpqs)
    print("=== 全部提取完成！ ===")

if __name__ == "__main__":
    main()
