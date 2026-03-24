import json
import pickle
from pathlib import Path
import re
import mmap
import struct

from ruamel.yaml import YAML


# 用来保留注释
yaml = YAML()


def check_path_exist(func):

  def inner(*args, **kwargs):
    p = args[0]
    if not isinstance(p, Path):
      p = Path(p)
      args = (p, *args[1:])
    if not p.is_file():
      raise FileNotFoundError(p)
    res = func(*args, **kwargs)
    return res

  return inner


@check_path_exist
def load_json(path, encoding='utf-8', **kwargs) -> dict:
  with path.open('r', encoding=encoding) as f:
    return json.load(f, **kwargs)


def write_json(path, obj, encoding='utf-8', **kwargs) -> None:
  with open(path, 'w', encoding=encoding) as f:
    json.dump(obj, f, ensure_ascii=False, **kwargs)


@check_path_exist
def load_yaml(path, encoding='utf-8') -> dict:
  with path.open('r', encoding=encoding) as f:
    return yaml.load(f)


def write_yaml(path, obj, encoding='utf-8') -> None:
  with open(path, 'w', encoding=encoding) as f:
    yaml.dump(obj, f)


@check_path_exist
def load_pickle(path, **kwargs):
  with path.open('rb') as f:
    return pickle.load(f, **kwargs)


def write_pickle(path, obj, **kwargs) -> None:
  with open(path, 'wb') as f:
    pickle.dump(obj, f, **kwargs)


def find_nth_sub_path(path: Path, r: str, index=-1) -> Path:
  r'''
    验证p是一个目录，然后对其中的每个路径名应用正则表达式r提取数字，
    过滤无法提取数字的路径，并按提取的数字排序返回有效路径列表。
  
    用于从权重目录下获取最新的训练权重

    :param path: 要扫描的目录路径
    :param r: 用于提取数字的正则表达式字符串，例如 r'model_(\d+)\.pt'
    :param index: 从排序好的（默认从小到大）路径中获取第index个
    :return: 按提取数字排序的有效子路径列表
    :raises: ValueError 如果p不是目录
    '''
  if not path.is_dir():
    raise ValueError(f'\'{path}\' is not a directory')

  compiled_regex = re.compile(r)
  valid_paths = []

  for p in path.iterdir():
    if not p.is_file():
      continue
    match = compiled_regex.search(p.name)
    if match is None:
      continue
    try:
      number = int(match.group(1))
      valid_paths.append((number, p))
    except (ValueError, IndexError):
      # 忽略无法转换为数字的匹配
      continue

  # 按数字排序并返回目标路径
  target = sorted(valid_paths, key=lambda x: x[0])
  return target[-1][index]



class MmapIO:
  '''
  内存映射文件读写工具，实现标准的MMAP协议。
  简单用法：
  - 写入：MmapIO(path).write(data, dtype='text', info='描述信息')
  - 读取：MmapIO(path).read()

  协议格式（默认小端序）：
  [1字节类型][2字节信息长度][4字节数据长度][UTF-8信息][数据]
  - 类型：0x00=二进制，0x01=UTF-8文本
  - 信息长度：uint16，表示描述信息的字节数（最大65535字节）
  - 数据长度：uint32，表示数据部分的字节数（最大4GB）
  - UTF-8信息：描述性文本（可选，长度为0时无内容）
  - 数据：实际内容
  '''
  TYPE_BINARY = 0x00
  TYPE_TEXT = 0x01
  # struct format 字符对应的字节数
  _SIZE_MAP = {'B': 1, 'H': 2, 'I': 4, 'Q': 8, 'b': 1, 'h': 2, 'i': 4, 'q': 8}

  def __init__(self, path: str | Path, header: dict[str, str] | None = None, endian: str = '<'):
    self.path = Path(path)
    # 默认协议头为 [1字节数据类型][2字节信息长度][4字节数据长度]
    # 然后接着数据部分 [UTF-8信息][数据]
    self.header = header or {'dtype': 'B', 'info': 'H', 'data': 'I'}
    # < 小端序, B 1字节无符号, H 2字节无符号, I 4字节无符号
    self.endian = endian

  def _header_format(self) -> str:
    # 返回用于 struct.pack/unpack 的格式字符串
    return self.endian + ''.join(self.header.values())

  def header_size(self) -> int:
    # 动态计算所有头部字段的总字节数
    return sum(self._SIZE_MAP[f] for f in self.header.values())

  def write(self, data: str | bytes, dtype: str = 'text', info: str = '') -> None:
    '''写入mmap，若对应mmap文件不存在/容量不够则会创建/扩展'''

    dtype_byte = self.TYPE_TEXT if dtype == 'text' else self.TYPE_BINARY
    payload = data.encode('utf-8') if isinstance(data, str) else (data if isinstance(data, bytes) else bytes(data))
    info_bytes = info.encode('utf-8') if info else b''

    header = struct.pack(self._header_format(), dtype_byte, len(info_bytes), len(payload))
    encoded = header + info_bytes + payload
    payload_size = len(encoded)

    # 先创建/扩展文件到所需大小，确保有用到 mmap
    if not self.path.exists():
      self.path.touch()
    current_size = self.path.stat().st_size
    if current_size < payload_size:
      # 预分配策略：needed + 4MB 或 needed * 1.10，取较大值
      # 适用于 10-100MB 数据，平衡空间浪费和 truncate 调用次数
      allocated_size = max(payload_size + 4 * 1024**2, int(payload_size * 1.10))
      with self.path.open('r+b') as f:
        f.truncate(allocated_size)
        print(f'将mmap临时文件<{self.path.name}>扩容至 {allocated_size/(1024**2):.2f} MB (实际需要 {payload_size/(1024**2):.2f} MB)')

    with self.path.open('r+b') as f:
      with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as m:
        if payload_size > len(m):
          raise ValueError(f'数据过大: {payload_size}字节 > mmap大小{len(m)}字节')
        m[:payload_size] = encoded

  def read(self) -> tuple[str | bytes, str, str]:
    with self.path.open('rb') as f:
      with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        hsize = self.header_size()
        if len(m) < hsize:
          raise ValueError(f'数据太短，至少需要{hsize}字节')

        dtype, info_len, data_len = struct.unpack(self._header_format(), m[:hsize])
        data_start = hsize + info_len
        data_end = data_start + data_len

        if len(m) < data_end:
          raise ValueError(f'数据不完整，期望{data_end}字节，实际{len(m)}字节')

        info = m[hsize:data_start].decode('utf-8') if info_len > 0 else ''
        payload = m[data_start:data_end]

        if dtype == self.TYPE_TEXT:
          return payload.decode('utf-8'), 'text', info
        elif dtype == self.TYPE_BINARY:
          return payload, 'binary', info
        else:
          raise ValueError(f'未知数据类型: 0x{dtype:02x}')


class MmapFallbackIO(MmapIO):
  '''
  支持降级的内存映射文件读写工具。
  当mmap失败时，自动降级为普通文件读写（不使用协议格式）。
  '''

  def write(self, data: str | bytes, dtype: str = 'text', info: str = '') -> None:
    try:
      super().write(data, dtype, info)
    except (ValueError, OSError):
      mode = 'w' if isinstance(data, str) else 'wb'
      with self.path.open(mode) as f:
        f.write(data if isinstance(data, (str, bytes)) else bytes(data))
      warning = f'警告: mmap失败，已降级为普通文件写入（{mode}模式）'
      if info:
        warning += f'，info信息将丢失: {info}'
      print(warning)

  def read(self, dtype: str = 'text') -> tuple[str | bytes, str, str]:
    try:
      return super().read()
    except (ValueError, OSError):
      mode = 'r' if dtype == 'text' else 'rb'
      with self.path.open(mode) as f:
        data = f.read()
      return data, dtype, ''


class MmapCompatibleIO(MmapIO):
  '''
  根据文件名后缀决定读写方式，兼容mmap/普通文件。
  若为 '.leaf.mmap' 后缀，作为 mmap 读写，失败不回退
  否则当作普通文件进行读写
  '''

  def __init__(self, path: str | Path, header: dict[str, str] | None = None, endian: str = '<'):
    path_str = str(path)
    self.use_mmap = path_str.endswith('.leaf.mmap')
    super().__init__(path_str, header, endian)

  def write(self, data: str | bytes, dtype: str = 'text', info: str = '') -> None:
    if self.use_mmap:
      super().write(data, dtype, info)
    else:
      mode = 'w' if isinstance(data, str) else 'wb'
      with self.path.open(mode) as f:
        f.write(data if isinstance(data, (str, bytes)) else bytes(data))

  def read(self, dtype: str = 'text') -> tuple[str | bytes, str, str]:
    if self.use_mmap:
      return super().read()
    else:
      mode = 'r' if dtype == 'text' else 'rb'
      with self.path.open(mode) as f:
        data = f.read()
      return data, dtype, ''
