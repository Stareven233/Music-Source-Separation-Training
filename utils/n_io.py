import io
import json
import pickle
from pathlib import Path
import re
import mmap
import struct
from dataclasses import dataclass
from urllib.parse import quote, unquote

from ruamel.yaml import YAML

# 用来保留注释
yaml = YAML()
# 避免长字符串按空格自动换行（ruamel 默认 width=80）
yaml.width = 10**9


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


def write_yaml(path, obj, encoding='utf-8', width: int | None = None) -> None:
  if width is None:
    emitter = yaml
  else:
    emitter = YAML()
    emitter.width = width

  with open(path, 'w', encoding=encoding) as f:
    emitter.dump(obj, f)


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

  @staticmethod
  def isMmap(path: str | Path) -> bool:
    p = path
    if isinstance(path, Path):
      p = path.as_posix()
    return p.endswith('.leaf.mmap')


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
    self.use_mmap = self.isMmap(path)
    super().__init__(path, header, endian)

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


@dataclass
class MmapFileExchangeResult:
  mode: str
  content: bytes | None
  paths: list[Path] | None
  info: str
  metadata: dict[str, str]


class MmapFileExchange(MmapCompatibleIO):
  '''
  通用 mmap 文件交换类。
  仅负责在 mmap 中存取两类载荷，不负责 mmap 之外的任何文件读写：
  - 文件内容（bytes）  
  - 文件路径列表（newline-separated text）  

  主要用法：
  1) 写入单个文件内容到 mmap：
     exchange = MmapFileExchange('a.leaf.mmap')
     exchange.write_content(b'file-bytes', metadata={'format': 'wav'})

  2) 写入多个已落盘文件的路径到 mmap：
     exchange = MmapFileExchange('a.leaf.mmap')
     exchange.write_paths(['a.wav', 'b.wav'], metadata={'instrument': 'vocals'})

  3) 从 mmap 读取载荷：
     result = exchange.read()
     if result.mode == MmapFileExchange.MODE_CONTENT:
       content = result.content
     else:
       paths = result.paths

  调用方负责决定写入“内容”还是“路径”，以及负责 mmap 之外的真实文件读写。  
  metadata本质是info的一部分，自动编解码为字符串从info存取
  '''

  MODE_CONTENT = 'content'
  MODE_PATHS = 'paths'
  METADATA_PREFIX = 'meta.'

  def __init__(self, path: str | Path, header: dict[str, str] | None = None, endian: str = '<'):
    super().__init__(path, header, endian)

  def _encode_info(self, mode: str, info: str = '', metadata: dict | None = None) -> str:
    parts: list[str] = [f'mmap_mode={quote(str(mode), safe="")}']
    if info:
      parts.append(f'info={quote(str(info), safe="")}')
    if metadata:
      for key, value in metadata.items():
        parts.append(f'{self.METADATA_PREFIX}{quote(str(key), safe="")}={quote(str(value), safe="")}')
    return ';'.join(parts)

  def _decode_info(self, info: str) -> tuple[str, str, dict[str, str]]:
    mode = ''
    decoded_info = ''
    metadata: dict[str, str] = {}
    if info:
      for part in info.split(';'):
        if '=' not in part:
          continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = unquote(value.strip())
        if key == 'mmap_mode':
          mode = value.lower()
        elif key == 'info':
          decoded_info = value
        elif key.startswith(self.METADATA_PREFIX):
          metadata[unquote(key[len(self.METADATA_PREFIX):])] = value
    return mode, decoded_info, metadata


  def write_content(self, content: bytes, *, info: str = '', metadata: dict | None = None) -> None:
    payload = content if isinstance(content, bytes) else bytes(content)
    super().write(payload, dtype='binary', info=self._encode_info(self.MODE_CONTENT, info=info, metadata=metadata))

  def write_paths(self, paths: list[str | Path], *, info: str = '', metadata: dict | None = None, append: bool = False) -> None:
    normalized_paths = [Path(p).expanduser().resolve() for p in paths if str(p).strip()]
    merged_paths: list[Path] = []
    merged_metadata = metadata

    if append and (existing := self.read()) is not None:
      if existing.mode != self.MODE_PATHS:
        raise ValueError(f'Cannot append file paths to mmap mode: {existing.mode}')
      merged_paths.extend(existing.paths or [])
      # 合并metadata，避免追加时只剩最后一次的路径映射。
      merged_metadata = dict(existing.metadata)
      merged_metadata.update(metadata or {})

    # 去重
    seen = {p.as_posix() for p in merged_paths}
    for path in normalized_paths:
      path_str = path.as_posix()
      if path_str in seen:
        continue
      merged_paths.append(path)
      seen.add(path_str)

    payload = '\n'.join(path.as_posix() for path in merged_paths)
    super().write(payload, dtype='text', info=self._encode_info(self.MODE_PATHS, info=info, metadata=merged_metadata))

  def write(
      self,
      path_buffer: dict[Path, io.BytesIO],
      temp_dir: str|None,
      *,
      info: str = '',
      metadata: dict | None = None,
      append: bool = False,
  ) -> None:
    '''
    path_buffer: 期望写入路径与buffer的映射，path可能是实际写入（temp_dir is None）的或期望写入的（temp_dir is not None）
    temp_dir: 指定临时文件夹目录名，主临时文件夹默认为mmap文件所在目录。如果不需要保存为临时文件（已正式落盘），则设为None
    '''

    if not path_buffer:
      # 没键值对的空字典视为False
      raise ValueError('path_buffer must contain at least one io.BytesIO instance')


    if len(path_buffer) == 1 and not append:
      # 只有一个输入文件的情况，直接将二进制内容存mmap里，metadata只需要期望/虚拟路径
      path, buffer = next(iter(path_buffer.items()))
      metadata.update(_construct_vpath_metadata(None, path))
      self.write_content(buffer.getvalue(), info=info, metadata=metadata)
      return

    self.path.parent.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for e_path, buffer in path_buffer.items():
      # temp_dir 给定，说明没中间输出，需要写缓存
      if temp_dir is not None and buffer is not None:
        # 构造实际写入路径（缓存路径）
        r_path = self.path.parent / temp_dir / e_path.name
        r_path.parent.mkdir(parents=True, exist_ok=True)
        buffer.seek(0)
        r_path.write_bytes(buffer.getvalue())
        # 真实(缓存路径) -> 期望路径映射靠metadata保存
        metadata.update(_construct_vpath_metadata(r_path, e_path))
      # temp_dir不存在，说明e_path就是真实写入路径，此时不写metadata，没有虚拟路径需要映射
      else:
        r_path = e_path
      # 真实保存路径写入mmap
      out_paths.append(r_path)
      

    self.write_paths(out_paths, info=info, metadata=metadata, append=append)

  def read(self) -> MmapFileExchangeResult|None:
    '''文件不存在/大小为0/存在但内容全0、头部不完整、协议字段异常、UTF-8 解码失败等都返回None'''

    if not self.path.exists() or self.path.stat().st_size == 0:
      return None

    try:
      payload, dtype, info = super().read()
      mode, info, metadata = self._decode_info(info)
    except (ValueError, OSError, UnicodeDecodeError, struct.error):
      return None

    if not mode:
      return None

    if mode == self.MODE_PATHS:
      if dtype != 'text':
        raise ValueError(f'Expected text payload for file paths, got {dtype}')
      paths = [Path(line.strip()).expanduser().resolve() for line in str(payload).splitlines() if line.strip()]
      return MmapFileExchangeResult(mode=mode, content=None, paths=paths, info=info, metadata=metadata)

    if mode == self.MODE_CONTENT:
      if dtype != 'binary':
        raise ValueError(f'Expected binary payload for file content, got {dtype}')
      return MmapFileExchangeResult(mode=mode, content=payload if isinstance(payload, bytes) else bytes(payload), paths=None, info=info, metadata=metadata)

    raise ValueError(f'Unsupported mmap mode: {mode}')


def _map_real_to_vpath(metadata: dict[str, str], r_path: Path|None) -> Path|None:
  # 真实路径是映射键的一部分
  key = f'real_path@None'
  if r_path is not None:
    key = f'real_path@{quote(r_path.as_posix(), safe="")}'
  v_path = metadata.get(key, None)
  # 若多输入的结果都有落盘，这里就找不到虚拟路径，因为不需要
  if v_path is None:
    return None
  return Path(v_path)


def _construct_vpath_metadata(real_path: Path|None, expect_path: Path):
  return {
    # 单输入时只需要虚拟/期望映射，键等于 real_path@None
    # 多输入时记录每个虚拟输出路径对应的真实输入路径，供 resolve_input_path 还原映射。
    f'real_path@{real_path and real_path.as_posix()}': expect_path.as_posix(),
  }


def resolve_input_path(paths: list[str]) -> tuple[list[Path], dict[Path, Path|io.BytesIO]]:
  '''
    Resolve input paths into virtual paths and path mappings.
    vr_path_map only stores virtual path -> real source path, value io.BytesIO means v_path is mmap file.
    resolved_paths: real path or virtual/expected path (need vr_path_map to find real path)
  '''
  resolved_paths: list[Path] = []
  # 保存虚拟路径到真实源路径的映射，仅mmap时有效
  vr_path_map: dict[Path, Path|io.BytesIO] = {}

  for p in paths:
    p = Path(p)

    if p.is_dir():
      for f in p.rglob('*'):
        if not f.is_file():
          continue
        resolved_paths.append(f)
      continue

    if not MmapIO.isMmap(p):
      resolved_paths.append(p)
      continue

    try:
      exchange = MmapFileExchange(p)
      result = exchange.read()
    except Exception as e:
      print(f'Error message: {str(e)}')
      continue

    if result.mode == MmapFileExchange.MODE_PATHS:
      # 多输入文件时paths就是每个结果的真实写入路径
      for r_path in result.paths or []:
        r_path = Path(r_path)
        v_path = _map_real_to_vpath(result.metadata, r_path)
        # 解析路径存虚拟/期望的，然后靠vr_path_map映射回真实
        resolved_paths.append(v_path or r_path)
        if v_path is None:
          continue
        vr_path_map[v_path] = r_path
      continue

    if result.mode != MmapFileExchange.MODE_CONTENT or result.content is None:
      print(f'Error message: Unsupported mmap payload mode: {result.mode}')
      continue

    # 单文件内容模式下，默认使用mmap存内容，因此mmap path指向虚拟路径
    v_path = _map_real_to_vpath(result.metadata, None)
    resolved_paths.append(v_path)
    # print('-------------------')
    # print(f'{v_path=}')
    # print(f'{result.metadata=}')
    # print('-------------------\n\n\n')
    buffer = io.BytesIO(result.content or b'')
    vr_path_map[v_path] = buffer

  return resolved_paths, vr_path_map
