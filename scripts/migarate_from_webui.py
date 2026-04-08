'''将 msst-webui 安装的模型、预设迁移到 leaf+msst。'''

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure local project modules can be imported when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inject_meta_to_config import inject_meta_to_config
from utils.n_io import load_json, load_yaml, write_yaml


@dataclass(slots=True)
class ModelInfo:
  # WebUI 的模型子目录名（pretrain/<model_class>/...）
  model_class: str
  # 模型权重完整文件名（含后缀）
  model_name: str
  # 推理模型类型（如 bs_roformer / mdx23c）
  model_type: str
  # WebUI 原始配置路径
  config_path: Path
  # WebUI 权重路径：webui/pretrain/model_class/model_name
  checkpoint_path: Path
  # 本项目生成的 merged yaml 路径
  merged_config_path: Path | None = None
  # 来源标记：official / unofficial
  source: str = ''


def _warn(message: str) -> None:
  print(f'Warning: {message}')


def _norm_text(v) -> str:
  if v is None:
    return ''
  return str(v).strip()


def _model_key(model_class: str, model_name: str) -> str:
  return f'{model_class}/{model_name}'


def _resolve_webui_path(webui_dir: Path, maybe_path: str) -> Path:
  '''把 WebUI 里的路径解析成绝对路径。'''
  p = Path(maybe_path).expanduser()
  if p.is_absolute():
    return p.resolve()
  return (webui_dir / p).resolve()


def _missing_keys(d: dict, keys: tuple[str, ...]) -> list[str]:
  '''简单字段检查：缺字段返回列表。'''
  missing: list[str] = []
  for key in keys:
    if not _norm_text(d.get(key)):
      missing.append(key)
  return missing


def _iter_model_defs(raw_map: dict, source: str) -> list[tuple[str, str, str, str]]:
  '''
  展平 model map 为四元组：
  (model_class, model_name, config_path, model_type)

  仅保留两种直接结构：
  1) {model_class: {...}}
  2) {model_class: [{...}, {...}]}
  缺字段则提示并跳过。
  '''
  result: list[tuple[str, str, str, str]] = []
  for model_class, value in raw_map.items():
    model_class = _norm_text(model_class)
    if not model_class:
      _warn(f'[{source}] skip entry: missing model_class')
      continue

    items = value if isinstance(value, list) else [value]
    for item in items:
      if not isinstance(item, dict):
        _warn(f'[{source}] skip {model_class}: invalid entry type={type(item).__name__}')
        continue

      # 与 WebUI 兼容：name 等价于 model_name
      if 'model_name' not in item and 'name' in item:
        item = dict(item)
        item['model_name'] = item.get('name')

      missing = _missing_keys(item, ('model_name', 'config_path', 'model_type'))
      if missing:
        _warn(
          f'[{source}] skip {model_class}/{_norm_text(item.get('model_name') or item.get('name'))}: '
          f'missing {', '.join(missing)}'
        )
        continue

      result.append((
        model_class,
        _norm_text(item['model_name']),
        _norm_text(item['config_path']),
        _norm_text(item['model_type']),
      ))
  return result


def _collect_official_models(webui_dir: Path) -> dict[str, ModelInfo]:
  models_info = load_json(webui_dir / 'data' / 'models_info.json')
  msst_model_map = load_json(webui_dir / 'data' / 'msst_model_map.json')

  # 只建立精确映射：model_class + model_name
  map_by_key = {
    _model_key(model_class, model_name): (config_path, model_type)
    for model_class, model_name, config_path, model_type in _iter_model_defs(
      msst_model_map, 'official-map'
    )
  }

  result: dict[str, ModelInfo] = {}
  for key_name, info in models_info.items():
    if not isinstance(info, dict):
      _warn(f'[official] skip {key_name}: invalid info type={type(info).__name__}')
      continue
    if info.get('is_installed') is not True:
      continue

    model_class = _norm_text(info.get('model_class'))
    model_name = _norm_text(info.get('model_name') or key_name)
    missing = []
    if not model_class:
      missing.append('model_class')
    if not model_name:
      missing.append('model_name')
    if missing:
      _warn(f'[official] skip {key_name}: missing {', '.join(missing)}')
      continue

    key = _model_key(model_class, model_name)
    mapped = map_by_key.get(key)
    if mapped is None:
      _warn(f'[official] skip {key}: no mapping found')
      continue

    config_path_raw, model_type = mapped
    result[key] = ModelInfo(
      model_class=model_class,
      model_name=model_name,
      model_type=model_type,
      config_path=_resolve_webui_path(webui_dir, config_path_raw),
      checkpoint_path=(webui_dir / 'pretrain' / model_class / model_name).resolve(),
      source='official',
    )
  return result


def _collect_unofficial_models(webui_dir: Path) -> dict[str, ModelInfo]:
  map_path = webui_dir / 'config_unofficial' / 'unofficial_msst_model.json'
  if not map_path.is_file():
    _warn(f'[unofficial] file not found: {map_path}')
    return {}

  raw_map = load_json(map_path)
  result: dict[str, ModelInfo] = {}
  for model_class, model_name, config_path_raw, model_type in _iter_model_defs(
    raw_map, 'unofficial-map'
  ):
    key = _model_key(model_class, model_name)
    result[key] = ModelInfo(
      model_class=model_class,
      model_name=model_name,
      model_type=model_type,
      config_path=_resolve_webui_path(webui_dir, config_path_raw),
      checkpoint_path=(webui_dir / 'pretrain' / model_class / model_name).resolve(),
      source='unofficial',
    )
  return result


def _inject_to_new_config(models: dict[str, ModelInfo]) -> dict[str, ModelInfo]:
  '''将目标配置目录写入新yaml'''
  injected: dict[str, ModelInfo] = {}
  for key, model in models.items():
    output_dir = (PROJECT_ROOT / 'configs' / 'merged' / model.model_class).resolve()
    output_name = Path(model.model_name).stem
    try:
      output_path = inject_meta_to_config(
        model_type=model.model_type,
        model_checkpoint=model.checkpoint_path,
        config_path=model.config_path,
        output_dir=output_dir,
        output_name=output_name,
      )
      model.merged_config_path = output_path
      injected[key] = model
      print(f'[info] generated: {output_path.relative_to(PROJECT_ROOT)}')
    except Exception as exc:
      _warn(
        f'[{model.source}] inject failed {model.model_class}/{model.model_name}: '
        f'{type(exc).__name__}: {exc}'
      )
  return injected


def _build_leaf_module(
  config_path: Path,
  input_to_next,
  output_to_storage: list[str],
  module_index_num: tuple[int],
) -> dict:
  '''Build a single MSST.infer module definition.'''

  arguments: list[dict] = [{
    'key': 'config_path',
    'value': config_path.as_posix(),
  }]

  # 模型支持输出的所有音轨
  training_config = load_yaml(config_path).get('training', {})
  all_instruments = None
  if insts := training_config.get('instruments'):
    all_instruments = insts
  elif inst := training_config.get('target_instrument'):
    all_instruments = [inst]

  index, num = module_index_num

  arguments.append({
    'key': 'input_path',
    # 在flow里，第一个模块必须有实际输入，后面的模块从mmap获取
    'method': 'mmap' if index > 0 else 'select',
  })

  arguments.append({
    'key': 'instrument',
    'value': output_to_storage,
    'options': all_instruments
  })

  if input_to_next and index < num-1:
    # 不是最后一个的模块都需要输出给mmap
    arguments.append({
      'key': 'temp-instrument',
      'value': input_to_next,
    })
  
  # if index < num-1:
  #   # 不是最后一个模块默认都不输出中间结果
  #   arguments.append({
  #     'key': 'filename_template',
  #     'value': '',
  #   })

  return {
    'key': 'MSST.infer',
    'desc': '',
    'arguments': arguments,
  }


def _build_leaf_flow(branches: list[dict], leaf_flow_path: Path) -> dict:
  '''Assemble a flow YAML document with all branches, based on previous msst yaml.'''
  template = load_yaml(leaf_flow_path)
  if not isinstance(template, dict):
    _warn(f'[flow] invalid template type={type(template).__name__}: {leaf_flow_path}')
    template = {'name': 'MSST', 'desc': '', 'meta': {'version': '1.0'}}

  return {
    'name': 'MSST-webui',
    'desc': template.get('desc', ''),
    'meta': template.get('meta', None),
    'branches': branches,
  }


def _build_leaf_branch(name: str, modules: list[dict]) -> dict:
  '''Build a branch entry for a single preset.'''
  return {
    'key': name,
    'name': name,
    'desc': '',
    'modules': modules,
  }


def _convert_preset_to_flow(webui_dir: Path, models: dict[str, ModelInfo], target_path: Path) -> Path | None:
  '''Build a single flow YAML with all preset branches.'''
  presets_dir = webui_dir / 'presets'
  if not presets_dir.is_dir():
    _warn(f'[preset] dir not found: {presets_dir}')
    return None

  target_path.parent.mkdir(parents=True, exist_ok=True)
  branches: list[dict] = []

  for preset_file in sorted(presets_dir.glob('*.json')):
    try:
      preset_data = load_json(preset_file)
    except Exception as exc:
      _warn(f'[preset] failed to read {preset_file.name}: {type(exc).__name__}: {exc}')
      continue

    flow = preset_data.get('flow', [])
    if not isinstance(flow, list):
      _warn(f'[preset] {preset_file.name}: invalid flow')
      continue

    modules: list[dict] = []
    for index, step in enumerate(flow):
      if not isinstance(step, dict):
        _warn(f'[preset] {preset_file.name}: skip invalid step type={type(step).__name__}')
        continue

      # 因为msst-webui的preset就是这么定义的，混用了model_type表示model_class
      missing = _missing_keys(step, ('model_type', 'model_name', 'input_to_next'))
      if missing:
        _warn(f'[preset] {preset_file.name}: skip step missing {', '.join(missing)}')
        continue

      step_model_class = _norm_text(step['model_type'])
      step_model_name = _norm_text(step['model_name'])
      matched = models.get(_model_key(step_model_class, step_model_name))
      if matched is None:
        _warn(
          f'[preset] {preset_file.name}: no matched model for '
          f'model_class={step_model_class}, model_name={step_model_name}'
        )
        continue

      if matched.merged_config_path is None:
        _warn(f'[preset] {preset_file.name}: merged config missing for {matched.model_name}')
        continue

      modules.append(
        _build_leaf_module(
          config_path=matched.merged_config_path,
          input_to_next=step.get('input_to_next'),
          output_to_storage=step.get('output_to_storage') or [],
          module_index_num=(index, len(flow)),
        )
      )

    if not modules:
      _warn(f'[preset] {preset_file.name}: no valid modules, skipped')
      continue

    branches.append(_build_leaf_branch(preset_file.stem, modules))

  if not branches:
    _warn('[preset] no valid branches generated')
    return None

  flow = _build_leaf_flow(branches, target_path.with_stem('MSST.flow'))
  write_yaml(target_path, flow)
  print(f'generated flow: {target_path}')


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Migrate models/presets from MSST-WebUI into this project.')
  parser.add_argument('--msst-webui-dir', required=True, type=Path)  # D:\Software\MSST WebUI
  parser.add_argument('--official-model', action='store_true', help='解析并迁移 msst webui 自带模型（默认安装的）')
  parser.add_argument('--unofficial-model', action='store_true', help='解析并迁移 msst webui 里第三方模型（自行安装的）')
  parser.add_argument('--preset-to-leaf-flow', action='store_true', help='将 msst webui 的预设（preset）迁移到 leaf ，生成 flow 配置 "MSST-webui.local.flow.yaml"')
  parser.add_argument('--leaf-dir', type=Path, default='./bud', help='leaf安装/解压目录')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  webui_dir = args.msst_webui_dir.expanduser().resolve()
  if not webui_dir.is_dir():
    raise FileNotFoundError(f'--msst-webui-dir not found: {webui_dir}')

  if not args.official_model and not args.unofficial_model and not args.preset_to_leaf_flow:
    raise ValueError('Nothing to do. Provide at least one of --official-model, --unofficial-model, --preset-to-leaf-flow')

  all_injected_models: dict[str, ModelInfo] = {}

  run_official = args.official_model or args.preset_to_leaf_flow
  run_unofficial = args.unofficial_model or args.preset_to_leaf_flow

  if run_official:
    official_models = _collect_official_models(webui_dir)
    injected = _inject_to_new_config(official_models)
    all_injected_models.update(injected)
    print(f'[info] official injected number: {len(injected)}')

  if run_unofficial:
    unofficial_models = _collect_unofficial_models(webui_dir)
    injected = _inject_to_new_config(unofficial_models)
    all_injected_models.update(injected)
    print(f'[info] unofficial injected number: {len(injected)}')

  if not args.preset_to_leaf_flow:
    return
    
  target_path: Path = args.leaf_dir / 'bud/sprig/MSST-webui.local.flow.yaml'
  print(target_path.resolve().absolute())
  _convert_preset_to_flow(webui_dir, all_injected_models, target_path)

if __name__ == '__main__':
  main()
