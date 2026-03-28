import argparse
import sys
from collections.abc import MutableMapping
from pathlib import Path

# Ensure local project modules can be imported when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from utils.n_io import load_yaml, write_yaml


def build_relative_weight_path(weight_path: Path, output_yaml_path: Path) -> str:
  """Build a checkpoint path relative to the output YAML directory when possible."""
  config_path = weight_path.expanduser().resolve()
  base_dir = output_yaml_path.parent.expanduser().resolve()

  # Windows跨盘符场景下无法计算相对路径，回退到绝对路径
  if config_path.anchor.casefold() != base_dir.anchor.casefold():
    return str(config_path)

  source_parts = config_path.parts
  base_parts = base_dir.parts

  shared_len = 0
  max_shared_len = min(len(source_parts), len(base_parts))
  while shared_len < max_shared_len and source_parts[shared_len].casefold() == base_parts[shared_len].casefold():
    shared_len += 1

  relative_parts = ['..'] * (len(base_parts) - shared_len) + list(source_parts[shared_len:])
  if not relative_parts:
    return '.'
  return str(Path(*relative_parts))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Inject _model_info into a YAML config and save as a new YAML file.')
  parser.add_argument('--model-type', required=True, help='Model type metadata.')
  parser.add_argument('--model-checkpoint', required=True, type=Path, help='Path to model weight file.')
  parser.add_argument('--config-path', required=True, type=Path, help='Path to source YAML config file.')
  parser.add_argument('--output-dir', default=None, type=Path, help='Directory to save the new YAML file.')
  parser.add_argument(
    '--output-name',
    default=None,
    help='New YAML filename (optional). If omitted, source YAML filename is used.',
  )
  return parser.parse_args()


def inject_meta_to_config(
  model_type: str,
  model_checkpoint: Path,
  config_path: Path,
  output_dir: Path | None = None,
  output_name: str | None = None,
) -> Path:
  """Inject _model_info into a YAML config and return output YAML path."""
  # expanduser() 把用户目录写法（~/xx/x）展开成真实绝对路径前缀
  config_path = config_path.expanduser().resolve()
  model_checkpoint = model_checkpoint.expanduser().resolve()
  if output_dir is None:
    # 不指定就覆盖源文件
    output_dir = config_path.parent
  else:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

  filename = output_name if output_name else config_path.name
  output_filename = Path(filename).with_suffix('.yaml')
  output_path = output_dir / output_filename

  yaml_obj = load_yaml(config_path)
  if yaml_obj is None:
    yaml_obj = {}
  if not isinstance(yaml_obj, MutableMapping):
    raise TypeError(f'YAML root must be a mapping, got: {type(yaml_obj)}')

  relative_weight_path = build_relative_weight_path(model_checkpoint, output_path)
  yaml_obj['_model_info'] = {
    'model_type': model_type,
    'model_checkpoint': relative_weight_path,
  }

  write_yaml(output_path, yaml_obj)
  return output_path


def main() -> None:
  args = parse_args()
  output_path = inject_meta_to_config(
    model_type=args.model_type,
    model_checkpoint=args.model_checkpoint,
    config_path=args.config_path,
    output_dir=args.output_dir,
    output_name=args.output_name,
  )
  print(f'Saved new YAML to: {output_path}')


if __name__ == '__main__':
  main()
