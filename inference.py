# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import io
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import argparse

# Using the embedded version of Python can also correctly import the utils module.
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.n_io import load_yaml, MmapFileExchange, resolve_input_path


def _parse_instrument_output(value: str) -> tuple[str, str]:
    """
    Parse `instrument:path` string from CLI.
    """
    if not value:
        return '', ''
    instr, sep, out_path = value.partition(':')
    if not sep:
        return '', ''
    return instr.strip(), out_path.strip()


def run(
    model: torch.nn.Module,
    args: argparse.Namespace,
    config: dict,
    device: torch.device,
    verbose: bool = False
) -> None:
    '''
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : argparse.Namespace
        Arguments containing input folder, output folder, and processing options.
    config : dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    '''

    start_time = time.time()
    model.eval()

    # Step 1) Build final mixture_paths and binary mmap payload cache.
    mixture_paths, vr_path_map = resolve_input_path(args.input_path)
    mixture_paths.sort(key=lambda p: p.as_posix())

    # Step 2) Resolve runtime settings and output intent.
    sample_rate: int = getattr(config.audio, 'sample_rate', 44100)
    instruments: list[str] = args.instrument or prefer_target_instrument(config)[:]
    mmap_instr, mmap_path = _parse_instrument_output(args.temporary_output or '')
    # 两个都存在，启用mmap输出
    use_mmap_output = bool(mmap_instr) and bool(mmap_path)
    print(f'Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}')

    # Step 3) Configure progress bars.
    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc='Total progress')
    detailed_pbar = not args.disable_detailed_pbar

    # Step 4) Process each input mixture independently.
    for path in mixture_paths:
        # vr_path_map 多输入时保存虚拟输入路径到真实源路径的映射，单输入时指向BytesIO
        real_obj = vr_path_map.get(path, path)
        mix, sr = librosa.load(real_obj, sr=sample_rate, mono=False)

        # Step 4.2) Prepare naming fields and channel layout.
        dir_name: str = path.parent.name
        file_name: str = path.stem

        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    print('Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        # Step 4.3) Optional normalization before demix.
        mix_orig = mix.copy()
        norm = config.inference.get('normalize') is True
        if norm:
            mix, norm_params = normalize_audio(mix)

        # Step 4.4) Run model inference and optional TTA.
        waveforms_orig = demix(
            config,
            model,
            mix,
            device,
            model_type=args.model_type,
            pbar=detailed_pbar
        )

        # Apply test-time augmentation if enabled
        if args.use_tta:
            waveforms_orig = apply_tta(
                config,
                model,
                mix,
                waveforms_orig,
                device,
                args.model_type
            )

        # Step 4.5) Optionally derive instrumental from target stem.
        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        # Step 4.6) Emit each requested stem to filesystem/mmap.
        subtype = args.pcm_type

        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Step 4.6.1) Restore original scale when normalization was enabled.
            if norm:
                estimates = denormalize_audio(estimates, norm_params)

            # Step 4.6.2) Decide whether a physical output file is required.
            output_path: Path | None = None
            # Step 4.6.3) Choose output codec by peak and PCM type.
            peak: float = float(np.abs(estimates).max())
            codec = 'flac' if peak <= 1.0 and subtype != 'FLOAT' else 'wav'

            if bool(args.filename_template):
                # Step 4.6.4-B) Normal output path generation from template.
                dirnames, fname = format_filename(
                    args.filename_template,
                    instr=instr,
                    start_time=int(start_time),
                    file_name=file_name,
                    dir_name=dir_name,
                    model_type=args.model_type,
                    model=Path(args.start_check_point).stem,
                )

                if not args.store_dir:
                    # 未指定则保存到输入文件的所在文件夹
                    output_dir = path.parent
                else:
                    output_dir = Path(args.store_dir).joinpath(*dirnames)

                output_dir.mkdir(parents=True, exist_ok=True)
                # Step 4.6.5) Write audio file and optional spectrogram image.
                output_path = output_dir / f'{fname}.{codec}'
                sf.write(output_path.as_posix(), estimates.T, sr, subtype=subtype)
                print(f'Save {instr} to {output_path}')

                if args.draw_spectro > 0:
                    output_img_path = output_dir / f'{fname}.jpg'
                    draw_spectrogram(estimates.T, sr, args.draw_spectro, str(output_img_path))
                    print('Wrote file:', output_img_path)

            # Step 4.6.6) 是否输出到mmap，命中则在循环内即时写入。
            if instr == mmap_instr and use_mmap_output:
                buffer = None
                len_input = len(mixture_paths)
                if output_path is None or len_input == 1:
                    # 多文件下没有实际输出需要写buffer，单文件也写，为了mmap加速读取
                    buffer = io.BytesIO()
                    sf.write(buffer, estimates.T, sr, subtype=subtype, format=codec.upper())
                # 期望保存路径，用于mmap多次调用下定位写入最终输出结果。若确实有则用output_path，否则上虚拟路径
                path = output_path or (path.parent / f'{file_name}_{instr}.{codec}')
                MmapFileExchange(mmap_path).write(
                    {path: buffer},
                    temp_dir= 'msst' if output_path is None else None,
                    metadata={'instrument': mmap_instr},
                    # 多个输入路径，说明需要多次迭代追加才写得完
                    append=len_input > 1,
                )

    # Step 5) Print final processing time.
    print(f'Elapsed time: {time.time() - start_time:.2f} seconds.')


def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g '{file_name}/{instr}'
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f'{{{k}}}', str(v))
    *dirnames, fname = result.split('/')
    return dirnames, fname


def _inject_meta_from_config(args) -> None:
    '''
    Merge `_model_info` values from config (args.config_path) into args, then remove `_model_info` from config.
    理论上应该直接使用这里加载的config，并去掉get_model_from_config里面加载config的逻辑，不过有点麻烦，暂且搁置
    '''

    try:
        config = load_yaml(args.config_path)
    except:
        return

    model_info = config.pop('_model_info', None)
    if not model_info or not isinstance(model_info, dict):
        return

    if (model_type := model_info.get('model_type', '')):
        args.model_type = model_type

    if (checkpoint := model_info.get('model_checkpoint', '')):
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute() and args.config_path:
            checkpoint_path = (
                Path(args.config_path).expanduser().resolve().parent / checkpoint_path
            ).resolve()
        args.start_check_point = str(checkpoint_path)


def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    _inject_meta_from_config(args)

    device = 'cpu'
    if args.force_cpu:
        device = 'cpu'
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print('Using device: ', device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print('Instruments: {}'.format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print('Model load time: {:.2f} sec'.format(time.time() - model_load_start_time))

    run(model, args, config, device, verbose=True)


if __name__ == '__main__':
    proc_folder(None)
