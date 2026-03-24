# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import io
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.n_io import load_yaml, MmapCompatibleIO

import warnings

warnings.filterwarnings('ignore')


def _load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int, Path]:
    '''
    Load audio from a normal file path or from a .leaf.mmap payload.
    Returns (audio, sr, source_path_for_naming).
    '''
    io_obj = MmapCompatibleIO(path)
    if not io_obj.use_mmap:
        mix, sr = librosa.load(path.as_posix(), sr=sample_rate, mono=False)
        return mix, sr, path

    payload, payload_type, _ = io_obj.read()
    if payload_type == 'text':
        # 若 payload 是 text，作为音频路径加载，一般也不会触发
        source_path = Path(str(payload).strip()).expanduser()
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        mix, sr = librosa.load(source_path.as_posix(), sr=sample_rate, mono=False)
        return mix, sr, source_path

    # 若 payload 是 binary，用字节流加载（BytesIO）作为中间层去load
    with io.BytesIO(payload) as audio_buffer:
        mix, sr = librosa.load(audio_buffer, sr=sample_rate, mono=False)
    return mix, sr, path


def _write_audio(path: str, audio: np.ndarray, sr: int, subtype: str, use_flac: bool) -> None:
    '''
    Write separated stem to normal file path or .leaf.mmap.
    '''
    io_obj = MmapCompatibleIO(path)
    io_obj.path.parent.mkdir(parents=True, exist_ok=True)

    audio_suffix = ''
    if not io_obj.use_mmap:
        audio_suffix = io_obj.path.suffix

    if audio_suffix == '':
        # 无后缀或mmap才由use_flac决定
        audio_format = 'FLAC' if use_flac else 'WAV'
    # 否则都是 .flac -> FLAC 后缀决定
    audio_format = audio_suffix[1:].uppper()

    if io_obj.use_mmap:
        with io.BytesIO() as audio_buffer:
            sf.write(audio_buffer, audio, sr, subtype=subtype, format=audio_format)
            io_obj.write(
                audio_buffer.getvalue(),
                dtype='binary',
                info=f'format={audio_format};sr={sr};subtype={subtype}'
            )
        return

    sf.write(path, audio, sr, subtype=subtype, format=audio_format)


def run(
    model: 'torch.nn.Module',
    args: 'argparse.Namespace',
    config: dict,
    device: 'torch.device',
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

    # Recursively collect all files from input directory
    mixture_paths = []
    for i in args.input_path:
        i = Path(i)
        if not i.is_dir():
            mixture_paths.append(i.resolve())
            continue
        mixture_paths.extend(f.resolve() for f in i.rglob('*') if f.is_file())
    mixture_paths.sort(key=lambda p: p.as_posix())

    sample_rate: int = getattr(config.audio, 'sample_rate', 44100)

    print(f'Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}')

    # 优先使用命令行参数指定的，否则获取配置中设置的目标输出乐器/训练时乐器列表，这决定了最终输出哪些乐器轨
    instruments: list[str] = args.instrument or prefer_target_instrument(config)[:]
    Path(args.store_dir).mkdir(parents=True, exist_ok=True)

    # Wrap paths with progress bar if not in verbose mode
    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc='Total progress')

    # Determine whether to use detailed progress bar
    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    for path in mixture_paths:
        path = Path(path)

        try:
            mix, sr, source_path = _load_audio(path, sample_rate)
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            continue

        # Get naming fields from original source path if input comes from mmap text payload.
        dir_name: str = source_path.parent.name
        file_name: str = source_path.stem
        if source_path.name.endswith('.leaf.mmap'):
            file_name = source_path.name[:-len('.leaf.mmap')]

        # Convert mono audio to expected channel format if needed
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    print('Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()

        # Normalize input audio if enabled
        norm = config.inference.get('normalize') is True
        if norm:
            mix, norm_params = normalize_audio(mix)

        # Perform source separation
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

        # Extract instrumental track if requested
        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')
        
        subtype = args.pcm_type
        # 处理单独的音轨输出
        if args.instrument_path:
            instr, p = args.instrument_path.split(':', 1)
            estimate = waveforms_orig[instr]
            if norm:
                estimate = denormalize_audio(estimate, norm_params)
            _write_audio(p, estimate.T, sr, subtype, args.flac_file)
        
        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Denormalize output audio if normalization was applied
            if norm:
                estimates = denormalize_audio(estimates, norm_params)

            peak: float = float(np.abs(estimates).max())
            if peak <= 1.0 and args.pcm_type != 'FLOAT':
                codec = 'flac'
            else:
                codec = 'wav'

            # Generate output directory structure using relative paths
            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=file_name,
                dir_name=dir_name,
                model_type=args.model_type,
                model=Path(args.start_check_point).stem,
            )

            # Create output directory
            output_dir = Path(args.store_dir).joinpath(*dirnames)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f'{fname}.{codec}'
            sf.write(str(output_path), estimates.T, sr, subtype=subtype)

            # Draw and save spectrogram if enabled
            if args.draw_spectro > 0:
                output_img_path = output_dir / f'{fname}.jpg'
                draw_spectrogram(estimates.T, sr, args.draw_spectro, str(output_img_path))
                print('Wrote file:', output_img_path)

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
    Merge `_leaf_meta` values from config (args.config_path) into args, then remove `_leaf_meta` from config.
    理论上应该直接使用这里加载的config，并去掉get_model_from_config里面加载config的逻辑，不过有点麻烦，暂且搁置
    '''

    try:
        config = load_yaml(args.config_path)
    except:
        return

    leaf_meta = config.pop('_model_info', None)
    if not leaf_meta or not isinstance(leaf_meta, dict):
        return

    if (model_type := leaf_meta.get('model_type', '')):
        args.model_type = model_type

    if (checkpoint := leaf_meta.get('model_checkpoint', '')):
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
