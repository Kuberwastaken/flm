"""Create browser/README derivatives of existing recordings, without simulation."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'public/research'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    outputs = []
    for name in ('closed-loop', 'learned-choice', 'physical-calibration'):
        source = MEDIA / f'{name}.mp4'
        target = MEDIA / f'{name}.webm'
        arguments = ['-an', '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '32',
                     '-deadline', 'good', '-cpu-used', '4', '-pix_fmt', 'yuv420p']
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(source), *arguments, str(target)], check=True)
        outputs.append(dict(source=source.name, source_sha256=digest(source),
                            file=target.name, sha256=digest(target), arguments=arguments))
    source = MEDIA / 'closed-loop.mp4'
    target = MEDIA / 'closed-loop-preview.gif'
    arguments = ['-filter_complex',
                 'fps=12,scale=400:-1:flags=lanczos,split[a][b];'
                 '[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
                 '-loop', '0']
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(source), *arguments, str(target)], check=True)
    outputs.append(dict(source=source.name, source_sha256=digest(source),
                        file=target.name, sha256=digest(target), arguments=arguments))
    record = dict(format='flm-demo-media-v1', generator_sha256=digest(Path(__file__)),
                  ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
                  scope='Format/resolution derivatives of the existing complete recordings. '
                        'No simulation, inference, new frames or timing changes; GIF frame rate is reduced.',
                  outputs=outputs)
    (MEDIA / 'demo-media.json').write_bytes((json.dumps(record, indent=2)+'\n').encode())
    print('Prepared three WebM videos and one animated README preview.')


if __name__ == '__main__':
    main()
