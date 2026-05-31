from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import re

import cv2
import h5py
import numpy as np
from tqdm import tqdm

'''
actions: 8bit (stored as decimal values)
MSB -> LSB = A, up, left, B, start, right, down, select
'''
RAW_TO_CLASS = {
    0: 0,      # NOOP
    16: 1,     # B
    20: 2,     # RIGHT+B
    148: 3,    # A+RIGHT+B
    48: 4,     # LEFT+B
    144: 5,    # A+B
    176: 6,    # A+LEFT+B
    4: 7,      # RIGHT
    18: 8,     # B+DOWN
    32: 9,     # LEFT
    128: 10,   # A
    132: 11,   # A+RIGHT
    80: 12,    # UP+B
    22: 13,    # RIGHT+B+DOWN
    84: 14,    # UP+RIGHT+B
}

# CLASS_NAMES = {
#     0: "NOOP",
#     1: "B",
#     2: "RIGHT+B",
#     3: "A+RIGHT+B",
#     4: "LEFT+B",
#     5: "A+B",
#     6: "A+LEFT+B",
#     7: "RIGHT",
#     8: "B+DOWN",
#     9: "LEFT",
#     10: "A",
#     11: "A+RIGHT",
#     12: "UP+B",
#     13: "RIGHT+B+DOWN",
#     14: "UP+RIGHT+B",
#     15: "OTHER",
# }

OTHER_CLASS = 15
INVALID_ACTION = 255

# <user>_<sessid>_e<episode>_<world>-<level>_f<frame>_a<action>_<datetime>.<outcome>.png
PNG_RE = re.compile(r"^.+?_[^_]+_e(?P<episode>\d+)_(?P<world>\d+)-(?P<level>\d+)_f(?P<frame>\d+)_a(?P<action>\d+)_.+\.(?:win|fail)$")

def parse_png(path):
    match = PNG_RE.match(path.stem)
    if match is None:
        return None

    return {
        "episode": int(match.group("episode")),
        "world_level": f'{match.group("world")}-{match.group("level")}',
        "frame": int(match.group("frame")),
        "raw_action": int(match.group("action")),
        "path": path,
    }

def load_frame(path, width, height):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to read image at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8, copy=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("--root", default="data/data-smb", help="Path to SMB dataset")
    parser.add_argument("--source-fps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--compression", type=str, default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--chunk-frames", type=int, default=16, help="Number of temporal frames per H5 frame chunk")
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    args.root = Path(args.root)

    # reduce number of samples
    assert args.source_fps % args.fps == 0
    stride = args.source_fps // args.fps

    if args.out is None:
        args.out = Path(f"data/smb_{args.fps}fps_{args.width}x{args.height}.h5")
    else:
        args.out = Path(args.out)

    pngs = list(args.root.rglob("*.png"))
    groups = defaultdict(list)

    print(f"Total number of frames: {len(pngs)}")

    print("Grouping episodes and frames")
    for path in pngs:
        rec = parse_png(path)
        if rec is None:
            continue
        groups[str(path.parent.relative_to(args.root))].append(rec)

    episodes = []
    total_kept_frames = 0

    for episode_dir, _frames in sorted(groups.items()):
        _frames = sorted(_frames, key=lambda r: int(r["frame"]))
        keep = list(range(0, len(_frames), stride))
        
        # each episode needs atleast 2 newly sampled frames
        if len(keep) < 2:
            continue

        episodes.append((episode_dir, _frames, keep))
        total_kept_frames += len(keep)

    print(f"Number of episodes: {len(episodes)}")
    print(f"Number of sampled frames: {total_kept_frames}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression
    string_dtype = h5py.string_dtype(encoding="utf-8") # h5py doesn't support str unicode??
    frame_chunk_frames = min(args.chunk_frames, total_kept_frames)
    action_chunk_frames = min(args.chunk_frames*1024, total_kept_frames)

    with h5py.File(args.out, "w") as h5:
        frames_ds = h5.create_dataset(
            "frames",
            shape=(total_kept_frames, args.height, args.width, 3),
            dtype=np.uint8,
            chunks=(frame_chunk_frames, args.height, args.width, 3),
            compression=compression,
        )

        actions_ds = h5.create_dataset(
            "action",
            shape=(total_kept_frames,),
            dtype=np.uint8,
            chunks=(action_chunk_frames,),
            compression=compression,
        )

        old_actions_ds = h5.create_dataset(
            "old_action",
            shape=(total_kept_frames,),
            dtype=np.uint8,
            chunks=(action_chunk_frames,),
            compression=compression,
        )

        frame_ids_ds = h5.create_dataset(
            "frame_ids",
            shape=(total_kept_frames,),
            dtype=np.int64,
            chunks=(action_chunk_frames,),
            compression=compression,
        )

        episode_offsets = [0]
        episode_numbers = []
        episode_world_levels = []

        curr_pos = 0

        print(f"frame chunks: ({frame_chunk_frames},{args.height},{args.width},3)")
        print("==== Writing *.h5 file =====")
        for episode_dir, _frames, kept in tqdm(episodes):
            start_pos = curr_pos
            episode_id = int(_frames[0]["episode"])
            world_level = str(_frames[0]["world_level"])
            n = len(kept)
            end_pos = start_pos + n

            episode_frames = np.empty((n,args.height,args.width,3), dtype=np.uint8)
            episode_actions = np.full((n), INVALID_ACTION, dtype=np.uint8)
            episode_old_actions = np.full((n), INVALID_ACTION, dtype=np.uint8)
            frame_ids = np.full((n), INVALID_ACTION, dtype=np.int64)

            for idx, keep_ in enumerate(kept):
                frame = _frames[keep_]
                old_cls = int(frame["raw_action"])
                cls = RAW_TO_CLASS.get(old_cls, OTHER_CLASS)

                episode_frames[idx] = load_frame(frame["path"], args.width, args.height)
                episode_old_actions[idx] = old_cls
                episode_actions[idx] = cls
                frame_ids[idx] = int(frame["frame"])

            # store the episode data as blocks
            frames_ds[start_pos:end_pos] = episode_frames
            actions_ds[start_pos:end_pos] = episode_actions
            old_actions_ds[start_pos:end_pos] = episode_old_actions
            frame_ids_ds[start_pos:end_pos] = frame_ids

            episode_offsets.append(end_pos)
            episode_numbers.append(episode_id)
            episode_world_levels.append(world_level)

            curr_pos = end_pos

        h5.create_dataset("episode_offsets", data=np.asarray(episode_offsets, dtype=np.int64))
        h5.create_dataset("episode_world_levels", data=episode_world_levels, dtype=string_dtype)
        h5.create_dataset("episode_numbers", data=np.asarray(episode_numbers, dtype=np.int64))

        h5.attrs["fps"] = args.fps
        h5.attrs["source_fps"] = args.source_fps
        h5.attrs["stride"] = stride
        h5.attrs["width"] = args.width
        h5.attrs["height"] = args.height
        h5.attrs["chunk_frames"] = args.chunk_frames
        h5.attrs["num_action_classes"] = 16
        h5.attrs["invalid_action"] = INVALID_ACTION

    print(f"Saved *.h5 file at {args.out}")

if __name__ == "__main__":
    main()
