"""Generate a self-contained SVG of two paired exploration trajectories."""
import argparse
from pathlib import Path
from .world import World, FREE
from .simulation import run_episode


def draw_svg(world: World, *, seed: int, radius: int, max_steps: int) -> str:
    tile = 19
    margin = 28
    header = 76
    width = world.width*tile + 2*margin
    height = world.height*tile + header + margin
    full_w = 2*width + 16
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{full_w}" height="{height}" viewBox="0 0 {full_w} {height}">',
             f'<rect width="{full_w}" height="{height}" fill="#0c1520"/>',
             '<style>text{font-family:Inter,Arial,sans-serif} .head{font-size:18px;font-weight:700;fill:#e8f1ff} .sub{font-size:12px;fill:#a9b9c8}</style>']
    for index, policy in enumerate(("frontier", "growth")):
        trace = []
        outcome = run_episode(world, policy, seed=seed, sensor_radius=radius,
                              max_steps=max_steps, trace=trace)
        ox = index*(width + 16)+margin
        oy = header
        name = "Nearest frontier BFS" if index == 0 else "SCA-inspired growth"
        path_color = "#83beff" if index == 0 else "#72edbd"
        parts.append(f'<text x="{ox}" y="30" class="head">{name}</text>')
        parts.append(f'<text x="{ox}" y="53" class="sub">{outcome.steps} steps · {outcome.coverage:.1%} coverage · {outcome.decisions} decisions</text>')
        parts.append(f'<rect x="{ox-2}" y="{oy-2}" width="{world.width*tile+4}" height="{world.height*tile+4}" rx="5" fill="#20303e"/>')
        for y in range(world.height):
            for x in range(world.width):
                fill = "#122533" if world.tile((x,y)) == FREE else "#455367"
                parts.append(f'<rect x="{ox+x*tile}" y="{oy+y*tile}" width="{tile-1}" height="{tile-1}" fill="{fill}"/>')
        if trace:
            polyline = " ".join(f"{ox+x*tile+tile/2:.1f},{oy+y*tile+tile/2:.1f}" for x,y in trace)
            parts.append(f'<polyline points="{polyline}" stroke="{path_color}" stroke-width="2.4" fill="none" stroke-linejoin="round" stroke-linecap="round" opacity=".80"/>')
            x, y = trace[-1]
            parts.append(f'<circle cx="{ox+x*tile+tile/2}" cy="{oy+y*tile+tile/2}" r="6" fill="{path_color}" stroke="#0a1720" stroke-width="2"/>')
        sx, sy = world.start
        parts.append(f'<circle cx="{ox+sx*tile+tile/2}" cy="{oy+sy*tile+tile/2}" r="6" fill="#ffc875" stroke="#0a1720" stroke-width="2"/>')
    parts.append('</svg>')
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=25)
    parser.add_argument("--height", type=int, default=19)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=Path("colony-trajectories.svg"))
    args = parser.parse_args()
    world = World.generated(args.seed, args.width, args.height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(draw_svg(world, seed=args.seed, radius=args.radius,
                                 max_steps=args.max_steps), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
