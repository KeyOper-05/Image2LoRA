#!/usr/bin/env python3
"""Build an HTML grid for batch inference/evaluation results.

The page is organized by prompt. Each prompt shows one generated image per
style, using paths relative to the output HTML file so the visualization is
portable across machines.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import textwrap
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOOD_STATUSES = {"generated", "skipped_existing"}
DEFAULT_EVAL_MANIFEST = Path("outputs/batch_eval/eval_manifest.jsonl")
FALLBACK_BATCH_MANIFEST = Path("outputs/batch_results/manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a prompt-by-style HTML visualization for batch_eval/batch_infer outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch_manifest",
        "--manifest",
        dest="batch_manifest",
        type=Path,
        default=DEFAULT_EVAL_MANIFEST,
        help="JSONL manifest written by scripts/batch_eval.py, or JSON manifest written by scripts/batch_infer.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/batch_eval/visualization.html"),
        help="Output HTML file.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("html", "png"),
        default=("html", "png"),
        help="Visualization formats to write.",
    )
    parser.add_argument(
        "--png_dir",
        type=Path,
        default=Path("outputs/batch_eval/visualization_pages"),
        help="Output folder for contact sheet PNG pages.",
    )
    parser.add_argument(
        "--prompts_per_page",
        type=int,
        default=12,
        help="Number of prompt rows per PNG page.",
    )
    parser.add_argument(
        "--thumb_size",
        type=int,
        default=192,
        help="Thumbnail size in pixels used before drawing PNG contact sheets.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for PNG contact sheets.",
    )
    parser.add_argument(
        "--generated_dir",
        type=Path,
        default=Path("outputs/batch_results"),
        help="Local/portable folder that contains generated image files. Absolute manifest paths are remapped here by basename.",
    )
    parser.add_argument(
        "--style_dir",
        type=Path,
        default=None,
        help="Optional local/portable folder for style reference images. Absolute style paths are remapped here by basename.",
    )
    parser.add_argument(
        "--expected_styles",
        type=int,
        default=9,
        help="Expected number of style columns. Used only for the summary warning.",
    )
    parser.add_argument(
        "--style_limit",
        type=int,
        default=None,
        help="Optionally show only the first N styles after sorting.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optionally show only the first N prompts.",
    )
    parser.add_argument(
        "--include_all_statuses",
        action="store_true",
        help="Include failed/dry_run records instead of only generated/skipped_existing records.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Batch Eval Style Grid",
        help="HTML page title.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def remap_path(raw_path: str, local_base: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        if local_base is None:
            return path
        return resolve_project_path(local_base) / path.name
    return resolve_project_path(path)


def html_rel_path(path: Path, html_file: Path) -> str:
    rel = os.path.relpath(path, start=html_file.parent)
    rel = rel.replace(os.sep, "/")
    return quote(rel, safe="/.-_~")


def display_rel_path(path: Path) -> str:
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    return str(rel).replace(os.sep, "/")


def load_records(manifest: Path, include_all_statuses: bool) -> tuple[list[dict[str, str]], Path]:
    manifest = resolve_project_path(manifest)
    if not manifest.exists() and manifest == resolve_project_path(DEFAULT_EVAL_MANIFEST):
        fallback = resolve_project_path(FALLBACK_BATCH_MANIFEST)
        if fallback.exists():
            manifest = fallback

    text = manifest.read_text(encoding="utf-8")
    if manifest.suffix.lower() == ".jsonl" or not text.lstrip().startswith("["):
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{manifest} must be a JSON array or JSONL manifest")

    records: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "generated")
        if not include_all_statuses and status not in GOOD_STATUSES:
            continue
        prompt = str(item.get("prompt") or item.get("caption") or "")
        style = str(item.get("style") or item.get("method") or "")
        output_image = str(item.get("output_image") or item.get("generated") or "")
        if not prompt or not style or not output_image:
            continue
        records.append({**item, "prompt": prompt, "style": style, "output_image": output_image})
    if not records:
        raise ValueError(f"No usable records found in {manifest}")
    return records, manifest


def build_grid(
    records: list[dict[str, str]],
    generated_dir: Path,
    style_dir: Path | None,
    style_limit: int | None,
    max_prompts: int | None,
) -> tuple[list[str], OrderedDict[str, dict[str, dict[str, str]]]]:
    style_order = sorted({record["style"] for record in records})
    if style_limit is not None:
        style_order = style_order[:style_limit]
    style_set = set(style_order)

    by_prompt: OrderedDict[str, dict[str, dict[str, str]]] = OrderedDict()
    for record in records:
        style = record["style"]
        if style not in style_set:
            continue
        prompt = record["prompt"]
        if prompt not in by_prompt:
            if max_prompts is not None and len(by_prompt) >= max_prompts:
                continue
            by_prompt[prompt] = {}
        if style in by_prompt[prompt]:
            continue

        generated = remap_path(record["output_image"], generated_dir)
        style_ref_raw = str(record.get("style_image") or record.get("style_ref") or "")
        style_ref = remap_path(style_ref_raw, style_dir) if style_ref_raw and style_dir is not None else None
        by_prompt[prompt][style] = {
            "generated": str(generated),
            "generated_display": display_rel_path(generated),
            "style_ref": str(style_ref) if style_ref is not None else "",
            "style_ref_display": display_rel_path(style_ref) if style_ref is not None else "",
            "status": str(record.get("status", "")),
        }

    return style_order, by_prompt


def render_card(record: dict[str, str] | None, style: str, html_file: Path) -> str:
    style_html = html.escape(style)
    if record is None:
        return f'<article class="cell missing"><div class="style">{style_html}</div><div class="empty">missing</div></article>'

    generated = Path(record["generated"])
    src = html_rel_path(generated, html_file)
    path_label = html.escape(record["generated_display"])
    title = html.escape(f"{style}: {record['generated_display']}")
    status = html.escape(record["status"])
    style_ref = ""
    if record["style_ref"]:
        ref_src = html_rel_path(Path(record["style_ref"]), html_file)
        ref_label = html.escape(record["style_ref_display"])
        style_ref = (
            f'<a class="ref" href="{ref_src}" target="_blank" rel="noreferrer">'
            f'<span>ref</span><span>{ref_label}</span></a>'
        )

    return f"""
      <article class="cell">
        <div class="style">{style_html}</div>
        <a class="thumb" href="{src}" target="_blank" rel="noreferrer" title="{title}">
          <img src="{src}" alt="{title}" loading="lazy" decoding="async"
               onerror="this.closest('.thumb').classList.add('is-missing')" />
        </a>
        <div class="meta">
          <span class="status">{status}</span>
          <span class="path">{path_label}</span>
        </div>
        {style_ref}
      </article>
    """


def render_html(
    title: str,
    manifest: Path,
    output: Path,
    styles: list[str],
    by_prompt: OrderedDict[str, dict[str, dict[str, str]]],
    expected_styles: int,
) -> str:
    title_html = html.escape(title)
    manifest_label = html.escape(display_rel_path(resolve_project_path(manifest)))
    prompt_count = len(by_prompt)
    style_count = len(styles)
    complete_count = sum(1 for style_map in by_prompt.values() if len(style_map) == style_count)
    incomplete_count = prompt_count - complete_count
    warning = ""
    if expected_styles and style_count != expected_styles:
        warning = f'<span class="warning">expected {expected_styles} styles, found {style_count}</span>'
    if incomplete_count:
        warning += f'<span class="warning">{incomplete_count} prompts have missing style cells</span>'

    sections = []
    for idx, (prompt, style_map) in enumerate(by_prompt.items(), start=1):
        prompt_html = html.escape(prompt)
        cells = "\n".join(render_card(style_map.get(style), style, output) for style in styles)
        sections.append(
            f"""
            <section class="prompt-section" data-prompt="{prompt_html.lower()}">
              <header class="prompt-header">
                <span class="index">{idx:03d}</span>
                <h2>{prompt_html}</h2>
              </header>
              <div class="grid" style="grid-template-columns: repeat({max(style_count, 1)}, minmax(150px, 1fr));">
                {cells}
              </div>
            </section>
            """
        )

    style_nav = "".join(f"<span>{html.escape(style)}</span>" for style in styles)
    body_sections = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_html}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f4;
      --ink: #1c211b;
      --muted: #657066;
      --line: #d9dfd5;
      --panel: #ffffff;
      --accent: #2f6f61;
      --warn: #a34919;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 244, 0.95);
      backdrop-filter: blur(10px);
    }}
    .topbar-inner {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(220px, 420px);
      gap: 20px;
      align-items: center;
      max-width: 1800px;
      margin: 0 auto;
      padding: 14px 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    .warning {{ color: var(--warn); font-weight: 680; }}
    .search {{
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 12px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    main {{
      max-width: 1800px;
      margin: 0 auto;
      padding: 18px;
    }}
    .styles {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .styles span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 4px 9px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .prompt-section {{
      margin-bottom: 26px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    .prompt-header {{
      display: grid;
      grid-template-columns: 48px 1fr;
      gap: 10px;
      align-items: start;
      margin-bottom: 10px;
    }}
    .index {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 760;
      padding-top: 3px;
    }}
    h2 {{
      margin: 0;
      font-size: 17px;
      line-height: 1.35;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .grid {{
      display: grid;
      gap: 10px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .cell {{
      min-width: 150px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }}
    .style {{
      min-height: 28px;
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      color: var(--accent);
      font-size: 12px;
      font-weight: 760;
    }}
    .thumb {{
      position: relative;
      display: block;
      aspect-ratio: 1 / 1;
      background:
        linear-gradient(45deg, #edf0ea 25%, transparent 25%),
        linear-gradient(-45deg, #edf0ea 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #edf0ea 75%),
        linear-gradient(-45deg, transparent 75%, #edf0ea 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
    }}
    .thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .thumb.is-missing::after {{
      content: "image not found";
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 10px;
      text-align: center;
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
      background: rgba(255, 255, 255, 0.82);
    }}
    .thumb.is-missing img {{ visibility: hidden; }}
    .meta {{
      display: grid;
      gap: 3px;
      padding: 7px 8px 8px;
      min-height: 74px;
    }}
    .status {{
      color: var(--accent);
      font-size: 11px;
      font-weight: 720;
    }}
    .path {{
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 10px;
      overflow-wrap: anywhere;
    }}
    .ref {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 6px;
      padding: 0 8px 8px;
      color: var(--muted);
      font-size: 10px;
      text-decoration: none;
      overflow-wrap: anywhere;
    }}
    .ref span:first-child {{
      color: var(--accent);
      font-weight: 760;
    }}
    .empty {{
      display: grid;
      min-height: 210px;
      place-items: center;
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
    }}
    @media (max-width: 760px) {{
      .topbar-inner {{ grid-template-columns: 1fr; }}
      main {{ padding: 12px; }}
      .grid {{ grid-template-columns: repeat({max(style_count, 1)}, minmax(132px, 1fr)) !important; }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div>
        <h1>{title_html}</h1>
        <div class="summary">
          <span>{prompt_count} prompts</span>
          <span>{style_count} styles</span>
          <span>{complete_count} complete</span>
          <span>{manifest_label}</span>
          {warning}
        </div>
      </div>
      <input class="search" id="search" type="search" placeholder="Filter prompts" />
    </div>
  </div>
  <main>
    <nav class="styles">{style_nav}</nav>
    {body_sections}
  </main>
  <script>
    const input = document.getElementById("search");
    const sections = Array.from(document.querySelectorAll(".prompt-section"));
    input.addEventListener("input", () => {{
      const needle = input.value.trim().toLowerCase();
      for (const section of sections) {{
        section.hidden = needle && !section.dataset.prompt.includes(needle);
      }}
    }});
  </script>
</body>
</html>
"""


def wrap_label(text: str, width: int = 38, max_lines: int = 3) -> str:
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    return "\n".join(lines)


def read_thumbnail(path: Path, thumb_size: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    return image


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered_text(draw, box: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=4, align="center")


def render_png_pages(
    png_dir: Path,
    styles: list[str],
    by_prompt: OrderedDict[str, dict[str, dict[str, str]]],
    prompts_per_page: int,
    thumb_size: int,
    dpi: int,
) -> list[Path]:
    from PIL import Image, ImageDraw

    if prompts_per_page < 1:
        raise ValueError("--prompts_per_page must be >= 1")

    png_dir.mkdir(parents=True, exist_ok=True)
    for old_page in png_dir.glob("page_*.png"):
        old_page.unlink()

    prompts = list(by_prompt.items())
    page_paths: list[Path] = []
    style_count = len(styles)
    margin = 18
    header_height = 42
    label_width = 360
    gap = 8
    cell_size = thumb_size
    cell_pitch = cell_size + gap
    page_width = margin * 2 + label_width + gap + style_count * cell_pitch
    header_font = load_font(16, bold=True)
    style_font = load_font(13, bold=True)
    prompt_font = load_font(12)
    small_font = load_font(12, bold=True)

    for page_idx, start in enumerate(range(0, len(prompts), prompts_per_page), start=1):
        page_prompts = prompts[start : start + prompts_per_page]
        page_height = margin * 2 + header_height + len(page_prompts) * cell_pitch
        sheet = Image.new("RGB", (page_width, page_height), "#f6f7f4")
        draw = ImageDraw.Draw(sheet)

        draw.text((margin, margin + 4), f"Batch Eval Style Grid - page {page_idx:03d}", fill="#1c211b", font=header_font)
        for col_idx, style in enumerate(styles):
            x = margin + label_width + gap + col_idx * cell_pitch
            draw_centered_text(draw, (x, margin + 8, x + cell_size, margin + header_height), style, style_font, "#2f6f61")

        for row_idx, (prompt, style_map) in enumerate(page_prompts):
            y = margin + header_height + row_idx * cell_pitch
            prompt_label = f"{start + row_idx + 1:03d}. {wrap_label(prompt, width=42, max_lines=5)}"
            draw.multiline_text(
                (margin, y + 8),
                prompt_label,
                fill="#1c211b",
                font=prompt_font,
                spacing=4,
            )
            for col_idx, style in enumerate(styles):
                x = margin + label_width + gap + col_idx * cell_pitch
                box = (x, y, x + cell_size, y + cell_size)
                draw.rectangle(box, fill="#ffffff", outline="#d9dfd5", width=1)

                record = style_map.get(style)
                if record is None:
                    draw_centered_text(draw, box, "missing", small_font, "#657066")
                    continue

                path = Path(record["generated"])
                if not path.exists():
                    draw_centered_text(draw, box, "not found", small_font, "#657066")
                    continue

                try:
                    image = read_thumbnail(path, thumb_size)
                except Exception:
                    draw_centered_text(draw, box, "load failed", small_font, "#657066")
                    continue
                paste_x = x + (cell_size - image.width) // 2
                paste_y = y + (cell_size - image.height) // 2
                sheet.paste(image, (paste_x, paste_y))

        page_path = png_dir / f"page_{page_idx:03d}.png"
        sheet.save(page_path, dpi=(dpi, dpi))
        page_paths.append(page_path)
        print(f"Wrote {page_path}")

    return page_paths


def main() -> None:
    args = parse_args()
    records, manifest = load_records(args.batch_manifest, args.include_all_statuses)
    styles, by_prompt = build_grid(
        records,
        args.generated_dir,
        args.style_dir,
        args.style_limit,
        args.max_prompts,
    )

    if "html" in args.formats:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        html_text = render_html(
            args.title,
            manifest,
            output,
            styles,
            by_prompt,
            args.expected_styles,
        )
        output.write_text(html_text, encoding="utf-8")
        print(f"Wrote {output}")
    if "png" in args.formats:
        try:
            render_png_pages(
                resolve_project_path(args.png_dir),
                styles,
                by_prompt,
                args.prompts_per_page,
                args.thumb_size,
                args.dpi,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "PIL":
                raise SystemExit(
                    "PNG output requires Pillow. Install it with `pip install pillow`, "
                    "or run with the project environment that has requirements.txt installed. "
                    "Use `--formats html` to write only the HTML view."
                ) from exc
            raise
    print(f"Prompts: {len(by_prompt)}")
    print(f"Styles: {len(styles)}")


if __name__ == "__main__":
    main()
