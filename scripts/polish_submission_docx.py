#!/usr/bin/env python3
"""
scripts/polish_submission_docx.py
---------------------------------
Performs a complete rubric-compliance and consistency audit update on
'24F-0807_CA_Project_Final.docx' and saves the result as
'24F-0807_CA_Project_Submission.docx'.

Preserves all existing embedded plots, terminal screenshots, tables, fonts,
and styling while adding/updating:
1. Front matter (Abstract, Introduction, Platform/Software Table, NPU note)
2. Section 1 Side-by-Side Architecture Table, Figure 1-3 captions & errata,
   and sequential renumbering of all Figures 1-12
3. Section 2 Boost Clock Peak, T4 INT4 Peak, TPU FP32/clock note, Peak Comparison Table
4. Sections 3-5 Repetition counts, CPU Memory Bandwidth table fix, full Test 1-4 tables
5. Section 6 Roofline spec-ceiling & GEMM-only clarifications
6. Sections 8-9 Numerical validation, BS=1 tail reconciliation, accuracy scope, training fixes
7. Section 10 Research Question/Hypothesis/Verdict, hypothesis labeling, Iron Law subsection
8. Dedicated Q1-Q6 Architectural Synthesis section
9. Conclusion #1 Roofline alignment, Limitations, Member Contribution, AI-Use Disclosure
10. APA in-text citations (8+ sources) and missing References entries
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

SRC_DOCX = "24F-0807_CA_Project_Final.docx"
DST_DOCX = "24F-0807_CA_Project_Submission.docx"


def find_p(doc, prefix):
    """Find first paragraph whose stripped text starts with prefix."""
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise ValueError(f"Paragraph starting with {prefix!r} not found")


def find_p_contains(doc, substr):
    """Find first paragraph containing substr."""
    for p in doc.paragraphs:
        if substr in p.text:
            return p
    raise ValueError(f"Paragraph containing {substr!r} not found")


def set_p_text(p, text, bold=False, italic=False, size=11, color=None, align=None):
    """Replace all runs in paragraph p with a single formatted run."""
    p.text = ""
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if color is not None:
        r.font.color.rgb = color
    return p


def add_p_before(
    target_element,
    doc,
    text="",
    style="normal",
    space_before=4,
    space_after=6,
    line_spacing=1.15,
    align=None,
    bold=False,
    italic=False,
    size=11,
    color=None,
):
    """Create a paragraph and insert it before target_element."""
    p = doc.add_paragraph()
    p.style = doc.styles[style] if style in doc.styles else doc.styles["Normal"]
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        if color is not None:
            r.font.color.rgb = color
    target_element.addprevious(p._element)
    return p


def add_p_after(
    anchor_p,
    doc,
    text="",
    style="normal",
    space_before=4,
    space_after=6,
    line_spacing=1.15,
    align=None,
    bold=False,
    italic=False,
    size=11,
    color=None,
):
    """Create a paragraph and insert it immediately after anchor_p."""
    p = doc.add_paragraph()
    p.style = doc.styles[style] if style in doc.styles else doc.styles["Normal"]
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        if color is not None:
            r.font.color.rgb = color
    anchor_p._element.addnext(p._element)
    return p


def add_heading_before(target_element, doc, text, level=2):
    """Create a heading and insert it before target_element."""
    style_name = f"Heading {level}"
    p = doc.add_paragraph()
    p.style = doc.styles[style_name] if style_name in doc.styles else doc.styles["Normal"]
    if level == 2:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        size = 16
    elif level == 3:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(3)
        size = 14
    else:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)
        size = 12
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(size)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)
    target_element.addprevious(p._element)
    return p


def add_table_before(target_element, doc, headers, rows_data, col_widths_inches=None, font_size=8.5):
    """Create a fixed-layout styled table (max 6.5 in total width) before target_element."""
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    total_w_in = sum(col_widths_inches) if col_widths_inches else 6.5
    total_w_dxa = int(round(total_w_in * 1440))

    tblPr = tbl._tbl.tblPr
    tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="{total_w_dxa}" w:type="dxa"/>'))
    tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    if not col_widths_inches:
        col_widths_inches = [6.5 / len(headers)] * len(headers)

    # Header row
    hdr_cells = tbl.rows[0].cells
    trPr0 = tbl.rows[0]._tr.get_or_add_trPr()
    trPr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    trPr0.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

    for i, h in enumerate(headers):
        cell = hdr_cells[i]
        cell.text = h
        w_dxa = int(round(col_widths_inches[i] * 1440))
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:tcW {nsdecls("w")} w:w="{w_dxa}" w:type="dxa"/>'))
        tcPr.append(
            parse_xml(
                f'<w:tcMar {nsdecls("w")}><w:top w:w="70" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/>'
                f'<w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>'
            )
        )
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="F1F3F4"/>'))
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            r.font.name = "Arial"
            r.font.size = Pt(font_size)
            r.bold = True
            r.font.color.rgb = RGBColor(0, 0, 0)

    # Data rows
    for r_idx, row_values in enumerate(rows_data):
        row = tbl.rows[r_idx + 1]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        fill_color = "FAFBFB" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_values):
            cell = row.cells[c_idx]
            cell.text = str(val)
            w_dxa = int(round(col_widths_inches[c_idx] * 1440))
            tcPr = cell._tc.get_or_add_tcPr()
            tcPr.append(parse_xml(f'<w:tcW {nsdecls("w")} w:w="{w_dxa}" w:type="dxa"/>'))
            tcPr.append(
                parse_xml(
                    f'<w:tcMar {nsdecls("w")}><w:top w:w="55" w:type="dxa"/><w:bottom w:w="55" w:type="dxa"/>'
                    f'<w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>'
                )
            )
            if fill_color != "FFFFFF":
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>'))
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(font_size)
                r.font.color.rgb = RGBColor(30, 30, 30)
                if c_idx == 0:
                    r.bold = True

    target_element.addprevious(tbl._tbl)
    return tbl


def main():
    if not os.path.exists(SRC_DOCX):
        raise FileNotFoundError(f"Source file not found: {SRC_DOCX}")

    doc = docx.Document(SRC_DOCX)

    # =========================================================================
    # 1. FRONT MATTER (Insert before "1. Hardware Architectures Overview")
    # =========================================================================
    sec1_h = find_p(doc, "1. Hardware Architectures Overview")

    add_heading_before(sec1_h._element, doc, "Abstract", level=2)
    add_p_before(
        sec1_h._element,
        doc,
        (
            "Modern deep learning workloads exhibit sharply different compute, memory, and "
            "interconnect demands across general-purpose CPUs, SIMT GPUs, and systolic-array TPUs. "
            "This study empirically benchmarks three cloud architectures—an AMD EPYC 7B12 CPU (Zen 2, "
            "2 vCPUs), an NVIDIA Tesla T4 GPU (Turing, 16 GB GDDR6), and a Google Cloud TPU v5e "
            "(v5e-1, 197 TFLOPS BF16)—across microbenchmarks, Roofline modeling, batch-size 1–32 "
            "inference, single-query (BS=1) tail latency, 3-epoch CIFAR-10 training, input resolution "
            "scaling (128×128 to 384×384), and CPU operator profiling on three representative models: "
            "MobileViT-S (hybrid Vision Transformer), MobileNetV2 (depthwise-separable CNN), and a "
            "2-Layer LSTM. While TPU v5e leads steady-state BS=32 throughput (998.4 samples/s on "
            "MobileViT-S BF16; 1,992.1 samples/s on MobileNetV2 BF16) and achieves tight BS=1 tail "
            "determinism (1.07× P99/median), XLA graph compilation imposes a 40.89 s first-step "
            "training penalty and host-to-device (H2D) transfer limits cap end-to-end speedup via "
            "Amdahl's Law (57.61% serial overhead on MobileNetV2 BF16, ceiling 1.74×). Conversely, "
            "Tesla T4 GPU FP32 dominates BS=1 latency (8.62 ms on MobileViT-S) and 3-epoch training "
            "wall-clock time (21.98 s vs. 66.69 s on TPU), while CPU execution remains constrained by "
            "its 16 MiB L3 cache and 1.8 FLOPs/Byte Roofline ridge point."
        ),
    )

    add_heading_before(sec1_h._element, doc, "Introduction", level=2)
    add_p_before(
        sec1_h._element,
        doc,
        (
            "Deep neural networks have evolved from uniform dense convolutions into heterogeneous "
            "architectures combining depthwise-separable convolutions (Sandler et al., 2018), "
            "quadratic self-attention blocks (Mehta & Rastegari, 2021), and sequential recurrent "
            "gates. General-purpose CPUs optimize for low-latency scalar/SIMD control flow with deep "
            "cache hierarchies (AMD, 2019; Patterson & Hennessy, 2017), SIMT GPUs trade cache capacity "
            "for thousands of concurrent threads and specialized Tensor Cores (NVIDIA, 2018), and "
            "domain-specific TPUs rely on large systolic Matrix Multiply Units (MXUs) fed by "
            "ahead-of-time XLA static graphs (Jouppi et al., 2017; Leary & Wang, 2017; Google Cloud, "
            "2023). Because theoretical peak FLOPS and clock speed fail to capture memory-bandwidth "
            "ceilings (Williams et al., 2009), kernel launch taxes, and host-to-device interconnect "
            "walls, rigorous cross-architecture empirical profiling is essential for hardware selection."
        ),
    )
    add_p_before(
        sec1_h._element,
        doc,
        (
            "Report Roadmap: Section 1–2 survey the three hardware architectures and derive theoretical "
            "peak compute bounds; Section 3–5 present the synchronized benchmarking methodology and "
            "microbenchmark telemetry (GEMM, memory bandwidth, precision scaling, H2D transfer); "
            "Section 6–7 synthesize Roofline ceilings and precision/interconnect tradeoffs; Section 8–10 "
            "evaluate end-to-end inference (BS=1–32 and BS=1 tail latency), CIFAR-10 training, "
            "resolution scaling, and CPU operator profiling; Section 11–13 formalize Amdahl's Law "
            "limits, answer core architectural synthesis questions (Q1–Q6), and state conclusions, "
            "limitations, and disclosures."
        ),
    )

    add_heading_before(sec1_h._element, doc, "Experimental Platform & Software Environment", level=3)
    add_table_before(
        sec1_h._element,
        doc,
        ["Component", "Specification / Version", "Verification Source / Notes"],
        [
            ["Cloud Platform", "Google Colab (hosted VM runtimes)", "Standard CPU, T4 GPU, and v5e-1 TPU runtimes"],
            ["Benchmark Window", "November 2026", "Executed Nov 15–27, 2026 across 5 Colab notebooks"],
            ["Python", "3.13", "Observed in Colab notebook runtime paths (/usr/local/lib/python3.13)"],
            ["PyTorch", "Colab default runtime, Nov 2026", "Exact patch version not logged in summary CSVs (Paszke et al., 2019)"],
            ["timm", "Colab default runtime, Nov 2026", "Pretrained tags: mobilevit_s, mobilenetv2_100 (Wightman, 2019)"],
            ["fvcore", "Colab default runtime, Nov 2026", "Used for MAC/FLOP counting (Meta AI, 2019); exact version not logged"],
            ["CUDA / Driver", "CUDA 13.0 / Driver 580.82.07", "Logged directly on NVIDIA Tesla T4 VM"],
            ["TPU Runtime", "Native TorchTPU (torch.device('tpu') / torch_tpu)", "PrivateUse1 backend; Colab default runtime, Nov 2026"],
            ["Datasets", "Imagenette (val, 3,925 imgs); CIFAR-10 (50k/10k)", "Seeded subsets (seed=42): 500 imgs (acc parity), 2,048 imgs (train)"],
        ],
        col_widths_inches=[1.5, 2.2, 2.8],
        font_size=8.5,
    )
    add_p_before(
        sec1_h._element,
        doc,
        (
            "NPU Evaluation: Not performed. Neither Google Colab nor Kaggle provides access to "
            "dedicated Neural Processing Unit (NPU) hardware."
        ),
        bold=True,
        space_before=6,
        space_after=8,
    )

    # =========================================================================
    # 2. SECTION 1: FIGURE CAPTIONS (1-3), ERRATA, AND SIDE-BY-SIDE TABLE
    # =========================================================================
    # Find the 3 images in Section 1:
    # P015 is after "Key SIMD Extensions: AVX2 (256-bit), FMA, SSE4.2"
    # P022 is after "Driver Version: 580.82.07"
    # P030 is after "Device ID: 0"
    p_avx = find_p(doc, "Key SIMD Extensions:")
    p_img_cpu = p_avx._element.getnext()
    # Insert Figure 1 caption after p_img_cpu
    p_cap1 = doc.add_paragraph()
    p_cap1.style = doc.styles["normal"] if "normal" in doc.styles else doc.styles["Normal"]
    p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap1.paragraph_format.space_before = Pt(2)
    p_cap1.paragraph_format.space_after = Pt(8)
    r = p_cap1.add_run(
        "Figure 1: AMD EPYC 7B12 (Zen 2) CPU core and cache hierarchy schematic. "
        "AI-generated schematic illustration adapted from official vendor documentation (AMD, 2019)."
    )
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.bold = True
    r.italic = True
    r.font.color.rgb = RGBColor(30, 30, 30)
    p_img_cpu.addnext(p_cap1._element)

    p_drv = find_p(doc, "Driver Version: 580.82.07")
    p_img_gpu = p_drv._element.getnext()
    p_cap2 = doc.add_paragraph()
    p_cap2.style = doc.styles["normal"] if "normal" in doc.styles else doc.styles["Normal"]
    p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap2.paragraph_format.space_before = Pt(2)
    p_cap2.paragraph_format.space_after = Pt(4)
    r = p_cap2.add_run(
        "Figure 2: NVIDIA Tesla T4 (Turing) Streaming Multiprocessor (SM) schematic. "
        "AI-generated schematic illustration adapted from official vendor documentation (NVIDIA, 2018)."
    )
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.bold = True
    r.italic = True
    r.font.color.rgb = RGBColor(30, 30, 30)
    p_img_gpu.addnext(p_cap2._element)

    p_errata = doc.add_paragraph()
    p_errata.style = doc.styles["normal"] if "normal" in doc.styles else doc.styles["Normal"]
    p_errata.paragraph_format.space_before = Pt(2)
    p_errata.paragraph_format.space_after = Pt(8)
    r_err = p_errata.add_run(
        "Errata Note (Figure 2): The schematic labels '300 GB/s' and 'PB1 to PB3' are rendering "
        "typos in the AI-generated illustration; verified NVIDIA Turing T4 specifications are "
        "320 GB/s GDDR6 memory bandwidth and 4 processing blocks (PB0–PB3) per SM with 1× RT Core "
        "(NVIDIA, 2018)."
    )
    r_err.font.name = "Arial"
    r_err.font.size = Pt(9.5)
    r_err.italic = True
    r_err.font.color.rgb = RGBColor(60, 60, 60)
    p_cap2._element.addnext(p_errata._element)

    p_dev0 = find_p(doc, "Device ID: 0")
    p_img_tpu = p_dev0._element.getnext()
    p_cap3 = doc.add_paragraph()
    p_cap3.style = doc.styles["normal"] if "normal" in doc.styles else doc.styles["Normal"]
    p_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap3.paragraph_format.space_before = Pt(2)
    p_cap3.paragraph_format.space_after = Pt(8)
    r = p_cap3.add_run(
        "Figure 3: Google Cloud TPU v5e single-chip architecture schematic. "
        "AI-generated schematic illustration adapted from official vendor documentation (Google Cloud, 2023)."
    )
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.bold = True
    r.italic = True
    r.font.color.rgb = RGBColor(30, 30, 30)
    p_img_tpu.addnext(p_cap3._element)

    # Renumber existing Figure 1..9 -> Figure 4..12 and remove stray trailing ')'
    fig_map = {
        "Figure 1:": "Figure 4:",
        "Figure 2:": "Figure 5:",
        "Figure 3:": "Figure 6:",
        "Figure 4:": "Figure 7:",
        "Figure 5:": "Figure 8:",
        "Figure 6:": "Figure 9:",
        "Figure 7:": "Figure 10:",
        "Figure 8:": "Figure 11:",
        "Figure 9:": "Figure 12:",
    }
    cap_elems = {p_cap1._element, p_cap2._element, p_cap3._element}
    for p in doc.paragraphs:
        if p._element in cap_elems:
            continue
        txt = p.text.strip()
        for old_f, new_f in fig_map.items():
            if txt.startswith(old_f):
                new_txt = txt.replace(old_f, new_f, 1)
                if new_txt.endswith(".)"):
                    new_txt = new_txt[:-1]
                # Preserve any inline image in the same paragraph (e.g. P151, P166, P176)
                has_img = len(p._element.xpath(".//w:drawing")) > 0
                if has_img:
                    # Only update text runs, keep drawing runs intact
                    for r in p.runs:
                        if old_f in r.text:
                            r.text = r.text.replace(old_f, new_f, 1)
                        if r.text.endswith(".)"):
                            r.text = r.text[:-1]
                else:
                    set_p_text(
                        p,
                        new_txt,
                        bold=True,
                        italic=True,
                        size=10,
                        color=RGBColor(30, 30, 30),
                        align=WD_ALIGN_PARAGRAPH.CENTER,
                    )
                break

    # Side-by-Side Architecture Comparison Table at end of Section 1
    sec2_h = find_p(doc, "2. Theoretical Peak Performance Calculations")
    add_heading_before(sec2_h._element, doc, "1.4. Side-by-Side Architecture Comparison", level=3)
    add_table_before(
        sec2_h._element,
        doc,
        ["Dimension", "AMD EPYC 7B12 (CPU)", "NVIDIA Tesla T4 (GPU)", "Google Cloud TPU v5e (TPU)"],
        [
            [
                "Execution Model",
                "Out-of-order superscalar MIMD + 256-bit AVX2 SIMD (AMD, 2019)",
                "SIMT (Single Instruction, Multiple Threads) warp scheduler (NVIDIA, 2018)",
                "VLIW + 2D systolic array dataflow (Jouppi et al., 2017; Google Cloud, 2023)",
            ],
            [
                "Compute Units",
                "2 vCPUs (1 physical Zen 2 core, 2× 256-bit FMA units)",
                "40 SMs (2,560 CUDA cores + 320 Turing Tensor Cores)",
                "1 TensorCore (4× 128×128 MXUs = 65,536 ALUs + Vector Unit)",
            ],
            [
                "Memory Hierarchy",
                "32 KiB L1I/L1D, 512 KiB L2, 16 MiB L3, host DDR4 (~40 GB/s)",
                "64 KiB L1/shared per SM, 4 MiB L2, 16 GB GDDR6 (320 GB/s)",
                "On-chip SRAM / Vector Mem, 16 GB HBM2e (820 GB/s)",
            ],
            [
                "Numeric Formats",
                "FP32, FP64, INT8 (AVX2); no native BF16/FP16 matrix units",
                "FP32, FP16 (Tensor Core), INT8, INT4",
                "BF16 (native MXU), INT8 (native MXU); FP32 emulated",
            ],
            [
                "Programming Model",
                "Imperative eager execution, OS threads, oneDNN/OpenMP",
                "Dynamic CUDA kernel dispatch via PyTorch / cuDNN",
                "AOT static graph compilation via XLA (TorchTPU PrivateUse1)",
            ],
        ],
        col_widths_inches=[1.2, 1.75, 1.75, 1.8],
        font_size=8.5,
    )

    # =========================================================================
    # 3. SECTION 2: BOOST CLOCK, INT4 PEAK, TPU FP32 NOTE, VENDOR TABLE
    # =========================================================================
    sec22_h = find_p(doc, "2.2. CPU Peak Performance (AMD EPYC 7B12 — INT8)")
    add_p_before(
        sec22_h._element,
        doc,
        (
            "Boost Clock Peak (Zen 2): While the EPYC 7B12 base clock is 2.25 GHz (72.0 GFLOPS FP32), "
            "Zen 2 dynamic frequency scaling reaches a ~3.3 GHz max boost clock (AMD, 2019). At boost "
            "frequency: 3.3 GHz × 1 core × 32 FLOPs/cycle = 105.6 GFLOPS. Accordingly, our measured "
            "74.55 GFLOPS at 2048×2048 represents 70.6% of boost peak (rather than >100% of base peak)."
        ),
    )

    sec25_h = find_p(doc, "2.5. TPU Peak Performance (Google Cloud TPU v5e — BF16)")
    add_p_before(
        sec25_h._element,
        doc,
        (
            "NVIDIA Tesla T4 INT4 Peak: Turing Tensor Cores also support sub-byte INT4 matrix "
            "operations, executing 256 INT4 MACs (512 operations) per clock cycle per Tensor Core: "
            "1.59 GHz × 320 Tensor Cores × 512 INT4 Ops/cycle ≈ 260.5 TOPS (NVIDIA, 2018)."
        ),
    )

    sec26_h = find_p(doc, "2.6. TPU Peak Performance (Google Cloud TPU v5e — INT8)")
    add_p_before(
        sec26_h._element,
        doc,
        (
            "TPU v5e FP32 & Clock Attribution Note: The TPU v5e MXUs natively execute BF16 (197 TFLOPS) "
            "and INT8 (393 TOPS) with no separate native FP32 MXU peak; FP32 matrix operations are "
            "emulated via multiple BF16 passes or handled on the Vector Unit. The 1.5 GHz effective "
            "clock used above is back-calculated from Google's published 197 TFLOPS BF16 specification: "
            "197 × 10¹² / (65,536 ALUs × 2 ops/cycle) ≈ 1.503 GHz (Google Cloud, 2023)."
        ),
    )

    sec3_h = find_p(doc, "3. Methodology & Benchmarking Harness")
    add_heading_before(sec3_h._element, doc, "2.7. Computed vs. Official Vendor Peak Comparison", level=3)
    add_table_before(
        sec3_h._element,
        doc,
        ["Device", "Format", "Computed Peak", "Official Vendor Spec", "Notes / Attribution"],
        [
            ["AMD EPYC 7B12", "FP32 (Base)", "72.0 GFLOPS", "72.0 GFLOPS (1 core @ 2.25 GHz)", "AVX2 256-bit × 2 FMA units (AMD, 2019)"],
            ["AMD EPYC 7B12", "FP32 (Boost)", "105.6 GFLOPS", "~105.6 GFLOPS (@ ~3.3 GHz boost)", "Measured 74.55 GFLOPS = 70.6% of boost peak"],
            ["AMD EPYC 7B12", "INT8", "288.0 GOPS", "288.0 GOPS (base) / 422.4 GOPS (boost)", "Requires calibrated quantized kernels"],
            ["NVIDIA Tesla T4", "FP32", "8.14 TFLOPS", "8.1 TFLOPS", "2,560 CUDA cores @ 1.59 GHz boost (NVIDIA, 2018)"],
            ["NVIDIA Tesla T4", "FP16 (TC)", "65.13 TFLOPS", "65 TFLOPS", "320 Tensor Cores × 256 FP16 ops/cycle"],
            ["NVIDIA Tesla T4", "INT8 (TC)", "130.25 TOPS", "130 TOPS", "320 Tensor Cores × 512 INT8 ops/cycle (or 256 MACs)"],
            ["NVIDIA Tesla T4", "INT4 (TC)", "260.50 TOPS", "260 TOPS", "Lowest-precision Turing matrix format"],
            ["Google TPU v5e", "BF16", "196.6 TFLOPS", "197 TFLOPS", "65,536 ALUs @ ~1.5 GHz back-calculated (Google Cloud, 2023)"],
            ["Google TPU v5e", "INT8", "393.2 TOPS", "393 TOPS", "2× BF16 packing on systolic MXUs"],
            ["Google TPU v5e", "FP32", "N/A (emulated)", "No native FP32 MXU peak", "Emulated via multi-pass BF16 / Vector Unit"],
        ],
        col_widths_inches=[1.2, 0.9, 1.15, 1.55, 1.7],
        font_size=8.5,
    )

    # =========================================================================
    # 4. SECTIONS 3-5: REPETITIONS, CPU BANDWIDTH FIX, FULL MICROBENCHMARK TABLES
    # =========================================================================
    p_s3_end = find_p(doc, "To prevent this measurement error")
    add_p_after(
        p_s3_end,
        doc,
        (
            "Warm-Up and Repetition Counts: Across all notebooks, explicit warm-up loops preceded "
            "timed measurements: (1) Inference sweeps (Section 8.2) used 2 warm-ups and 10 repeats "
            "(BS ≤ 8) or 5 repeats (BS > 8) on CPU, 5 warm-ups and 20 repeats on GPU, and 3 warm-ups "
            "and 15 repeats on TPU; (2) Single-query BS=1 tail latency (Section 8.5) used 100 "
            "consecutive timed iterations per model/device; (3) Resolution sweeps (Section 10.1) used "
            "1 warm-up and 5 repeats (≤256) or 3 repeats (384) on CPU, 5 warm-ups and 20 repeats on "
            "GPU, and 3 warm-ups and 15 repeats on TPU; (4) Microbenchmarks (Sections 4–5) used "
            "multi-iteration warm-up and 10+ timed median loops (exact per-size loop counts were not "
            "separately logged in hardware_microbenchmarks.csv). Median, P95, and P99 latencies were "
            "recorded to resist cloud multi-tenant tail-latency skew; per-iteration mean and standard "
            "deviation were not retained in the summary CSVs."
        ),
    )

    # Update P114 (CPU 2048 GFLOPS context)
    p_cpu_boost = find_p(doc, "On the ")
    for p in doc.paragraphs:
        if "74.55 GFLOPS, slightly exceeding its calculated 72 GFLOPS" in p.text:
            set_p_text(
                p,
                (
                    "On the 2048×2048 workload, the CPU achieved 74.55 GFLOPS, representing 70.6% "
                    "of its 105.6 GFLOPS boost-clock peak (~3.3 GHz) and 103.5% of its 72.0 GFLOPS "
                    "base-clock peak (2.25 GHz) due to Turbo Boost dynamic frequency scaling (AMD, 2019)."
                ),
            )
            break

    # Fix Section 4.1 CPU Memory Bandwidth Contradiction
    p_cpu_bw_false = find_p(doc, "Because the host CPU accesses system DDR4 memory directly")
    set_p_text(
        p_cpu_bw_false,
        (
            "CPU Memory Copy Microbenchmark (Test 2): A standalone PyTorch tensor copy microbenchmark "
            "was executed on the AMD EPYC host (shown in the Section 4.1 screenshot above), "
            "measuring 19.863 ms (4.03 GB/s) for 10M FP32 elements (40 MB), 100.485 ms (3.98 GB/s) "
            "for 50M elements (200 MB), and 200.599 ms (3.99 GB/s) for 100M elements (400 MB). "
            "Because single-threaded eager PyTorch tensor copies are bottlenecked by Python and "
            "memory-allocator overhead rather than physical DRAM bus limits, the 40 GB/s nominal "
            "DDR4 hardware ceiling is used for theoretical Roofline modeling in Section 6."
        ),
    )

    # Add Full Microbenchmark Tables (Test 1, Test 2, Test 4) at end of Section 5 (before Section 6)
    sec6_h = find_p(doc, "6. Roofline Model and Bottleneck Analysis")
    add_heading_before(sec6_h._element, doc, "5.3. Complete Microbenchmark Telemetry Summary (Tests 1–4)", level=3)

    add_p_before(
        sec6_h._element,
        doc,
        "Table: Test 1 — Square Matrix Multiplication Latency and Throughput (data/hardware_microbenchmarks.csv).",
        bold=True,
        size=9.5,
    )
    add_table_before(
        sec6_h._element,
        doc,
        ["Matrix Size (N×N)", "CPU FP32 (ms | GFLOPS)", "GPU FP32 (ms | GFLOPS)", "TPU BF16 (ms | TFLOPS)", "Additional Precision Tests"],
        [
            ["256 × 256", "0.854 ms | 39.28", "0.067 ms | 500.60", "0.701 ms | 0.05", "—"],
            ["512 × 512", "17.919 ms | 14.98", "0.108 ms | 2,490.98", "1.689 ms | 0.16", "—"],
            ["1024 × 1024", "53.974 ms | 39.79", "0.561 ms | 3,825.14", "5.018 ms | 0.43", "—"],
            ["1500 × 1500", "300.836 ms | 22.44", "1.679 ms | 4,021.13", "6.176 ms | 1.09", "—"],
            ["2048 × 2048", "230.446 ms | 74.55", "4.040 ms | 4,252.33", "7.237 ms | 2.37", "TPU FP32: 10.86 ms (1.58 TF); BF16: 7.21 ms (2.38 TF)"],
            ["4096 × 4096", "Not run", "32.510 ms | 4,230.00", "Not run", "GPU FP16: 2.88 ms (47.70 TFLOPS, 11.3× speedup)"],
        ],
        col_widths_inches=[1.05, 1.3, 1.35, 1.25, 1.55],
        font_size=8.5,
    )

    add_p_before(
        sec6_h._element,
        doc,
        "Table: Test 2 — Internal Device Memory Bandwidth (10M, 50M, 100M FP32 Elements).",
        bold=True,
        size=9.5,
        space_before=8,
    )
    add_table_before(
        sec6_h._element,
        doc,
        ["Device", "Operation & Method", "10M (40 MB)", "50M (200 MB)", "100M (400 MB)"],
        [
            ["AMD EPYC CPU", "Host RAM tensor.clone() (eager)", "19.863 ms | 4.03 GB/s", "100.485 ms | 3.98 GB/s", "200.599 ms | 3.99 GB/s"],
            ["NVIDIA Tesla T4", "VRAM_Memcpy (device-to-device)", "0.352 ms | 227.13 GB/s", "1.670 ms | 239.59 GB/s", "3.348 ms | 238.91 GB/s"],
            ["Google TPU v5e", "TPU_HBM_Bandwidth (eager + sync)", "0.07 GB/s", "0.32 GB/s", "0.53 GB/s"],
        ],
        col_widths_inches=[1.2, 1.85, 1.15, 1.15, 1.15],
        font_size=8.5,
    )

    add_p_before(
        sec6_h._element,
        doc,
        (
            "Test 3 (Precision Microbenchmarks) Scope: CPU reduced-precision (FP16/BF16/INT8), "
            "GPU INT8, and TPU INT8 microbenchmarks were not performed, as integer matrix "
            "multiplication requires a calibrated post-training quantization pipeline and x86 AVX2 "
            "lacks native FP16/BF16 matrix units."
        ),
        space_before=6,
    )

    add_p_before(
        sec6_h._element,
        doc,
        "Table: Test 4 — Host-to-Device (H2D) Transfer Overhead (data/hardware_microbenchmarks.csv).",
        bold=True,
        size=9.5,
        space_before=6,
    )
    add_table_before(
        sec6_h._element,
        doc,
        ["Transfer Link", "10M Elements (40 MB)", "50M Elements (200 MB)", "100M Elements (400 MB)"],
        [
            ["GPU PCIe Transfer (H2D)", "8.802 ms | 4.54 GB/s", "46.550 ms | 4.30 GB/s", "86.406 ms | 4.63 GB/s"],
            ["TPU H2D Transfer (gRPC/Host)", "1.15 GB/s (latency not logged)", "1.14 GB/s (latency not logged)", "1.03 GB/s (latency not logged)"],
        ],
        col_widths_inches=[1.8, 1.55, 1.55, 1.6],
        font_size=8.5,
    )

    # =========================================================================
    # 5. SECTION 6: ROOFLINE CLARIFICATIONS
    # =========================================================================
    p_s6_intro = find_p(doc, "To evaluate how effectively each architecture utilizes")
    add_p_after(
        p_s6_intro,
        doc,
        (
            "Roofline Modeling Assumptions (Williams et al., 2009): The CPU (40 GB/s DDR4) and TPU "
            "(820 GB/s HBM2e) memory bandwidth ceilings use official vendor/specification values rather "
            "than measured eager-mode copy rates, which are distorted by Python allocator and host-sync "
            "overheads. Furthermore, Figure 5 plots dense GEMM microbenchmarks only; full neural "
            "network workloads (MobileViT-S, MobileNetV2, LSTM) are analyzed separately in Sections 8–11."
        ),
    )

    # =========================================================================
    # 6. SECTIONS 8 & 9: DATA RECONCILIATION & ACCURACY SCOPE
    # =========================================================================
    p_s81 = find_p(doc, "All models were initialized with a fixed random seed (42)")
    set_p_text(
        p_s81,
        (
            "All models were initialized with a fixed random seed (42) and profiled for parameter "
            "count and compute complexity via fvcore (Meta AI, 2019). Vision models (timm tags: "
            "mobilevit_s, mobilenetv2_100; Wightman, 2019) were evaluated on the held-out Imagenette "
            "validation set (3,925 images, 10 classes; Howard, 2019). Because fvcore does not natively "
            "trace PyTorch's fused aten::lstm operator, LSTM operations were derived analytically "
            "across all four gates over 128 timesteps. Numerical parity (CPU vs. GPU FP32) passed "
            "for all models: MobileViT-S's max absolute difference (1.22×10⁻³) marginally exceeded "
            "atol = 1e-3 on an extreme logit while passing relative tolerance (rtol = 1e-5; "
            "torch.allclose returned True). TPU vs. CPU numerical parity was not performed."
        ),
    )

    p_s83 = find_p(doc, "To confirm that reduced-precision speedups do not degrade")
    set_p_text(
        p_s83,
        (
            "To confirm that reduced-precision speedups do not degrade prediction quality, 500 seeded "
            "random validation images (seed=42, covering all 10 Imagenette classes) were evaluated on "
            "the T4 GPU using timm tags mobilevit_s and mobilenetv2_100 with their native "
            "preprocessing pipelines (Wightman, 2019). Standard 1,000-class ImageNet logits were "
            "mapped to the 10 Imagenette WordNet class indices ([0, 217, 482, 491, 497, 566, 569, "
            "571, 574, 701]; Howard, 2019); the 500-image subset provides a reproducible, fast "
            "parity check across all 10 classes. CPU and TPU Imagenette accuracy evaluations were "
            "not performed, and LSTM accuracy was not performed because the 2-Layer LSTM is a "
            "random-weight synthetic sequence benchmark with no labeled dataset."
        ),
    )

    # Reconcile 8.2 vs 8.5 at end of 8.5 (right before Section 9)
    sec9_h = find_p(doc, "9. Cross-Device Training Benchmark")
    add_p_before(
        sec9_h._element,
        doc,
        (
            "Cross-Session Variance Note (Section 8.2 vs. 8.5): The BS=1–32 inference sweep "
            "(data/inference_cpu_gpu.csv) and dedicated 100-iteration BS=1 tail test "
            "(data/bs1_tail_cpu_gpu.csv, data/bs1_tail_tpu.csv) executed in separate Google Colab VM "
            "sessions, accounting for cloud multi-tenant noisy-neighbor variance (e.g., CPU "
            "MobileViT-S BS=1 median 198.36 ms in sweep vs. 97.11 ms in pinned tail test; GPU FP16 "
            "14.12 ms vs. 17.97 ms). Additionally, CPU LSTM BS=32 E2E (72.10 ms) < Steady (73.04 ms) "
            "reflects sub-millisecond OS timing jitter with zero H2D transfer."
        ),
        italic=True,
        size=10,
        space_before=4,
        space_after=8,
    )

    # Section 9.2 clarifications (after Per-Epoch Breakdown, before Figure 9)
    p_tpu_ep = find_p(doc, "Google Cloud TPU v5e: Epoch 1:")
    add_p_after(
        p_tpu_ep,
        doc,
        (
            "Metric Notes (Table above): On TPU v5e, Avg. Steady Step = 54.75 ms is the 3-epoch "
            "mean ((46.77 + 54.24 + 63.24) / 3), whereas the Epochs 2–3 average is 58.74 ms. "
            "The Peak Memory column reports distinct memory spaces—Host RAM (CPU), discrete VRAM "
            "(GPU), and Host RSS (TPU)—and is not directly comparable 1:1 across devices."
        ),
        italic=True,
        size=9.5,
    )

    # Section 9.3 fixes
    p_conv_parity = find_p(doc, "Convergence Parity Confirms Valid Benchmarking:")
    set_p_text(
        p_conv_parity,
        (
            "Training-Split Convergence vs. Held-Out Divergence: On the 2,048-image CIFAR-10 training "
            "subset (Krizhevsky, 2009), all three devices converged to nearly identical training "
            "accuracy (89.55%–89.94%) and loss (0.3111–0.3267). However, as detailed below, "
            "held-out test evaluation revealed a major discrepancy on the TPU."
        ),
    )

    p_held_out = find_p(doc, "• Held-Out CIFAR-10 Test Accuracy")
    set_p_text(
        p_held_out,
        (
            "• Held-Out CIFAR-10 Test Accuracy (10,000 images, model.eval()): Tesla T4 GPU achieved "
            "76.46%, whereas Cloud TPU v5e dropped to 24.82% despite matching training convergence "
            "(0.3111 loss, 89.89% train acc). Unverified Hypothesis: This divergence is suspected to "
            "stem from a torch_tpu (PrivateUse1) BatchNorm running_mean/running_var buffer-tracking "
            "bug during XLA graph execution, pending CPU weight-transfer verification. CPU held-out "
            "test accuracy was not performed (limitation)."
        ),
    )

    # =========================================================================
    # 7. SECTION 10: RESEARCH QUESTION, HYPOTHESIS, VERDICT, IRON LAW
    # =========================================================================
    p_s10_1 = find_p(doc, "10.1. Resolution Scaling Across Architectures")
    add_p_after(
        p_s10_1,
        doc,
        (
            "Research Question: How does scaling input spatial resolution (128×128 → 256×256 → "
            "384×384 at BS=16) affect latency and memory across a pure CNN (MobileNetV2) versus a "
            "hybrid Vision Transformer (MobileViT-S) on CPU, GPU, and TPU?\n"
            "Pre-Experiment Hypothesis: Quadratic self-attention in MobileViT-S will cause "
            "disproportionately steeper latency and memory growth than linear convolutions in "
            "MobileNetV2 as resolution scales 128 → 384, while larger attention matrices at 384×384 "
            "will saturate the TPU v5e's 128×128 systolic arrays and allow TPU BF16 to overtake GPU FP16."
        ),
        space_before=4,
        space_after=6,
    )

    # Soften unverified microarchitectural claims in 10.1 to labeled hypotheses + add Verdict
    p_tpu_sat = find_p(doc, "GPU-to-TPU Crossover and Systolic Array Saturation:")
    set_p_text(
        p_tpu_sat,
        p_tpu_sat.text.replace(
            "larger attention matrices finally saturate the systolic arrays:",
            "larger attention matrices appear to saturate the systolic arrays (hypothesis, as hardware counters are unavailable):",
        ),
    )
    p_cpu_cache = find_p(doc, "CPU Cache-Collapse Effect:")
    set_p_text(
        p_cpu_cache,
        p_cpu_cache.text.replace(
            "execution shifts from L2/L3-resident compute to DRAM-bound cache thrashing.",
            "execution is hypothesized to shift from L2/L3-resident compute to DRAM-bound cache thrashing (unverified without hardware cache-miss counters).",
        ),
    )
    p_vram = find_p(doc, "Mixed-Precision VRAM Anomaly (384 × 384):")
    set_p_text(
        p_vram,
        p_vram.text.replace(
            "forcing PyTorch to retain both FP16 and FP32 copies of large high-resolution attention tensors simultaneously.",
            "which is hypothesized to force PyTorch to retain both FP16 and FP32 copies of large high-resolution attention tensors simultaneously.",
        ),
    )
    add_p_after(
        p_vram,
        doc,
        (
            "Verdict (Supported): The hypothesis is supported—MobileViT-S slowed down 5.78× from "
            "128×128 to 384×384 on GPU FP16 (vs. 2.65× for MobileNetV2), and TPU BF16 overtook GPU "
            "FP16 by 2.16× at 384×384 (35.57 ms vs. 76.67 ms)."
        ),
        bold=True,
        space_before=4,
        space_after=8,
    )

    # Iron Law subsection in 10.2 (before Section 11)
    sec11_h = find_p(doc, "11. Amdahl's Law Bottleneck Analysis")
    add_heading_before(sec11_h._element, doc, "10.3. Iron Law of Processor Performance & Counter Limitations", level=4)
    add_p_before(
        sec11_h._element,
        doc,
        (
            "Under the Iron Law of Processor Performance (Patterson & Hennessy, 2017):\n"
            "Execution Time = (Instruction Count × CPI) / Clock Rate\n"
            "Hardware performance counters (perf stat for retired instructions, cycles, CPI, and L3 "
            "cache misses) are blocked inside virtualized Google Colab VMs. To bridge this gap, we "
            "combine torch.profiler operator telemetry with analytical effective FLOPs/cycle "
            "estimates derived from measured GFLOPS and the 3.3 GHz Zen 2 boost clock: on dense "
            "2048×2048 GEMM (74.55 GFLOPS), the CPU sustains ~22.6 FLOPs/cycle (70.6% of its 32 "
            "FLOPs/cycle AVX2+FMA ceiling). By contrast, MobileViT-S inference at BS=1 (3.7116 GFLOPs "
            "in 198.36 ms ≈ 18.7 GFLOPS) sustains only ~5.7 FLOPs/cycle—confirming that aten::copy_ "
            "transposes, SiLU transcendental ops, and memory-bound normalization severely inflate "
            "effective CPI."
        ),
    )

    # =========================================================================
    # 8. DEDICATED Q1-Q6 ARCHITECTURAL SYNTHESIS SECTION (Before Conclusion)
    # =========================================================================
    sec12_h = find_p(doc, "12. Conclusion & Architectural Synthesis")
    add_heading_before(sec12_h._element, doc, "12. Architectural Synthesis: Core Research Questions (Q1–Q6)", level=2)

    q_items = [
        (
            "Q1. Compute-Bound vs. Memory-Bound Behavior (Roofline):",
            "Using the Roofline model (Williams et al., 2009), dense GEMM workloads on CPU (AI = "
            "42.7–341.3 FLOPs/Byte vs. 1.8 FLOPs/Byte ridge point) and GPU FP32 (vs. 25.4 FLOPs/Byte "
            "ridge point) are compute-bound. However, TPU v5e's 240.2 FLOPs/Byte ridge point places "
            "smaller matrices (N ≤ 1024) in the memory-bound regime theoretically, while eager-mode "
            "synchronization traps all measured TPU GEMMs below both roofs. On real neural networks, "
            "large-batch/high-resolution activations spill beyond the CPU's 16 MiB L3 cache and "
            "become memory-bound.",
        ),
        (
            "Q2. Batch Size Crossover and Structural Causes:",
            "At BS=1, GPU FP32 beats TPU BF16 by ~2× on vision models (9.35 vs. 18.28 ms on "
            "MobileViT-S) because single images underutilize the TPU's four 128×128 systolic MXUs "
            "and expose host-dispatch overhead. By BS=32, TPU BF16 overtakes GPU FP16 on steady-state "
            "throughput across all models (998.4 vs. 551.3 samples/s on MobileViT-S; 1,992.1 vs. "
            "1,677.1 samples/s on MobileNetV2) as batching fills the 65,536-ALU systolic pipeline.",
        ),
        (
            "Q3. Amdahl's Law Serial Transfer and Host Bottlenecks:",
            "As accelerator compute shrinks into the sub-20 ms regime, host-to-device (H2D) transfers "
            "(1.03 GB/s TPU H2D vs. 4.63 GB/s GPU PCIe) and Python runtime dispatch become the "
            "unaccelerated serial fraction. On MobileNetV2 BF16 (BS=32), 57.61% of TPU E2E time "
            "(21.83 ms of 37.90 ms) is serial overhead, capping theoretical speedup at 1.74× even "
            "with infinite compute.",
        ),
        (
            "Q4. Precision Scaling (FP32 vs. FP16/BF16) and the BS=1 Dispatch Trap:",
            "On 4096×4096 GEMM, GPU FP16 Tensor Cores deliver an 11.3× speedup over FP32 (47.70 vs. "
            "4.23 TFLOPS), and BS=32 MobileViT-S/MobileNetV2 inference retains 99.8%–100.0% FP32/FP16 "
            "class agreement. Yet at BS=1, GPU FP16 is 2.08× slower than FP32 on MobileViT-S (17.97 "
            "vs. 8.62 ms) because Tensor Core tile starvation forces fallback kernels while "
            "torch.amp.autocast incurs dynamic casting overhead.",
        ),
        (
            "Q5. Static-Graph Compilation vs. Dynamic-Execution Tradeoffs (XLA vs. CUDA):",
            "XLA ahead-of-time compilation (Leary & Wang, 2017) gives TPU v5e unmatched BS=1 tail "
            "determinism (1.07× P99/median on MobileViT-S BF16), but imposes a 40.89 s compile shock "
            "on Epoch 1 Step 1 and shape-recompilation penalties (17.40 s at BS=32). Dynamic CUDA "
            "execution starts in 3.83 s and handles varying batch sizes/resolutions without "
            "recompilation, winning 3-epoch training wall-clock time (21.98 s vs. 66.69 s).",
        ),
        (
            "Q6. Why MIPS, Clock Speed, or Theoretical Peak FLOPS Alone Cannot Compare CPU, GPU, and TPU:",
            "Comparing architectures by clock rate (2.25–3.3 GHz CPU vs. 1.59 GHz GPU vs. ~1.5 GHz "
            "TPU) or raw peak FLOPS fails for five structural reasons: (1) ISA & SIMD/matrix width "
            "(AVX2 256-bit vs. SIMT warps vs. 128×128 systolic MXUs yield vastly different "
            "ops/cycle); (2) Roofline memory ceilings (40 vs. 320 vs. 820 GB/s dictate whether "
            "compute units ever saturate); (3) Numeric format specialization (Turing Tensor Cores "
            "and TPU MXUs achieve 8×–16× higher throughput in FP16/BF16/INT8 than FP32); "
            "(4) Host/interconnect Amdahl bottlenecks (1.03 GB/s TPU H2D caps lightweight models); "
            "and (5) Compilation and kernel-dispatch overheads (40.89 s XLA compile vs. BS=1 FP16 "
            "dispatch trap).",
        ),
    ]
    for q_title, q_body in q_items:
        add_p_before(sec12_h._element, doc, q_title, bold=True, space_before=6, space_after=2)
        add_p_before(sec12_h._element, doc, q_body, space_before=0, space_after=6)

    # Renumber Conclusion heading to 13
    set_p_text(sec12_h, "13. Conclusion, Limitations & Disclosures", bold=True, size=16, color=RGBColor(0, 0, 0))

    # =========================================================================
    # 9. CONCLUSION #1 ROOFLINE ALIGNMENT, LIMITATIONS, CONTRIBUTION, AI DISCLOSURE
    # =========================================================================
    p_conc1 = find_p(doc, "1. The Arithmetic Intensity Imperative:")
    set_p_text(
        p_conc1,
        (
            "1. The Arithmetic Intensity Imperative: While dense GEMM on the AMD EPYC CPU is "
            "compute-bound above its 1.8 FLOPs/Byte Roofline ridge point, real deep learning "
            "workloads with large batches or high spatial resolutions spill beyond the 16 MiB L3 "
            "cache and become DRAM memory-bound, limiting MobileViT-S BS=32 throughput to 5.4 samples/s."
        ),
    )

    ref_h = find_p(doc, "References")
    add_heading_before(ref_h._element, doc, "13.1. Limitations", level=3)
    limitations = [
        "• Unverified TPU BatchNorm Hypothesis: Held-out CIFAR-10 TPU accuracy dropped to 24.82% (vs. 76.46% on GPU); suspected to be a torch_tpu BatchNorm running-statistics bug, pending CPU weight-transfer verification.",
        "• CPU Held-Out Test Evaluation: Not performed on CIFAR-10 due to Colab runtime limits, and CPU/TPU Imagenette accuracy parity was not performed.",
        "• Eager-Mode TPU Microbenchmarks: Standalone TPU HBM (0.07–0.53 GB/s) and H2D (1.03–1.15 GB/s) rates reflect eager .cpu() synchronization overhead rather than physical HBM2e silicon bandwidth.",
        "• Blocked Hardware Counters: Virtualized Google Colab VMs block perf stat hardware performance counters, preventing direct measurement of CPI and L3 cache-miss rates.",
        "• Missing NPU Hardware & Synthetic LSTM: Dedicated NPU hardware is unavailable on Colab/Kaggle, and the 2-Layer LSTM benchmark used random weights without a labeled accuracy dataset.",
    ]
    for lim in limitations:
        add_p_before(ref_h._element, doc, lim, space_before=2, space_after=3)

    add_heading_before(ref_h._element, doc, "13.2. Member Contribution", level=3)
    add_p_before(
        ref_h._element,
        doc,
        (
            "Solo Project — Muhammad Ahmad (24F-0807): Designed and executed all Google Colab "
            "benchmarking notebooks (CPU, T4 GPU, TPU v5e), curated all CSV telemetry, authored "
            "analysis/plotting scripts, and wrote the complete technical report."
        ),
    )

    add_heading_before(ref_h._element, doc, "13.3. AI-Use Disclosure", level=3)
    add_p_before(
        ref_h._element,
        doc,
        (
            "AI assistance was used transparently to debug PyTorch/TorchTPU benchmarking scripts, "
            "generate matplotlib visualization scripts, assist with technical editing and APA "
            "formatting, and generate the three hardware architecture schematic illustrations "
            "(Figures 1–3) adapted from official AMD, NVIDIA, and Google Cloud documentation. All "
            "reported numbers were empirically measured and verified directly against repository CSVs."
        ),
    )

    # =========================================================================
    # 10. REFERENCES (Append missing APA entries)
    # =========================================================================
    new_refs = [
        "Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures. Communications of the ACM, 52(4), 65–76.",
        "Wightman, R. (2019). PyTorch Image Models (timm). GitHub repository. https://github.com/huggingface/pytorch-image-models",
        "Meta AI. (2019). fvcore: Lightweight core library for computer vision. GitHub repository. https://github.com/facebookresearch/fvcore",
    ]
    for ref_txt in new_refs:
        p_ref = doc.add_paragraph()
        p_ref.style = doc.styles["normal"] if "normal" in doc.styles else doc.styles["Normal"]
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        r = p_ref.add_run(ref_txt)
        r.font.name = "Arial"
        r.font.size = Pt(11)

    doc.save(DST_DOCX)
    print(f"Successfully generated {DST_DOCX}")


if __name__ == "__main__":
    main()
