#!/usr/bin/env python3
"""
scripts/build_final_docx.py
---------------------------
Updates 'Copy of 24F-0807_CA_Project.docx' to integrate:
- Day 9 analysis and missing Task 3 metrics
- All 5 publication-ready plots (Figures 5 to 9)
- Section 8.5: Single-Query (BS=1) Tail Latency (P95 & P99)
- Section 9.2: Figure 6 training comparison
- Section 9.3: Held-Out CIFAR-10 Test Accuracy and BatchNorm diagnosis
- Section 10.1: Figure 7 resolution scaling
- Section 10.2: Figure 8 CPU profiler breakdown
- Section 11: Amdahl's Law bottleneck analysis and hardware selection matrix
- Section 12: Conclusion & architectural synthesis
- References: Retention of all 8 existing citations + 3 verified additions

Output is saved to '24F-0807_CA_Project_Final.docx' without modifying the original.
"""

import os
import sys
import csv
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

SRC_DOCX = "Copy of 24F-0807_CA_Project.docx"
DST_DOCX = "24F-0807_CA_Project_Final.docx"

def create_element_before(target_element, new_element):
    """Inserts new_element before target_element in the XML DOM."""
    target_element.addprevious(new_element)

def add_p_before(target_element, doc, text="", style='normal', space_before=6, space_after=6, line_spacing=1.15, align=None, bold=False, italic=False):
    """Creates a paragraph and moves it before target_element."""
    p = doc.add_paragraph()
    p.style = doc.styles[style] if style in doc.styles else doc.styles['Normal']
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(11)
        if bold:
            r.bold = True
        if italic:
            r.italic = True
    target_element.addprevious(p._element)
    return p

def add_heading_before(target_element, doc, text, level=2):
    """Creates a heading and moves it before target_element."""
    style_name = f'Heading {level}'
    p = doc.add_paragraph()
    p.style = doc.styles[style_name] if style_name in doc.styles else doc.styles['Normal']
    
    # Precise heading spacing
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
    r.font.name = 'Arial'
    r.font.size = Pt(size)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)
    
    target_element.addprevious(p._element)
    return p

def add_image_and_caption_before(target_element, doc, img_path, caption_text, width=Inches(6.2)):
    """Inserts a centered image followed by a centered caption before target_element."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
        
    # Image paragraph
    p_img = doc.add_paragraph()
    p_img.style = doc.styles['normal'] if 'normal' in doc.styles else doc.styles['Normal']
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(3)
    r_img = p_img.add_run()
    r_img.add_picture(img_path, width=width)
    target_element.addprevious(p_img._element)
    
    # Caption paragraph
    p_cap = doc.add_paragraph()
    p_cap.style = doc.styles['normal'] if 'normal' in doc.styles else doc.styles['Normal']
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = 'Arial'
    r_cap.font.size = Pt(10)
    r_cap.bold = True
    r_cap.italic = True
    r_cap.font.color.rgb = RGBColor(30, 30, 30)
    target_element.addprevious(p_cap._element)
    
    return p_img, p_cap

def add_table_before(target_element, doc, headers, rows_data, col_widths=None):
    """Creates a styled table and moves it before target_element."""
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Add subtle border XML
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    
    # Populate Header
    hdr_cells = tbl.rows[0].cells
    trPr0 = tbl.rows[0]._tr.get_or_add_trPr()
    trPr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    trPr0.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
    
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        if col_widths and i < len(col_widths):
            hdr_cells[i].width = col_widths[i]
        tcPr = hdr_cells[i]._tc.get_or_add_tcPr()
        tcMar = parse_xml(
            f'<w:tcMar {nsdecls("w")}><w:top w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/>'
            f'<w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
        )
        tcPr.append(tcMar)
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F1F3F4"/>')
        tcPr.append(shd)
        
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            r.font.name = 'Arial'
            r.font.size = Pt(9.5)
            r.bold = True
            r.font.color.rgb = RGBColor(0, 0, 0)
            
    # Populate Data
    for r_idx, row_values in enumerate(rows_data):
        row = tbl.rows[r_idx + 1]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        # Slight alternating row tint for readability
        fill_color = "FAFBFB" if r_idx % 2 == 1 else "FFFFFF"
        
        for c_idx, val in enumerate(row_values):
            cell = row.cells[c_idx]
            cell.text = str(val)
            if col_widths and c_idx < len(col_widths):
                cell.width = col_widths[c_idx]
            tcPr = cell._tc.get_or_add_tcPr()
            tcMar = parse_xml(
                f'<w:tcMar {nsdecls("w")}><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/>'
                f'<w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
            )
            tcPr.append(tcMar)
            if fill_color != "FFFFFF":
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
                tcPr.append(shd)
                
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.font.name = 'Arial'
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(30, 30, 30)
                # Bold the model or first column if appropriate
                if c_idx == 0:
                    r.bold = True
                    
    target_element.addprevious(tbl._tbl)
    return tbl

def remove_p(p):
    """Removes a paragraph from document tree."""
    parent = p._element.getparent()
    if parent is not None:
        parent.remove(p._element)

def main():
    print(f"Loading {SRC_DOCX}...")
    doc = docx.Document(SRC_DOCX)
    
    # ----------------------------------------------------
    # Verification: Verify raw CSVs exist and are non-empty
    # ----------------------------------------------------
    csv_paths = [
        "data/bs1_tail_cpu_gpu.csv",
        "data/bs1_tail_tpu.csv",
        "data/amdahl_analysis.csv",
        "data/training_cpu.csv",
        "data/training_gpu.csv",
        "data/training_tpu.csv"
    ]
    for p in csv_paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            raise FileNotFoundError(f"Missing or empty required CSV: {p}")
            
    # ----------------------------------------------------
    # Load and Parse BS1 Tail Latency Data
    # ----------------------------------------------------
    tail_rows = []
    # Read CPU and GPU
    with open("data/bs1_tail_cpu_gpu.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            model = r["Model"]
            dev = r["Device"]
            dev_str = "AMD EPYC CPU" if dev == "CPU" else ("Tesla T4 GPU" if dev == "GPU" else dev)
            prec = r["Precision"]
            med = float(r["BS1_Median_ms"])
            p95 = float(r["BS1_P95_ms"])
            p99 = float(r["BS1_P99_ms"])
            ratio = p99 / med
            tail_rows.append([
                model, dev_str, prec,
                f"{med:.3f} ms", f"{p95:.3f} ms", f"{p99:.3f} ms", f"{ratio:.2f}×"
            ])
            
    # Read TPU
    with open("data/bs1_tail_tpu.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            model = r["Model"]
            prec = r["Precision"]
            med = float(r["BS1_Median_ms"])
            p95 = float(r["BS1_P95_ms"])
            p99 = float(r["BS1_P99_ms"])
            ratio = p99 / med
            tail_rows.append([
                model, "Google Cloud TPU v5e", prec,
                f"{med:.3f} ms", f"{p95:.3f} ms", f"{p99:.3f} ms", f"{ratio:.2f}×"
            ])
            
    # ----------------------------------------------------
    # Load and Parse Amdahl Analysis Data
    # ----------------------------------------------------
    amdahl_rows = []
    with open("data/amdahl_analysis.csv", mode="r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            w_name = r["Workload"]
            dev = r["Device"]
            dev_str = "Tesla T4 GPU" if dev == "GPU" else ("Cloud TPU v5e" if dev == "TPU" else dev)
            prec = r["Precision"]
            batch_epoch = r["Batch_Size_or_Epoch"]
            compute_ms = float(r["Compute_ms"])
            e2e_ms = float(r["E2E_ms"])
            serial_ms = float(r["Serial_Overhead_ms"])
            serial_frac = float(r["Serial_Fraction_%"])
            ceiling = float(r["Speedup_Ceiling"])
            realized_2x = float(r["Speedup_2x_Compute"])
            
            amdahl_rows.append([
                w_name, dev_str, prec, batch_epoch,
                f"{compute_ms:.3f} ms", f"{e2e_ms:.3f} ms", f"{serial_ms:.3f} ms",
                f"{serial_frac:.2f}%", f"{ceiling:.2f}×", f"{realized_2x:.2f}×"
            ])

    # ----------------------------------------------------
    # 1. Section 8.1: Clean remove model_specs screenshot placeholder
    # ----------------------------------------------------
    for p in doc.paragraphs:
        if "[INSERT SCREENSHOT HERE: Colab output from model_setup.ipynb showing model_specs.csv]" in p.text:
            print("Found Section 8.1 screenshot placeholder; removing...")
            remove_p(p)
            break
            
    # ----------------------------------------------------
    # 2. Section 8.2: Replace throughput placeholder with Figure 5
    # ----------------------------------------------------
    for p in doc.paragraphs:
        if "[INSERT FIGURE HERE: Day 9 Throughput vs. Batch Size chart" in p.text:
            print("Found Section 8.2 throughput placeholder; inserting Figure 5...")
            add_image_and_caption_before(
                p._element, doc,
                "plots/throughput_vs_batch_size.png",
                "Figure 5: Steady-state inference throughput vs. batch size (1–32) across AMD EPYC CPU, NVIDIA Tesla T4 GPU, and Google Cloud TPU v5e for MobileViT-S, MobileNetV2, and 2-Layer LSTM."
            )
            remove_p(p)
            break
            
    # ----------------------------------------------------
    # 3. Section 8.5: Single-Query (BS=1) Tail Latency (P95 & P99)
    # Insert right before Section 9 heading
    # ----------------------------------------------------
    p_sec9 = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("9. Cross-Device Training Benchmark"):
            p_sec9 = p
            break
            
    if p_sec9 is None:
        raise ValueError("Could not find Section 9 heading paragraph!")
        
    print("Inserting Section 8.5 before Section 9...")
    add_heading_before(p_sec9._element, doc, "8.5. Single-Query (BS=1) Tail Latency (P95 & P99)", level=3)
    
    add_p_before(
        p_sec9._element, doc,
        "Deterministic execution is critical for real-time inference systems where Service Level Agreements (SLAs) are defined at P95 or P99 latencies. We conducted dedicated single-query (BS=1) latency profiling across 100 consecutive iterations (data/bs1_tail_cpu_gpu.csv, data/bs1_tail_tpu.csv)."
    )
    
    tail_headers = ["Model", "Execution Hardware", "Precision", "Median Latency", "P95 Latency", "P99 Latency", "P99 / Median"]
    tail_widths = [Inches(1.3), Inches(1.5), Inches(0.8), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9)]
    add_table_before(p_sec9._element, doc, tail_headers, tail_rows, col_widths=tail_widths)
    
    # Add spacing after table
    add_p_before(p_sec9._element, doc, "", space_before=2, space_after=4)
    
    # 3 concise bullets
    p_b1 = add_p_before(p_sec9._element, doc, "", space_before=3, space_after=3)
    r = p_b1.add_run("• Flat Determinism on Google Cloud TPU v5e: ")
    r.bold = True
    p_b1.add_run("Ahead-of-time (AOT) static graph compilation via XLA delivers extraordinary tail latency stability (1.07× P99/Median on MobileViT-S BF16, 20.98 ms P99 vs. 19.64 ms median; 1.17× on 2-Layer LSTM BF16, 2.29 ms P99). With static memory allocations and fixed execution graphs, the TPU executes with zero kernel launch jitter, zero dynamic memory allocation, and zero warp scheduler preemption.")
    
    p_b2 = add_p_before(p_sec9._element, doc, "", space_before=3, space_after=3)
    r = p_b2.add_run("• The GPU FP16 Batch Size 1 Dispatch Trap: ")
    r.bold = True
    p_b2.add_run("In stark contrast to large batches, GPU FP16 is substantially slower than FP32 at BS=1 across every evaluated model. On MobileViT-S, GPU FP32 delivers 8.62 ms median latency, whereas GPU FP16 takes 17.97 ms (2.08× slower). Tensor Cores require a minimum matrix tile dimension (16 × 16 × 16) to engage MMA instructions; at BS=1, tile starvation forces fallback routines while the CUDA runtime incurs dynamic FP32-to-FP16 buffer casting and kernel launch overhead.")
    
    p_b3 = add_p_before(p_sec9._element, doc, "", space_before=3, space_after=6)
    r = p_b3.add_run("• Dissecting CPU Jitter (Pinned Benchmarks vs. Multi-Tenant Cloud Spikes): ")
    r.bold = True
    p_b3.add_run("In dedicated pinned microbenchmarks, MobileViT-S on the AMD EPYC CPU exhibits a tight 1.17× tail expansion (97.11 ms median, 111.32 ms P95, 113.46 ms P99). However, during unpinned multi-tenant cloud execution (data/inference_cpu_gpu.csv), CPU tail latency spikes dramatically to 477.29 ms (4.9× tail expansion). In shared cloud hosts, co-located process bursts, OS core migration across CCX boundaries, and L3 cache thrashing destroy instruction locality, mandating strict core affinity pinning for production CPU inference SLAs.")

    # ----------------------------------------------------
    # 4. Section 9.2 & 9.3: Training Comparison Plot & Held-Out Test Eval
    # ----------------------------------------------------
    # Insert Figure 6 before Section 9.3 heading
    p_sec9_3 = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("9.3. Architectural Analysis of Training Bottlenecks"):
            p_sec9_3 = p
            break
            
    if p_sec9_3 is None:
        raise ValueError("Could not find Section 9.3 heading paragraph!")
        
    print("Inserting Figure 6 before Section 9.3...")
    add_image_and_caption_before(
        p_sec9_3._element, doc,
        "plots/training_comparison.png",
        "Figure 6: MobileViT-S CIFAR-10 training dynamics across CPU, GPU, and TPU (epoch time, first-step XLA compile shock, loss, and training accuracy)."
    )
    
    # Insert Held-Out CIFAR-10 Test Accuracy bullet at the end of Section 9.3 (before Section 10)
    p_sec10 = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("10. Input Resolution Scaling and CPU Operator Profiling"):
            p_sec10 = p
            break
            
    if p_sec10 is None:
        raise ValueError("Could not find Section 10 heading paragraph!")
        
    print("Appending Held-Out Test Eval bullet to Section 9.3...")
    p_eval = add_p_before(p_sec10._element, doc, "", space_before=3, space_after=6)
    r = p_eval.add_run("• Held-Out CIFAR-10 Test Accuracy (10,000 images, model.eval()): ")
    r.bold = True
    p_eval.add_run("Tesla T4 GPU achieved ")
    r_gpu = p_eval.add_run("76.46%")
    r_gpu.bold = True
    p_eval.add_run("; Cloud TPU v5e dropped to ")
    r_tpu = p_eval.add_run("24.82%")
    r_tpu.bold = True
    p_eval.add_run(" despite identical training convergence (0.3111 loss, 89.89% train acc), diagnosed as a ")
    r_pkg = p_eval.add_run("torch_tpu")
    r_pkg.font.name = "Courier New"
    p_eval.add_run(" (PrivateUse1) BatchNorm ")
    r_code = p_eval.add_run("running_mean/running_var")
    r_code.font.name = "Courier New"
    p_eval.add_run(" buffer-tracking failure; CPU held-out eval marked as not performed separately due to identical seeded convergence.")

    # ----------------------------------------------------
    # 5. Section 10.1: Resolution Scaling Plot (Figure 7)
    # ----------------------------------------------------
    for p in doc.paragraphs:
        if "[INSERT FIGURE HERE: Day 9 Resolution Scaling chart" in p.text:
            print("Found Section 10.1 resolution placeholder; inserting Figure 7...")
            add_image_and_caption_before(
                p._element, doc,
                "plots/resolution_scaling.png",
                "Figure 7: Input resolution scaling (128×128, 256×256, 384×384 at BS=16) comparing pure CNN (MobileNetV2) vs. hybrid Vision Transformer (MobileViT-S) across CPU, GPU, and TPU."
            )
            remove_p(p)
            break

    # ----------------------------------------------------
    # 6. Section 10.2: CPU Profiler Operator Breakdown (Figure 8)
    # Insert right below Table 6 (before paragraph 'Operator-Level Diagnosis:')
    # ----------------------------------------------------
    p_op_diag = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("Operator-Level Diagnosis:"):
            p_op_diag = p
            break
            
    if p_op_diag is None:
        raise ValueError("Could not find 'Operator-Level Diagnosis:' paragraph!")
        
    print("Inserting Figure 8 before Operator-Level Diagnosis...")
    add_image_and_caption_before(
        p_op_diag._element, doc,
        "plots/cpu_profiler_breakdown.png",
        "Figure 8: AMD EPYC CPU operator-level execution breakdown (Self CPU %) via torch.profiler for MobileNetV2 and MobileViT-S (256×256, BS=1)."
    )

    # ----------------------------------------------------
    # 7. Clean up empty placeholder paragraphs before References
    # ----------------------------------------------------
    p_refs = None
    for p in doc.paragraphs:
        if p.text.strip() == "References" and p.style.name.startswith("Heading"):
            p_refs = p
            break
            
    if p_refs is None:
        raise ValueError("Could not find 'References' heading paragraph!")
        
    # Inspect all paragraphs before p_refs back to the last non-empty paragraph of Section 10.2
    # In the original document, P238-P247 were empty. Remove them.
    curr_el = p_refs._element.getprevious()
    while curr_el is not None:
        # Check if curr_el is an empty paragraph
        tag = curr_el.tag.split("}")[-1]
        if tag == "p":
            text_val = "".join(curr_el.itertext()).strip()
            drawings = curr_el.xpath(".//w:drawing")
            if text_val == "" and len(drawings) == 0:
                prev_el = curr_el.getprevious()
                curr_el.getparent().remove(curr_el)
                curr_el = prev_el
                continue
        break
        
    # ----------------------------------------------------
    # 8. Insert Section 11 & Section 12 before References
    # ----------------------------------------------------
    print("Inserting Section 11: Amdahl's Law Bottleneck Analysis...")
    add_heading_before(p_refs._element, doc, "11. Amdahl's Law Bottleneck Analysis & Hardware Selection Guide", level=2)
    
    add_p_before(
        p_refs._element, doc,
        "Amdahl's Law dictates that the maximum theoretical speedup achievable by accelerating a portion of a workload is strictly bounded by the fraction of the workload that remains unaccelerated (serial overhead):"
    )
    
    p_eq1 = add_p_before(p_refs._element, doc, "", space_before=4, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p_eq1.add_run("Speedup = 1 / [f_serial + (1 - f_serial) / s]")
    r.bold = True
    
    add_p_before(
        p_refs._element, doc,
        "Where f_serial = (E2E - Compute) / E2E represents the serial fraction (host dispatch, PCIe transfer, data loader latency, framework runtime overhead), and (1 - f_serial) is the parallelizable compute fraction. As compute acceleration approaches infinity (s → ∞), the theoretical speedup ceiling collapses to:"
    )
    
    p_eq2 = add_p_before(p_refs._element, doc, "", space_before=4, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    r = p_eq2.add_run("Speedup_ceiling = 1 / f_serial")
    r.bold = True
    
    add_heading_before(p_refs._element, doc, "11.1. Empirical Amdahl's Law Analysis", level=3)
    
    # Figure 9
    add_image_and_caption_before(
        p_refs._element, doc,
        "plots/amdahl_bottleneck.png",
        "Figure 9: Amdahl's Law system bottleneck partition (compute vs. serial transfer/host overhead) and theoretical speedup ceilings across inference and training."
    )
    
    add_p_before(
        p_refs._element, doc,
        "Evaluating inference at BS=32 and training at Epoch 2 reveals striking differences in serial overhead fractions across architectures (data/amdahl_analysis.csv):"
    )
    
    amdahl_headers = [
        "Evaluated Workload", "Device", "Precision", "Batch / Epoch",
        "Compute Time", "End-to-End Time", "Serial Overhead",
        "Serial Fraction", "Speedup Ceiling", "Realized Speedup (2×)"
    ]
    amdahl_widths = [
        Inches(1.2), Inches(0.9), Inches(0.6), Inches(0.7),
        Inches(0.7), Inches(0.7), Inches(0.7),
        Inches(0.6), Inches(0.6), Inches(0.7)
    ]
    add_table_before(p_refs._element, doc, amdahl_headers, amdahl_rows, col_widths=amdahl_widths)
    
    # Spacing after table
    add_p_before(p_refs._element, doc, "", space_before=2, space_after=4)
    
    add_p_before(
        p_refs._element, doc,
        "Architectural Explanation of the TPU Amdahl Trap: Why does the Cloud TPU v5e encounter such severe Amdahl speedup limits despite possessing massive raw systolic compute power?",
        bold=True
    )
    
    p_trap1 = add_p_before(p_refs._element, doc, "", space_before=3, space_after=3)
    r = p_trap1.add_run("1. Lightweight Workload Dispatch Mismatch: ")
    r.bold = True
    p_trap1.add_run("On MobileNetV2 BF16 at BS=32, the TPU executes the entire forward pass in just 16.06 ms. However, total end-to-end latency is 37.90 ms, meaning 21.83 ms (57.61% of total execution time) is consumed outside the TPU accelerator core. This non-compute time is dominated by PyTorch host tensor packaging, Python interpreter runtime, and host-to-device transport across the virtualized Cloud TPU interconnect. Because f_serial = 57.61%, the theoretical speedup ceiling is capped at 1.74×. Even if a next-generation accelerator possessed infinite computational speed (s → ∞), end-to-end execution could never drop below 21.83 ms.")
    
    p_trap2 = add_p_before(p_refs._element, doc, "", space_before=3, space_after=6)
    r = p_trap2.add_run("2. GPU Low-Latency Advantage in PyTorch: ")
    r.bold = True
    p_trap2.add_run("On MobileViT-S FP16 (BS=32), the Tesla T4 GPU achieves an end-to-end time of 61.32 ms with 58.04 ms compute time, leaving only 3.28 ms of serial overhead (f_serial = 5.35%). Direct CUDA runtime driver integration eliminates the IPC and RPC marshalling overheads inherent to the virtualized TPU runtime, yielding a theoretical ceiling of 18.70× and achieving a 1.90× speedup under a hypothetical 2× compute increase.")
    
    add_heading_before(p_refs._element, doc, "11.2. Workload-to-Hardware Selection Matrix", level=3)
    
    add_p_before(
        p_refs._element, doc,
        "Based on our empirical telemetry across microbenchmarks, inference sweeps, and training dynamics, we synthesize the optimal operational selection criteria:"
    )
    
    matrix_headers = ["Production Workload Profile", "Recommended Hardware", "Key Architectural Rationale", "Practical Operational Constraints"]
    matrix_rows = [
        [
            "Real-Time Single-Stream Serving (BS=1, SLA < 15 ms)",
            "Tesla T4 GPU (FP32)",
            "Low PCIe dispatch overhead, fast CUDA kernel launch; avoids the GPU FP16 dispatch trap.",
            "Ensure FP32 precision is selected to avoid Tensor Core underfill penalties."
        ],
        [
            "Ultra-Low Tail Latency SLAs (BS=1, P99 Guarantee)",
            "Cloud TPU v5e (BF16)",
            "Flat deterministic execution (1.07× P99/median ratio, 20.98 ms P99) due to AOT XLA graph compilation.",
            "High cold-start penalty on first request; requires pre-warmed static graph."
        ],
        [
            "Edge / Low-Volume Offline Processing",
            "AMD EPYC CPU",
            "Zero accelerator provisioning cost; handles non-standard operators without fallback penalties.",
            "Unviable for large batch sizes or high-resolution attention; throughput strictly limited to ~5 samples/s."
        ],
        [
            "Dynamic Resolution / Interactive Vision",
            "Tesla T4 GPU (FP16)",
            "Dynamic tensor shaping handled without recompilation; superior GPU memory caching.",
            "Watch VRAM capacity at extreme resolutions (384×384 BS=16 approaches 1 GB)."
        ],
        [
            "High-Throughput Batch Serving (BS ≥ 32)",
            "Cloud TPU v5e (BF16)",
            "Outperforms T4 GPU in pure compute throughput (1,992.1 samples/s on MobileNetV2, 998.4 samples/s on MobileViT-S).",
            "Requires client-side request batching to amortize the 21–45 ms host serial overhead."
        ],
        [
            "Large-Scale Distributed Training",
            "Cloud TPU v5e Cluster",
            "High-bandwidth HBM2 (819 GB/s theoretical) and dedicated inter-chip interconnects (ICI).",
            "Must verify custom layer support (e.g., BatchNorm running statistics) in the torch_tpu runtime."
        ]
    ]
    matrix_widths = [Inches(1.5), Inches(1.3), Inches(1.8), Inches(1.8)]
    add_table_before(p_refs._element, doc, matrix_headers, matrix_rows, col_widths=matrix_widths)
    
    # Spacing after matrix table
    add_p_before(p_refs._element, doc, "", space_before=2, space_after=6)
    
    # ----------------------------------------------------
    # Insert Section 12: Conclusion & Architectural Synthesis
    # ----------------------------------------------------
    print("Inserting Section 12: Conclusion & Architectural Synthesis...")
    add_heading_before(p_refs._element, doc, "12. Conclusion & Architectural Synthesis", level=2)
    
    add_p_before(
        p_refs._element, doc,
        "This comprehensive benchmarking project provides empirical proof of fundamental computer architecture principles applied to deep learning accelerators:"
    )
    
    takeaways = [
        ("1. The Arithmetic Intensity Imperative: ", "General-purpose CPUs cannot escape the memory wall. With an empirical roofline ridge point at 1.8 FLOPs/byte, host CPUs spend the majority of execution cycles stalled on memory fetches during modern deep learning workloads."),
        ("2. The Precision Dichotomy: ", "Reducing precision from FP32 to FP16/BF16 is the single most effective hardware optimization for deep neural networks, yielding up to 11.3× higher compute throughput on GPU Tensor Cores with zero loss in categorical accuracy (100.0% class agreement on MobileViT-S). However, low precision is an accelerator optimization: at BS=1, GPU FP16 triggers a severe dispatch trap (2.08× slower than FP32) due to Tensor Core tile starvation."),
        ("3. Static Graph Compilation Tradeoffs: ", "The Google Cloud TPU v5e demonstrates the power of domain-specific compilation. By fixing shapes and compiling graphs via XLA ahead of time, the TPU achieves unparalleled tail latency stability (P99 within 7% of median). However, this introduces substantial compilation shocks (40.8s on first training step) and exposes users to silent runtime bugs (BatchNorm buffer tracking failure)."),
        ("4. The Amdahl Bottleneck in Modern Accelerators: ", "As accelerator compute times shrink into the sub-20 ms regime, host communication, DataLoader pipelines, and framework runtime dispatch rapidly become the dominant system bottleneck. On lightweight vision models, host serial overhead consumes up to 57.6% of end-to-end time on the TPU, strictly capping speedup at 1.74× regardless of future TPU compute scaling."),
        ("5. System Engineering Implication: ", "Hardware acceleration cannot be evaluated in a vacuum. Achieving true production speedups requires matching neural workload structures (batch size, resolution, and operator distribution) to the microarchitectural strengths and interconnect realities of the underlying compute platform.")
    ]
    
    for prefix, body in takeaways:
        p_t = add_p_before(p_refs._element, doc, "", space_before=3, space_after=3)
        r = p_t.add_run(prefix)
        r.bold = True
        p_t.add_run(body)
        
    add_p_before(p_refs._element, doc, "", space_before=2, space_after=6)

    # ----------------------------------------------------
    # 9. References: Append 3 APA citations
    # ----------------------------------------------------
    print("Appending new APA citations to References...")
    new_refs = [
        "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520.",
        "Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto.",
        "Howard, J. (2019). Imagenette: A Smaller Subset of 10 Easily Classified Classes from ImageNet. Fast.ai Documentation."
    ]
    
    for ref_text in new_refs:
        p_ref = doc.add_paragraph()
        p_ref.style = doc.styles['normal'] if 'normal' in doc.styles else doc.styles['Normal']
        p_ref.paragraph_format.space_before = Pt(3)
        p_ref.paragraph_format.space_after = Pt(3)
        p_ref.paragraph_format.line_spacing = 1.15
        r = p_ref.add_run(ref_text)
        r.font.name = 'Arial'
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0, 0, 0)

    # ----------------------------------------------------
    # Save Output Document
    # ----------------------------------------------------
    print(f"Saving final document to {DST_DOCX}...")
    doc.save(DST_DOCX)
    print("Document successfully created and saved!")

if __name__ == "__main__":
    main()

