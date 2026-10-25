from src.reading_priority import reading_priority
from src.saturation import saturation_map

def student_brief(frame, labels, vectors, topic: str) -> str:
    priority = reading_priority(frame, vectors, labels)[:3]
    sm = saturation_map(frame, labels, vectors)
    lines = ["=" * 64, "RESEARCHLENS - YOUR CORPUS IN ONE SCREEN", f"Topic: {topic}", "=" * 64, "", "READ THESE FIRST:"]
    for i, r in enumerate(priority, 1):
        lines.append(f"   {i}. [{r['paper_id']}] {r['title']}")
    lines += ["", "STRONG ON:"]
    for r in sm["saturated"]:
        lines.append(f"   - {r['label']} ({r['papers']} papers)")
    lines += ["", "THIN / EMERGING:"]
    for r in sm["emerging"]:
        lines.append(f"   - {r['label']} ({r['papers']} paper)")
    lines += ["", "SILENT KEYWORDS:"]
    for r in sm["silent"][:6]:
        lines.append(f"   - '{r['keyword']}'")
    lines += ["", "All findings are corpus-internal.", "=" * 64]
    return "\n".join(lines)
