"""The written explanation and reliability statement of one graded image, built only from that image's outputs and
the validation rates in the reliability calibration file.

Reliability rule (a wording aid for the demo, not a validated method): "review recommended" when any of four checks
fails -- Stage 1 rates the image Reject, the three models give different grades, a flip / rotation changes a model's
grade, or the ensemble entropy is above the threshold of the 20 % least certain validation images. The statement
quotes how often the grade was exact / within one grade on validation images in the same certainty band; those
validation images also selected the checkpoints, so the rates are optimistic.

Lines may contain **bold** markers.
"""
import json

from .config import GRADES
from .lesions import NAMES as LESION_NAMES

#: ICDR severity scale (Wilkinson et al., Ophthalmology 2003), as wording for the predicted grade.
ICDR = ("no apparent retinopathy",
        "microaneurysms only",
        "more than microaneurysms only, but less than severe non-proliferative DR",
        "severe non-proliferative DR: many intraretinal haemorrhages in all four quadrants, venous beading, or "
        "intraretinal microvascular abnormalities, without signs of proliferation",
        "proliferative DR: neovascularisation, or vitreous / preretinal haemorrhage")
LESION_FOUND = 1e-4                  # a lesion class counts as found when it covers at least 0.01 % of the frame
METHOD_NAMES = {"grad_cam": "Grad-CAM", "integrated_gradients": "Integrated Gradients", "occlusion": "Occlusion"}


def load_calibration(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reliability(quality_label, seed_grades, tta_grade_changes, entropy, calibration):
    """(review recommended, reasons, the validation exact / within-one rates of this image's certainty band).
    `calibration`: one arm's entry of the calibration file."""
    reasons = []
    if quality_label == "Reject":
        reasons.append("Stage 1 rated the image quality Reject")
    if len(set(seed_grades)) > 1:
        reasons.append(f"the three models give different grades {list(seed_grades)}")
    if tta_grade_changes:
        reasons.append("flipping or rotating the image changes the grade of at least one model")
    certain = entropy <= calibration["entropy_threshold"]
    if not certain:
        reasons.append(f"the ensemble is less certain (entropy {entropy:.2f}) than {100 * calibration['coverage']:.0f} % of the validation images")
    return bool(reasons), reasons, calibration["certain" if certain else "uncertain"]


def written_summary(grade, probabilities, lesion_area, overlaps, review, reasons, rates, calibration_note):
    """Plain-language lines. `lesion_area`: {class: share of the frame above 0.5}; `overlaps`: {method name: (share of
    the map's top 10 % on Stage-4 lesions, lesion share of the frame)}."""
    lines = [f"**Grade {grade} - {GRADES[grade]}** ({ICDR[grade]}). The ensemble gives this grade a probability of "
             f"{100 * probabilities[grade]:.0f} %" + ("" if grade == 0 and probabilities[0] > 0.5 else
                                                     f"; referable (grade 2 or worse): {'yes' if grade >= 2 else 'no'}") + "."]
    found = {k: v for k, v in lesion_area.items() if v >= LESION_FOUND}
    if found:
        lines.append("Stage 4 marks " + ", ".join(f"{LESION_NAMES[k]} ({100 * v:.2f} % of the frame)"
                                                 for k, v in sorted(found.items(), key=lambda kv: -kv[1])) + ".")
    else:
        lines.append("Stage 4 marks no lesions.")
    if grade >= 2 and not found:
        lines.append("The grade and the lesion maps disagree (a referable grade with no marked lesion): worth a closer look.")
    if grade == 0 and found and sum(found.values()) >= 10 * LESION_FOUND:
        lines.append("The grade and the lesion maps disagree (grade 0 with marked lesions): worth a closer look.")
    for name, (on, chance) in overlaps.items():
        lines.append(f"{name}: {100 * on:.0f} % of its most influential area lies on Stage-4 lesions "
                     f"(a map unrelated to the lesions would give about {100 * chance:.1f} %).")
    lines.append("**Review recommended**: " + "; ".join(reasons) + "." if review else
                 "**No reliability flag**: image quality acceptable, the three models agree, the grade is stable under "
                 "flips and rotations, and the certainty is in the upper 80 % of the validation images.")
    lines.append(f"On validation images in this certainty band the grade was exact in {100 * rates['exact']:.0f} % and within one "
                 f"grade in {100 * rates['within_one']:.0f} % ({rates['images']} images; {calibration_note}). "
                 "Images from other cameras or clinics than the APTOS and EyePACS validation sets can be less reliable than this.")
    lines.append("The grade comes from the RGB grader alone; the lesion maps and explanation maps are supporting views, "
                 "and overlap with lesions does not show that the grader detected them. Not a diagnosis.")
    return lines
