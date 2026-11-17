"""Human-readable names and pathogen facts for PlantVillage class folders.

Class folder names differ slightly between PlantVillage mirrors (GitHub raw/color vs. the
Kaggle re-uploads), so lookup works on a normalised string and keyword rules instead of
exact names.  Only the causal agent and its type are stored; management text is kept
generic on purpose, because chemical recommendations depend on crop, country and
regulation.
"""
import re

# (keywords that must ALL appear in the normalised class name) -> (pathogen, kind)
_RULES = [
    (("apple", "scab"), ("Venturia inaequalis", "fungal")),
    (("apple", "blackrot"), ("Botryosphaeria obtusa", "fungal")),
    (("apple", "cedar"), ("Gymnosporangium juniperi-virginianae", "fungal")),
    (("cherry", "powdery"), ("Podosphaera clandestina", "fungal")),
    (("corn", "cercospora"), ("Cercospora zeae-maydis", "fungal")),
    (("corn", "graylea"), ("Cercospora zeae-maydis", "fungal")),
    (("corn", "commonrust"), ("Puccinia sorghi", "fungal")),
    (("corn", "northern"), ("Exserohilum turcicum", "fungal")),
    (("grape", "blackrot"), ("Guignardia bidwellii", "fungal")),
    (("grape", "esca"), ("Fungal complex (e.g. Phaeomoniella chlamydospora)", "fungal")),
    (("grape", "isariopsis"), ("Pseudocercospora vitis", "fungal")),
    (("grape", "leafblight"), ("Pseudocercospora vitis", "fungal")),
    (("orange", "haunglongbing"), ("Candidatus Liberibacter asiaticus (psyllid-vectored)", "bacterial")),
    (("peach", "bacterialspot"), ("Xanthomonas arboricola pv. pruni", "bacterial")),
    (("pepper", "bacterialspot"), ("Xanthomonas spp.", "bacterial")),
    (("potato", "earlyblight"), ("Alternaria solani", "fungal")),
    (("potato", "lateblight"), ("Phytophthora infestans", "oomycete")),
    (("squash", "powdery"), ("Podosphaera xanthii", "fungal")),
    (("strawberry", "scorch"), ("Diplocarpon earlianum", "fungal")),
    (("tomato", "bacterialspot"), ("Xanthomonas spp.", "bacterial")),
    (("tomato", "earlyblight"), ("Alternaria solani", "fungal")),
    (("tomato", "lateblight"), ("Phytophthora infestans", "oomycete")),
    (("tomato", "leafmold"), ("Passalora fulva", "fungal")),
    (("tomato", "septoria"), ("Septoria lycopersici", "fungal")),
    (("tomato", "spider"), ("Tetranychus urticae (two-spotted spider mite)", "mite")),
    (("tomato", "target"), ("Corynespora cassiicola", "fungal")),
    (("tomato", "curl"), ("Tomato yellow leaf curl virus (whitefly-transmitted)", "viral")),
    (("tomato", "mosaic"), ("Tomato mosaic virus", "viral")),
]

ADVICE = {
    "fungal": "Remove and bag badly affected leaves, keep foliage dry (water at the base), "
              "and improve air flow. Fungicide choice depends on crop and local rules.",
    "oomycete": "Spreads fast in cool, wet weather. Remove infected plants early and avoid "
                "overhead irrigation. Confirm any treatment with local extension advice.",
    "bacterial": "Use clean seed or transplants, avoid handling wet plants, and remove "
                 "infected material. Sprays only slow spread; check local guidance.",
    "viral": "There is no cure. Remove infected plants, sanitise tools and hands, and control "
             "insect vectors where relevant.",
    "mite": "Check leaf undersides with a hand lens. Mites favour hot, dry conditions; "
            "consult local advice on miticides or predatory mites.",
    "healthy": "No disease pattern detected by the classifier.",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def split_name(class_name: str):
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')."""
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    elif "__" in class_name:
        crop, disease = class_name.split("__", 1)
    else:
        crop, disease = class_name, ""
    crop = re.sub(r"\(.*?\)", "", crop).replace(",", " ").replace("_", " ").strip()
    crop = re.sub(r"\s+", " ", crop)
    disease = re.sub(r"\s+", " ", disease.replace("_", " ")).strip()
    disease = re.sub(r"\(.*?\)", lambda m: m.group(0), disease)
    return crop, disease


def describe(class_name: str) -> dict:
    crop, disease = split_name(class_name)
    n = _norm(class_name)
    if "healthy" in n:
        return {"crop": crop, "disease": "Healthy", "pathogen": None, "kind": "healthy",
                "healthy": True, "advice": ADVICE["healthy"]}
    for keys, (pathogen, kind) in _RULES:
        if all(k in n for k in keys):
            return {"crop": crop, "disease": disease, "pathogen": pathogen, "kind": kind,
                    "healthy": False, "advice": ADVICE[kind]}
    return {"crop": crop, "disease": disease, "pathogen": None, "kind": "unknown",
            "healthy": False, "advice": "Consult a local agricultural extension service."}
