"""
Patient generators for the Clinical Triage Agent environment.

All generators are fully seeded and deterministic. The same (task_id, seed)
combination always produces the same patient set, ensuring reproducibility.
"""

from __future__ import annotations

import random
from typing import List, Optional

from .models import (
    GroundTruth,
    Patient,
    PatientHidden,
    PatientVisible,
    TriageLevel,
    TriagePathway,
    Vitals,
)

# ---------------------------------------------------------------------------
# Question bank for ASK actions
# ---------------------------------------------------------------------------

QUESTION_BANK: dict[str, str] = {
    "pain_scale": "What is the patient's pain level on a scale of 0-10?",
    "duration": "How long has the patient had these symptoms?",
    "history": "What is the patient's relevant medical history?",
    "medications": "What medications is the patient currently taking?",
}

VALID_QUESTION_KEYS = set(QUESTION_BANK.keys())

# ---------------------------------------------------------------------------
# Clinical data pools (realistic values)
# ---------------------------------------------------------------------------

_CHIEF_COMPLAINTS_CRITICAL = [
    "Severe chest pain with diaphoresis",
    "Unresponsive, found collapsed at home",
    "Active seizure not self-resolving",
    "Acute respiratory distress, unable to speak",
    "Massive haematemesis",
    "Anaphylaxis following bee sting",
    "Major trauma – MVC at high speed",
]

_CHIEF_COMPLAINTS_EMERGENT = [
    "Chest pain radiating to the left arm",
    "Sudden severe headache – worst of life",
    "Shortness of breath with SpO2 88%",
    "Altered mental status, confused",
    "Right-sided weakness and facial droop",
    "Severe abdominal pain, rigid abdomen",
    "High fever (40°C) with neck stiffness",
    "Eclampsia – 32 weeks pregnant",
]

_CHIEF_COMPLAINTS_URGENT = [
    "Moderate chest pain, no radiation",
    "Worsening shortness of breath",
    "Broken arm after fall",
    "Diabetic with blood sugar 400 mg/dL",
    "Flank pain with haematuria",
    "Severe migraine with nausea and vomiting",
    "Laceration to hand with active bleeding",
    "Dislocation of right shoulder",
]

_CHIEF_COMPLAINTS_LESS_URGENT = [
    "Sore throat for 3 days",
    "Sprained ankle, weight bearing",
    "UTI symptoms – dysuria, frequency",
    "Mild abdominal cramps",
    "Ear pain for 2 days",
    "Minor lacerations requiring sutures",
    "Allergic rash – urticaria, no angioedema",
    "Low-grade fever 38°C for 1 day",
]

_CHIEF_COMPLAINTS_NON_URGENT = [
    "Prescription refill request",
    "Mild cold symptoms",
    "Insect bite, no signs of infection",
    "Back pain – chronic, no red flags",
    "Small contusion after minor fall",
    "Skin rash for several weeks",
]

ARRIVAL_MODES = ["walk-in", "ambulance", "unknown"]

_MEDICAL_HISTORIES = [
    ["Hypertension", "Type 2 Diabetes"],
    ["COPD", "Chronic heart failure"],
    ["Asthma"],
    ["No significant history"],
    ["Atrial fibrillation", "Warfarin therapy"],
    ["Epilepsy"],
    ["Coronary artery disease", "Prior MI"],
    ["Liver cirrhosis", "Portal hypertension"],
    ["End-stage renal disease on dialysis"],
    ["Depression", "Anxiety disorder"],
    [],
]

_MEDICATIONS_POOLS = [
    ["Metoprolol", "Metformin", "Aspirin"],
    ["Lisinopril", "Atorvastatin"],
    ["Salbutamol inhaler", "Prednisolone"],
    ["None"],
    ["Warfarin", "Digoxin", "Furosemide"],
    ["Levetiracetam"],
    ["Clopidogrel", "Ramipril"],
    ["Spironolactone", "Propranolol"],
    ["Erythropoietin", "Phosphate binders"],
    ["Sertraline", "Lorazepam"],
    [],
]


# ---------------------------------------------------------------------------
# Vital sign generators per acuity level
# ---------------------------------------------------------------------------

def _vitals_level1(rng: random.Random) -> Vitals:
    """Resuscitation – critically abnormal vitals."""
    return Vitals(
        heart_rate=rng.choice([rng.randint(130, 180), rng.randint(20, 40)]),
        spo2=round(rng.uniform(72, 84), 1),
        respiratory_rate=rng.choice([rng.randint(30, 40), rng.randint(4, 8)]),
        systolic_bp=rng.choice([rng.randint(60, 80), rng.randint(200, 240)]),
        temperature=round(rng.choice([rng.uniform(39.5, 41.5), rng.uniform(33.0, 35.0)]), 1),
    )


def _vitals_level2(rng: random.Random) -> Vitals:
    """Emergent – significantly abnormal."""
    return Vitals(
        heart_rate=rng.randint(110, 140),
        spo2=round(rng.uniform(85, 91), 1),
        respiratory_rate=rng.randint(24, 32),
        systolic_bp=rng.randint(80, 100),
        temperature=round(rng.uniform(38.5, 40.0), 1),
    )


def _vitals_level3(rng: random.Random) -> Vitals:
    """Urgent – moderately abnormal."""
    return Vitals(
        heart_rate=rng.randint(95, 115),
        spo2=round(rng.uniform(92, 95), 1),
        respiratory_rate=rng.randint(20, 26),
        systolic_bp=rng.randint(100, 140),
        temperature=round(rng.uniform(37.8, 39.0), 1),
    )


def _vitals_level4(rng: random.Random) -> Vitals:
    """Less urgent – mildly abnormal."""
    return Vitals(
        heart_rate=rng.randint(80, 100),
        spo2=round(rng.uniform(95, 97), 1),
        respiratory_rate=rng.randint(16, 20),
        systolic_bp=rng.randint(120, 150),
        temperature=round(rng.uniform(37.2, 38.4), 1),
    )


def _vitals_level5(rng: random.Random) -> Vitals:
    """Non-urgent – near-normal vitals."""
    return Vitals(
        heart_rate=rng.randint(60, 85),
        spo2=round(rng.uniform(97, 100), 1),
        respiratory_rate=rng.randint(12, 17),
        systolic_bp=rng.randint(110, 140),
        temperature=round(rng.uniform(36.5, 37.5), 1),
    )


_VITALS_GENERATORS = {
    TriageLevel.RESUSCITATION: _vitals_level1,
    TriageLevel.EMERGENT: _vitals_level2,
    TriageLevel.URGENT: _vitals_level3,
    TriageLevel.LESS_URGENT: _vitals_level4,
    TriageLevel.NON_URGENT: _vitals_level5,
}


# ---------------------------------------------------------------------------
# Level-to-pathway mapping (deterministic canonical answer)
# ---------------------------------------------------------------------------

LEVEL_TO_PATHWAY: dict[TriageLevel, TriagePathway] = {
    TriageLevel.RESUSCITATION: TriagePathway.RESUS,
    TriageLevel.EMERGENT: TriagePathway.MAJORS,
    TriageLevel.URGENT: TriagePathway.MAJORS,
    TriageLevel.LESS_URGENT: TriagePathway.FAST_TRACK,
    TriageLevel.NON_URGENT: TriagePathway.FAST_TRACK,
}

LEVEL_TO_COMPLAINTS: dict[TriageLevel, list[str]] = {
    TriageLevel.RESUSCITATION: _CHIEF_COMPLAINTS_CRITICAL,
    TriageLevel.EMERGENT: _CHIEF_COMPLAINTS_EMERGENT,
    TriageLevel.URGENT: _CHIEF_COMPLAINTS_URGENT,
    TriageLevel.LESS_URGENT: _CHIEF_COMPLAINTS_LESS_URGENT,
    TriageLevel.NON_URGENT: _CHIEF_COMPLAINTS_NON_URGENT,
}


def _build_patient(
    rng: random.Random,
    patient_id: str,
    level: TriageLevel,
    arrival_step: int = 0,
) -> Patient:
    """Build a single realistic patient at the given triage level."""
    complaints = LEVEL_TO_COMPLAINTS[level]
    chief_complaint = rng.choice(complaints)
    vitals = _VITALS_GENERATORS[level](rng)
    pathway = LEVEL_TO_PATHWAY[level]
    history_idx = rng.randint(0, len(_MEDICAL_HISTORIES) - 1)
    med_idx = rng.randint(0, len(_MEDICATIONS_POOLS) - 1)

    arrival_mode = rng.choice(ARRIVAL_MODES)
    if level in (TriageLevel.RESUSCITATION, TriageLevel.EMERGENT):
        arrival_mode = "ambulance"

    age_ranges = {
        TriageLevel.RESUSCITATION: (40, 85),
        TriageLevel.EMERGENT: (35, 80),
        TriageLevel.URGENT: (20, 75),
        TriageLevel.LESS_URGENT: (15, 65),
        TriageLevel.NON_URGENT: (5, 60),
    }
    age = rng.randint(*age_ranges[level])

    pain_map = {
        TriageLevel.RESUSCITATION: (8, 10),
        TriageLevel.EMERGENT: (7, 9),
        TriageLevel.URGENT: (5, 7),
        TriageLevel.LESS_URGENT: (3, 5),
        TriageLevel.NON_URGENT: (1, 3),
    }
    pain_scale = rng.randint(*pain_map[level])

    duration_map = {
        TriageLevel.RESUSCITATION: (1, 30),
        TriageLevel.EMERGENT: (10, 120),
        TriageLevel.URGENT: (30, 480),
        TriageLevel.LESS_URGENT: (120, 1440),
        TriageLevel.NON_URGENT: (240, 4320),
    }
    duration = rng.randint(*duration_map[level])

    rationale = (
        f"Level {level.value} ({level.name}): {chief_complaint}. "
        f"Vitals: HR={vitals.heart_rate}, SpO2={vitals.spo2}%, "
        f"RR={vitals.respiratory_rate}. Pathway: {pathway.value}."
    )

    return Patient(
        visible=PatientVisible(
            patient_id=patient_id,
            age=age,
            chief_complaint=chief_complaint,
            vitals=vitals,
            arrival_mode=arrival_mode,
            arrival_step=arrival_step,
        ),
        hidden=PatientHidden(
            pain_scale=pain_scale,
            symptom_duration_minutes=duration,
            medical_history=_MEDICAL_HISTORIES[history_idx],
            current_medications=_MEDICATIONS_POOLS[med_idx],
        ),
        ground_truth=GroundTruth(
            level=level,
            pathway=pathway,
            rationale=rationale,
        ),
    )


# ---------------------------------------------------------------------------
# Task-specific generators
# ---------------------------------------------------------------------------

def generate_task1_patients(rng: random.Random) -> List[Patient]:
    """
    Task 1 (easy): 3 patients, spread across levels 2/3/4.
    Mostly visible info, low complexity.
    """
    configs = [
        (TriageLevel.EMERGENT, "PT-001"),
        (TriageLevel.URGENT, "PT-002"),
        (TriageLevel.LESS_URGENT, "PT-003"),
    ]
    patients = [_build_patient(rng, pid, level) for level, pid in configs]
    for p in patients:
        p.visible.revealed_info["pain_scale"] = p.hidden.pain_scale
        p.visible.revealed_info["duration"] = f"{p.hidden.symptom_duration_minutes} minutes"
        p.visible.revealed_info["history"] = p.hidden.medical_history or ["None"]
        p.visible.revealed_info["medications"] = p.hidden.current_medications or ["None"]
    return patients


def generate_task2_patients(rng: random.Random) -> List[Patient]:
    """
    Task 2 (medium): 8 patients with overlapping symptoms.
    Mix of all levels, requires ASK to differentiate.
    """
    level_pool = [
        TriageLevel.RESUSCITATION,
        TriageLevel.EMERGENT,
        TriageLevel.EMERGENT,
        TriageLevel.URGENT,
        TriageLevel.URGENT,
        TriageLevel.URGENT,
        TriageLevel.LESS_URGENT,
        TriageLevel.NON_URGENT,
    ]
    rng.shuffle(level_pool)
    return [
        _build_patient(rng, f"PT-{i + 1:03d}", level)
        for i, level in enumerate(level_pool)
    ]


def generate_task3_patients(
    rng: random.Random,
    wave: int = 0,
) -> List[Patient]:
    """
    Task 3 (hard): initial 12 patients; wave adds 8 more at step 10.
    Dynamic patient arrival mid-episode.
    """
    if wave == 0:
        level_pool = [
            TriageLevel.RESUSCITATION,
            TriageLevel.RESUSCITATION,
            TriageLevel.EMERGENT,
            TriageLevel.EMERGENT,
            TriageLevel.EMERGENT,
            TriageLevel.URGENT,
            TriageLevel.URGENT,
            TriageLevel.URGENT,
            TriageLevel.LESS_URGENT,
            TriageLevel.LESS_URGENT,
            TriageLevel.NON_URGENT,
            TriageLevel.NON_URGENT,
        ]
        rng.shuffle(level_pool)
        return [
            _build_patient(rng, f"PT-W0-{i + 1:03d}", level)
            for i, level in enumerate(level_pool)
        ]
    else:
        # Wave 2 – mass casualty event patients
        level_pool = [
            TriageLevel.RESUSCITATION,
            TriageLevel.RESUSCITATION,
            TriageLevel.EMERGENT,
            TriageLevel.EMERGENT,
            TriageLevel.URGENT,
            TriageLevel.URGENT,
            TriageLevel.LESS_URGENT,
            TriageLevel.NON_URGENT,
        ]
        rng.shuffle(level_pool)
        return [
            _build_patient(rng, f"PT-W2-{i + 1:03d}", level, arrival_step=10)
            for i, level in enumerate(level_pool)
        ]


def generate_patients(
    task_id: int,
    seed: Optional[int] = None,
    wave: int = 0,
) -> List[Patient]:
    """
    Top-level patient generator. Fully deterministic given (task_id, seed).

    Args:
        task_id: 0=easy, 1=medium, 2=hard
        seed: Random seed for reproducibility
        wave: For task 3, wave 0 = initial, wave 1 = reinforcement wave

    Returns:
        List of Patient objects
    """
    rng = random.Random(seed)
    if task_id == 0:
        return generate_task1_patients(rng)
    elif task_id == 1:
        return generate_task2_patients(rng)
    elif task_id == 2:
        return generate_task3_patients(rng, wave=wave)
    else:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 0, 1, or 2.")
