"""
Fuzzy Logic Disease Severity Classifier
Based on: Automatic detection and severity analysis of plant disease using deep learning and fuzzy logic
Inputs: POI (Percentage of Infection), ROI_normalized (normalized diseased pixel ratio)
Output: severity_score (0-100), severity_grade (Healthy/Mild/Medium/Severe)
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyDiseaseClassifier:
    def __init__(self):
        # --- Antecedents (Inputs) ---
        # POI: Percentage of Infection (0–100%)
        self.poi_var = ctrl.Antecedent(np.arange(0, 101, 1), 'poi')
        # ROI_norm: normalized ROI (0–1.0)
        self.roi_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'roi')

        # --- Consequent (Output) ---
        # severity_score: 0=Severe ... 100=Healthy
        self.severity_var = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

        # --- Membership functions: POI ---
        self.poi_var['low']    = fuzz.trapmf(self.poi_var.universe,    [0,  0, 10, 25])
        self.poi_var['medium'] = fuzz.trimf(self.poi_var.universe,     [15, 40, 65])
        self.poi_var['high']   = fuzz.trapmf(self.poi_var.universe,    [55, 75, 100, 100])

        # --- Membership functions: ROI_norm ---
        self.roi_var['small']  = fuzz.trapmf(self.roi_var.universe,    [0,    0,    0.10, 0.25])
        self.roi_var['medium'] = fuzz.trimf(self.roi_var.universe,     [0.15, 0.35, 0.55])
        self.roi_var['large']  = fuzz.trapmf(self.roi_var.universe,    [0.45, 0.65, 1.0,  1.0])

        # --- Membership functions: severity (higher = healthier) ---
        self.severity_var['healthy'] = fuzz.trapmf(self.severity_var.universe, [70, 82, 100, 100])
        self.severity_var['mild']    = fuzz.trimf(self.severity_var.universe,  [48, 60,  75])
        self.severity_var['medium']  = fuzz.trimf(self.severity_var.universe,  [25, 38,  53])
        self.severity_var['severe']  = fuzz.trapmf(self.severity_var.universe, [0,   0,  18, 32])

        # --- Fuzzy Rules (Mamdani, 9 rules) ---
        rules = [
            ctrl.Rule(self.poi_var['low']    & self.roi_var['small'],  self.severity_var['healthy']),
            ctrl.Rule(self.poi_var['low']    & self.roi_var['medium'], self.severity_var['mild']),
            ctrl.Rule(self.poi_var['low']    & self.roi_var['large'],  self.severity_var['mild']),
            ctrl.Rule(self.poi_var['medium'] & self.roi_var['small'],  self.severity_var['mild']),
            ctrl.Rule(self.poi_var['medium'] & self.roi_var['medium'], self.severity_var['medium']),
            ctrl.Rule(self.poi_var['medium'] & self.roi_var['large'],  self.severity_var['medium']),
            ctrl.Rule(self.poi_var['high']   & self.roi_var['small'],  self.severity_var['medium']),
            ctrl.Rule(self.poi_var['high']   & self.roi_var['medium'], self.severity_var['severe']),
            ctrl.Rule(self.poi_var['high']   & self.roi_var['large'],  self.severity_var['severe']),
        ]

        severity_ctrl = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(severity_ctrl)

    def classify(self, poi: float, roi_normalized: float) -> tuple:
        """
        Run fuzzy inference.
        Args:
            poi: Percentage of Infection (0–100)
            roi_normalized: diseased pixels / total pixels (0–1.0)
        Returns:
            (severity_score: float, severity_grade: str)
        """
        poi = float(max(0, min(100, poi)))
        roi_normalized = float(max(0, min(1.0, roi_normalized)))

        try:
            self.simulation.input['poi'] = poi
            self.simulation.input['roi'] = roi_normalized
            self.simulation.compute()
            score = float(self.simulation.output['severity'])
        except Exception:
            # Linear fallback if fuzzy computation fails
            score = max(0.0, 100.0 - (poi * 0.65 + roi_normalized * 35.0))

        score = round(score, 2)

        if score >= 75:
            grade = 'Healthy'
        elif score >= 50:
            grade = 'Mild'
        elif score >= 25:
            grade = 'Medium'
        else:
            grade = 'Severe'

        return score, grade
