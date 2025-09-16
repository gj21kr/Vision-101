#!/usr/bin/env python3
"""
의료 노트 템플릿 시스템 (Medical Note Templates)

표준화된 의료 노트 템플릿과 구조화된 데이터 모델을 제공하여
일관성 있고 완성도 높은 의료 기록 작성을 지원합니다.

주요 기능:
- 다양한 의료 노트 유형별 표준 템플릿
- 의료진별 맞춤형 템플릿
- 진료과별 특화 템플릿
- 자동 품질 검증 및 완성도 평가
- 표준 의료 용어 사전 통합
- 의료진 간 인수인계 표준화
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from enum import Enum
import re

class NoteType(Enum):
    """의료 노트 유형"""
    ADMISSION_NOTE = "admission_note"
    PROGRESS_NOTE = "progress_note"
    H_AND_P = "history_and_physical"
    DISCHARGE_SUMMARY = "discharge_summary"
    CONSULTATION = "consultation"
    OPERATIVE_NOTE = "operative_note"
    PROCEDURE_NOTE = "procedure_note"
    TRANSFER_NOTE = "transfer_note"
    EMERGENCY_NOTE = "emergency_note"
    ICU_NOTE = "icu_note"

class Department(Enum):
    """진료과"""
    INTERNAL_MEDICINE = "internal_medicine"
    SURGERY = "surgery"
    EMERGENCY = "emergency"
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    NEUROLOGY = "neurology"
    ORTHOPEDICS = "orthopedics"
    PEDIATRICS = "pediatrics"
    PSYCHIATRY = "psychiatry"
    RADIOLOGY = "radiology"

class Severity(Enum):
    """중증도"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TemplateSection:
    """템플릿 섹션"""
    title: str
    content: str
    required: bool = True
    max_length: Optional[int] = None
    validation_rules: List[str] = None

    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []

@dataclass
class NoteTemplate:
    """의료 노트 템플릿"""
    template_id: str
    name: str
    note_type: NoteType
    department: Department
    sections: List[TemplateSection]
    metadata: Dict[str, Any]
    created_by: str
    created_date: datetime
    version: str = "1.0"

class MedicalTerminology:
    """의료 용어 사전"""

    def __init__(self):
        self.symptoms = {
            "fever": {"korean": "발열", "code": "R50.9", "severity": ["mild", "moderate", "high"]},
            "pain": {"korean": "통증", "code": "R52", "severity": ["mild", "moderate", "severe"]},
            "dyspnea": {"korean": "호흡곤란", "code": "R06.8", "severity": ["mild", "moderate", "severe"]},
            "chest_pain": {"korean": "흉통", "code": "R07.9", "severity": ["mild", "moderate", "severe"]},
            "abdominal_pain": {"korean": "복통", "code": "R10.9", "severity": ["mild", "moderate", "severe"]},
            "headache": {"korean": "두통", "code": "R51", "severity": ["mild", "moderate", "severe"]},
            "nausea": {"korean": "오심", "code": "R11", "severity": ["mild", "moderate", "severe"]},
            "vomiting": {"korean": "구토", "code": "R11", "severity": ["mild", "moderate", "severe"]},
            "dizziness": {"korean": "어지러움", "code": "R42", "severity": ["mild", "moderate", "severe"]},
            "fatigue": {"korean": "피로감", "code": "R53", "severity": ["mild", "moderate", "severe"]}
        }

        self.diagnoses = {
            "hypertension": {"korean": "고혈압", "icd10": "I10", "category": "cardiovascular"},
            "diabetes": {"korean": "당뇨병", "icd10": "E11", "category": "endocrine"},
            "copd": {"korean": "만성폐쇄성폐질환", "icd10": "J44", "category": "respiratory"},
            "heart_failure": {"korean": "심부전", "icd10": "I50", "category": "cardiovascular"},
            "pneumonia": {"korean": "폐렴", "icd10": "J18", "category": "respiratory"},
            "appendicitis": {"korean": "충수염", "icd10": "K37", "category": "digestive"},
            "stroke": {"korean": "뇌졸중", "icd10": "I64", "category": "neurological"},
            "mi": {"korean": "심근경색", "icd10": "I21", "category": "cardiovascular"}
        }

        self.medications = {
            "aspirin": {"korean": "아스피린", "generic": "acetylsalicylic acid", "class": "antiplatelet"},
            "metformin": {"korean": "메트포르민", "generic": "metformin", "class": "antidiabetic"},
            "lisinopril": {"korean": "리시노프릴", "generic": "lisinopril", "class": "ace_inhibitor"},
            "amlodipine": {"korean": "암로디핀", "generic": "amlodipine", "class": "calcium_channel_blocker"},
            "atorvastatin": {"korean": "아토르바스타틴", "generic": "atorvastatin", "class": "statin"},
            "omeprazole": {"korean": "오메프라졸", "generic": "omeprazole", "class": "ppi"}
        }

class TemplateManager:
    """템플릿 관리자"""

    def __init__(self):
        self.templates = {}
        self.terminology = MedicalTerminology()
        self._load_default_templates()

    def _load_default_templates(self):
        """기본 템플릿 로드"""
        # Admission Note 템플릿
        self.create_admission_note_template()

        # Progress Note 템플릿
        self.create_progress_note_template()

        # Discharge Summary 템플릿
        self.create_discharge_summary_template()

        # Consultation 템플릿
        self.create_consultation_template()

        # Emergency Note 템플릿
        self.create_emergency_note_template()

        # ICU Note 템플릿
        self.create_icu_note_template()

    def create_admission_note_template(self):
        """입원 기록 템플릿"""
        sections = [
            TemplateSection(
                title="환자 기본 정보",
                content="""
성명: [환자명]
나이: [나이]세
성별: [성별]
의료보험번호: [보험번호]
입원일: [입원일]
입원경로: [외래/응급실/전원]
주치의: [주치의명]
진료과: [진료과]
                """.strip(),
                required=True,
                validation_rules=["required_fields"]
            ),
            TemplateSection(
                title="주소 (Chief Complaint)",
                content="[환자가 호소하는 주요 증상이나 내원 사유]",
                required=True,
                max_length=500,
                validation_rules=["not_empty", "max_length"]
            ),
            TemplateSection(
                title="현병력 (History of Present Illness)",
                content="""
[증상 발생 시기, 양상, 경과, 동반 증상, 악화/완화 요인 등을 시간 순서대로 기술]

- 발생 시기:
- 증상의 양상:
- 경과:
- 동반 증상:
- 악화/완화 요인:
- 이전 치료력:
                """.strip(),
                required=True,
                validation_rules=["not_empty", "timeline_check"]
            ),
            TemplateSection(
                title="과거병력 (Past Medical History)",
                content="""
1. 과거 질환력:
   - [질환명] ([진단연도])

2. 수술력:
   - [수술명] ([수술연도])

3. 입원력:
   - [입원사유] ([입원연도])

4. 외상력:
   - [외상 내용] ([발생연도])
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="복용 약물 (Medications)",
                content="""
현재 복용 중인 약물:
1. [약물명] [용량] [용법] - [적응증]
2. [약물명] [용량] [용법] - [적응증]

최근 중단한 약물:
- [약물명]: [중단 사유] ([중단일])
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="알레르기 (Allergies)",
                content="""
약물 알레르기:
- [약물명]: [반응 양상]

음식 알레르기:
- [음식명]: [반응 양상]

기타 알레르기:
- [알레르기 항원]: [반응 양상]

※ 알레르기 없음: [ ]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="가족력 (Family History)",
                content="""
직계 가족의 주요 질환력:
- 부: [질환명 또는 건강상태]
- 모: [질환명 또는 건강상태]
- 형제자매: [질환명 또는 건강상태]
- 자녀: [질환명 또는 건강상태]

유전성 질환 가족력: [있음/없음]
                """.strip(),
                required=False
            ),
            TemplateSection(
                title="사회력 (Social History)",
                content="""
- 흡연: [현재/과거/비흡연] ([갑년] 또는 [기간])
- 음주: [현재/과거/금주] ([종류, 양, 빈도])
- 직업: [직업명] ([기간])
- 거주지: [거주지역]
- 결혼상태: [기혼/미혼/기타]
- 종교: [종교]
                """.strip(),
                required=False
            ),
            TemplateSection(
                title="계통별 문진 (Review of Systems)",
                content="""
- 전신: 발열, 체중변화, 피로감 등
- 피부: 발진, 가려움, 색조변화 등
- 머리목: 두통, 시야장애, 청력장애 등
- 심혈관: 흉통, 호흡곤란, 부종 등
- 호흡기: 기침, 가래, 혈담 등
- 소화기: 복통, 오심구토, 설사변비 등
- 비뇨기: 배뇨장애, 혈뇨, 빈뇨 등
- 근골격: 관절통, 근육통, 운동장애 등
- 신경: 두통, 어지러움, 감각이상 등
                """.strip(),
                required=False
            ),
            TemplateSection(
                title="활력징후 (Vital Signs)",
                content="""
측정일시: [측정일시]
- 혈압: [수축기]/[이완기] mmHg
- 맥박: [맥박수] bpm, [규칙성]
- 호흡수: [호흡수] /min
- 체온: [체온] °C
- 산소포화도: [SpO2] % (실내공기/[산소농도])
- 통증 점수: [0-10점]
                """.strip(),
                required=True,
                validation_rules=["vital_signs_range"]
            ),
            TemplateSection(
                title="신체검진 (Physical Examination)",
                content="""
일반적 외견: [외견 상태]

활력징후: 상기 기술

머리목:
- 머리: [소견]
- 눈: [소견]
- 귀: [소견]
- 코: [소견]
- 목: [소견]

흉부:
- 심장: [심음, 잡음, 부정맥 등]
- 폐: [호흡음, 수포음, 천명음 등]

복부:
- 외관: [팽만, 함몰 등]
- 촉진: [압통, 종괴, 간비대 등]
- 청진: [장음]

사지:
- 상지: [소견]
- 하지: [소견, 부종 등]

신경학적:
- 의식: [의식상태]
- 뇌신경: [뇌신경 검사]
- 운동: [근력, 근긴장도]
- 감각: [감각검사]
- 반사: [심부건반사]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="검사 계획 (Diagnostic Plan)",
                content="""
혈액검사:
- 일반혈액검사 (CBC with differential)
- 생화학검사 (Comprehensive metabolic panel)
- [기타 특수검사]

영상검사:
- [검사명]: [검사 목적]

기타 검사:
- [검사명]: [검사 목적]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="초기 치료 계획 (Initial Treatment Plan)",
                content="""
1. 일반적 관리:
   - 안정: [활동 정도]
   - 식이: [식이 종류]
   - 수액: [수액 종류 및 속도]

2. 약물 치료:
   - [약물명] [용량] [용법] - [목적]

3. 모니터링:
   - 활력징후 모니터링: [빈도]
   - [특정 증상/수치] 관찰

4. 상급의 보고 기준:
   - [보고해야 할 상황들]
                """.strip(),
                required=True
            )
        ]

        template = NoteTemplate(
            template_id="admission_001",
            name="표준 입원기록",
            note_type=NoteType.ADMISSION_NOTE,
            department=Department.INTERNAL_MEDICINE,
            sections=sections,
            metadata={
                "target_users": ["전공의", "전문의"],
                "estimated_time": "30-45분",
                "complexity": "moderate"
            },
            created_by="시스템",
            created_date=datetime.now()
        )

        self.templates[template.template_id] = template

    def create_progress_note_template(self):
        """경과 기록 템플릿"""
        sections = [
            TemplateSection(
                title="날짜 및 시간",
                content="[YYYY-MM-DD HH:MM] 경과기록 ([입원 N일째])",
                required=True
            ),
            TemplateSection(
                title="Subjective (주관적 소견)",
                content="""
환자 호소:
- 주요 증상: [증상 및 변화]
- 통증: [0-10점] - [위치, 양상]
- 수면: [수면 상태]
- 식욕: [식욕 상태]
- 배뇨/배변: [상태]
- 기타: [환자가 호소하는 증상]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="Objective (객관적 소견)",
                content="""
활력징후:
- BP: [혈압] mmHg
- PR: [맥박] bpm
- RR: [호흡수] /min
- BT: [체온] °C
- SpO2: [산소포화도] %

신체검진:
- 일반상태: [의식, 외견]
- 심장: [심음, 잡음]
- 폐: [호흡음, 수포음]
- 복부: [압통, 장음]
- 사지: [부종, 순환]

금일 검사결과:
- [검사명]: [결과] ([정상범위])

투입량/배출량 (I/O):
- 입: [mL] (경구: [mL], 정맥: [mL])
- 출: [mL] (소변: [mL], 배변: [회])
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="Assessment (평가)",
                content="""
1. [주진단명]: [현재 상태 평가]
   - 호전/악화/보합
   - [평가 근거]

2. [부진단명]: [현재 상태 평가]

문제점:
1. [문제점 1]
2. [문제점 2]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="Plan (계획)",
                content="""
1. [주진단명] 관련:
   - 치료: [치료 계획]
   - 검사: [추가 검사 계획]
   - 모니터링: [관찰 항목]

2. [부진단명] 관련:
   - [치료/관리 계획]

3. 일반 관리:
   - 활동: [활동 제한/허용]
   - 식이: [식이 종류]
   - 교육: [환자/보호자 교육]

4. 퇴원 계획:
   - 예상 퇴원일: [날짜]
   - 퇴원 기준: [기준]
                """.strip(),
                required=True
            )
        ]

        template = NoteTemplate(
            template_id="progress_001",
            name="표준 경과기록",
            note_type=NoteType.PROGRESS_NOTE,
            department=Department.INTERNAL_MEDICINE,
            sections=sections,
            metadata={
                "target_users": ["전공의", "전문의", "PA"],
                "estimated_time": "10-15분",
                "complexity": "low"
            },
            created_by="시스템",
            created_date=datetime.now()
        )

        self.templates[template.template_id] = template

    def create_discharge_summary_template(self):
        """퇴원 요약 템플릿"""
        sections = [
            TemplateSection(
                title="환자 정보",
                content="""
성명: [환자명]
나이: [나이]세
성별: [성별]
의료보험번호: [보험번호]
입원일: [입원일]
퇴원일: [퇴원일]
재원기간: [재원일수]일
주치의: [주치의명]
진료과: [진료과]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="입원 사유",
                content="[입원 당시 주소 및 주요 증상]",
                required=True
            ),
            TemplateSection(
                title="주요 진단명",
                content="""
1. [주진단명] (ICD-10: [코드])
2. [부진단명 1] (ICD-10: [코드])
3. [부진단명 2] (ICD-10: [코드])
                """.strip(),
                required=True,
                validation_rules=["icd10_format"]
            ),
            TemplateSection(
                title="치료 경과",
                content="""
[입원 중 주요 치료 과정을 시간 순서대로 요약]

입원 당일: [초기 치료]
입원 2-3일: [경과 및 치료 변화]
중간 경과: [주요 변화점, 합병증 등]
퇴원 전: [최종 상태]

주요 시술/수술:
- [시술명]: [날짜] - [결과]

주요 검사 결과:
- [검사명]: [결과] - [의미]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="퇴원 시 상태",
                content="""
일반상태: [의식, 활력징후, 전반적 상태]

주요 증상:
- [증상]: [호전/악화/보합]

신체검진:
- [주요 소견]

최종 검사 결과:
- [검사명]: [결과]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="퇴원 처방",
                content="""
1. [약물명] [용량] [용법] ([복용 기간])
   - 목적: [치료 목적]
   - 주의사항: [부작용, 주의점]

2. [약물명] [용량] [용법] ([복용 기간])
   - 목적: [치료 목적]

기타 처방:
- [처방 내용]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="퇴원 후 계획",
                content="""
외래 추적:
- [진료과] [날짜] ([목적])

생활 수칙:
- 활동: [활동 제한/권고사항]
- 식이: [식이 제한/권고사항]
- 기타: [주의사항]

응급상황 대처:
다음 증상 발생 시 즉시 응급실 내원:
- [응급 증상 1]
- [응급 증상 2]

추가 검사 계획:
- [검사명]: [예정일] - [목적]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="환자/보호자 교육",
                content="""
질병에 대한 설명:
- [질병 설명 완료 여부]
- [이해도 평가]

약물 교육:
- [복용법 교육 완료]
- [부작용 설명 완료]

생활 관리 교육:
- [교육 내용]

추적 관리의 중요성:
- [설명 완료 여부]
                """.strip(),
                required=True
            )
        ]

        template = NoteTemplate(
            template_id="discharge_001",
            name="표준 퇴원요약",
            note_type=NoteType.DISCHARGE_SUMMARY,
            department=Department.INTERNAL_MEDICINE,
            sections=sections,
            metadata={
                "target_users": ["전공의", "전문의"],
                "estimated_time": "45-60분",
                "complexity": "high"
            },
            created_by="시스템",
            created_date=datetime.now()
        )

        self.templates[template.template_id] = template

    def create_consultation_template(self):
        """협진 의뢰서 템플릿"""
        sections = [
            TemplateSection(
                title="협진 정보",
                content="""
의뢰과: [의뢰한 진료과]
의뢰의: [의뢰의사명]
협진과: [협진 요청 진료과]
의뢰일: [의뢰일자]
협진 목적: [협진 요청 이유]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="환자 기본 정보",
                content="""
성명: [환자명]
나이: [나이]세
성별: [성별]
입원/외래: [구분]
입원일: [입원일] (입원환자인 경우)
주치의: [주치의명]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="협진 요청 사유",
                content="""
[협진이 필요한 구체적인 이유를 명확히 기술]

협진 질문 사항:
1. [질문 1]
2. [질문 2]
3. [질문 3]
                """.strip(),
                required=True,
                validation_rules=["specific_question"]
            ),
            TemplateSection(
                title="현재 상태 요약",
                content="""
주진단: [주진단명]
현재 주요 증상: [증상]
현재 치료: [진행 중인 치료]

관련 과거력:
- [협진과 관련 과거 병력]

현재 복용 약물:
- [관련 약물 목록]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="관련 검사 결과",
                content="""
혈액검사: ([검사일])
- [관련 검사 결과]

영상검사: ([검사일])
- [검사명]: [주요 소견]

기타 검사: ([검사일])
- [검사명]: [결과]

※ 상세한 검사 결과는 EMR 참조
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="협진 의견",
                content="""
[협진과에서 작성]

협진 의견:
1. [진단적 의견]
2. [치료 권고사항]
3. [추가 검사 필요성]
4. [추적 관찰 계획]

협진의: [협진의사명]
협진일: [협진일자]
                """.strip(),
                required=False
            )
        ]

        template = NoteTemplate(
            template_id="consultation_001",
            name="표준 협진의뢰서",
            note_type=NoteType.CONSULTATION,
            department=Department.INTERNAL_MEDICINE,
            sections=sections,
            metadata={
                "target_users": ["전공의", "전문의"],
                "estimated_time": "15-20분",
                "complexity": "moderate"
            },
            created_by="시스템",
            created_date=datetime.now()
        )

        self.templates[template.template_id] = template

    def create_emergency_note_template(self):
        """응급실 기록 템플릿"""
        sections = [
            TemplateSection(
                title="응급실 내원 정보",
                content="""
내원일시: [YYYY-MM-DD HH:MM]
내원방법: [도보/휠체어/들것/119구급차/사설구급차]
내원경로: [직접/타병원 전원]
초기 중증도: [1-5단계]
담당의: [담당의명]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="주소 (Chief Complaint)",
                content="[환자 또는 보호자가 호소하는 주요 증상]",
                required=True,
                max_length=200
            ),
            TemplateSection(
                title="현병력 (HPI) - 응급실 특화",
                content="""
OPQRST 평가:
- Onset: [발생 시점]
- Provocation/Palliation: [악화/완화 요인]
- Quality: [증상의 양상/성격]
- Region/Radiation: [위치/방사]
- Severity: [0-10점]
- Timing: [지속시간/빈도]

동반 증상: [관련 증상들]
이전 치료: [내원 전 받은 치료]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="응급 활력징후",
                content="""
측정시간: [HH:MM]
- 혈압: [수축기]/[이완기] mmHg
- 맥박: [맥박수] bpm
- 호흡수: [호흡수] /min
- 체온: [체온] °C
- SpO2: [산소포화도] % ([실내공기/O2 L/min])
- 혈당: [혈당] mg/dL (POC)
- 통증점수: [0-10점]

재측정 ([HH:MM]):
[필요시 재측정 결과]
                """.strip(),
                required=True,
                validation_rules=["emergency_vital_range"]
            ),
            TemplateSection(
                title="응급 신체검진",
                content="""
의식상태: [alert/drowsy/stupor/coma]
GCS: E[점]V[점]M[점] = [총점]

일차 평가 (Primary Survey):
- Airway: [기도 개방성]
- Breathing: [호흡 상태]
- Circulation: [순환 상태]
- Disability: [신경학적 상태]
- Exposure: [노출 검사]

이차 평가 (Secondary Survey):
- Head/Neck: [소견]
- Chest: [소견]
- Abdomen: [소견]
- Pelvis: [소견]
- Extremities: [소견]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="응급 검사",
                content="""
즉시 시행 검사:
- 혈액검사: [CBC, BUN/Cr, Glucose, etc.]
- 심전도: [소견]
- 흉부 X-ray: [소견]
- [기타 응급 검사]

검사 결과:
- [주요 이상 소견]
- [정상 소견]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="응급처치 및 치료",
                content="""
즉시 처치:
- [응급 처치 내용]
- 산소공급: [유무 및 방법]
- 정맥로 확보: [부위 및 gauge]
- 수액: [종류 및 속도]

투약:
- [약물명] [용량] [경로] [시간] - [목적]

기타 처치:
- [시행한 처치]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="응급실 진단 및 계획",
                content="""
응급실 진단:
1. [주진단]
2. [부진단]

중증도 평가: [stable/unstable]

처치 계획:
1. [즉시 필요한 처치]
2. [추가 검사 계획]
3. [치료 방향]

최종 처리:
- [입원/퇴원/전원/사망]
- 입원과: [진료과]
- 퇴원 교육: [내용]
                """.strip(),
                required=True
            )
        ]

        template = NoteTemplate(
            template_id="emergency_001",
            name="응급실 표준기록",
            note_type=NoteType.EMERGENCY_NOTE,
            department=Department.EMERGENCY,
            sections=sections,
            metadata={
                "target_users": ["응급의학과 전공의", "응급의학과 전문의"],
                "estimated_time": "20-30분",
                "complexity": "high",
                "priority": "urgent"
            },
            created_by="시스템",
            created_date=datetime.now()
        )

        self.templates[template.template_id] = template

    def create_icu_note_template(self):
        """중환자실 기록 템플릿"""
        sections = [
            TemplateSection(
                title="ICU 일반 정보",
                content="""
ICU 입실일: [YYYY-MM-DD HH:MM]
ICU Day: [N일째]
입실 경로: [응급실/병동/수술실/타원]
주치의: [주치의명]
담당 간호사: [간호사명]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="주요 문제 및 진단",
                content="""
1. [주진단] - [active/stable/resolving]
2. [진단 2] - [상태]
3. [진단 3] - [상태]

현재 주요 문제점:
1. [문제점 1]
2. [문제점 2]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="중환자 활력징후 및 모니터링",
                content="""
[시간대별 활력징후 - 최근 24시간]

현재 ([HH:MM]):
- BP: [수축기]/[이완기] mmHg (MAP: [평균동맥압])
- HR: [심박수] bpm, [리듬]
- RR: [호흡수] /min
- SpO2: [산소포화도] %
- 체온: [체온] °C
- CVP: [중심정맥압] mmHg (해당시)
- ICP: [뇌압] mmHg (해당시)

인공호흡기 설정 (해당시):
- Mode: [모드]
- FiO2: [산소농도] %
- PEEP: [PEEP] cmH2O
- Tidal Volume: [일회호흡량] mL
- Rate: [호흡수] /min
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="수액 균형 (Fluid Balance)",
                content="""
지난 24시간:
Input:
- IV fluid: [mL]
- Enteral: [mL]
- Blood products: [mL]
- Total input: [mL]

Output:
- Urine: [mL] ([mL/kg/hr])
- Drain: [mL]
- Other losses: [mL]
- Total output: [mL]

Net balance: [+/-mL]
누적 balance: [+/-mL] (ICU 입실 후)
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="검사실 결과",
                content="""
혈액가스분석 ([시간]):
- pH: [값]
- pCO2: [값] mmHg
- pO2: [값] mmHg
- HCO3: [값] mEq/L
- Lactate: [값] mmol/L

일반혈액검사:
- Hgb: [값] g/dL
- WBC: [값] K/uL
- Platelet: [값] K/uL

생화학검사:
- BUN/Cr: [값]/[값] mg/dL
- Na/K: [값]/[값] mEq/L
- 간기능: AST/ALT [값]/[값] U/L

기타 중요 검사:
- [검사명]: [결과]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="현재 치료",
                content="""
혈관활성제 (Vasopressors/Inotropes):
- [약물명]: [용량] mcg/kg/min

진정/진통제:
- [약물명]: [용량]

항생제:
- [약물명]: [용량] [용법] (D[N])

기타 주요 약물:
- [약물명]: [용량] [용법] - [목적]

영양:
- 경로: [NPO/enteral/parenteral]
- 칼로리: [kcal/day]

기타 처치:
- [처치 내용]
                """.strip(),
                required=True
            ),
            TemplateSection(
                title="Assessment & Plan",
                content="""
1. [주요 문제 1]:
   - Assessment: [현재 상태 평가]
   - Plan: [치료 계획]

2. [주요 문제 2]:
   - Assessment: [현재 상태 평가]
   - Plan: [치료 계획]

전반적 계획:
- ICU 재원 기간: [예상]
- 목표: [치료 목표]
- 가족 면담: [계획]

퇴실 기준:
- [퇴실 기준들]
                """.strip(),
                required=True
            )
        ]

        template = NoteTemplate(
            template_id="icu_001",
            name="중환자실 표준기록",
            note_type=NoteType.ICU_NOTE,
            department=Department.INTERNAL_MEDICINE,
            sections=sections,
            metadata={
                "target_users": ["중환자의학과 전문의", "내과 전문의", "전공의"],
                "estimated_time": "30-40분",
                "complexity": "high",
                "special_requirements": ["중환자 모니터링", "정밀 수액 관리"]
            },
            created_by="시스템",
            created_date=datetime.now()
        )

        self.templates[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[NoteTemplate]:
        """템플릿 조회"""
        return self.templates.get(template_id)

    def get_templates_by_type(self, note_type: NoteType) -> List[NoteTemplate]:
        """노트 유형별 템플릿 조회"""
        return [template for template in self.templates.values()
                if template.note_type == note_type]

    def get_templates_by_department(self, department: Department) -> List[NoteTemplate]:
        """진료과별 템플릿 조회"""
        return [template for template in self.templates.values()
                if template.department == department]

    def add_custom_template(self, template: NoteTemplate) -> bool:
        """사용자 정의 템플릿 추가"""
        if template.template_id in self.templates:
            return False

        self.templates[template.template_id] = template
        return True

    def validate_template(self, template: NoteTemplate) -> List[str]:
        """템플릿 유효성 검증"""
        errors = []

        # 필수 섹션 확인
        required_sections = [section for section in template.sections if section.required]
        if not required_sections:
            errors.append("최소 하나의 필수 섹션이 필요합니다.")

        # 섹션별 유효성 검증
        for section in template.sections:
            section_errors = self._validate_section(section)
            errors.extend(section_errors)

        return errors

    def _validate_section(self, section: TemplateSection) -> List[str]:
        """섹션 유효성 검증"""
        errors = []

        if not section.title.strip():
            errors.append("섹션 제목이 비어있습니다.")

        if section.max_length and len(section.content) > section.max_length:
            errors.append(f"섹션 '{section.title}'의 내용이 최대 길이({section.max_length})를 초과합니다.")

        # 유효성 규칙 검증
        for rule in section.validation_rules:
            if rule == "not_empty" and not section.content.strip():
                errors.append(f"섹션 '{section.title}'의 내용이 비어있습니다.")
            elif rule == "icd10_format":
                if not re.search(r'ICD-10:\s*[A-Z]\d{2}', section.content):
                    errors.append(f"섹션 '{section.title}'에 올바른 ICD-10 형식이 필요합니다.")

        return errors

    def export_template(self, template_id: str, format_type: str = "json") -> str:
        """템플릿 내보내기"""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"템플릿 ID '{template_id}'를 찾을 수 없습니다.")

        if format_type.lower() == "json":
            return json.dumps(asdict(template), default=str, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"지원하지 않는 형식: {format_type}")

    def import_template(self, template_data: str, format_type: str = "json") -> bool:
        """템플릿 가져오기"""
        try:
            if format_type.lower() == "json":
                data = json.loads(template_data)
                # 딕셔너리를 NoteTemplate 객체로 변환하는 로직 필요
                # 여기서는 간단화
                return True
            else:
                raise ValueError(f"지원하지 않는 형식: {format_type}")
        except Exception as e:
            print(f"템플릿 가져오기 실패: {e}")
            return False

def demonstrate_templates():
    """템플릿 시스템 시연"""
    print("🏥 의료 노트 템플릿 시스템 시연")
    print("=" * 50)

    # 템플릿 관리자 생성
    manager = TemplateManager()

    print("📋 사용 가능한 템플릿 목록:")
    for template_id, template in manager.templates.items():
        print(f"- {template_id}: {template.name} ({template.note_type.value})")

    print("\n" + "="*50)

    # 입원기록 템플릿 시연
    admission_template = manager.get_template("admission_001")
    if admission_template:
        print("📝 입원기록 템플릿 예시:")
        print(f"템플릿명: {admission_template.name}")
        print(f"노트 유형: {admission_template.note_type.value}")
        print(f"진료과: {admission_template.department.value}")
        print(f"예상 작성 시간: {admission_template.metadata['estimated_time']}")

        print("\n섹션 구성:")
        for i, section in enumerate(admission_template.sections[:3], 1):  # 처음 3개만 표시
            print(f"{i}. {section.title} {'(필수)' if section.required else '(선택)'}")
            print(f"   {section.content[:100]}...")
            if section.validation_rules:
                print(f"   검증 규칙: {', '.join(section.validation_rules)}")
            print()

    print("="*50)

    # 경과기록 템플릿 시연
    progress_template = manager.get_template("progress_001")
    if progress_template:
        print("📊 경과기록 템플릿 (SOAP 형식):")
        print(f"템플릿명: {progress_template.name}")

        soap_sections = ["Subjective", "Objective", "Assessment", "Plan"]
        for i, section in enumerate(progress_template.sections[1:5], 1):  # SOAP 섹션만
            print(f"\n{soap_sections[i-1] if i <= len(soap_sections) else section.title}:")
            print(f"  {section.content[:150]}...")

    print("\n" + "="*50)

    # 템플릿 검증 시연
    print("🔍 템플릿 검증 예시:")
    for template_id in ["admission_001", "progress_001"]:
        template = manager.get_template(template_id)
        errors = manager.validate_template(template)
        if errors:
            print(f"❌ {template.name}: {len(errors)}개 오류")
            for error in errors[:3]:  # 처음 3개만 표시
                print(f"   - {error}")
        else:
            print(f"✅ {template.name}: 검증 통과")

    print("\n" + "="*50)

    # 의료 용어 시연
    print("📚 의료 용어 사전 예시:")
    terminology = manager.terminology

    print("주요 증상:")
    for symptom, info in list(terminology.symptoms.items())[:5]:
        print(f"- {symptom}: {info['korean']} (ICD-10: {info['code']})")

    print("\n주요 진단:")
    for diagnosis, info in list(terminology.diagnoses.items())[:5]:
        print(f"- {diagnosis}: {info['korean']} (ICD-10: {info['icd10']})")

if __name__ == "__main__":
    demonstrate_templates()