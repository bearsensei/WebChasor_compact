# src/toolset.py
from dataclasses import dataclass

@dataclass
class Toolset:
    router: any
    planner: any = None
    serp: any = None
    visitor: any = None
    extractor: any = None
    synthesizer: any = None
    code_exec: any = None
    image_ocr: any = None