"""
Model dữ liệu vi phạm và kết quả nhận diện.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple


class ViolationType(str, Enum):
    """Loại vi phạm trang bị bảo hộ (theo dataset: Non-Helmet, bare-arms, ...)."""
    MISSING_HELMET = "missing_helmet"    # Non-Helmet → Thiếu mũ bảo hộ
    MISSING_VEST = "missing_vest"       # bare-arms → Thiếu áo bảo hộ / để lộ tay
    MISSING_BOTH = "missing_both"       # Thiếu cả hai (tính từ detection)


@dataclass
class Violation:
    """Một bản ghi vi phạm."""
    id: str
    violation_type: ViolationType
    confidence: float  # 0-1
    timestamp: datetime
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2

    @property
    def label_vi(self) -> str:
        if self.violation_type == ViolationType.MISSING_HELMET:
            return "Thiếu mũ bảo hộ"
        if self.violation_type == ViolationType.MISSING_VEST:
            return "Thiếu áo bảo hộ / Để lộ tay"
        return "Thiếu cả hai"

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    @property
    def confidence_pct(self) -> str:
        return f"{int(self.confidence * 100)}%"


@dataclass
class DetectionResult:
    """Kết quả nhận diện trên một frame."""
    frame: any  # numpy array (BGR)
    violations: List[Violation] = field(default_factory=list)
    fps: float = 0.0

    @property
    def total(self) -> int:
        return len(self.violations)

    @property
    def missing_helmet_count(self) -> int:
        return sum(1 for v in self.violations if v.violation_type == ViolationType.MISSING_HELMET)

    @property
    def missing_vest_count(self) -> int:
        return sum(1 for v in self.violations if v.violation_type == ViolationType.MISSING_VEST)

    @property
    def missing_both_count(self) -> int:
        return sum(1 for v in self.violations if v.violation_type == ViolationType.MISSING_BOTH)
