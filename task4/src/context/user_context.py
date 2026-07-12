from dataclasses import dataclass
from typing import Optional

@dataclass
class UserContext:
    role: str
    department: str
    state: str
    district: str
    city: Optional[str] = None
    
    def dict(self):
        return {
            "role": self.role,
            "department": self.department,
            "state": self.state,
            "district": self.district,
            "city": self.city
        }
