import math
from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def quantize(self, grid: float = 5.0) -> "Point":
        return Point(x=round(self.x / grid) * grid, y=round(self.y / grid) * grid)


class Wall(BaseModel):
    id: str
    start: Point
    end: Point
    thickness: float = 0
    exterior: bool = False
    center: Point | None = None
    radius: float = 0
    start_angle: float = 0
    end_angle: float = 0

    @property
    def is_arc(self) -> bool:
        return self.center is not None and self.radius > 0

    def length(self) -> float:
        if self.is_arc:
            angle_span = abs(self.end_angle - self.start_angle)
            if angle_span > math.pi:
                angle_span = 2 * math.pi - angle_span
            return self.radius * angle_span
        return self.start.distance_to(self.end)

    def is_horizontal(self, tol: float = 0.5) -> bool:
        return abs(self.start.y - self.end.y) < tol

    def is_vertical(self, tol: float = 0.5) -> bool:
        return abs(self.start.x - self.end.x) < tol

    def angle(self) -> float:
        return math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)

    def midpoint(self) -> Point:
        if self.is_arc:
            mid_angle = (self.start_angle + self.end_angle) / 2
            return Point(
                x=self.center.x + self.radius * math.cos(mid_angle),
                y=self.center.y + self.radius * math.sin(mid_angle),
            )
        return Point(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def as_tuple(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.start.as_tuple(), self.end.as_tuple())


class Door(BaseModel):
    id: str = ""
    wall_id: str = ""
    offset: float = 0.0
    width: float = 0.0
    # Arc swing geometry (quarter-circle door indicator)
    center: Point | None = None
    radius: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0
    # Door type: "swing" | "sliding" | "double_swing" | "bifold" | "unknown"
    door_type: str = "unknown"


class Window(BaseModel):
    wall_id: str
    offset: float
    width: float


class WallGraph(BaseModel):
    walls: list[Wall] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
