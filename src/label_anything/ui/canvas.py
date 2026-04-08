from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tkinter as tk
from PIL import Image, ImageTk


def fit_scale_to_bounds(width: int, height: int, max_width: int, max_height: int) -> float:
    if width <= 0 or height <= 0:
        return 1.0

    bounded_width = max(max_width, 1)
    bounded_height = max(max_height, 1)
    return min(1.0, bounded_width / width, bounded_height / height)


@dataclass(slots=True)
class Binding:
    widget: tk.Misc
    sequence: str
    funcid: str


class BindingManager:
    def __init__(self) -> None:
        self._bindings: list[Binding] = []

    def bind(self, widget: tk.Misc, sequence: str, callback: Callable[[tk.Event], object]) -> None:
        funcid = widget.bind(sequence, callback)
        if funcid:
            self._bindings.append(Binding(widget=widget, sequence=sequence, funcid=funcid))

    def clear(self) -> None:
        for binding in self._bindings:
            binding.widget.unbind(binding.sequence, binding.funcid)
        self._bindings.clear()


class ZoomableCanvasFrame:
    def __init__(
        self,
        root: tk.Tk,
        width: int,
        height: int,
        *,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
        initial_scale: float = 1.0,
    ) -> None:
        self.root = root
        self.frame = tk.Frame(root)
        self.frame.grid(row=0, column=0, sticky="nsew")
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        requested_canvas_width = max(canvas_width or width, 1)
        requested_canvas_height = max(canvas_height or height, 1)
        self.canvas = tk.Canvas(
            self.frame,
            width=requested_canvas_width,
            height=requested_canvas_height,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.sidebar = tk.Frame(self.frame, padx=12, pady=12)
        self.sidebar.grid(row=0, column=1, sticky="ns")

        self.width = width
        self.height = height
        self.imscale = initial_scale
        self.min_imscale = min(0.9, initial_scale)
        self.max_imscale = 10.0
        self.zoom_delta = 0.9
        self.x_0_img = 0.0
        self.y_0_img = 0.0
        self.display_image = Image.new("RGB", (width, height))
        self._canvas_photo: ImageTk.PhotoImage | None = None
        self._image_item_id: int | None = None

        scaled_width = max(round(width * initial_scale), 1)
        scaled_height = max(round(height * initial_scale), 1)
        self.container = self.canvas.create_rectangle(0, 0, scaled_width, scaled_height, width=0)
        self.canvas.update()
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)

    def enable_pan_controls(self, include_middle_button: bool) -> None:
        self.canvas.bind("<Control-ButtonPress-1>", self.move_from)
        self.canvas.bind("<Control-B1-Motion>", self.move_to)
        self.canvas.bind("<Control-ButtonRelease-1>", self.move_from)
        if include_middle_button:
            self.canvas.bind("<ButtonPress-2>", self.move_from)
            self.canvas.bind("<B2-Motion>", self.move_to)
            self.canvas.bind("<ButtonRelease-2>", self.move_from)

    def clear_sidebar(self) -> None:
        for child in self.sidebar.winfo_children():
            child.destroy()

    def destroy(self) -> None:
        self.frame.destroy()

    def event_to_image_coords(self, event: tk.Event) -> tuple[float, float]:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return (x / self.imscale) - self.x_0_img, (y / self.imscale) - self.y_0_img

    def set_display_image(self, image: Image.Image) -> None:
        self.display_image = image
        self.show_image()

    def zoom(self, event: tk.Event) -> None:
        # The canvas zoom / pan logic is intentionally kept very close to the
        # original implementation because the coordinate math is easy to break.
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)
        if bbox is None or not (bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]):
            return

        scale = 1.0
        if event.num == 4 or event.delta > 0:
            next_imscale = self.imscale / self.zoom_delta
            if next_imscale > self.max_imscale:
                return
            scale = next_imscale / self.imscale
            self.imscale = next_imscale
        elif event.num == 5 or event.delta < 0:
            next_imscale = self.imscale * self.zoom_delta
            if next_imscale < self.min_imscale:
                return
            scale = next_imscale / self.imscale
            self.imscale = next_imscale
        else:
            return

        self.canvas.scale("all", x, y, scale, scale)
        self.show_image()

    def move_from(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()

    def show_image(self) -> None:
        bbox1 = self.canvas.bbox(self.container)
        if bbox1 is None:
            return

        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (
            self.canvas.canvasx(0),
            self.canvas.canvasy(0),
            self.canvas.canvasx(self.canvas.winfo_width()),
            self.canvas.canvasy(self.canvas.winfo_height()),
        )
        bbox = [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3]),
        ]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]

        x1 = max(bbox2[0] - bbox1[0], 0)
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]

        self.x_0_img = bbox1[0] / self.imscale
        self.y_0_img = bbox1[1] / self.imscale

        if int(x2 - x1) <= 0 or int(y2 - y1) <= 0:
            return

        crop_x = min(int(x2 / self.imscale), self.width)
        crop_y = min(int(y2 / self.imscale), self.height)
        image_tile = self.display_image.crop((int(x1 / self.imscale), int(y1 / self.imscale), crop_x, crop_y))
        canvas_image = image_tile.resize((int(x2 - x1), int(y2 - y1)))
        self._canvas_photo = ImageTk.PhotoImage(canvas_image)

        if self._image_item_id is not None:
            self.canvas.delete(self._image_item_id)
        self._image_item_id = self.canvas.create_image(
            max(bbox2[0], bbox1[0]),
            max(bbox2[1], bbox1[1]),
            anchor="nw",
            image=self._canvas_photo,
        )
        self.canvas.lower(self._image_item_id)
