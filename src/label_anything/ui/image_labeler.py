from __future__ import annotations

import tkinter as tk
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..files import find_existing_mask, final_image_output_path, final_mask_output_path, label_output_name, list_image_files
from ..masks import (
    BoolMask,
    apply_morphology,
    draw_mask_output,
    empty_mask,
    load_binary_mask,
    load_rgb_image,
    mask_to_overlay,
    remove_from_mask,
    strokes_to_mask,
)
from ..model import load_image_predictor, sam_inference_mode
from .canvas import BindingManager, ZoomableCanvasFrame, fit_scale_to_bounds

Point = tuple[float, float]
INITIAL_VIEW_MAX_WIDTH = 1000
INITIAL_VIEW_MIN_WIDTH = 200
INITIAL_VIEW_MIN_HEIGHT = 200
WINDOW_WIDTH_MARGIN = 280
WINDOW_HEIGHT_MARGIN = 120


@dataclass(slots=True)
class ProposalLayer:
    positive_points: list[Point] = field(default_factory=list)
    negative_points: list[Point] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    mask: BoolMask | None = None


class ImageLabelerSession(ZoomableCanvasFrame):
    def __init__(
        self,
        root: tk.Tk,
        model,
        image_path: Path,
        mask_path: Path | None = None,
        rgb_to_bgr: bool = False,
    ) -> None:
        self.original_image, self.original_size = load_rgb_image(image_path, rgb_to_bgr=rgb_to_bgr)
        max_canvas_width = min(
            INITIAL_VIEW_MAX_WIDTH,
            max(root.winfo_screenwidth() - WINDOW_WIDTH_MARGIN, INITIAL_VIEW_MIN_WIDTH),
        )
        max_canvas_height = max(root.winfo_screenheight() - WINDOW_HEIGHT_MARGIN, INITIAL_VIEW_MIN_HEIGHT)
        initial_scale = fit_scale_to_bounds(
            self.original_image.width,
            self.original_image.height,
            max_canvas_width,
            max_canvas_height,
        )
        initial_canvas_width = max(round(self.original_image.width * initial_scale), 1)
        initial_canvas_height = max(round(self.original_image.height * initial_scale), 1)

        super().__init__(
            root=root,
            width=self.original_image.width,
            height=self.original_image.height,
            canvas_width=initial_canvas_width,
            canvas_height=initial_canvas_height,
            initial_scale=initial_scale,
        )
        self.enable_pan_controls(include_middle_button=True)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.model = model
        self.image_path = image_path
        self.mask_path = mask_path
        self.mode = "init"
        self.exit_state = "quit"
        self.model_image_ready = False

        self.bindings = BindingManager()
        self.mask: BoolMask = (
            load_binary_mask(mask_path, self.original_image.size) if mask_path is not None else empty_mask(self.original_image.size)
        )
        self.unsubmitted_brush_strokes: list[Image.Image] = []
        self.unsubmitted_eraser_strokes: list[Image.Image] = []
        self.edit_actions: list[str] = []
        self.brush_size = 25

        self.proposal_layers: list[ProposalLayer] = []
        self.current_layer_index = -1
        self.max_num_masks = 10
        self.kp_radius = 5
        self.mask_alpha = 128
        self.mask_colors = (
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 255, 0),
            (255, 0, 255),
            (128, 0, 255),
            (0, 128, 255),
            (128, 255, 0),
            (0, 255, 128),
            (255, 128, 0),
        )
        self.current_brush_stroke_img: Image.Image | None = None
        self.current_eraser_stroke_img: Image.Image | None = None
        self.last_x = 0.0
        self.last_y = 0.0
        self.current_mask_label: tk.Label | None = None

        self.render_image()
        self.main_menu()

    def run(self) -> None:
        self.root.mainloop()

    def destroy(self) -> None:
        self.bindings.clear()
        super().destroy()

    def _clear_mode(self) -> None:
        self.bindings.clear()
        self.clear_sidebar()

    def _reset_edit_state(self) -> None:
        self.unsubmitted_brush_strokes = []
        self.unsubmitted_eraser_strokes = []
        self.edit_actions = []

    def _commit_pending_proposals(self) -> None:
        if not self.proposal_layers:
            return

        for layer in self.proposal_layers:
            if layer.mask is not None:
                self.mask = np.logical_or(self.mask, layer.mask)

        self.proposal_layers = []
        self.current_layer_index = -1

    def _draw_points(self, image: Image.Image, points: list[Point], color: tuple[int, int, int]) -> None:
        for x, y in points:
            point_img = Image.new("RGBA", image.size)
            draw = ImageDraw.Draw(point_img)
            draw.ellipse(
                (x - self.kp_radius, y - self.kp_radius, x + self.kp_radius, y + self.kp_radius),
                fill=(*color, 128),
            )
            image.paste(point_img, (0, 0), point_img)

    def _current_layer(self) -> ProposalLayer:
        return self.proposal_layers[self.current_layer_index]

    def _update_current_mask_badge(self) -> None:
        if self.current_mask_label is None or self.current_layer_index < 0:
            return
        label_color = "#%02x%02x%02x" % self.mask_colors[self.current_layer_index]
        self.current_mask_label.config(text=" Current Mask ", bg=label_color, fg="black")

    def _compose_image(self) -> Image.Image:
        composed = deepcopy(self.original_image)
        main_mask_overlay = mask_to_overlay(self.mask, color=(0, 0, 255), alpha=self.mask_alpha)
        composed.paste(main_mask_overlay, (0, 0), main_mask_overlay)

        if self.mode == "proposal":
            for index, layer in enumerate(self.proposal_layers):
                if layer.mask is None:
                    continue
                layer_overlay = mask_to_overlay(layer.mask, color=self.mask_colors[index], alpha=self.mask_alpha)
                composed.paste(layer_overlay, (0, 0), layer_overlay)

            if self.current_layer_index >= 0:
                current_layer = self._current_layer()
                self._draw_points(composed, current_layer.positive_points, (0, 255, 0))
                self._draw_points(composed, current_layer.negative_points, (255, 0, 0))

        for brush_stroke in self.unsubmitted_brush_strokes:
            composed.paste(brush_stroke, (0, 0), brush_stroke)
        for eraser_stroke in self.unsubmitted_eraser_strokes:
            composed.paste(eraser_stroke, (0, 0), eraser_stroke)

        return composed

    def main_menu(self) -> None:
        self._commit_pending_proposals()
        self._reset_edit_state()
        self.mode = "main_menu"
        self._clear_mode()
        self.render_image()

        tk.Button(self.sidebar, text="Prev Job", command=self.prev_image).pack(fill="x", pady=(0, 8))
        tk.Button(self.sidebar, text="Next Job", command=self.next_image).pack(fill="x", pady=(0, 24))
        tk.Button(self.sidebar, text="Propose Mask", bg="cornflower blue", command=self.proposal_mode).pack(
            fill="x",
            pady=(0, 8),
        )
        tk.Button(self.sidebar, text="Edit Mask", bg="medium purple", command=self.editing_mode).pack(
            fill="x",
            pady=(0, 24),
        )
        tk.Button(self.sidebar, text="Mark as Done", bg="pale green", command=self.finish_labeling_job).pack(
            fill="x",
            pady=(0, 8),
        )
        tk.Button(self.sidebar, text="Delete Mask", bg="tomato", command=self.delete_mask).pack(fill="x", pady=(0, 24))
        tk.Button(self.sidebar, text="Quit", bg="AntiqueWhite4", command=self.quit).pack(fill="x")

    def proposal_mode(self) -> None:
        self.mode = "proposal"
        self._clear_mode()

        self.bindings.bind(self.canvas, "<ButtonPress-1>", self.new_positive_keypoint)
        self.bindings.bind(self.canvas, "<ButtonPress-3>", self.new_negative_keypoint)
        self.bindings.bind(self.root, "p", self.propose_mask)
        self.bindings.bind(self.root, "u", self.undo_last_proposal_action)
        self.bindings.bind(self.root, "c", self.create_new_mask)
        self.bindings.bind(self.root, "n", self.next_mask)
        self.bindings.bind(self.root, "r", self.reset_keypoints)

        self.current_mask_label = tk.Label(self.sidebar, text=" Current Mask ")
        self.current_mask_label.pack(fill="x", pady=(0, 16))

        tk.Button(self.sidebar, text="Undo Keypoint (U)", bg="dark salmon", command=self.undo_last_proposal_action).pack(
            fill="x",
            pady=(0, 8),
        )
        tk.Button(self.sidebar, text="Create New Mask (C)", command=self.create_new_mask).pack(fill="x", pady=(0, 8))
        tk.Button(self.sidebar, text="Next Mask (N)", command=self.next_mask).pack(fill="x", pady=(0, 24))
        tk.Button(self.sidebar, text="Submit Current Mask", bg="pale green", command=self.submit_mask).pack(
            fill="x",
            pady=(0, 8),
        )
        tk.Button(self.sidebar, text="Reset Current Mask (R)", bg="tomato", command=self.reset_keypoints).pack(
            fill="x",
            pady=(0, 24),
        )
        tk.Button(self.sidebar, text="Main Menu", bg="AntiqueWhite4", command=self.main_menu).pack(fill="x", pady=(0, 24))

        self.mask_alpha_label = ttk.Label(self.sidebar, text=f"Alpha: {self.mask_alpha}")
        self.mask_alpha_label.pack(fill="x")
        self.alpha_slider = ttk.Scale(self.sidebar, from_=1, to=255, orient="horizontal", command=self.change_mask_alpha)
        self.alpha_slider.set(self.mask_alpha)
        self.alpha_slider.pack(fill="x", pady=(4, 0))

        self.proposal_layers = []
        self.current_layer_index = -1
        self.create_new_mask()

        if not self.model_image_ready:
            with sam_inference_mode():
                self.model.set_image(np.asarray(self.original_image))
            self.model_image_ready = True

        self.render_image()

    def new_positive_keypoint(self, event: tk.Event) -> None:
        image_x, image_y = self.event_to_image_coords(event)
        layer = self._current_layer()
        layer.positive_points.append((image_x, image_y))
        layer.actions.append("pos")
        self.propose_mask()

    def new_negative_keypoint(self, event: tk.Event) -> None:
        image_x, image_y = self.event_to_image_coords(event)
        layer = self._current_layer()
        layer.negative_points.append((image_x, image_y))
        layer.actions.append("neg")
        self.propose_mask()

    def undo_last_proposal_action(self, _event: tk.Event | None = None) -> None:
        layer = self._current_layer()
        if not layer.actions:
            return
        last_action = layer.actions.pop()
        if last_action == "pos":
            layer.positive_points.pop()
        else:
            layer.negative_points.pop()
        self.propose_mask()

    def submit_mask(self) -> None:
        layer = self._current_layer()
        if layer.mask is not None:
            self.mask = np.logical_or(self.mask, layer.mask)
        layer.positive_points = []
        layer.negative_points = []
        layer.actions = []
        layer.mask = None
        self.propose_mask()

    def reset_keypoints(self, _event: tk.Event | None = None) -> None:
        layer = self._current_layer()
        layer.positive_points = []
        layer.negative_points = []
        layer.actions = []
        layer.mask = None
        self.propose_mask()

    def propose_mask(self, _event: tk.Event | None = None) -> None:
        layer = self._current_layer()
        if not layer.positive_points and not layer.negative_points:
            layer.mask = None
            self.render_image()
            return

        points: list[Point] = []
        labels: list[int] = []
        points.extend(layer.positive_points)
        labels.extend([1] * len(layer.positive_points))
        points.extend(layer.negative_points)
        labels.extend([0] * len(layer.negative_points))

        with sam_inference_mode():
            masks, scores, _ = self.model.predict(
                point_coords=np.asarray(points, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=True,
            )

        best_index = int(np.argmax(scores))
        layer.mask = np.asarray(masks[best_index]).astype(bool)
        self.render_image()

    def next_mask(self, _event: tk.Event | None = None) -> None:
        self.current_layer_index += 1
        if self.current_layer_index >= len(self.proposal_layers):
            self.current_layer_index = 0
        self._update_current_mask_badge()
        self.render_image()

    def create_new_mask(self, _event: tk.Event | None = None) -> None:
        if len(self.proposal_layers) >= self.max_num_masks:
            print("You have reached the maximum number of masks.")
            return
        self.proposal_layers.append(ProposalLayer())
        self.current_layer_index = len(self.proposal_layers) - 1
        self._update_current_mask_badge()
        self.render_image()

    def editing_mode(self) -> None:
        self.mode = "editing"
        self._clear_mode()

        self.bindings.bind(self.canvas, "<ButtonPress-1>", self.activate_brush)
        self.bindings.bind(self.canvas, "<ButtonRelease-1>", self.deactivate_brush)
        self.bindings.bind(self.canvas, "<ButtonPress-3>", self.activate_eraser)
        self.bindings.bind(self.canvas, "<ButtonRelease-3>", self.deactivate_eraser)
        self.bindings.bind(self.root, "u", self.undo_last_edit_action)
        self.bindings.bind(self.root, "s", self.submit_edit_actions)

        ttk.Label(self.sidebar, text=f"Brush Size: {self.brush_size}").pack(fill="x")
        self.brush_size_label = self.sidebar.winfo_children()[-1]
        self.brush_slider = ttk.Scale(self.sidebar, from_=1, to=75, orient="horizontal", command=self.change_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(fill="x", pady=(4, 16))

        self.mask_alpha_label = ttk.Label(self.sidebar, text=f"Alpha: {self.mask_alpha}")
        self.mask_alpha_label.pack(fill="x")
        self.alpha_slider = ttk.Scale(self.sidebar, from_=1, to=255, orient="horizontal", command=self.change_mask_alpha)
        self.alpha_slider.set(self.mask_alpha)
        self.alpha_slider.pack(fill="x", pady=(4, 16))

        tk.Button(self.sidebar, text="Remove Imperfections", command=self.remove_imperfections).pack(fill="x", pady=(0, 8))
        tk.Button(self.sidebar, text="Submit (S)", bg="pale green", command=self.submit_edit_actions).pack(
            fill="x",
            pady=(0, 8),
        )
        tk.Button(self.sidebar, text="Undo (U)", bg="tomato", command=self.undo_last_edit_action).pack(
            fill="x",
            pady=(0, 8),
        )
        tk.Button(self.sidebar, text="Invert Mask", bg="medium purple", command=self.invert_mask).pack(
            fill="x",
            pady=(0, 24),
        )
        tk.Button(self.sidebar, text="Main Menu", bg="AntiqueWhite4", command=self.main_menu).pack(fill="x")

        self.canvas.update()

    def morph_close(self) -> None:
        self.mask = apply_morphology(self.mask, cv2.MORPH_CLOSE)
        self.render_image()

    def morph_open(self) -> None:
        self.mask = apply_morphology(self.mask, cv2.MORPH_OPEN)
        self.render_image()

    def remove_imperfections(self) -> None:
        self.mask = apply_morphology(apply_morphology(self.mask, cv2.MORPH_OPEN), cv2.MORPH_CLOSE)
        self.render_image()

    def invert_mask(self) -> None:
        self.mask = np.logical_not(self.mask)
        self.render_image()

    def activate_brush(self, event: tk.Event) -> None:
        self.canvas.bind("<B1-Motion>", self.brush)
        self.last_x, self.last_y = self.event_to_image_coords(event)
        self.current_brush_stroke_img = Image.new("RGBA", self.original_image.size)
        self.draw = ImageDraw.Draw(self.current_brush_stroke_img)

    def brush(self, event: tk.Event) -> None:
        if self.current_brush_stroke_img is None:
            return
        image_x, image_y = self.event_to_image_coords(event)
        width = max(int(self.brush_size / self.imscale), 1)
        radius = max(int(self.brush_size / 2 / self.imscale), 1)
        self.draw.line((self.last_x, self.last_y, image_x, image_y), width=width, fill=(0, 255, 0, 128))
        self.draw.ellipse((image_x - radius, image_y - radius, image_x + radius, image_y + radius), fill=(0, 255, 0, 128))
        preview_image = self._compose_image()
        preview_image.paste(self.current_brush_stroke_img, (0, 0), self.current_brush_stroke_img)
        self.set_display_image(preview_image)
        self.last_x, self.last_y = image_x, image_y

    def deactivate_brush(self, _event: tk.Event) -> None:
        if self.current_brush_stroke_img is None:
            return
        self.unsubmitted_brush_strokes.append(self.current_brush_stroke_img)
        self.edit_actions.append("brush")
        self.current_brush_stroke_img = None
        self.render_image()

    def activate_eraser(self, event: tk.Event) -> None:
        self.canvas.bind("<B3-Motion>", self.erase)
        self.last_x, self.last_y = self.event_to_image_coords(event)
        self.current_eraser_stroke_img = Image.new("RGBA", self.original_image.size)
        self.draw = ImageDraw.Draw(self.current_eraser_stroke_img)

    def erase(self, event: tk.Event) -> None:
        if self.current_eraser_stroke_img is None:
            return
        image_x, image_y = self.event_to_image_coords(event)
        width = max(int(self.brush_size / self.imscale), 1)
        radius = max(int(self.brush_size / 2 / self.imscale), 1)
        self.draw.line((self.last_x, self.last_y, image_x, image_y), width=width, fill=(255, 0, 0, 128))
        self.draw.ellipse((image_x - radius, image_y - radius, image_x + radius, image_y + radius), fill=(255, 0, 0, 128))
        preview_image = self._compose_image()
        preview_image.paste(self.current_eraser_stroke_img, (0, 0), self.current_eraser_stroke_img)
        self.set_display_image(preview_image)
        self.last_x, self.last_y = image_x, image_y

    def deactivate_eraser(self, _event: tk.Event) -> None:
        if self.current_eraser_stroke_img is None:
            return
        self.unsubmitted_eraser_strokes.append(self.current_eraser_stroke_img)
        self.edit_actions.append("eraser")
        self.current_eraser_stroke_img = None
        self.render_image()

    def undo_last_edit_action(self, _event: tk.Event | None = None) -> None:
        if not self.edit_actions:
            return
        last_action = self.edit_actions.pop()
        if last_action == "brush":
            self.unsubmitted_brush_strokes.pop()
        else:
            self.unsubmitted_eraser_strokes.pop()
        self.render_image()

    def submit_edit_actions(self, _event: tk.Event | None = None) -> None:
        brush_mask = strokes_to_mask(self.unsubmitted_brush_strokes, self.original_image.size, channel_index=1)
        eraser_mask = strokes_to_mask(self.unsubmitted_eraser_strokes, self.original_image.size, channel_index=0)

        self.mask = np.logical_or(self.mask, brush_mask)
        self.mask = remove_from_mask(self.mask, eraser_mask)
        self._reset_edit_state()
        self.render_image()

    def change_brush_size(self, value: str) -> None:
        self.brush_size = round(float(value))
        self.brush_size_label.config(text=f"Brush Size: {self.brush_size}")

    def change_mask_alpha(self, value: str) -> None:
        self.mask_alpha = round(float(value))
        self.mask_alpha_label.config(text=f"Alpha: {self.mask_alpha}")
        self.render_image()

    def save_mask(self, *, final: bool = False) -> None:
        mask_image = draw_mask_output(self.mask, self.original_size)
        if not final:
            mask_image.save(self.image_path.with_name(label_output_name(self.image_path)))
            return

        image_output = final_image_output_path(self.image_path)
        mask_output = final_mask_output_path(self.image_path)
        if image_output.exists():
            image_output.unlink()
        self.image_path.replace(image_output)
        mask_image.save(mask_output)

    def delete_mask(self) -> None:
        self.mask = empty_mask(self.original_image.size)
        self.render_image()

    def next_image(self) -> None:
        self.save_mask()
        self.exit_state = "next"
        self.root.quit()

    def prev_image(self) -> None:
        self.save_mask()
        self.exit_state = "prev"
        self.root.quit()

    def finish_labeling_job(self) -> None:
        self.save_mask(final=True)
        self.exit_state = "finish"
        self.root.quit()

    def quit(self) -> None:
        self.save_mask()
        self.exit_state = "quit"
        self.root.quit()

    def render_image(self) -> None:
        self.set_display_image(self._compose_image())


def start_image_labeling(
    tasks_dir: Path,
    rgb_to_bgr: bool = False,
    model_type: str = "base_plus",
    sam2_dir: Path | None = None,
    custom_weights: Path | None = None,
    fullscreen: bool = False,
) -> None:
    image_files = list_image_files(tasks_dir)
    if not image_files:
        raise FileNotFoundError(f"No labelable images found in {tasks_dir}")

    root = tk.Tk()
    if fullscreen:
        root.attributes("-fullscreen", True)

    model = load_image_predictor(model_type=model_type, sam2_dir=sam2_dir, custom_weights=custom_weights)
    image_index = 0

    try:
        while image_files:
            image_path = image_files[image_index]
            mask_path = find_existing_mask(image_path)
            root.title(f"Job {image_index + 1}/{len(image_files)} - ({image_path.stem})")

            session = ImageLabelerSession(
                root=root,
                model=model,
                image_path=image_path,
                mask_path=mask_path,
                rgb_to_bgr=rgb_to_bgr,
            )
            session.run()
            exit_state = session.exit_state
            session.destroy()

            if exit_state == "next":
                image_index = min(image_index + 1, len(image_files) - 1)
            elif exit_state == "prev":
                image_index = max(image_index - 1, 0)
            elif exit_state == "finish":
                image_files = list_image_files(tasks_dir)
                if not image_files:
                    break
                image_index = min(image_index, len(image_files) - 1)
            elif exit_state == "quit":
                break
    finally:
        root.destroy()
