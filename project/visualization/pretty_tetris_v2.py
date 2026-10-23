import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tetris_gymnasium.envs import Tetris


# ============================================================
# Window / Layout
# ============================================================

WINDOW_NAME = "DSA5102 - Tetris RL Project"

CANVAS_W = 1100
CANVAS_H = 820

CELL = 28

BOARD_X = 410
BOARD_Y = 130


# ============================================================
# Color Palette
# OpenCV uses BGR
# ============================================================

BG_TOP = (46, 36, 24)
BG_BOTTOM = (24, 17, 11)

PANEL = (58, 47, 33)
PANEL_LIGHT = (68, 57, 42)

BORDER = (105, 92, 72)
GRID = (65, 57, 45)

ACCENT = (235, 196, 72)
ACCENT_SOFT = (190, 160, 74)

WHITE = (245, 248, 252)
TEXT = (226, 232, 240)
MUTED = (156, 170, 186)


# ============================================================
# Font Handling
# ============================================================

_font_cache = {}


def get_font(size, bold=False):

    key = (size, bold)

    if key in _font_cache:
        return _font_cache[key]

    if bold:
        candidates = [
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]

    for path in candidates:

        if os.path.exists(path):

            font = ImageFont.truetype(
                path,
                size,
            )

            _font_cache[key] = font

            return font

    font = ImageFont.load_default()

    _font_cache[key] = font

    return font


def add_text(
    commands,
    text,
    x,
    y,
    size=20,
    color=TEXT,
    bold=False,
    align="left",
):

    commands.append(
        {
            "text": str(text),
            "x": x,
            "y": y,
            "size": size,
            "color": color,
            "bold": bold,
            "align": align,
        }
    )


def apply_text(
    img,
    commands,
):

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB,
    )

    pil_img = Image.fromarray(rgb)

    draw = ImageDraw.Draw(
        pil_img
    )

    for cmd in commands:

        font = get_font(
            cmd["size"],
            cmd["bold"],
        )

        text = cmd["text"]

        x = cmd["x"]
        y = cmd["y"]

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_w = (
            bbox[2] - bbox[0]
        )

        if (
            cmd["align"]
            == "center"
        ):

            x -= text_w // 2

        elif (
            cmd["align"]
            == "right"
        ):

            x -= text_w

        b, g, r = cmd["color"]

        draw.text(
            (x, y),
            text,
            font=font,
            fill=(r, g, b),
        )

    result = cv2.cvtColor(
        np.array(pil_img),
        cv2.COLOR_RGB2BGR,
    )

    return result


# ============================================================
# Background / Panels
# ============================================================

def create_background():

    img = np.zeros(
        (
            CANVAS_H,
            CANVAS_W,
            3,
        ),
        dtype=np.uint8,
    )

    top = np.array(
        BG_TOP,
        dtype=np.float32,
    )

    bottom = np.array(
        BG_BOTTOM,
        dtype=np.float32,
    )

    for y in range(
        CANVAS_H
    ):

        t = (
            y
            / max(
                1,
                CANVAS_H - 1,
            )
        )

        color = (
            top * (1 - t)
            + bottom * t
        )

        img[
            y,
            :,
            :,
        ] = color.astype(
            np.uint8
        )

    return img


def draw_panel(
    img,
    x,
    y,
    w,
    h,
    accent=False,
):

    cv2.rectangle(
        img,
        (x, y),
        (
            x + w,
            y + h,
        ),
        PANEL,
        -1,
    )

    cv2.rectangle(
        img,
        (x, y),
        (
            x + w,
            y + h,
        ),
        BORDER,
        1,
    )

    if accent:

        cv2.line(
            img,
            (
                x + 1,
                y + 1,
            ),
            (
                x + w - 1,
                y + 1,
            ),
            ACCENT,
            3,
        )


# ============================================================
# Tetromino Drawing
# ============================================================

def get_piece_color_bgr(
    env,
    pixel_id,
):

    pixel_id = int(
        pixel_id
    )

    if (
        pixel_id <= 0
        or pixel_id
        >= len(env.pixels)
    ):

        return (
            100,
            100,
            100,
        )

    r, g, b = (
        env.pixels[
            pixel_id
        ].color_rgb
    )

    return (
        int(b),
        int(g),
        int(r),
    )


def draw_block(
    img,
    x,
    y,
    size,
    color,
):

    pad = 2

    x1 = x + pad
    y1 = y + pad

    x2 = (
        x
        + size
        - pad
    )

    y2 = (
        y
        + size
        - pad
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        -1,
    )

    bright = tuple(
        min(
            255,
            int(c) + 55,
        )
        for c in color
    )

    dark = tuple(
        max(
            0,
            int(c) - 65,
        )
        for c in color
    )

    cv2.line(
        img,
        (x1, y1),
        (x2, y1),
        bright,
        2,
    )

    cv2.line(
        img,
        (x1, y1),
        (x1, y2),
        bright,
        2,
    )

    cv2.line(
        img,
        (x1, y2),
        (x2, y2),
        dark,
        2,
    )

    cv2.line(
        img,
        (x2, y1),
        (x2, y2),
        dark,
        2,
    )

    inner = tuple(
        min(
            255,
            int(c) + 18,
        )
        for c in color
    )

    cv2.rectangle(
        img,
        (
            x1 + 4,
            y1 + 4,
        ),
        (
            x2 - 4,
            y2 - 4,
        ),
        inner,
        1,
    )


# ============================================================
# Game Board
# ============================================================

def draw_board(
    img,
    env,
    observation,
):

    board = env.crop_padding(
        observation["board"]
    )

    rows, cols = (
        board.shape
    )

    board_w = (
        cols * CELL
    )

    board_h = (
        rows * CELL
    )

    # Outer accent border
    cv2.rectangle(
        img,
        (
            BOARD_X - 8,
            BOARD_Y - 8,
        ),
        (
            BOARD_X
            + board_w
            + 8,
            BOARD_Y
            + board_h
            + 8,
        ),
        ACCENT_SOFT,
        2,
    )

    # Dark backing
    cv2.rectangle(
        img,
        (
            BOARD_X - 4,
            BOARD_Y - 4,
        ),
        (
            BOARD_X
            + board_w
            + 4,
            BOARD_Y
            + board_h
            + 4,
        ),
        (
            10,
            13,
            20,
        ),
        -1,
    )

    for r in range(rows):

        for c in range(cols):

            x = (
                BOARD_X
                + c * CELL
            )

            y = (
                BOARD_Y
                + r * CELL
            )

            cv2.rectangle(
                img,
                (x, y),
                (
                    x + CELL,
                    y + CELL,
                ),
                (
                    22,
                    28,
                    38,
                ),
                -1,
            )

            cv2.rectangle(
                img,
                (x, y),
                (
                    x + CELL,
                    y + CELL,
                ),
                GRID,
                1,
            )

            pixel_id = int(
                board[r, c]
            )

            if pixel_id > 0:

                color = (
                    get_piece_color_bgr(
                        env,
                        pixel_id,
                    )
                )

                draw_block(
                    img,
                    x,
                    y,
                    CELL,
                    color,
                )


# ============================================================
# HOLD / NEXT Preview
# ============================================================

def draw_tetromino_preview(
    img,
    tetromino,
    x,
    y,
    w,
    h,
    cell=21,
):

    if tetromino is None:
        return

    matrix = (
        tetromino.matrix
    )

    occupied = (
        np.argwhere(
            matrix > 0
        )
    )

    if occupied.size == 0:
        return

    r0, c0 = (
        occupied.min(
            axis=0
        )
    )

    r1, c1 = (
        occupied.max(
            axis=0
        )
    )

    shape = matrix[
        r0:r1 + 1,
        c0:c1 + 1,
    ]

    piece_h, piece_w = (
        shape.shape
    )

    start_x = (
        x
        + (
            w
            - piece_w
            * cell
        )
        // 2
    )

    start_y = (
        y
        + (
            h
            - piece_h
            * cell
        )
        // 2
    )

    r, g, b = (
        tetromino.color_rgb
    )

    color = (
        int(b),
        int(g),
        int(r),
    )

    for rr in range(
        piece_h
    ):

        for cc in range(
            piece_w
        ):

            if (
                shape[
                    rr,
                    cc,
                ]
                > 0
            ):

                draw_block(
                    img,
                    start_x
                    + cc
                    * cell,
                    start_y
                    + rr
                    * cell,
                    cell,
                    color,
                )


# ============================================================
# Main Gameplay UI
# ============================================================

def render_game_ui(
    env,
    observation,
    total_lines,
    display_score,
    total_reward,
    steps,
    episode,
    seed,
):

    img = create_background()

    texts = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    add_text(
        texts,
        "TETRIS RL",
        42,
        24,
        37,
        WHITE,
        True,
    )

    add_text(
        texts,
        "ENVIRONMENT & EVALUATION",
        44,
        68,
        15,
        MUTED,
        True,
    )

    # Controller box
    cv2.rectangle(
        img,
        (
            807,
            28,
        ),
        (
            1048,
            68,
        ),
        PANEL_LIGHT,
        -1,
    )

    cv2.rectangle(
        img,
        (
            807,
            28,
        ),
        (
            1048,
            68,
        ),
        BORDER,
        1,
    )

    add_text(
        texts,
        "CONTROLLER  HUMAN",
        927,
        37,
        17,
        WHITE,
        True,
        "center",
    )

    add_text(
        texts,
        f"Seed {seed}",
        1048,
        82,
        14,
        MUTED,
        False,
        "right",
    )

    # --------------------------------------------------------
    # Board title
    # --------------------------------------------------------

    add_text(
        texts,
        "GAME BOARD",
        BOARD_X,
        98,
        16,
        MUTED,
        True,
    )

    draw_board(
        img,
        env,
        observation,
    )

    # --------------------------------------------------------
    # HOLD Panel
    # --------------------------------------------------------

    hold_x = 45
    hold_y = 130
    hold_w = 295
    hold_h = 195

    draw_panel(
        img,
        hold_x,
        hold_y,
        hold_w,
        hold_h,
        True,
    )

    add_text(
        texts,
        "HOLD",
        hold_x + 20,
        hold_y + 17,
        22,
        WHITE,
        True,
    )

    held = (
        env.holder
        .get_tetrominoes()
    )

    held_piece = (
        held[0]
        if held
        else None
    )

    if (
        held_piece
        is None
    ):

        add_text(
            texts,
            "Empty",
            hold_x
            + hold_w // 2,
            hold_y + 104,
            18,
            MUTED,
            False,
            "center",
        )

    else:

        draw_tetromino_preview(
            img,
            held_piece,
            hold_x + 15,
            hold_y + 48,
            hold_w - 30,
            hold_h - 60,
            23,
        )

    # --------------------------------------------------------
    # GAME INFO
    # --------------------------------------------------------

    info_x = 45
    info_y = 350
    info_w = 295
    info_h = 335

    draw_panel(
        img,
        info_x,
        info_y,
        info_w,
        info_h,
        True,
    )

    add_text(
        texts,
        "GAME INFO",
        info_x + 20,
        info_y + 18,
        22,
        WHITE,
        True,
    )

    stats = [
        (
            "Score",
            display_score,
        ),
        (
            "Lines",
            total_lines,
        ),
        (
            "Steps",
            steps,
        ),
        (
            "Reward",
            f"{total_reward:.2f}",
        ),
        (
            "Episode",
            episode,
        ),
    ]

    yy = (
        info_y + 75
    )

    for (
        label,
        value,
    ) in stats:

        add_text(
            texts,
            label,
            info_x + 25,
            yy,
            18,
            MUTED,
        )

        add_text(
            texts,
            value,
            info_x + 260,
            yy - 2,
            24,
            WHITE,
            True,
            "right",
        )

        cv2.line(
            img,
            (
                info_x + 22,
                yy + 34,
            ),
            (
                info_x
                + info_w
                - 22,
                yy + 34,
            ),
            BORDER,
            1,
        )

        yy += 52

    # --------------------------------------------------------
    # NEXT Panel
    # --------------------------------------------------------

    next_x = 755
    next_y = 130
    next_w = 300
    next_h = 555

    draw_panel(
        img,
        next_x,
        next_y,
        next_w,
        next_h,
        True,
    )

    add_text(
        texts,
        "NEXT",
        next_x + 20,
        next_y + 17,
        22,
        WHITE,
        True,
    )

    queue_ids = (
        env.queue
        .get_queue()
    )

    preview_y = (
        next_y + 58
    )

    for (
        i,
        piece_id,
    ) in enumerate(
        queue_ids[:4]
    ):

        piece = (
            env.tetrominoes[
                int(piece_id)
            ]
        )

        draw_tetromino_preview(
            img,
            piece,
            next_x + 20,
            preview_y,
            next_w - 40,
            95,
            20,
        )

        if i < 3:

            cv2.line(
                img,
                (
                    next_x + 30,
                    preview_y
                    + 98,
                ),
                (
                    next_x
                    + next_w
                    - 30,
                    preview_y
                    + 98,
                ),
                BORDER,
                1,
            )

        preview_y += 118

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    ctrl_x = 45
    ctrl_y = 720
    ctrl_w = 1010
    ctrl_h = 70

    draw_panel(
        img,
        ctrl_x,
        ctrl_y,
        ctrl_w,
        ctrl_h,
        False,
    )

    controls = [
        (
            "A / D",
            "Move",
            68,
            124,
        ),
        (
            "W / E",
            "Rotate",
            205,
            262,
        ),
        (
            "SPACE",
            "Hard Drop",
            353,
            420,
        ),
        (
            "Q",
            "Hold",
            550,
            575,
        ),
        (
            "P",
            "Pause",
            654,
            680,
        ),
        (
            "R",
            "Restart",
            770,
            796,
        ),
        (
            "ESC",
            "Menu",
            909,
            956,
        ),
    ]

    for (
        key_name,
        description,
        key_x,
        desc_x,
    ) in controls:

        add_text(
            texts,
            key_name,
            key_x,
            739,
            16,
            WHITE,
            True,
        )

        add_text(
            texts,
            description,
            desc_x,
            740,
            15,
            MUTED,
        )

    return apply_text(
        img,
        texts,
    )


# ============================================================
# Start Screen
# ============================================================

def render_start_screen():

    img = (
        create_background()
    )

    texts = []

    block_size = 42

    demo_blocks = [
        (
            408,
            185,
            (
                255,
                210,
                50,
            ),
        ),
        (
            450,
            185,
            (
                255,
                210,
                50,
            ),
        ),
        (
            492,
            185,
            (
                255,
                210,
                50,
            ),
        ),
        (
            492,
            143,
            (
                255,
                210,
                50,
            ),
        ),
        (
            534,
            185,
            (
                190,
                90,
                240,
            ),
        ),
        (
            534,
            143,
            (
                190,
                90,
                240,
            ),
        ),
        (
            576,
            185,
            (
                190,
                90,
                240,
            ),
        ),
    ]

    for (
        x,
        y,
        color_rgb,
    ) in demo_blocks:

        r, g, b = (
            color_rgb
        )

        draw_block(
            img,
            x,
            y,
            block_size,
            (
                b,
                g,
                r,
            ),
        )

    add_text(
        texts,
        "TETRIS",
        CANVAS_W // 2,
        285,
        58,
        WHITE,
        True,
        "center",
    )

    add_text(
        texts,
        "REINFORCEMENT LEARNING PROJECT",
        CANVAS_W // 2,
        355,
        25,
        ACCENT,
        True,
        "center",
    )

    add_text(
        texts,
        "DSA5102  •  Environment  •  Training  •  Evaluation",
        CANVAS_W // 2,
        405,
        18,
        MUTED,
        False,
        "center",
    )

    # Research cards
    card_y = 485
    card_w = 220
    card_h = 72

    card_xs = [
        190,
        440,
        690,
    ]

    card_titles = [
        "ENVIRONMENT",
        "RL AGENTS",
        "EVALUATION",
    ]

    for (
        x,
        title,
    ) in zip(
        card_xs,
        card_titles,
    ):

        draw_panel(
            img,
            x,
            card_y,
            card_w,
            card_h,
            False,
        )

        add_text(
            texts,
            title,
            x + card_w // 2,
            card_y + 25,
            16,
            WHITE,
            True,
            "center",
        )

    # Pulsing prompt
    pulse = (
        np.sin(
            time.time()
            * 3.2
        )
        + 1
    ) / 2

    brightness = int(
        170
        + 70
        * pulse
    )

    prompt_color = (
        brightness,
        brightness,
        70,
    )

    add_text(
        texts,
        "PRESS ENTER TO START",
        CANVAS_W // 2,
        630,
        24,
        prompt_color,
        True,
        "center",
    )

    add_text(
        texts,
        "SPACE also starts",
        CANVAS_W // 2,
        672,
        15,
        MUTED,
        False,
        "center",
    )

    add_text(
        texts,
        "ESC  Quit",
        CANVAS_W // 2,
        710,
        16,
        MUTED,
        False,
        "center",
    )

    return apply_text(
        img,
        texts,
    )


# ============================================================
# Overlay Helpers
# ============================================================

def darken_background(
    img,
):

    overlay = (
        np.zeros_like(
            img
        )
    )

    return (
        cv2.addWeighted(
            img,
            0.35,
            overlay,
            0.65,
            0,
        )
    )


def render_pause_overlay(
    img,
):

    img = (
        darken_background(
            img
        )
    )

    texts = []

    panel_x = 350
    panel_y = 290
    panel_w = 400
    panel_h = 205

    draw_panel(
        img,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        True,
    )

    add_text(
        texts,
        "PAUSED",
        CANVAS_W // 2,
        325,
        38,
        WHITE,
        True,
        "center",
    )

    add_text(
        texts,
        "P  Resume",
        CANVAS_W // 2,
        392,
        19,
        TEXT,
        True,
        "center",
    )

    add_text(
        texts,
        "R  Restart     ESC  Menu",
        CANVAS_W // 2,
        438,
        16,
        MUTED,
        False,
        "center",
    )

    return apply_text(
        img,
        texts,
    )


def render_game_over_overlay(
    img,
    score,
    lines,
    reward,
    steps,
):

    img = (
        darken_background(
            img
        )
    )

    texts = []

    panel_x = 315
    panel_y = 205
    panel_w = 470
    panel_h = 395

    draw_panel(
        img,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        True,
    )

    add_text(
        texts,
        "GAME OVER",
        CANVAS_W // 2,
        245,
        40,
        WHITE,
        True,
        "center",
    )

    add_text(
        texts,
        "FINAL RESULTS",
        CANVAS_W // 2,
        307,
        16,
        ACCENT,
        True,
        "center",
    )

    results = [
        (
            "Score",
            score,
        ),
        (
            "Lines",
            lines,
        ),
        (
            "Steps",
            steps,
        ),
        (
            "Reward",
            f"{reward:.2f}",
        ),
    ]

    yy = 355

    for (
        label,
        value,
    ) in results:

        add_text(
            texts,
            label,
            405,
            yy,
            18,
            MUTED,
        )

        add_text(
            texts,
            value,
            690,
            yy - 2,
            23,
            WHITE,
            True,
            "right",
        )

        yy += 45

    add_text(
        texts,
        "R / ENTER   Play Again",
        CANVAS_W // 2,
        535,
        17,
        WHITE,
        True,
        "center",
    )

    add_text(
        texts,
        "ESC   Back to Menu",
        CANVAS_W // 2,
        568,
        15,
        MUTED,
        False,
        "center",
    )

    return apply_text(
        img,
        texts,
    )


# ============================================================
# Reset Game
# ============================================================

def reset_game(
    env,
    seed,
):

    observation, info = (
        env.reset(
            seed=seed
        )
    )

    total_lines = 0

    display_score = 0

    total_reward = 0.0

    steps = 0

    return (
        observation,
        total_lines,
        display_score,
        total_reward,
        steps,
    )


# ============================================================
# Return To Menu
# ============================================================

def return_to_menu(
    env,
    seed,
):

    (
        observation,
        total_lines,
        display_score,
        total_reward,
        steps,
    ) = reset_game(
        env,
        seed,
    )

    return (
        observation,
        total_lines,
        display_score,
        total_reward,
        steps,
        "start",
    )


# ============================================================
# Main Program
# ============================================================

def main():

    env = Tetris(
        render_mode=None,
        gravity=True,
    )

    seed = 42

    episode = 1

    (
        observation,
        total_lines,
        display_score,
        total_reward,
        steps,
    ) = reset_game(
        env,
        seed,
    )

    # Possible states:
    # start
    # playing
    # paused
    # game_over

    screen_state = "start"

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        CANVAS_W,
        CANVAS_H,
    )

    try:

        while True:

            # =================================================
            # Render Screen
            # =================================================

            if (
                screen_state
                == "start"
            ):

                frame = (
                    render_start_screen()
                )

            else:

                frame = (
                    render_game_ui(
                        env,
                        observation,
                        total_lines,
                        display_score,
                        total_reward,
                        steps,
                        episode,
                        seed,
                    )
                )

                if (
                    screen_state
                    == "paused"
                ):

                    frame = (
                        render_pause_overlay(
                            frame
                        )
                    )

                elif (
                    screen_state
                    == "game_over"
                ):

                    frame = (
                        render_game_over_overlay(
                            frame,
                            display_score,
                            total_lines,
                            total_reward,
                            steps,
                        )
                    )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            # Detect window closed manually
            visible = (
                cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                )
            )

            if visible < 1:
                break

            key = (
                cv2.waitKey(30)
                & 0xFF
            )

            # =================================================
            # START SCREEN
            # =================================================

            if (
                screen_state
                == "start"
            ):

                # ENTER or SPACE
                if key in (
                    13,
                    10,
                    ord(" "),
                ):

                    screen_state = (
                        "playing"
                    )

                # ESC = quit program
                elif key == 27:

                    break

                continue

            # =================================================
            # PAUSED
            # =================================================

            if (
                screen_state
                == "paused"
            ):

                # P = resume
                if (
                    key
                    == ord("p")
                ):

                    screen_state = (
                        "playing"
                    )

                # R = restart
                elif (
                    key
                    == ord("r")
                ):

                    episode += 1

                    (
                        observation,
                        total_lines,
                        display_score,
                        total_reward,
                        steps,
                    ) = reset_game(
                        env,
                        seed,
                    )

                    screen_state = (
                        "playing"
                    )

                # ESC = return to menu
                elif key == 27:

                    (
                        observation,
                        total_lines,
                        display_score,
                        total_reward,
                        steps,
                        screen_state,
                    ) = return_to_menu(
                        env,
                        seed,
                    )

                continue

            # =================================================
            # GAME OVER
            # =================================================

            if (
                screen_state
                == "game_over"
            ):

                # R or ENTER = new game
                if key in (
                    ord("r"),
                    13,
                    10,
                ):

                    episode += 1

                    (
                        observation,
                        total_lines,
                        display_score,
                        total_reward,
                        steps,
                    ) = reset_game(
                        env,
                        seed,
                    )

                    screen_state = (
                        "playing"
                    )

                # ESC = return to menu
                elif key == 27:

                    (
                        observation,
                        total_lines,
                        display_score,
                        total_reward,
                        steps,
                        screen_state,
                    ) = return_to_menu(
                        env,
                        seed,
                    )

                continue

            # =================================================
            # PLAYING
            # =================================================

            # ESC = return to menu
            if key == 27:

                (
                    observation,
                    total_lines,
                    display_score,
                    total_reward,
                    steps,
                    screen_state,
                ) = return_to_menu(
                    env,
                    seed,
                )

                continue

            # P = pause
            if (
                key
                == ord("p")
            ):

                screen_state = (
                    "paused"
                )

                continue

            # R = restart
            if (
                key
                == ord("r")
            ):

                episode += 1

                (
                    observation,
                    total_lines,
                    display_score,
                    total_reward,
                    steps,
                ) = reset_game(
                    env,
                    seed,
                )

                continue

            action = None

            # -------------------------------------------------
            # Human Controls
            # -------------------------------------------------

            if (
                key
                == ord("a")
            ):

                action = (
                    env.actions
                    .move_left
                )

            elif (
                key
                == ord("d")
            ):

                action = (
                    env.actions
                    .move_right
                )

            elif (
                key
                == ord("s")
            ):

                action = (
                    env.actions
                    .move_down
                )

            elif (
                key
                == ord("w")
            ):

                action = (
                    env.actions
                    .rotate_counterclockwise
                )

            elif (
                key
                == ord("e")
            ):

                action = (
                    env.actions
                    .rotate_clockwise
                )

            elif (
                key
                == ord(" ")
            ):

                action = (
                    env.actions
                    .hard_drop
                )

            elif (
                key
                == ord("q")
            ):

                action = (
                    env.actions
                    .swap
                )

            if action is None:
                continue

            # -------------------------------------------------
            # Environment Step
            # -------------------------------------------------

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            lines = int(
                info.get(
                    "lines_cleared",
                    0,
                )
            )

            total_lines += lines

            # Display-only score
            # Does NOT modify RL reward
            if lines > 0:

                display_score += (
                    env.score(
                        lines
                    )
                )

            total_reward += (
                float(reward)
            )

            steps += 1

            if (
                terminated
                or truncated
            ):

                screen_state = (
                    "game_over"
                )

    finally:

        env.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()