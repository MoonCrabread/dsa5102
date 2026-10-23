import cv2
import numpy as np

from tetris_gymnasium.envs import Tetris


# =============================
# UI settings
# =============================
WINDOW_NAME = "DSA5102 - Tetris RL Demo"

CANVAS_W = 960
CANVAS_H = 760

CELL = 28
BOARD_X = 340
BOARD_Y = 120

# OpenCV uses BGR colors
BG = (18, 22, 30)
PANEL = (28, 34, 44)
PANEL_BORDER = (65, 76, 92)
TEXT = (235, 239, 245)
MUTED = (160, 170, 185)
GRID = (48, 55, 68)
WHITE = (245, 245, 245)


def put_text(img, text, x, y, scale=0.6, color=TEXT, thickness=1):
    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL_BORDER, 1)

    if title:
        put_text(img, title, x + 14, y + 28, 0.6, WHITE, 2)


def get_piece_color_bgr(env, pixel_id):
    pixel_id = int(pixel_id)

    if pixel_id <= 0 or pixel_id >= len(env.pixels):
        return (90, 90, 90)

    r, g, b = env.pixels[pixel_id].color_rgb
    return (int(b), int(g), int(r))


def draw_block(img, x, y, size, color):
    pad = 2

    cv2.rectangle(
        img,
        (x + pad, y + pad),
        (x + size - pad, y + size - pad),
        color,
        -1,
    )

    # Small highlight/shadow to make blocks easier to see
    hi = tuple(min(255, c + 45) for c in color)
    lo = tuple(max(0, c - 55) for c in color)

    cv2.line(
        img,
        (x + pad, y + pad),
        (x + size - pad, y + pad),
        hi,
        1,
    )

    cv2.line(
        img,
        (x + pad, y + pad),
        (x + pad, y + size - pad),
        hi,
        1,
    )

    cv2.line(
        img,
        (x + pad, y + size - pad),
        (x + size - pad, y + size - pad),
        lo,
        1,
    )

    cv2.line(
        img,
        (x + size - pad, y + pad),
        (x + size - pad, y + size - pad),
        lo,
        1,
    )


def draw_board(img, env, observation):
    # Remove collision padding and keep the real 20 x 10 board
    board = env.crop_padding(observation["board"])

    rows, cols = board.shape

    board_w = cols * CELL
    board_h = rows * CELL

    # Outer frame
    cv2.rectangle(
        img,
        (BOARD_X - 4, BOARD_Y - 4),
        (BOARD_X + board_w + 4, BOARD_Y + board_h + 4),
        (8, 10, 15),
        -1,
    )

    cv2.rectangle(
        img,
        (BOARD_X - 4, BOARD_Y - 4),
        (BOARD_X + board_w + 4, BOARD_Y + board_h + 4),
        PANEL_BORDER,
        2,
    )

    for r in range(rows):
        for c in range(cols):

            x = BOARD_X + c * CELL
            y = BOARD_Y + r * CELL

            # Empty board cell
            cv2.rectangle(
                img,
                (x, y),
                (x + CELL, y + CELL),
                BG,
                -1,
            )

            # Grid line
            cv2.rectangle(
                img,
                (x, y),
                (x + CELL, y + CELL),
                GRID,
                1,
            )

            pixel_id = int(board[r, c])

            if pixel_id > 0:
                color = get_piece_color_bgr(env, pixel_id)
                draw_block(img, x, y, CELL, color)


def draw_tetromino_preview(
    img,
    tetromino,
    x,
    y,
    w,
    h,
    cell=22,
):
    if tetromino is None:
        put_text(
            img,
            "EMPTY",
            x + 20,
            y + h // 2,
            0.5,
            MUTED,
            1,
        )
        return

    matrix = tetromino.matrix

    occupied = np.argwhere(matrix > 0)

    if occupied.size == 0:
        return

    r0, c0 = occupied.min(axis=0)
    r1, c1 = occupied.max(axis=0)

    shape = matrix[
        r0:r1 + 1,
        c0:c1 + 1,
    ]

    piece_h, piece_w = shape.shape

    start_x = x + (w - piece_w * cell) // 2
    start_y = y + (h - piece_h * cell) // 2

    r, g, b = tetromino.color_rgb
    color = (int(b), int(g), int(r))

    for rr in range(piece_h):
        for cc in range(piece_w):

            if shape[rr, cc] > 0:
                draw_block(
                    img,
                    start_x + cc * cell,
                    start_y + rr * cell,
                    cell,
                    color,
                )


def render_ui(
    env,
    observation,
    total_lines,
    display_score,
    total_reward,
    steps,
    episode,
    seed,
    done=False,
):

    img = np.full(
        (CANVAS_H, CANVAS_W, 3),
        BG,
        dtype=np.uint8,
    )

    # =============================
    # Header
    # =============================
    put_text(
        img,
        "TETRIS RL PROJECT",
        36,
        50,
        1.05,
        WHITE,
        2,
    )

    put_text(
        img,
        "Environment & Evaluation Demo",
        38,
        80,
        0.55,
        MUTED,
        1,
    )

    put_text(
        img,
        "Mode: Human Demo",
        735,
        48,
        0.55,
        TEXT,
        1,
    )

    put_text(
        img,
        f"Seed: {seed}",
        735,
        76,
        0.5,
        MUTED,
        1,
    )

    # =============================
    # Game board
    # =============================
    put_text(
        img,
        "GAME BOARD",
        BOARD_X,
        104,
        0.55,
        MUTED,
        1,
    )

    draw_board(
        img,
        env,
        observation,
    )

    # =============================
    # HOLD
    # =============================
    hold_x = 36
    hold_y = 120
    hold_w = 245
    hold_h = 170

    draw_panel(
        img,
        hold_x,
        hold_y,
        hold_w,
        hold_h,
        "HOLD",
    )

    held = env.holder.get_tetrominoes()

    held_piece = held[0] if held else None

    draw_tetromino_preview(
        img,
        held_piece,
        hold_x + 12,
        hold_y + 42,
        hold_w - 24,
        hold_h - 54,
    )

    # =============================
    # Game statistics
    # =============================
    stat_x = 36
    stat_y = 315
    stat_w = 245
    stat_h = 245

    draw_panel(
        img,
        stat_x,
        stat_y,
        stat_w,
        stat_h,
        "GAME INFO",
    )

    labels = [
        ("Score", display_score),
        ("Lines", total_lines),
        ("Steps", steps),
        ("Reward", f"{total_reward:.2f}"),
        ("Episode", episode),
    ]

    yy = stat_y + 68

    for label, value in labels:

        put_text(
            img,
            label,
            stat_x + 18,
            yy,
            0.53,
            MUTED,
            1,
        )

        put_text(
            img,
            value,
            stat_x + 145,
            yy,
            0.58,
            WHITE,
            2,
        )

        yy += 34

    # =============================
    # NEXT pieces
    # =============================
    next_x = 680
    next_y = 120
    next_w = 240
    next_h = 440

    draw_panel(
        img,
        next_x,
        next_y,
        next_w,
        next_h,
        "NEXT",
    )

    queue_ids = env.queue.get_queue()

    preview_y = next_y + 46

    for i, piece_id in enumerate(queue_ids[:4]):

        piece = env.tetrominoes[int(piece_id)]

        if i > 0:
            cv2.line(
                img,
                (next_x + 20, preview_y - 7),
                (next_x + next_w - 20, preview_y - 7),
                PANEL_BORDER,
                1,
            )

        draw_tetromino_preview(
            img,
            piece,
            next_x + 15,
            preview_y,
            next_w - 30,
            82,
            cell=18,
        )

        preview_y += 92

    # =============================
    # Controls
    # =============================
    ctrl_y = 610

    draw_panel(
        img,
        36,
        ctrl_y,
        884,
        108,
        "CONTROLS",
    )

    put_text(
        img,
        "A / D : Move    S : Down    W / E : Rotate    SPACE : Hard Drop",
        58,
        ctrl_y + 58,
        0.52,
        TEXT,
        1,
    )

    put_text(
        img,
        "Q : Hold / Swap    R : Restart    ESC : Quit",
        58,
        ctrl_y + 88,
        0.52,
        MUTED,
        1,
    )

    # =============================
    # Game Over overlay
    # =============================
    if done:

        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (300, 310),
            (660, 455),
            (8, 10, 15),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.88,
            img,
            0.12,
            0,
            img,
        )

        cv2.rectangle(
            img,
            (300, 310),
            (660, 455),
            PANEL_BORDER,
            2,
        )

        put_text(
            img,
            "GAME OVER",
            365,
            365,
            1.0,
            WHITE,
            2,
        )

        put_text(
            img,
            "Press R to restart",
            365,
            405,
            0.58,
            TEXT,
            1,
        )

        put_text(
            img,
            "Press ESC to quit",
            365,
            435,
            0.5,
            MUTED,
            1,
        )

    return img


def reset_game(env, seed):

    observation, info = env.reset(seed=seed)

    total_lines = 0
    display_score = 0
    total_reward = 0.0
    steps = 0
    done = False

    return (
        observation,
        total_lines,
        display_score,
        total_reward,
        steps,
        done,
    )


def main():

    # Important:
    # This is only a new visualization layer.
    # We are NOT changing the original game's
    # state/action/reward rules.

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
        done,
    ) = reset_game(env, seed)

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

            frame = render_ui(
                env,
                observation,
                total_lines,
                display_score,
                total_reward,
                steps,
                episode,
                seed,
                done,
            )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            # User closes the window
            if (
                cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                )
                < 1
            ):
                break

            key = cv2.waitKey(30) & 0xFF

            # No key pressed
            if key == 255:
                continue

            # ESC
            if key == 27:
                break

            # Restart
            if key == ord("r"):

                episode += 1

                (
                    observation,
                    total_lines,
                    display_score,
                    total_reward,
                    steps,
                    done,
                ) = reset_game(
                    env,
                    seed,
                )

                continue

            # After game over:
            # only R and ESC work
            if done:
                continue

            action = None

            if key == ord("a"):
                action = env.actions.move_left

            elif key == ord("d"):
                action = env.actions.move_right

            elif key == ord("s"):
                action = env.actions.move_down

            elif key == ord("w"):
                action = env.actions.rotate_counterclockwise

            elif key == ord("e"):
                action = env.actions.rotate_clockwise

            elif key == ord(" "):
                action = env.actions.hard_drop

            elif key == ord("q"):
                action = env.actions.swap

            if action is None:
                continue

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            lines = int(
                info.get(
                    "lines_cleared",
                    0,
                )
            )

            total_lines += lines

            # Display-only score.
            # Does NOT modify the RL reward.
            if lines > 0:
                display_score += env.score(lines)

            total_reward += float(reward)

            steps += 1

            done = bool(
                terminated or truncated
            )

    finally:

        env.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()