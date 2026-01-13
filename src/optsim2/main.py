"""
メインアプリケーション
2画面GUI（横図・上面図）を表示し、光学シミュレーションを実行
"""
import pygame
import sys
import numpy as np
from typing import Tuple, List
from .optics_engine import OpticsEngine, Ray


class OpticsSimulator:
    """光学シミュレーターのメインクラス"""

    # 色定義
    COLOR_BG = (240, 240, 250)
    COLOR_WATER = (100, 150, 255, 100)
    COLOR_AIR = (200, 220, 255, 50)
    COLOR_BALL = (255, 100, 100)
    COLOR_LIGHT_SOURCE = (255, 255, 0)
    COLOR_RAY = (255, 200, 0)
    COLOR_TEXT = (50, 50, 50)
    COLOR_GRID = (200, 200, 200)

    def __init__(self, width: int = 1600, height: int = 800):
        """
        Args:
            width: ウィンドウの幅
            height: ウィンドウの高さ
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("OptSim2 - 光学シミュレーション")

        # フォント（日本語対応）
        # Windowsの標準日本語フォントを使用
        try:
            self.font = pygame.font.SysFont('meiryo', 20)
            self.title_font = pygame.font.SysFont('meiryo', 28)
        except:
            # メイリオがない場合は他の日本語フォントを試す
            try:
                self.font = pygame.font.SysFont('msgothic', 20)
                self.title_font = pygame.font.SysFont('msgothic', 28)
            except:
                # それでもない場合はデフォルト
                self.font = pygame.font.Font(None, 24)
                self.title_font = pygame.font.Font(None, 32)

        # 2つのビュー（横図と上面図）
        self.view_width = width // 2 - 40
        self.view_height = height - 100

        # 光学エンジン
        self.engine = OpticsEngine(self.view_width, self.view_height)

        # デフォルトの設定
        self.setup_default_scene()

        # シミュレーション状態
        self.running = True
        self.clock = pygame.time.Clock()
        self.light_position = (self.view_width // 2, 50)
        self.light_angle = 0.0  # 光の角度（ラジアン、0は下向き）
        self.light_spread = np.pi / 2  # 光の広がり角度
        self.dragging_light = False

        # 球の光強度マップ（角度ごとの強度を記録）
        self.ball_intensity_map = {}

        # ズーム設定
        self.side_view_zoom = 1.0
        self.top_view_zoom = 1.0

        # パン（平行移動）設定
        self.side_view_offset = [0, 0]
        self.top_view_offset = [0, 0]
        self.dragging_side_view = False
        self.dragging_top_view = False
        self.drag_start_pos = None

    def setup_default_scene(self):
        """デフォルトのシーンを設定"""
        # 水面の位置を設定
        self.engine.set_water_level(self.view_height * 0.5)

        # 球を追加（水中）
        ball_y = self.view_height * 0.7
        self.engine.add_ball((self.view_width // 2, ball_y), 40)

    def calculate_ball_intensity(self):
        """球に当たった光の強度を計算"""
        self.ball_intensity_map = {}

        if not self.engine.balls:
            return

        ball = self.engine.balls[0]
        ball_pos = np.array(ball['position'])
        ball_radius = ball['radius']

        # 各光線について、球に当たった位置と強度を記録
        for ray in self.engine.rays:
            if len(ray.path) < 2:
                continue

            for i in range(len(ray.path) - 1):
                p1 = np.array(ray.path[i], dtype=float)
                p2 = np.array(ray.path[i + 1], dtype=float)

                # 球との交点を計算
                d = p2 - p1
                f = p1 - ball_pos

                a = np.dot(d, d)
                if a == 0:
                    continue

                b = 2 * np.dot(f, d)
                c = np.dot(f, f) - ball_radius ** 2

                discriminant = b * b - 4 * a * c

                if discriminant >= 0:
                    discriminant = np.sqrt(discriminant)
                    t1 = (-b - discriminant) / (2 * a)

                    if 0 <= t1 <= 1:
                        # 交点の位置
                        hit_point = p1 + t1 * d

                        # 球の中心からの角度を計算
                        diff = hit_point - ball_pos
                        angle = np.arctan2(diff[1], diff[0])

                        # 角度を度数に変換（-180～180）
                        angle_deg = int(np.degrees(angle))

                        # その角度の強度を累積
                        if angle_deg not in self.ball_intensity_map:
                            self.ball_intensity_map[angle_deg] = 0
                        self.ball_intensity_map[angle_deg] += ray.intensity

    def get_intensity_color(self, intensity: float, max_intensity: float) -> Tuple[int, int, int]:
        """強度から色を計算（青→緑→黄→赤）"""
        if max_intensity == 0 or intensity == 0:
            return (0, 0, 255)  # 青（光が当たっていない）

        # 正規化 (0.0 ~ 1.0)
        normalized = min(1.0, intensity / max_intensity)

        if normalized < 0.33:
            # 青 → 緑
            ratio = normalized / 0.33
            r = 0
            g = int(255 * ratio)
            b = int(255 * (1 - ratio))
        elif normalized < 0.66:
            # 緑 → 黄
            ratio = (normalized - 0.33) / 0.33
            r = int(255 * ratio)
            g = 255
            b = 0
        else:
            # 黄 → 赤
            ratio = (normalized - 0.66) / 0.34
            r = 255
            g = int(255 * (1 - ratio))
            b = 0

        return (r, g, b)

    def draw_grid(self, surface: pygame.Surface, offset_x: int, offset_y: int):
        """グリッドを描画"""
        grid_spacing = 50

        for x in range(0, self.view_width, grid_spacing):
            pygame.draw.line(
                surface, self.COLOR_GRID,
                (offset_x + x, offset_y),
                (offset_x + x, offset_y + self.view_height), 1
            )

        for y in range(0, self.view_height, grid_spacing):
            pygame.draw.line(
                surface, self.COLOR_GRID,
                (offset_x, offset_y + y),
                (offset_x + self.view_width, offset_y + y), 1
            )

    def draw_side_view(self):
        """横図ビューを描画"""
        offset_x = 20
        offset_y = 60

        # ズーム適用したサーフェスを作成
        zoom = self.side_view_zoom
        zoomed_width = int(self.view_width * zoom)
        zoomed_height = int(self.view_height * zoom)

        # 背景
        view_surface = pygame.Surface((zoomed_width, zoomed_height), pygame.SRCALPHA)

        # グリッド
        self.draw_grid(self.screen, offset_x, offset_y)

        # 空気領域
        air_rect = pygame.Rect(0, 0, int(zoomed_width), int(self.engine.water_level * zoom))
        pygame.draw.rect(view_surface, self.COLOR_AIR, air_rect)

        # 水領域
        water_rect = pygame.Rect(
            0, int(self.engine.water_level * zoom),
            int(zoomed_width), int(zoomed_height - self.engine.water_level * zoom)
        )
        pygame.draw.rect(view_surface, self.COLOR_WATER, water_rect)

        # 水面の線
        pygame.draw.line(
            view_surface, (0, 100, 200),
            (0, int(self.engine.water_level * zoom)),
            (int(zoomed_width), int(self.engine.water_level * zoom)), 3
        )

        # 球を描画（光強度に応じて色付け）
        for ball in self.engine.balls:
            pos = ball['position']
            radius = ball['radius']

            # ズーム適用した座標と半径
            zoomed_pos = (pos[0] * zoom, pos[1] * zoom)
            zoomed_radius = radius * zoom

            # 最大強度を取得
            max_intensity = max(self.ball_intensity_map.values()) if self.ball_intensity_map else 1.0

            # 球を角度ごとに分割して描画（ヒートマップ）
            # 角度は-180～180度の範囲
            for angle_deg in range(-180, 180):
                # この角度の強度を取得
                intensity = self.ball_intensity_map.get(angle_deg, 0)

                # 強度から色を計算
                color = self.get_intensity_color(intensity, max_intensity)

                # 扇形を描画
                angle_rad = np.radians(angle_deg)
                next_angle_rad = np.radians(angle_deg + 1)

                # 扇形の頂点を計算（ズーム適用）
                points = [
                    zoomed_pos,
                    (zoomed_pos[0] + zoomed_radius * np.cos(angle_rad), zoomed_pos[1] + zoomed_radius * np.sin(angle_rad)),
                    (zoomed_pos[0] + zoomed_radius * np.cos(next_angle_rad), zoomed_pos[1] + zoomed_radius * np.sin(next_angle_rad))
                ]
                points_int = [(int(p[0]), int(p[1])) for p in points]

                pygame.draw.polygon(view_surface, color, points_int)

            # 球の輪郭
            pygame.draw.circle(view_surface, (200, 50, 50), (int(zoomed_pos[0]), int(zoomed_pos[1])), int(zoomed_radius), 2)

        # 光線を描画
        for ray in self.engine.rays:
            if len(ray.path) > 1:
                points = [(int(p[0] * zoom), int(p[1] * zoom)) for p in ray.path]
                # 光線の強度に応じて色を変える
                alpha = int(ray.intensity * 255)
                color = (*self.COLOR_RAY[:3], min(alpha, 255))

                for i in range(len(points) - 1):
                    pygame.draw.line(view_surface, color, points[i], points[i + 1], 2)

        # 光源を描画
        zoomed_light_pos = (int(self.light_position[0] * zoom), int(self.light_position[1] * zoom))
        pygame.draw.circle(view_surface, self.COLOR_LIGHT_SOURCE, zoomed_light_pos, int(10 * zoom))
        pygame.draw.circle(view_surface, (200, 200, 0), zoomed_light_pos, int(10 * zoom), 2)

        # 光の方向を示す矢印を描画
        arrow_length = 30 * zoom
        arrow_end = (
            int(zoomed_light_pos[0] + arrow_length * np.sin(self.light_angle)),
            int(zoomed_light_pos[1] + arrow_length * np.cos(self.light_angle))
        )
        pygame.draw.line(view_surface, (255, 255, 0), zoomed_light_pos, arrow_end, int(3 * zoom))

        # 光の広がり範囲を示す扇形の線
        spread_left = self.light_angle - self.light_spread / 2
        spread_right = self.light_angle + self.light_spread / 2
        spread_len = 25 * zoom
        spread_left_end = (
            int(zoomed_light_pos[0] + spread_len * np.sin(spread_left)),
            int(zoomed_light_pos[1] + spread_len * np.cos(spread_left))
        )
        spread_right_end = (
            int(zoomed_light_pos[0] + spread_len * np.sin(spread_right)),
            int(zoomed_light_pos[1] + spread_len * np.cos(spread_right))
        )
        pygame.draw.line(view_surface, (255, 200, 0, 150), zoomed_light_pos, spread_left_end, max(1, int(zoom)))
        pygame.draw.line(view_surface, (255, 200, 0, 150), zoomed_light_pos, spread_right_end, max(1, int(zoom)))

        # ビューサーフェスを画面に描画（中央部分を切り取って表示）
        if zoom > 1.0:
            # ズームした画像の中央部分を切り取る（パンオフセットを適用）
            crop_x = max(0, min(zoomed_width - self.view_width,
                                (zoomed_width - self.view_width) // 2 - int(self.side_view_offset[0] * zoom)))
            crop_y = max(0, min(zoomed_height - self.view_height,
                                (zoomed_height - self.view_height) // 2 - int(self.side_view_offset[1] * zoom)))
            crop_width = min(self.view_width, zoomed_width)
            crop_height = min(self.view_height, zoomed_height)
            crop_rect = pygame.Rect(crop_x, crop_y, crop_width, crop_height)
            cropped_surface = view_surface.subsurface(crop_rect)
            self.screen.blit(cropped_surface, (offset_x, offset_y))
        elif zoom < 1.0:
            # 縮小時は中央に配置
            scaled_surface = pygame.transform.smoothscale(view_surface, (int(self.view_width * zoom), int(self.view_height * zoom)))
            center_x = offset_x + (self.view_width - int(self.view_width * zoom)) // 2
            center_y = offset_y + (self.view_height - int(self.view_height * zoom)) // 2
            self.screen.blit(scaled_surface, (center_x, center_y))
        else:
            self.screen.blit(view_surface, (offset_x, offset_y))

        # 枠線
        pygame.draw.rect(self.screen, (100, 100, 100), (offset_x, offset_y, self.view_width, self.view_height), 2)

        # タイトル
        title = self.title_font.render("横図（側面図）", True, self.COLOR_TEXT)
        self.screen.blit(title, (offset_x, 20))

    def draw_top_view(self):
        """上面図ビューを描画（真上から見た水槽）"""
        offset_x = self.width // 2 + 10
        offset_y = 60

        # ズーム適用
        zoom = self.top_view_zoom
        zoomed_width = int(self.view_width * zoom)
        zoomed_height = int(self.view_height * zoom)

        # 背景
        view_surface = pygame.Surface((zoomed_width, zoomed_height), pygame.SRCALPHA)

        # グリッド
        self.draw_grid(self.screen, offset_x, offset_y)

        # 水面（全体が水面）
        pygame.draw.rect(view_surface, self.COLOR_WATER, (0, 0, zoomed_width, zoomed_height))

        # 上面図の座標系（真上から見下ろした図）：
        # X軸 = 左右方向（画面中央に固定）
        # Y軸 = 上から下方向（横図のY座標（高さ）に対応）
        #      横図で光源が上にある → 上面図では光源が上（球との距離が遠い）
        #      横図で光源が下にある → 上面図では光源が下（球との距離が近い）

        # 光源（ハロゲン）の位置: 横図のY座標（高さ）を上面図のY座標に変換
        top_light_x = (self.view_width // 2) * zoom  # 画面中央（X軸は固定）
        # 横図のY座標を上面図のY座標に変換（横図で上にあるほど、上面図でも上）
        # 横図: Y=0が上、Y=view_heightが下
        # 上面図: Y=0が上、Y=view_heightが下
        top_light_y = int(self.light_position[1] * zoom)

        # ハロゲンは横長の矩形として描画
        halogen_width = 60 * zoom  # X方向の幅（横に長い）
        halogen_depth = 30 * zoom  # Y方向の奥行き
        halogen_rect = pygame.Rect(
            int(top_light_x - halogen_width // 2),
            int(top_light_y - halogen_depth // 2),
            int(halogen_width),
            int(halogen_depth)
        )
        pygame.draw.rect(view_surface, (255, 255, 200), halogen_rect)
        pygame.draw.rect(view_surface, (200, 200, 0), halogen_rect, int(3 * zoom))

        # 球を描画（横図のY座標を上面図のY座標に変換）
        for ball in self.engine.balls:
            pos = ball['position']
            radius = ball['radius']
            top_x = int((self.view_width // 2) * zoom)  # 画面中央（X軸は固定）
            top_y = int(pos[1] * zoom)  # 横図のY座標をそのまま使用
            pygame.draw.circle(view_surface, self.COLOR_BALL, (top_x, top_y), int(radius * zoom))
            pygame.draw.circle(view_surface, (200, 50, 50), (top_x, top_y), int(radius * zoom), max(2, int(2 * zoom)))

        # 光線を縦線として描画（上から下に向かう、画面中央から広がる）
        num_rays = len(self.engine.rays)

        # 球のY座標を取得
        ball_y = int(self.engine.balls[0]['position'][1] * zoom) if self.engine.balls else int((self.view_height - 150) * zoom)

        for idx, ray in enumerate(self.engine.rays):
            if len(ray.path) > 1:
                # 光線のX座標を中央からの広がりとして計算
                # 横図での光線の広がり具合を上面図でも反映
                ray_offset_ratio = (idx - num_rays / 2) / (num_rays / 2)  # -1.0 ～ 1.0
                spread_width = 150 * zoom  # 光の広がり幅
                ray_x = int(top_light_x + ray_offset_ratio * spread_width * np.sin(self.light_spread / 2))

                # 縦線として描画（ハロゲンの下端から球の位置まで）
                start_y = int(top_light_y + halogen_depth // 2)
                end_y = ball_y  # 球のY座標まで

                # 線の色
                alpha = int(ray.intensity * 200)
                color = (*self.COLOR_RAY[:3], min(alpha, 255))

                # 縦線として描画
                pygame.draw.line(view_surface, color,
                               (ray_x, start_y),
                               (ray_x, end_y), max(1, int(zoom)))

        # ビューサーフェスを画面に描画（中央部分を切り取って表示）
        if zoom > 1.0:
            # ズームした画像の中央部分を切り取る（パンオフセットを適用）
            crop_x = max(0, min(zoomed_width - self.view_width,
                                (zoomed_width - self.view_width) // 2 - int(self.top_view_offset[0] * zoom)))
            crop_y = max(0, min(zoomed_height - self.view_height,
                                (zoomed_height - self.view_height) // 2 - int(self.top_view_offset[1] * zoom)))
            crop_width = min(self.view_width, zoomed_width)
            crop_height = min(self.view_height, zoomed_height)
            crop_rect = pygame.Rect(crop_x, crop_y, crop_width, crop_height)
            cropped_surface = view_surface.subsurface(crop_rect)
            self.screen.blit(cropped_surface, (offset_x, offset_y))
        elif zoom < 1.0:
            # 縮小時は中央に配置
            scaled_surface = pygame.transform.smoothscale(view_surface, (int(self.view_width * zoom), int(self.view_height * zoom)))
            center_x = offset_x + (self.view_width - int(self.view_width * zoom)) // 2
            center_y = offset_y + (self.view_height - int(self.view_height * zoom)) // 2
            self.screen.blit(scaled_surface, (center_x, center_y))
        else:
            self.screen.blit(view_surface, (offset_x, offset_y))

        # 枠線
        pygame.draw.rect(self.screen, (100, 100, 100), (offset_x, offset_y, self.view_width, self.view_height), 2)

        # タイトル
        title = self.title_font.render("上面図（真上から）", True, self.COLOR_TEXT)
        self.screen.blit(title, (offset_x, 20))

    def draw_ui(self):
        """UI要素を描画"""
        angle_deg = int(np.degrees(self.light_angle))
        spread_deg = int(np.degrees(self.light_spread))
        info_texts = [
            f"光源位置: ({self.light_position[0]}, {self.light_position[1]})",
            f"光の角度: {angle_deg}° / 広がり: {spread_deg}°",
            f"水面: {int(self.engine.water_level)}",
            f"光線数: {len(self.engine.rays)}",
            "左クリック: 光源を移動",
            "左右キー: 光の角度調整",
            "Q/E: 光の広がり調整",
            "上下キー: 水面の位置調整",
            "R: リセット / ESC: 終了"
        ]

        y_offset = self.height - 150
        for i, text in enumerate(info_texts):
            surface = self.font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (20, y_offset + i * 25))

    def handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # リセット
                    self.engine.balls.clear()
                    self.light_angle = 0.0
                    self.light_spread = np.pi / 2
                    self.setup_default_scene()
                    self.update_simulation()
                elif event.key == pygame.K_UP:
                    # 水面を上げる
                    self.engine.water_level = max(50, self.engine.water_level - 10)
                    self.update_simulation()
                elif event.key == pygame.K_DOWN:
                    # 水面を下げる
                    self.engine.water_level = min(self.view_height - 50, self.engine.water_level + 10)
                    self.update_simulation()
                elif event.key == pygame.K_LEFT:
                    # 光の角度を左に
                    self.light_angle -= np.pi / 18  # 10度ずつ
                    self.update_simulation()
                elif event.key == pygame.K_RIGHT:
                    # 光の角度を右に
                    self.light_angle += np.pi / 18  # 10度ずつ
                    self.update_simulation()
                elif event.key == pygame.K_q:
                    # 光の広がりを狭く
                    self.light_spread = max(np.pi / 18, self.light_spread - np.pi / 18)
                    self.update_simulation()
                elif event.key == pygame.K_e:
                    # 光の広がりを広く
                    self.light_spread = min(np.pi, self.light_spread + np.pi / 18)
                    self.update_simulation()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左クリック
                    mouse_pos = pygame.mouse.get_pos()
                    # 横図ビュー内かチェック
                    if (20 <= mouse_pos[0] <= 20 + self.view_width and
                        60 <= mouse_pos[1] <= 60 + self.view_height):
                        # ズーム・オフセットを考慮した光源の表示位置を計算
                        zoom = self.side_view_zoom
                        if zoom > 1.0:
                            # ズーム時の切り取り位置を計算
                            crop_x = max(0, min(int(self.view_width * zoom) - self.view_width,
                                                (int(self.view_width * zoom) - self.view_width) // 2 - int(self.side_view_offset[0] * zoom)))
                            crop_y = max(0, min(int(self.view_height * zoom) - self.view_height,
                                                (int(self.view_height * zoom) - self.view_height) // 2 - int(self.side_view_offset[1] * zoom)))
                            # 画面上の光源位置
                            display_light_x = int(self.light_position[0] * zoom) - crop_x + 20
                            display_light_y = int(self.light_position[1] * zoom) - crop_y + 60
                        elif zoom < 1.0:
                            # 縮小時は中央配置
                            center_x = 20 + (self.view_width - int(self.view_width * zoom)) // 2
                            center_y = 60 + (self.view_height - int(self.view_height * zoom)) // 2
                            display_light_x = int(self.light_position[0] * zoom) + center_x
                            display_light_y = int(self.light_position[1] * zoom) + center_y
                        else:
                            # 等倍時
                            display_light_x = int(self.light_position[0]) + 20
                            display_light_y = int(self.light_position[1]) + 60

                        # 光源をドラッグ開始（当たり判定）
                        if np.linalg.norm(np.array(mouse_pos) - np.array([display_light_x, display_light_y])) < int(10 * zoom):
                            self.dragging_light = True
                elif event.button == 2:  # マウスホイールクリック（中クリック）
                    mouse_pos = pygame.mouse.get_pos()
                    self.drag_start_pos = mouse_pos
                    # 横図ビュー内かチェック
                    if (20 <= mouse_pos[0] <= 20 + self.view_width and
                        60 <= mouse_pos[1] <= 60 + self.view_height):
                        self.dragging_side_view = True
                    # 上面図ビュー内
                    elif (self.width // 2 + 10 <= mouse_pos[0] <= self.width // 2 + 10 + self.view_width and
                          60 <= mouse_pos[1] <= 60 + self.view_height):
                        self.dragging_top_view = True
                elif event.button == 4:  # マウスホイール上
                    mouse_pos = pygame.mouse.get_pos()
                    # 横図ビュー内
                    if (20 <= mouse_pos[0] <= 20 + self.view_width and
                        60 <= mouse_pos[1] <= 60 + self.view_height):
                        self.side_view_zoom = min(3.0, self.side_view_zoom * 1.1)
                    # 上面図ビュー内
                    elif (self.width // 2 + 10 <= mouse_pos[0] <= self.width // 2 + 10 + self.view_width and
                          60 <= mouse_pos[1] <= 60 + self.view_height):
                        self.top_view_zoom = min(3.0, self.top_view_zoom * 1.1)
                elif event.button == 5:  # マウスホイール下
                    mouse_pos = pygame.mouse.get_pos()
                    # 横図ビュー内
                    if (20 <= mouse_pos[0] <= 20 + self.view_width and
                        60 <= mouse_pos[1] <= 60 + self.view_height):
                        self.side_view_zoom = max(0.5, self.side_view_zoom / 1.1)
                    # 上面図ビュー内
                    elif (self.width // 2 + 10 <= mouse_pos[0] <= self.width // 2 + 10 + self.view_width and
                          60 <= mouse_pos[1] <= 60 + self.view_height):
                        self.top_view_zoom = max(0.5, self.top_view_zoom / 1.1)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_light = False
                elif event.button == 2:  # マウスホイールクリック（中クリック）
                    self.dragging_side_view = False
                    self.dragging_top_view = False
                    self.drag_start_pos = None

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_light:
                    mouse_pos = pygame.mouse.get_pos()

                    # ズーム・オフセットを考慮してワールド座標に変換
                    zoom = self.side_view_zoom
                    if zoom > 1.0:
                        # ズーム時の切り取り位置を計算
                        crop_x = max(0, min(int(self.view_width * zoom) - self.view_width,
                                            (int(self.view_width * zoom) - self.view_width) // 2 - int(self.side_view_offset[0] * zoom)))
                        crop_y = max(0, min(int(self.view_height * zoom) - self.view_height,
                                            (int(self.view_height * zoom) - self.view_height) // 2 - int(self.side_view_offset[1] * zoom)))
                        # マウス位置をワールド座標に変換
                        world_x = (mouse_pos[0] - 20 + crop_x) / zoom
                        world_y = (mouse_pos[1] - 60 + crop_y) / zoom
                    elif zoom < 1.0:
                        # 縮小時
                        center_x = (self.view_width - int(self.view_width * zoom)) // 2
                        center_y = (self.view_height - int(self.view_height * zoom)) // 2
                        world_x = (mouse_pos[0] - 20 - center_x) / zoom
                        world_y = (mouse_pos[1] - 60 - center_y) / zoom
                    else:
                        # 等倍時
                        world_x = mouse_pos[0] - 20
                        world_y = mouse_pos[1] - 60

                    # ワールド座標の範囲制限
                    world_x = max(0, min(self.view_width, world_x))
                    world_y = max(0, min(self.view_height, world_y))

                    self.light_position = (world_x, world_y)
                    self.update_simulation()
                elif self.dragging_side_view and self.drag_start_pos:
                    mouse_pos = pygame.mouse.get_pos()
                    # マウスの移動量を計算
                    dx = mouse_pos[0] - self.drag_start_pos[0]
                    dy = mouse_pos[1] - self.drag_start_pos[1]
                    # オフセットを更新
                    self.side_view_offset[0] += dx
                    self.side_view_offset[1] += dy
                    self.drag_start_pos = mouse_pos
                elif self.dragging_top_view and self.drag_start_pos:
                    mouse_pos = pygame.mouse.get_pos()
                    # マウスの移動量を計算
                    dx = mouse_pos[0] - self.drag_start_pos[0]
                    dy = mouse_pos[1] - self.drag_start_pos[1]
                    # オフセットを更新
                    self.top_view_offset[0] += dx
                    self.top_view_offset[1] += dy
                    self.drag_start_pos = mouse_pos

    def update_simulation(self):
        """シミュレーションを更新"""
        self.engine.create_light_source(
            self.light_position,
            num_rays=50,  # 光線の本数を増やす
            spread_angle=self.light_spread,
            center_angle=self.light_angle
        )
        # 球の光強度を計算
        self.calculate_ball_intensity()

    def run(self):
        """メインループ"""
        self.update_simulation()

        while self.running:
            self.handle_events()

            # 描画
            self.screen.fill(self.COLOR_BG)
            self.draw_side_view()
            self.draw_top_view()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    """エントリーポイント"""
    simulator = OpticsSimulator()
    simulator.run()


if __name__ == "__main__":
    main()
