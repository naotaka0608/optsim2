"""
メインアプリケーション
2画面GUI（横図・上面図）を表示し、光学シミュレーションを実行
3D表示モードも利用可能
"""
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import numpy as np
from typing import Tuple, List, Callable
from .optics_engine import OpticsEngine, Ray
import math


class Slider:
    """スライダーUIコンポーネント"""
    def __init__(self, x: int, y: int, width: int, min_val: float, max_val: float,
                 initial_val: float, label: str, callback: Callable[[float], None]):
        self.rect = pygame.Rect(x, y, width, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.callback = callback
        self.dragging = False
        self.knob_radius = 8

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        # ラベル
        label_surf = font.render(self.label, True, (50, 50, 50))
        surface.blit(label_surf, (self.rect.x, self.rect.y - 20))

        # スライダーバー
        pygame.draw.rect(surface, (180, 180, 180), self.rect, border_radius=3)

        # つまみの位置を計算
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        knob_x = int(self.rect.x + ratio * self.rect.width)
        knob_y = self.rect.y + self.rect.height // 2

        # つまみ
        pygame.draw.circle(surface, (70, 130, 220), (knob_x, knob_y), self.knob_radius)
        pygame.draw.circle(surface, (50, 100, 180), (knob_x, knob_y), self.knob_radius, 2)

        # 値の表示
        if isinstance(self.value, int):
            value_text = str(self.value)
        elif self.max_val <= 2.0 and self.min_val >= 1.0:
            # 屈折率などの小さい範囲は小数第2位まで
            value_text = f"{self.value:.2f}"
        else:
            value_text = f"{self.value:.1f}"
        value_surf = font.render(value_text, True, (80, 80, 80))
        surface.blit(value_surf, (self.rect.x + self.rect.width + 10, self.rect.y))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            knob_pos = self._get_knob_pos()
            if ((mouse_pos[0] - knob_pos[0]) ** 2 + (mouse_pos[1] - knob_pos[1]) ** 2) <= self.knob_radius ** 2:
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = event.pos[0]
            ratio = max(0, min(1, (mouse_x - self.rect.x) / self.rect.width))
            new_value = self.min_val + ratio * (self.max_val - self.min_val)
            if isinstance(self.min_val, int) and isinstance(self.max_val, int):
                new_value = int(new_value)
            self.value = new_value
            self.callback(self.value)
            return True
        return False

    def _get_knob_pos(self) -> Tuple[int, int]:
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        knob_x = int(self.rect.x + ratio * self.rect.width)
        knob_y = self.rect.y + self.rect.height // 2
        return (knob_x, knob_y)


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

    def __init__(self, width: int = 1800, height: int = 900):
        """
        Args:
            width: ウィンドウの幅
            height: ウィンドウの高さ
        """
        pygame.init()
        # キーボードのリピート設定（長押しで連続入力）
        # delay: 最初のリピートまでの待機時間(ms)
        # interval: リピート間隔(ms)
        pygame.key.set_repeat(200, 50)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("OptSim2 - 光学シミュレーション")

        # フォント（日本語対応）
        # Windowsの標準日本語フォントを使用
        try:
            self.font = pygame.font.SysFont('meiryo', 16)
            self.title_font = pygame.font.SysFont('meiryo', 24)
            self.small_font = pygame.font.SysFont('meiryo', 14)
        except:
            # メイリオがない場合は他の日本語フォントを試す
            try:
                self.font = pygame.font.SysFont('msgothic', 16)
                self.title_font = pygame.font.SysFont('msgothic', 24)
                self.small_font = pygame.font.SysFont('msgothic', 14)
            except:
                # それでもない場合はデフォルト
                self.font = pygame.font.Font(None, 20)
                self.title_font = pygame.font.Font(None, 28)
                self.small_font = pygame.font.Font(None, 18)

        # UIパネルとビューのレイアウト
        self.ui_panel_width = 250
        self.view_margin = 10
        remaining_width = width - self.ui_panel_width - self.view_margin * 3
        self.view_width = remaining_width // 2
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
        self.light_intensity = 1.0  # 光の強度（0.0〜2.0）
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

        # スライダーの初期化
        self.sliders = []
        slider_x = 20
        slider_y_start = 120
        slider_width = 180
        slider_spacing = 50

        # 光の角度スライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start, slider_width,
            -90, 90, int(np.degrees(self.light_angle)),
            "光の角度 (°)",
            lambda v: self._set_light_angle(v)
        ))

        # 光の広がりスライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing, slider_width,
            0, 180, int(np.degrees(self.light_spread)),
            "光の広がり (°)",
            lambda v: self._set_light_spread(v)
        ))

        # 水面位置スライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 2, slider_width,
            50, self.view_height - 50, int(self.engine.water_level),
            "水面位置 (px)",
            lambda v: self._set_water_level(v)
        ))

        # 屈折率スライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 3, slider_width,
            1.00, 2.00, self.engine.water_refractive_index,
            "屈折率",
            lambda v: self._set_refractive_index(v)
        ))

        # 光の強度スライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 4, slider_width,
            0.0, 2.0, self.light_intensity,
            "光の強度",
            lambda v: self._set_light_intensity(v)
        ))

        # 3Dビューモード設定
        self.view_mode_3d = False  # False=2Dモード, True=3Dモード
        self.view_mode_raytracing = False  # レイトレーシング風3D描画モード（未使用）
        self.view_mode_natural_3d = False  # 自然光3Dモード（キー4）
        self.raytracing_image = None  # レイトレーシング結果のサーフェス
        self.camera_rotation = [20.0, 45.0]  # [pitch, yaw] in degrees
        self.camera_distance = 800.0
        self.camera_target = [0.0, 0.0, 0.0]  # カメラの注視点（平行移動用）
        self.dragging_camera = False
        self.dragging_camera_pan = False
        self.camera_drag_start = None
        self.camera_pan_start = None

    def setup_default_scene(self):
        """デフォルトのシーンを設定"""
        # 水面の位置を設定
        self.engine.set_water_level(self.view_height * 0.5)

        # 球を追加（水中、3D座標）
        ball_y = self.view_height * 0.7
        ball_x = self.view_width // 2
        ball_z = 0  # Z軸中心
        self.engine.add_ball((ball_x, ball_y, ball_z), 40)

    def init_opengl(self):
        """OpenGLの初期化"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # ライトの設定
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])

        glClearColor(0.1, 0.1, 0.15, 1.0)

    def init_opengl_natural(self):
        """自然光3DモードのOpenGL初期化"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glDisable(GL_LIGHT1)  # フィルライトは無効
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # メインライト（点光源として使用、位置は後で更新）
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1])  # 環境光を抑える
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 0.9, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 0.95, 1])

        # 背景色（スカイブルーのグラデーション風）
        glClearColor(0.6, 0.75, 0.9, 1.0)

    def setup_3d_perspective(self):
        """3D透視投影の設定"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 2000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # カメラ位置の計算（注視点からの相対位置）
        pitch_rad = math.radians(self.camera_rotation[0])
        yaw_rad = math.radians(self.camera_rotation[1])

        cam_x = self.camera_target[0] + self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(pitch_rad)
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

        gluLookAt(cam_x, cam_y, cam_z,                      # カメラ位置
                  self.camera_target[0],
                  self.camera_target[1],
                  self.camera_target[2],                     # 注視点
                  0, 1, 0)                                   # 上方向

    def draw_sphere_3d(self, x, y, z, radius, color):
        """3D球体を描画"""
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(*color)

        # GLUクアドリックで球体を描画
        quad = gluNewQuadric()
        gluSphere(quad, radius, 32, 32)
        gluDeleteQuadric(quad)

        glPopMatrix()

    def draw_line_3d(self, p1, p2, color, width=2.0):
        """3D線分を描画"""
        glDisable(GL_LIGHTING)
        glLineWidth(width)
        glColor3f(*color)
        glBegin(GL_LINES)
        glVertex3f(*p1)
        glVertex3f(*p2)
        glEnd()
        glEnable(GL_LIGHTING)

    def draw_water_plane_3d(self):
        """3D水面を描画（半透明）"""
        # 水面のY座標を計算（2D座標系を3D座標系に変換）
        water_y = -(self.engine.water_level - self.view_height / 2)

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 水面の平面（半透明）
        glColor4f(0.2, 0.5, 0.8, 0.4)
        size = 500
        glBegin(GL_QUADS)
        glVertex3f(-size, water_y, -size)
        glVertex3f(size, water_y, -size)
        glVertex3f(size, water_y, size)
        glVertex3f(-size, water_y, size)
        glEnd()

        # グリッド線
        glColor4f(0.3, 0.6, 0.9, 0.5)
        glLineWidth(1.0)
        step = 50
        glBegin(GL_LINES)
        for i in range(-10, 11):
            # X方向の線
            glVertex3f(i * step, water_y, -size)
            glVertex3f(i * step, water_y, size)
            # Z方向の線
            glVertex3f(-size, water_y, i * step)
            glVertex3f(size, water_y, i * step)
        glEnd()

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def draw_3d_view(self):
        """3Dビュー全体を描画"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_3d_perspective()

        # 球を描画（3D座標を使用）- 不透明なものを先に描画
        for ball in self.engine.balls:
            x_3d, y_3d, z_3d = ball['position']
            # ビュー座標系に変換
            x_3d_view = x_3d - self.view_width / 2
            y_3d_view = -(y_3d - self.view_height / 2)

            # 球の色（水中なので少し青みがかった色）
            self.draw_sphere_3d(x_3d_view, y_3d_view, z_3d, ball['radius'], (0.7, 0.8, 0.9))

        # 光源を描画
        light_x_2d, light_y_2d = self.light_position
        light_x_3d = light_x_2d - self.view_width / 2
        light_y_3d = -(light_y_2d - self.view_height / 2)
        light_z_3d = 0

        # 光源（小さな黄色い球）
        self.draw_sphere_3d(light_x_3d, light_y_3d, light_z_3d, 10, (1.0, 1.0, 0.3))

        # 光線を描画（3D座標を使用）
        for ray in self.engine.rays:
            if len(ray.path) < 2:
                continue

            # 光線の強度に応じて色を変える
            intensity_factor = ray.intensity
            color = (1.0 * intensity_factor, 0.8 * intensity_factor, 0.2 * intensity_factor)

            for i in range(len(ray.path) - 1):
                p1 = ray.path[i]
                p2 = ray.path[i + 1]

                # ビュー座標系に変換
                p1_3d = (p1[0] - self.view_width / 2,
                        -(p1[1] - self.view_height / 2),
                        p1[2])
                p2_3d = (p2[0] - self.view_width / 2,
                        -(p2[1] - self.view_height / 2),
                        p2[2])

                self.draw_line_3d(p1_3d, p2_3d, color, 1.5)

        # 座標軸を描画（デバッグ用）
        axis_length = 100
        # X軸（赤）
        self.draw_line_3d((0, 0, 0), (axis_length, 0, 0), (1, 0, 0), 2)
        # Y軸（緑）
        self.draw_line_3d((0, 0, 0), (0, axis_length, 0), (0, 1, 0), 2)
        # Z軸（青）
        self.draw_line_3d((0, 0, 0), (0, 0, axis_length), (0, 0, 1), 2)

        # 水面を最後に描画（半透明なので）
        self.draw_water_plane_3d()

    def draw_light_glow_3d(self, x, y, z, light_angle, light_spread, intensity=1.0):
        """光源からのグロー効果（光の放射）を描画"""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # 加算ブレンド

        # 強度に応じた明るさ
        bright = min(1.0, intensity)

        # 光源本体（明るい黄色の球）
        glColor4f(bright, bright, bright * 0.8, 1.0)
        glPushMatrix()
        glTranslatef(x, y, z)
        quadric = gluNewQuadric()
        gluSphere(quadric, 12, 16, 16)
        gluDeleteQuadric(quadric)
        glPopMatrix()

        # グロー効果（複数の半透明の球で表現）- 強度に応じてサイズと明るさ変化
        for i in range(3):
            alpha = (0.3 - i * 0.08) * intensity
            radius = (20 + i * 15) * (0.5 + intensity * 0.5)
            glColor4f(bright, bright * 0.95, bright * 0.7, alpha)
            glPushMatrix()
            glTranslatef(x, y, z)
            quadric = gluNewQuadric()
            gluSphere(quadric, radius, 12, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()

        # 光線（2Dの角度に合わせた方向に照射）
        # light_angle: 0が下向き、正が時計回り
        # 3D座標系: Y軸が上向きなので、下向きは-Y方向
        center_dir_x = math.sin(light_angle)
        center_dir_y = -math.cos(light_angle)  # 下向きが基準

        glLineWidth(2.0)
        num_rays = 12
        ray_length = 200 * (0.5 + intensity * 0.5)  # 強度に応じて長さ変化

        for i in range(num_rays):
            # 円周上の角度
            phi = (2 * math.pi * i) / num_rays
            # 広がり角度（中心からの角度）
            theta = light_spread / 2 * 0.8  # 少し狭めに

            # 中心方向からの広がりを計算
            # X-Y平面での広がり
            spread_x = math.sin(theta) * math.cos(phi)
            spread_y = math.sin(theta) * math.sin(phi)

            # 光線の終点方向
            dir_x = center_dir_x + spread_x * math.cos(light_angle)
            dir_y = center_dir_y - spread_y
            dir_z = spread_x * math.sin(light_angle) + math.sin(theta) * math.sin(phi)

            # 正規化
            length = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            if length > 0:
                dir_x /= length
                dir_y /= length
                dir_z /= length

            end_x = x + ray_length * dir_x
            end_y = y + ray_length * dir_y
            end_z = z + ray_length * dir_z

            # グラデーション効果のある光線（強度に応じた明るさ）
            ray_alpha = 0.5 * intensity
            glBegin(GL_LINES)
            glColor4f(bright, bright, bright * 0.6, ray_alpha)
            glVertex3f(x, y, z)
            glColor4f(bright, bright * 0.9, bright * 0.5, 0.0)
            glVertex3f(end_x, end_y, end_z)
            glEnd()

        glDisable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)

    def draw_3d_view_natural(self):
        """自然光3Dビュー全体を描画（光線なし、2D光源位置に連動）"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_3d_perspective()

        # 2Dの光源位置から3Dライト位置を計算
        light_x_2d, light_y_2d = self.light_position
        light_x_3d = light_x_2d - self.view_width / 2
        light_y_3d = -(light_y_2d - self.view_height / 2)
        light_z_3d = 0

        # ライト位置を更新（点光源として設定）
        glLightfv(GL_LIGHT0, GL_POSITION, [light_x_3d, light_y_3d, light_z_3d, 1.0])

        # 光の強度を反映（diffuseとspecularを調整）
        intensity = self.light_intensity
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [intensity, intensity, intensity * 0.9, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [intensity, intensity, intensity * 0.95, 1])

        # 光源のグロー効果を描画（2Dの角度と広がりを反映）
        self.draw_light_glow_3d(light_x_3d, light_y_3d, light_z_3d, self.light_angle, self.light_spread, self.light_intensity)

        # 球を描画（3D座標を使用）- 不透明なものを先に描画
        for ball in self.engine.balls:
            x_3d, y_3d, z_3d = ball['position']
            # ビュー座標系に変換
            x_3d_view = x_3d - self.view_width / 2
            y_3d_view = -(y_3d - self.view_height / 2)

            # 光の角度から、光源の方向を計算
            # light_angle: 0が下向き、正が時計回り（2D座標系）
            # 光線が進む方向（2D）: (sin(angle), cos(angle)) - Y正が下向き
            # OpenGLの方向光源(w=0)では「光源がある方向」を指定
            # 光が下に進む = 光源は上にある = 3DでY+方向
            light_dir_x = -math.sin(self.light_angle)  # 光線の進む方向の逆
            light_dir_y = math.cos(self.light_angle)   # 光線が下に進む→光源は上
            light_dir_z = -0.2  # 少し奥から

            glLightfv(GL_LIGHT0, GL_POSITION, [light_dir_x, light_dir_y, light_dir_z, 0.0])

            # 球の色（自然な色合い）
            self.draw_sphere_3d(x_3d_view, y_3d_view, z_3d, ball['radius'], (0.85, 0.75, 0.7))

        # 座標軸を描画（デバッグ用）
        axis_length = 100
        # X軸（赤）
        self.draw_line_3d((0, 0, 0), (axis_length, 0, 0), (1, 0, 0), 2)
        # Y軸（緑）
        self.draw_line_3d((0, 0, 0), (0, axis_length, 0), (0, 1, 0), 2)
        # Z軸（青）
        self.draw_line_3d((0, 0, 0), (0, 0, axis_length), (0, 0, 1), 2)

        # 水面を最後に描画（半透明なので）
        self.draw_water_plane_3d()

    def render_raytracing(self):
        """フォンシェーディングで光源・水面・球を高速描画"""
        render_width = 500
        render_height = 400

        self.raytracing_image = pygame.Surface((render_width, render_height))

        # 背景グラデーション（上は明るく、下は暗く）
        for y in range(render_height):
            t = y / render_height
            r = int(60 + 40 * (1 - t))
            g = int(70 + 50 * (1 - t))
            b = int(100 + 60 * (1 - t))
            pygame.draw.line(self.raytracing_image, (r, g, b), (0, y), (render_width, y))

        # 水面の位置（シミュレーションの水面位置に対応）
        water_ratio = self.engine.water_level / self.view_height
        water_screen_y = int(render_height * water_ratio * 0.8 + render_height * 0.1)

        # 球の位置（シミュレーションの球位置に対応）
        ball_screen_x = render_width // 2
        ball_screen_y = int(water_screen_y + render_height * 0.25)
        ball_screen_radius = 50

        # 光源位置
        light_ratio_x = self.light_position[0] / self.view_width
        light_ratio_y = self.light_position[1] / self.view_height
        light_x = int(render_width * light_ratio_x)
        light_y = int(render_height * light_ratio_y * 0.5)

        # 床を描画
        floor_y = render_height - 40
        for y in range(floor_y, render_height):
            t = (y - floor_y) / (render_height - floor_y)
            gray = int(50 + 30 * (1 - t))
            pygame.draw.line(self.raytracing_image, (gray, gray + 10, gray + 20), (0, y), (render_width, y))

        # 影を描画
        shadow_x = ball_screen_x + (ball_screen_x - light_x) // 4
        shadow_w = int(ball_screen_radius * 1.2)
        shadow_h = int(ball_screen_radius * 0.25)
        shadow_surf = pygame.Surface((shadow_w * 2, shadow_h * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (20, 25, 30, 120), (0, 0, shadow_w * 2, shadow_h * 2))
        self.raytracing_image.blit(shadow_surf, (shadow_x - shadow_w, floor_y - shadow_h))

        # 水面を描画（半透明）
        water_surf = pygame.Surface((render_width, floor_y - water_screen_y), pygame.SRCALPHA)
        water_surf.fill((70, 120, 180, 80))
        self.raytracing_image.blit(water_surf, (0, water_screen_y))

        # 水面の境界線
        pygame.draw.line(self.raytracing_image, (100, 150, 200), (0, water_screen_y), (render_width, water_screen_y), 2)

        # 球をフォンシェーディングで描画
        self.draw_phong_sphere(ball_screen_x, ball_screen_y, ball_screen_radius, light_x, light_y)

        # 光源を描画（グロー効果付き）
        for r in range(25, 0, -3):
            alpha = int(200 * (1 - r / 25))
            pygame.draw.circle(self.raytracing_image, (255, 255, 200), (light_x, light_y), r)
        pygame.draw.circle(self.raytracing_image, (255, 255, 240), (light_x, light_y), 8)

        # 光線を描画
        num_rays = 9
        if self.light_spread > 0.01:
            for i in range(num_rays):
                angle = self.light_angle + self.light_spread * (i / (num_rays - 1) - 0.5)

                # 水面との交点を計算
                if math.cos(angle) > 0.01:
                    t_water = (water_screen_y - light_y) / math.cos(angle)
                    water_x = int(light_x + t_water * math.sin(angle))

                    if 0 <= water_x <= render_width:
                        # 空気中の光線（黄色）
                        pygame.draw.line(self.raytracing_image, (255, 220, 100),
                                        (light_x, light_y), (water_x, water_screen_y), 2)

                        # 屈折計算
                        sin_i = math.sin(angle)
                        sin_r = sin_i / self.engine.water_refractive_index
                        if abs(sin_r) <= 1:
                            refract_angle = math.asin(sin_r)
                            length = 150
                            end_x = int(water_x + length * math.sin(refract_angle))
                            end_y = int(water_screen_y + length * math.cos(refract_angle))
                            # 水中の光線（オレンジがかった色）
                            pygame.draw.line(self.raytracing_image, (255, 180, 80),
                                            (water_x, water_screen_y), (end_x, end_y), 2)

    def draw_phong_sphere(self, cx, cy, radius, light_x, light_y):
        """フォンシェーディングで球を描画（高画質）"""
        light_z = -150
        light_dir = np.array([light_x - cx, light_y - cy, light_z], dtype=float)
        light_dir = light_dir / np.linalg.norm(light_dir)
        view_dir = np.array([0, 0, -1], dtype=float)

        # シェーディングパラメータ
        ambient = 0.15
        diffuse_k = 0.55
        specular_k = 0.4
        shininess = 40
        base_color = np.array([140, 150, 170], dtype=float)

        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                dist_sq = x * x + y * y
                if dist_sq <= radius * radius:
                    z = math.sqrt(radius * radius - dist_sq)
                    normal = np.array([x / radius, y / radius, -z / radius], dtype=float)

                    # 環境光
                    color = base_color * ambient

                    # 拡散光
                    diff = max(0, np.dot(normal, -light_dir))
                    color = color + base_color * diffuse_k * diff

                    # 鏡面反射
                    reflect = 2 * np.dot(normal, -light_dir) * normal - (-light_dir)
                    spec = max(0, np.dot(reflect, -view_dir)) ** shininess
                    color = color + np.array([255, 255, 255]) * specular_k * spec

                    r = int(min(255, max(0, color[0])))
                    g = int(min(255, max(0, color[1])))
                    b = int(min(255, max(0, color[2])))

                    px, py = cx + x, cy + y
                    if 0 <= px < self.raytracing_image.get_width() and 0 <= py < self.raytracing_image.get_height():
                        self.raytracing_image.set_at((px, py), (r, g, b))

    def draw_raytracing_view(self):
        """レイトレーシング結果を表示（フルスクリーン）"""
        if self.raytracing_image:
            self.screen.blit(self.raytracing_image, (0, 0))

        # 操作説明
        help_text = self.font.render("2: 2Dモード  3: 3Dモード  4: レイトレ再描画", True, (255, 255, 255))
        self.screen.blit(help_text, (10, 10))

    def draw_raytracing_2d(self):
        """レイトレーシング結果を2Dビューエリアに表示"""
        offset_x = self.ui_panel_width + self.view_margin
        offset_y = 60

        if self.raytracing_image:
            # ビューエリアに合わせてスケール
            scaled = pygame.transform.smoothscale(
                self.raytracing_image,
                (self.view_width * 2 + self.view_margin, self.view_height)
            )
            self.screen.blit(scaled, (offset_x, offset_y))

        # 枠線
        pygame.draw.rect(self.screen, (100, 100, 100),
                        (offset_x, offset_y, self.view_width * 2 + self.view_margin, self.view_height), 2)

        # タイトル
        title = self.title_font.render("レイトレーシング（4キーで再描画）", True, self.COLOR_TEXT)
        self.screen.blit(title, (offset_x, 20))

    def _set_light_angle(self, angle_deg: float):
        """光の角度を設定（スライダー用コールバック）"""
        self.light_angle = np.radians(angle_deg)
        self.update_simulation()

    def _set_light_spread(self, spread_deg: float):
        """光の広がりを設定（スライダー用コールバック）"""
        self.light_spread = np.radians(spread_deg)
        self.update_simulation()

    def _set_water_level(self, level: float):
        """水面位置を設定（スライダー用コールバック）"""
        self.engine.water_level = level
        self.update_simulation()

    def _set_refractive_index(self, value: float):
        """屈折率を設定（スライダー用コールバック）"""
        # 小数第2位で丸める
        self.engine.water_refractive_index = round(value, 2)
        self.update_simulation()

    def _set_light_intensity(self, value: float):
        """光の強度を設定（スライダー用コールバック）"""
        self.light_intensity = round(value, 2)

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
        offset_x = self.ui_panel_width + self.view_margin
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
            pos_3d = ball['position']
            radius = ball['radius']

            # 3D座標からX-Y平面（正面図）を取得
            pos_2d = (pos_3d[0], pos_3d[1])

            # ズーム適用した座標と半径
            zoomed_pos = (pos_2d[0] * zoom, pos_2d[1] * zoom)
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

        # 光線を描画（X-Y平面への投影）
        for ray in self.engine.rays:
            if len(ray.path) > 1:
                # 3D座標からX-Y平面（正面図）へ投影
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
        offset_x = self.ui_panel_width + self.view_width + self.view_margin * 2
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
            pos_3d = ball['position']
            radius = ball['radius']
            top_x = int((self.view_width // 2) * zoom)  # 画面中央（X軸は固定）
            top_y = int(pos_3d[1] * zoom)  # 3D座標のY座標をそのまま使用
            pygame.draw.circle(view_surface, self.COLOR_BALL, (top_x, top_y), int(radius * zoom))
            pygame.draw.circle(view_surface, (200, 50, 50), (top_x, top_y), int(radius * zoom), max(2, int(2 * zoom)))

        # 光線を縦線として描画（上から下に向かう、画面中央から広がる）
        num_rays = len(self.engine.rays)

        # 球のY座標を取得（3D座標のY成分）
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
        # UIパネルの背景
        panel_rect = pygame.Rect(0, 0, self.ui_panel_width, self.height)
        pygame.draw.rect(self.screen, (240, 240, 245), panel_rect)
        pygame.draw.line(self.screen, (180, 180, 180), (self.ui_panel_width, 0), (self.ui_panel_width, self.height), 2)

        y = 20

        # タイトル
        title = self.title_font.render("パラメータ", True, self.COLOR_TEXT)
        self.screen.blit(title, (15, y))
        y += 45

        # 光源位置（読み取り専用）
        text = self.small_font.render("光源位置", True, self.COLOR_TEXT)
        self.screen.blit(text, (15, y))
        y += 18
        pos_text = self.small_font.render(f"X: {int(self.light_position[0])}, Y: {int(self.light_position[1])}", True, (100, 100, 100))
        self.screen.blit(pos_text, (20, y))
        y += 30

        # スライダーを描画
        for slider in self.sliders:
            slider.draw(self.screen, self.small_font)

        y = 300

        # 光線数（読み取り専用）
        text = self.small_font.render(f"光線数: {len(self.engine.rays)} 本", True, (100, 100, 100))
        self.screen.blit(text, (15, y))
        y += 50

        # 操作説明
        help_title = self.font.render("操作方法", True, self.COLOR_TEXT)
        self.screen.blit(help_title, (15, y))
        y += 30

        help_texts = [
            "左クリック: 光源移動(2D)",
            "ホイール: ズーム",
            "中クリック: 平行移動",
            "右クリック: 回転(3D)",
            "左右キー: 角度",
            "Q/E: 広がり",
            "↑↓: 水面",
            "2/3キー: 2D/3D切替",
            "R: リセット",
        ]

        for text in help_texts:
            surface = self.small_font.render(text, True, (100, 100, 100))
            self.screen.blit(surface, (20, y))
            y += 22

    def handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            # スライダーのイベント処理を優先
            slider_handled = False
            for slider in self.sliders:
                if slider.handle_event(event):
                    slider_handled = True
                    break

            if slider_handled:
                continue

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
                    self.engine.water_level = max(50, self.engine.water_level - 1)
                    self.sliders[2].value = int(self.engine.water_level)
                    self.update_simulation()
                elif event.key == pygame.K_DOWN:
                    # 水面を下げる
                    self.engine.water_level = min(self.view_height - 50, self.engine.water_level + 1)
                    self.sliders[2].value = int(self.engine.water_level)
                    self.update_simulation()
                elif event.key == pygame.K_LEFT:
                    # 光の角度を左に
                    self.light_angle -= np.pi / 180  # 1度ずつ
                    self.sliders[0].value = int(np.degrees(self.light_angle))
                    self.update_simulation()
                elif event.key == pygame.K_RIGHT:
                    # 光の角度を右に
                    self.light_angle += np.pi / 180  # 1度ずつ
                    self.sliders[0].value = int(np.degrees(self.light_angle))
                    self.update_simulation()
                elif event.key == pygame.K_q:
                    # 光の広がりを狭く
                    self.light_spread = max(0, self.light_spread - np.pi / 180)  # 1度ずつ
                    self.sliders[1].value = int(np.degrees(self.light_spread))
                    self.update_simulation()
                elif event.key == pygame.K_e:
                    # 光の広がりを広く
                    self.light_spread = min(np.pi, self.light_spread + np.pi / 180)  # 1度ずつ
                    self.sliders[1].value = int(np.degrees(self.light_spread))
                    self.update_simulation()
                elif event.key == pygame.K_n:
                    # 屈折率を下げる（0.01刻み）
                    new_index = max(1.00, self.engine.water_refractive_index - 0.01)
                    self.engine.water_refractive_index = round(new_index, 2)
                    self.sliders[3].value = self.engine.water_refractive_index
                    self.update_simulation()
                elif event.key == pygame.K_m:
                    # 屈折率を上げる（0.01刻み）
                    new_index = min(2.00, self.engine.water_refractive_index + 0.01)
                    self.engine.water_refractive_index = round(new_index, 2)
                    self.sliders[3].value = self.engine.water_refractive_index
                    self.update_simulation()
                elif event.key == pygame.K_2:
                    # 2Dモードに切り替え
                    if self.view_mode_3d or self.view_mode_raytracing or self.view_mode_natural_3d:
                        self.view_mode_3d = False
                        self.view_mode_raytracing = False
                        self.view_mode_natural_3d = False
                        # Pygameの2D描画モードに戻す
                        pygame.display.set_mode((self.width, self.height))
                elif event.key == pygame.K_3:
                    # 3Dモード（光線あり）に切り替え
                    if not self.view_mode_3d:
                        self.view_mode_3d = True
                        self.view_mode_raytracing = False
                        self.view_mode_natural_3d = False
                        # OpenGL有効のウィンドウに切り替え
                        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
                        self.init_opengl()
                elif event.key == pygame.K_4:
                    # 自然光3Dモード（光線なし）に切り替え
                    if not self.view_mode_natural_3d:
                        self.view_mode_3d = False
                        self.view_mode_raytracing = False
                        self.view_mode_natural_3d = True
                        # OpenGL有効のウィンドウに切り替え
                        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
                        self.init_opengl_natural()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左クリック
                    # 2Dモード時のみ光源ドラッグ
                    if not self.view_mode_3d:
                        mouse_pos = pygame.mouse.get_pos()
                        side_view_x = self.ui_panel_width + self.view_margin
                        side_view_y = 60
                        # 横図ビュー内かチェック
                        if (side_view_x <= mouse_pos[0] <= side_view_x + self.view_width and
                            side_view_y <= mouse_pos[1] <= side_view_y + self.view_height):
                            # ズーム・オフセットを考慮した光源の表示位置を計算
                            zoom = self.side_view_zoom
                            if zoom > 1.0:
                                # ズーム時の切り取り位置を計算
                                crop_x = max(0, min(int(self.view_width * zoom) - self.view_width,
                                                    (int(self.view_width * zoom) - self.view_width) // 2 - int(self.side_view_offset[0] * zoom)))
                                crop_y = max(0, min(int(self.view_height * zoom) - self.view_height,
                                                    (int(self.view_height * zoom) - self.view_height) // 2 - int(self.side_view_offset[1] * zoom)))
                                # 画面上の光源位置
                                display_light_x = int(self.light_position[0] * zoom) - crop_x + side_view_x
                                display_light_y = int(self.light_position[1] * zoom) - crop_y + side_view_y
                            elif zoom < 1.0:
                                # 縮小時は中央配置
                                center_x = side_view_x + (self.view_width - int(self.view_width * zoom)) // 2
                                center_y = side_view_y + (self.view_height - int(self.view_height * zoom)) // 2
                                display_light_x = int(self.light_position[0] * zoom) + center_x
                                display_light_y = int(self.light_position[1] * zoom) + center_y
                            else:
                                # 等倍時
                                display_light_x = int(self.light_position[0]) + side_view_x
                                display_light_y = int(self.light_position[1]) + side_view_y

                            # 光源をドラッグ開始（当たり判定）
                            if np.linalg.norm(np.array(mouse_pos) - np.array([display_light_x, display_light_y])) < int(10 * zoom):
                                self.dragging_light = True
                elif event.button == 2:  # マウスホイールクリック（中クリック）
                    mouse_pos = pygame.mouse.get_pos()
                    self.drag_start_pos = mouse_pos
                    if self.view_mode_3d or self.view_mode_natural_3d:
                        # 3Dモード時はカメラ平行移動
                        self.dragging_camera_pan = True
                        self.camera_pan_start = mouse_pos
                    else:
                        # 2Dモード時は平行移動
                        side_view_x = self.ui_panel_width + self.view_margin
                        side_view_y = 60
                        top_view_x = self.ui_panel_width + self.view_width + self.view_margin * 2
                        top_view_y = 60
                        # 横図ビュー内かチェック
                        if (side_view_x <= mouse_pos[0] <= side_view_x + self.view_width and
                            side_view_y <= mouse_pos[1] <= side_view_y + self.view_height):
                            self.dragging_side_view = True
                        # 上面図ビュー内
                        elif (top_view_x <= mouse_pos[0] <= top_view_x + self.view_width and
                              top_view_y <= mouse_pos[1] <= top_view_y + self.view_height):
                            self.dragging_top_view = True
                elif event.button == 3:  # 右クリック
                    if self.view_mode_3d or self.view_mode_natural_3d:
                        # 3Dモード時はカメラ回転
                        mouse_pos = pygame.mouse.get_pos()
                        self.dragging_camera = True
                        self.camera_drag_start = mouse_pos
                elif event.button == 4:  # マウスホイール上（ズームイン）
                    if self.view_mode_3d or self.view_mode_natural_3d:
                        # 3Dモード時はカメラ距離を縮める（ズームイン）
                        self.camera_distance = max(200.0, self.camera_distance - 30.0)
                    else:
                        # 2Dモード時
                        mouse_pos = pygame.mouse.get_pos()
                        side_view_x = self.ui_panel_width + self.view_margin
                        side_view_y = 60
                        top_view_x = self.ui_panel_width + self.view_width + self.view_margin * 2
                        top_view_y = 60
                        # 横図ビュー内
                        if (side_view_x <= mouse_pos[0] <= side_view_x + self.view_width and
                            side_view_y <= mouse_pos[1] <= side_view_y + self.view_height):
                            self.side_view_zoom = min(3.0, self.side_view_zoom * 1.1)
                        # 上面図ビュー内
                        elif (top_view_x <= mouse_pos[0] <= top_view_x + self.view_width and
                              top_view_y <= mouse_pos[1] <= top_view_y + self.view_height):
                            self.top_view_zoom = min(3.0, self.top_view_zoom * 1.1)
                elif event.button == 5:  # マウスホイール下（ズームアウト）
                    if self.view_mode_3d or self.view_mode_natural_3d:
                        # 3Dモード時はカメラ距離を伸ばす（ズームアウト）
                        self.camera_distance = min(2000.0, self.camera_distance + 30.0)
                    else:
                        # 2Dモード時
                        mouse_pos = pygame.mouse.get_pos()
                        side_view_x = self.ui_panel_width + self.view_margin
                        side_view_y = 60
                        top_view_x = self.ui_panel_width + self.view_width + self.view_margin * 2
                        top_view_y = 60
                        # 横図ビュー内
                        if (side_view_x <= mouse_pos[0] <= side_view_x + self.view_width and
                            side_view_y <= mouse_pos[1] <= side_view_y + self.view_height):
                            self.side_view_zoom = max(0.5, self.side_view_zoom / 1.1)
                        # 上面図ビュー内
                        elif (top_view_x <= mouse_pos[0] <= top_view_x + self.view_width and
                              top_view_y <= mouse_pos[1] <= top_view_y + self.view_height):
                            self.top_view_zoom = max(0.5, self.top_view_zoom / 1.1)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_light = False
                elif event.button == 2:  # マウスホイールクリック（中クリック）
                    self.dragging_side_view = False
                    self.dragging_top_view = False
                    self.dragging_camera_pan = False
                    self.drag_start_pos = None
                    self.camera_pan_start = None
                elif event.button == 3:  # 右クリック
                    self.dragging_camera = False
                    self.camera_drag_start = None

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_camera and self.camera_drag_start:
                    # 3Dモード時のカメラ回転
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - self.camera_drag_start[0]
                    dy = mouse_pos[1] - self.camera_drag_start[1]

                    # カメラの回転を更新（マウスの動きに合わせて直感的に）
                    self.camera_rotation[1] -= dx * 0.5  # yaw（左右反転）
                    self.camera_rotation[0] += dy * 0.5  # pitch（上下反転）

                    # pitch を -89 ~ 89 度に制限
                    self.camera_rotation[0] = max(-89, min(89, self.camera_rotation[0]))

                    self.camera_drag_start = mouse_pos
                elif self.dragging_camera_pan and self.camera_pan_start:
                    # 3Dモード時のカメラ平行移動
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - self.camera_pan_start[0]
                    dy = mouse_pos[1] - self.camera_pan_start[1]

                    # カメラの向きに基づいて平行移動方向を計算
                    yaw_rad = math.radians(self.camera_rotation[1])

                    # 右方向ベクトル（X軸周り）
                    right_x = math.cos(yaw_rad)
                    right_z = -math.sin(yaw_rad)

                    # 上方向は常にY軸
                    up_x = 0
                    up_y = 1
                    up_z = 0

                    # 移動量を計算（感度調整、マウスの動きに合わせて直感的に）
                    move_speed = 0.5
                    self.camera_target[0] -= (right_x * dx) * move_speed
                    self.camera_target[1] += dy * move_speed
                    self.camera_target[2] -= (right_z * dx) * move_speed

                    self.camera_pan_start = mouse_pos
                elif self.dragging_light:
                    mouse_pos = pygame.mouse.get_pos()
                    side_view_x = self.ui_panel_width + self.view_margin
                    side_view_y = 60

                    # ズーム・オフセットを考慮してワールド座標に変換
                    zoom = self.side_view_zoom
                    if zoom > 1.0:
                        # ズーム時の切り取り位置を計算
                        crop_x = max(0, min(int(self.view_width * zoom) - self.view_width,
                                            (int(self.view_width * zoom) - self.view_width) // 2 - int(self.side_view_offset[0] * zoom)))
                        crop_y = max(0, min(int(self.view_height * zoom) - self.view_height,
                                            (int(self.view_height * zoom) - self.view_height) // 2 - int(self.side_view_offset[1] * zoom)))
                        # マウス位置をワールド座標に変換
                        world_x = (mouse_pos[0] - side_view_x + crop_x) / zoom
                        world_y = (mouse_pos[1] - side_view_y + crop_y) / zoom
                    elif zoom < 1.0:
                        # 縮小時
                        center_x = (self.view_width - int(self.view_width * zoom)) // 2
                        center_y = (self.view_height - int(self.view_height * zoom)) // 2
                        world_x = (mouse_pos[0] - side_view_x - center_x) / zoom
                        world_y = (mouse_pos[1] - side_view_y - center_y) / zoom
                    else:
                        # 等倍時
                        world_x = mouse_pos[0] - side_view_x
                        world_y = mouse_pos[1] - side_view_y

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
        # 光源位置を3Dに変換
        light_pos_3d = (self.light_position[0], self.light_position[1], 0)

        # 3D光源を生成
        self.engine.create_light_source_3d(
            light_pos_3d,
            num_rays_radial=20,  # 放射方向の光線数
            num_rays_circular=12,  # 円周方向の光線数
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
            if self.view_mode_3d:
                # 3Dモード（光線あり）
                self.draw_3d_view()
                pygame.display.flip()
            elif self.view_mode_natural_3d:
                # 自然光3Dモード（光線なし）
                self.draw_3d_view_natural()
                pygame.display.flip()
            elif self.view_mode_raytracing:
                # レイトレーシング風2Dモード（未使用）
                self.screen.fill(self.COLOR_BG)
                self.draw_raytracing_2d()
                self.draw_ui()
                pygame.display.flip()
            else:
                # 通常の2Dモード
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
