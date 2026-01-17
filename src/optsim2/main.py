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


class TabGroup:
    """タブUIコンポーネント"""
    def __init__(self, x: int, y: int, width: int, tabs: List[str]):
        self.x = x
        self.y = y
        self.width = width
        self.tabs = tabs
        self.active_tab = 0
        self.tab_height = 28
        self.tab_width = width // len(tabs)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        for i, tab_name in enumerate(self.tabs):
            tab_x = self.x + i * self.tab_width
            tab_rect = pygame.Rect(tab_x, self.y, self.tab_width, self.tab_height)

            # アクティブタブは明るく、非アクティブは暗く
            if i == self.active_tab:
                pygame.draw.rect(surface, (240, 240, 245), tab_rect)
                pygame.draw.rect(surface, (70, 130, 220), tab_rect, 2)
                text_color = (50, 50, 50)
            else:
                pygame.draw.rect(surface, (200, 200, 210), tab_rect)
                pygame.draw.rect(surface, (150, 150, 160), tab_rect, 1)
                text_color = (100, 100, 100)

            # タブ名
            text_surf = font.render(tab_name, True, text_color)
            text_x = tab_x + (self.tab_width - text_surf.get_width()) // 2
            text_y = self.y + (self.tab_height - text_surf.get_height()) // 2
            surface.blit(text_surf, (text_x, text_y))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            for i in range(len(self.tabs)):
                tab_x = self.x + i * self.tab_width
                tab_rect = pygame.Rect(tab_x, self.y, self.tab_width, self.tab_height)
                if tab_rect.collidepoint(mouse_pos):
                    self.active_tab = i
                    return True
        return False


class Slider:
    """スライダーUIコンポーネント"""
    def __init__(self, x: int, y: int, width: int, min_val: float, max_val: float,
                 initial_val: float, label: str, callback: Callable[[float], None],
                 tab_index: int = 0):
        self.rect = pygame.Rect(x, y, width, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.callback = callback
        self.dragging = False
        self.knob_radius = 8
        self.tab_index = tab_index  # どのタブに属するか

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
        elif self.max_val <= 30.0 and self.min_val < 1.0:
            # mm単位など小数第2位まで必要なもの
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
        self.light_position = (300, 450)
        self.light_angle = np.radians(45)  # 光の角度（ラジアン、初期値45°）
        self.light_spread = np.radians(5)  # 光の広がり角度（初期値5°）
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

        # タブグループの初期化（光源、球、環境の3タブ）
        self.tab_group = TabGroup(10, 50, self.ui_panel_width - 20, ["光源", "球", "環境"])

        # スライダーの初期化
        self.sliders = []
        slider_x = 20
        slider_y_start = 95  # タブの下から開始
        slider_width = 180
        slider_spacing = 55

        # === タブ0: 光源設定 ===
        # 光の角度スライダー（小数第1位まで）
        self.sliders.append(Slider(
            slider_x, slider_y_start, slider_width,
            -90.0, 90.0, round(np.degrees(self.light_angle), 1),
            "光の角度 (°)",
            lambda v: self._set_light_angle(v),
            tab_index=0
        ))

        # 光の広がりスライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing, slider_width,
            0, 180, int(np.degrees(self.light_spread)),
            "光の広がり (°)",
            lambda v: self._set_light_spread(v),
            tab_index=0
        ))

        # 光の強度スライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 2, slider_width,
            0.0, 2.0, self.light_intensity,
            "光の強度",
            lambda v: self._set_light_intensity(v),
            tab_index=0
        ))

        # 光源の個数
        self.light_count = 15
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 3, slider_width,
            1, 30, self.light_count,
            "光源の個数",
            lambda v: self._set_light_count(int(v)),
            tab_index=0
        ))

        # 光源の間隔（mm単位）
        self.light_spacing_mm = 3.0  # デフォルト3mm
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 4, slider_width,
            1.0, 100.0, self.light_spacing_mm,
            "光源の間隔 (mm)",
            lambda v: self._set_light_spacing_mm(v),
            tab_index=0
        ))

        # === タブ1: 球設定 ===
        # 球の個数（初期値2）
        self.ball_count = 2
        self.sliders.append(Slider(
            slider_x, slider_y_start, slider_width,
            1, 30, self.ball_count,
            "球の個数",
            lambda v: self._set_ball_count(int(v)),
            tab_index=1
        ))

        # 球の大きさ（mm単位、内部ではピクセルに変換）
        # 1mm = 4ピクセル
        self.ball_radius_mm = 8.73  # デフォルト8.73mm
        self.mm_to_pixel = 4.0  # 1mm = 4ピクセル
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing, slider_width,
            0.1, 30.0, self.ball_radius_mm,
            "球の半径 (mm)",
            lambda v: self._set_ball_radius_mm(v),
            tab_index=1
        ))

        # 球の間隔（mm単位）
        self.ball_spacing_mm = 25.0  # デフォルト25mm
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 2, slider_width,
            1.0, 100.0, self.ball_spacing_mm,
            "球の間隔 (mm)",
            lambda v: self._set_ball_spacing_mm(v),
            tab_index=1
        ))

        # 球の回転速度（rpm、Z軸中心）
        self.ball_rotation_rpm = 0.0  # 初期値は回転なし
        self.ball_rotation_angle = 0.0  # 現在の回転角度（ラジアン）
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 3, slider_width,
            0.0, 500.0, self.ball_rotation_rpm,
            "回転速度 (rpm)",
            lambda v: self._set_ball_rotation_rpm(v),
            tab_index=1
        ))

        # === タブ2: 環境設定 ===
        # 水面位置スライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start, slider_width,
            50, self.view_height - 50, int(self.engine.water_level),
            "水面位置 (px)",
            lambda v: self._set_water_level(v),
            tab_index=2
        ))

        # 屈折率スライダー（初期値1.47）
        self.engine.water_refractive_index = 1.47
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing, slider_width,
            1.00, 2.00, self.engine.water_refractive_index,
            "屈折率",
            lambda v: self._set_refractive_index(v),
            tab_index=2
        ))

        # 水面ゆらぎ強度スライダー
        self.sliders.append(Slider(
            slider_x, slider_y_start + slider_spacing * 2, slider_width,
            0.0, 1.0, self.engine.water_ripple_strength,
            "水面ゆらぎ",
            lambda v: self._set_water_ripple(v),
            tab_index=2
        ))

        # スライダーを名前で参照するための辞書
        self.slider_map = {
            'light_angle': self.sliders[0],
            'light_spread': self.sliders[1],
            'light_intensity': self.sliders[2],
            'light_count': self.sliders[3],
            'light_spacing': self.sliders[4],
            'ball_count': self.sliders[5],
            'ball_radius': self.sliders[6],
            'ball_spacing': self.sliders[7],
            'water_level': self.sliders[8],
            'refractive_index': self.sliders[9],
            'water_ripple': self.sliders[10],
        }

        # 初期状態で球を再構築
        self._rebuild_balls()

        # 3Dビューモード設定
        self.view_mode_3d = False  # False=2Dモード, True=3Dモード
        self.view_mode_raytracing = False  # レイトレーシング風3D描画モード（未使用）
        self.view_mode_natural_3d = False  # 自然光3Dモード（キー4）
        self.heatmap_mode = False  # ヒートマップ表示モード（キーH）
        self.show_light_source = True  # 光源表示モード（キーL）
        self.heatmap_cache = {}  # ヒートマップのキャッシュ
        self.heatmap_max_intensity = 1  # ヒートマップの最大強度
        self.raytracing_image = None  # レイトレーシング結果のサーフェス
        self.camera_rotation = [20.0, 45.0]  # [pitch, yaw] in degrees
        self.camera_distance = 800.0
        self.camera_target = [0.0, 0.0, 0.0]  # カメラの注視点（平行移動用）
        self.dragging_camera = False
        self.dragging_camera_pan = False
        self.camera_drag_start = None
        self.camera_pan_start = None

        # 3D視点切り替えボタン定義 (Label, Name, [Pitch, Yaw])
        self.orientation_buttons = [
            {'label': 'Front',  'angle': [0.0, 0.0]},     # XY Plane (+Z)
            {'label': 'Back',   'angle': [0.0, 180.0]},   # XY Plane (-Z)
            {'label': 'Right',  'angle': [0.0, 90.0]},    # YZ Plane (+X)
            {'label': 'Left',   'angle': [0.0, -90.0]},   # YZ Plane (-X)
            {'label': 'Top',    'angle': [90.0, 0.0]},    # XZ Plane (+Y)
            {'label': 'Bottom', 'angle': [-90.0, 0.0]},   # XZ Plane (-Y)
        ]

        # 90度回転ボタン定義
        self.rotation_buttons = [
            {'label': 'Rot R', 'delta': 90.0},
            {'label': 'Rot L', 'delta': -90.0},
        ]

        # 輝度プロファイル機能の状態
        self.profile_mode = False  # プロファイル表示モード
        self.profile_scan_axis = 'Y'  # 'X' or 'Y'
        self.profile_pos = 0  # スキャン位置
        self.dragging_profile_line = False  # プロファイルラインのドラッグ中フラグ

    def setup_default_scene(self):
        """デフォルトのシーンを設定"""
        # 水面の位置を設定
        self.engine.set_water_level(self.view_height * 0.5)

        # 球を追加（水中、3D座標）
        ball_y = self.view_height * 0.7
        ball_x = self.view_width // 2
        ball_z = 0  # Z軸中心
        self.engine.add_ball((ball_x, ball_y, ball_z), 40)

    def draw_orientation_buttons_overlay(self):
        """3Dビュー上に視点切り替えボタンを描画"""
        # 画面右上に配置
        btn_width = 50
        btn_height = 25
        spacing_x = 5
        spacing_y = 5
        
        start_x = self.width - (btn_width * 2 + spacing_x) - 20
        start_y = 60 # タイトルバーの下あたり

        mouse_pos = pygame.mouse.get_pos()
        
        # オーバーレイ用のSurfaceを作成
        # 5行分確保 (Orientation 3行 + Rotation 1行 + Analysis 1行)
        w = btn_width * 2 + spacing_x + 20
        h = btn_height * 5 + spacing_y * 4 + 20
        overlay_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # overlay_surf.fill((0, 0, 0, 100)) # 背景なし、ボタンのみ描画

        # ボタン描画
        for i, btn in enumerate(self.orientation_buttons):
            row = i // 2
            col = i % 2
            
            # Surface内の相対座標
            bx = 10 + col * (btn_width + spacing_x)
            by = 10 + row * (btn_height + spacing_y)
            
            # 実際の画面上での座標（ホバー判定用）
            screen_bx = start_x + bx - 10 # start_xはパネル左上ではなくボタン配置基準なので調整が必要
            # start_xはパネルの左端ではない。
            # パネル左端 = start_x (ここではパネル左上を基準に描画ロジックを組むべきだった)
            
            # ロジック整理: パネルの左上座標
            panel_x = self.width - (btn_width * 2 + spacing_x) - 20
            panel_y = 60
            
            rect = pygame.Rect(bx, by, btn_width, btn_height)
            screen_rect = pygame.Rect(panel_x + bx, panel_y + by, btn_width, btn_height)
            
            # ホバー判定
            color = (80, 80, 90, 200)
            if screen_rect.collidepoint(mouse_pos):
                color = (120, 120, 130, 230)
            
            pygame.draw.rect(overlay_surf, color, rect)
            pygame.draw.rect(overlay_surf, (200, 200, 200), rect, 1)
            
            # テキスト
            text = self.small_font.render(btn['label'], True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            overlay_surf.blit(text, text_rect)

        # 回転ボタン描画
        panel_x = self.width - (btn_width * 2 + spacing_x) - 20
        panel_y = 60
        rotation_start_row = 3
        
        for i, btn in enumerate(self.rotation_buttons):
            row = rotation_start_row
            col = i % 2
            
            # Surface内の相対座標
            bx = 10 + col * (btn_width + spacing_x)
            by = 10 + row * (btn_height + spacing_y)
            
            rect = pygame.Rect(bx, by, btn_width, btn_height)
            screen_rect = pygame.Rect(panel_x + bx, panel_y + by, btn_width, btn_height)
            
            # ホバー判定
            color = (80, 80, 90, 200)
            if screen_rect.collidepoint(mouse_pos):
                color = (120, 120, 130, 230)
            
            pygame.draw.rect(overlay_surf, color, rect)
            pygame.draw.rect(overlay_surf, (200, 200, 200), rect, 1)
            
            text = self.small_font.render(btn['label'], True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            overlay_surf.blit(text, text_rect)
            
            text_rect = text.get_rect(center=rect.center)
            overlay_surf.blit(text, text_rect)
            
        # Profile/Axisボタン描画 (最下段)
        profile_row = 4
        # 1行に2つのボタンを配置
        # 左: Profile ON/OFF
        # 右: Axis X/Y
        
        # Profile Button
        p1_width = btn_width
        bx1 = 10
        by = 10 + profile_row * (btn_height + spacing_y)
        rect1 = pygame.Rect(bx1, by, p1_width, btn_height)
        screen_rect1 = pygame.Rect(panel_x + bx1, panel_y + by, p1_width, btn_height)
        
        if self.profile_mode:
            c1 = (100, 150, 200, 230)
        elif screen_rect1.collidepoint(mouse_pos):
            c1 = (120, 120, 130, 230)
        else:
            c1 = (80, 80, 90, 200)
            
        pygame.draw.rect(overlay_surf, c1, rect1)
        pygame.draw.rect(overlay_surf, (200, 200, 200), rect1, 1)
        
        t1 = "Prof:ON" if self.profile_mode else "Prof:OFF"
        text1 = self.small_font.render(t1, True, (255, 255, 255))
        tr1 = text1.get_rect(center=rect1.center)
        overlay_surf.blit(text1, tr1)
        
        # Axis Button
        bx2 = 10 + btn_width + spacing_x
        rect2 = pygame.Rect(bx2, by, p1_width, btn_height)
        screen_rect2 = pygame.Rect(panel_x + bx2, panel_y + by, p1_width, btn_height)
        
        if screen_rect2.collidepoint(mouse_pos):
            c2 = (120, 120, 130, 230)
        else:
            c2 = (80, 80, 90, 200)
            
        pygame.draw.rect(overlay_surf, c2, rect2)
        pygame.draw.rect(overlay_surf, (200, 200, 200), rect2, 1)
        
        t2 = f"Axis: {self.profile_scan_axis}"
        text2 = self.small_font.render(t2, True, (255, 255, 255))
        tr2 = text2.get_rect(center=rect2.center)
        overlay_surf.blit(text2, tr2)

        # OpenGL上に描画
        # パネル左上の座標を指定
        self._blit_pygame_surface_to_opengl(overlay_surf, self.width - (btn_width * 2 + spacing_x) - 20, 60)

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

    def draw_sphere_3d(self, x, y, z, radius, color, rotation_angle=0.0):
        """3D球体を描画（Z軸中心の回転対応）"""
        glPushMatrix()
        glTranslatef(x, y, z)
        # Z軸中心で回転（度単位に変換）
        if rotation_angle != 0.0:
            glRotatef(math.degrees(rotation_angle), 0, 0, 1)
        glColor3f(*color)

        # GLUクアドリックで球体を描画
        quad = gluNewQuadric()
        gluSphere(quad, radius, 32, 32)
        gluDeleteQuadric(quad)

        glPopMatrix()

    def draw_sphere_heatmap_3d(self, ball_idx, ball_world_pos, radius, view_x, view_y, view_z):
        """3D球体をヒートマップで描画（キャッシュされたヒット情報を使用）"""
        glPushMatrix()
        glTranslatef(view_x, view_y, view_z)
        glDisable(GL_LIGHTING)

        # キャッシュからヒットマップを取得
        heatmap = self.heatmap_cache.get(ball_idx, {})
        max_intensity = self.heatmap_max_intensity

        # 球を描画（セグメント数を減らして軽量化）
        slices = 16
        stacks = 12

        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z0 = math.sin(lat0)
            z1 = math.sin(lat1)
            r0 = math.cos(lat0)
            r1 = math.cos(lat1)

            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x_n = math.cos(lng)
                y_n = math.sin(lng)

                for z_val, r_val, stack_idx in [(z0, r0, i), (z1, r1, i + 1)]:
                    nx = x_n * r_val
                    ny = y_n * r_val
                    nz = z_val

                    # キャッシュからこのセグメントの強度を取得
                    seg_key = (stack_idx, j % slices)
                    intensity = heatmap.get(seg_key, 0)
                    color = self.get_heatmap_color(intensity, max_intensity)

                    glColor3f(*color)
                    glVertex3f(nx * radius, ny * radius, nz * radius)

            glEnd()

        glEnable(GL_LIGHTING)
        glPopMatrix()

    def calculate_heatmap_cache(self):
        """全球のヒートマップ情報を事前計算してキャッシュ"""
        self.heatmap_cache = {}
        self.heatmap_max_intensity = 1

        slices = 16
        stacks = 12

        all_intensities = []

        for ball_idx, ball in enumerate(self.engine.balls):
            ball_cx, ball_cy, ball_cz = ball['position']
            ball_r = ball['radius']
            ball_heatmap = {}

            for i in range(stacks + 1):
                # 描画時と同じ計算方法を使用
                lat = math.pi * (-0.5 + float(i) / stacks)
                lat_sin = math.sin(lat)  # Z方向（OpenGLでのローカルZ）
                lat_cos = math.cos(lat)  # X-Y平面での半径

                for j in range(slices):
                    lng = 2 * math.pi * float(j) / slices
                    lng_cos = math.cos(lng)
                    lng_sin = math.sin(lng)

                    # OpenGLローカル座標系での法線（描画時と同じ）
                    # nx = lng_cos * lat_cos
                    # ny = lng_sin * lat_cos
                    # nz = lat_sin
                    nx = lng_cos * lat_cos
                    ny = lng_sin * lat_cos
                    nz = lat_sin

                    # 球の表面のワールド座標（2D座標系）
                    # 注意：2D座標系ではY軸が下向き、OpenGLではY軸が上向き
                    # OpenGLの(nx, ny, nz)を2Dワールド座標に変換
                    # OpenGL Y軸上向き → 2D Y軸下向き（反転）
                    world_x = ball_cx + nx * ball_r
                    world_y = ball_cy - ny * ball_r  # Y軸反転
                    world_z = ball_cz + nz * ball_r

                    # この点に当たる光線をカウント
                    hit_count = 0
                    tolerance = ball_r * 0.4

                    for ray in self.engine.rays:
                        if len(ray.path) < 2:
                            continue

                        for k in range(len(ray.path) - 1):
                            p1 = ray.path[k]
                            p2 = ray.path[k + 1]
                            p1x, p1y = float(p1[0]), float(p1[1])
                            p2x, p2y = float(p2[0]), float(p2[1])
                            p1z = float(p1[2]) if len(p1) > 2 else 0.0
                            p2z = float(p2[2]) if len(p2) > 2 else 0.0

                            dx = p2x - p1x
                            dy = p2y - p1y
                            dz = p2z - p1z
                            line_len_sq = dx * dx + dy * dy + dz * dz
                            if line_len_sq < 0.001:
                                continue

                            t = max(0, min(1, ((world_x - p1x) * dx + (world_y - p1y) * dy + (world_z - p1z) * dz) / line_len_sq))
                            closest_x = p1x + t * dx
                            closest_y = p1y + t * dy
                            closest_z = p1z + t * dz
                            dist = math.sqrt((world_x - closest_x) ** 2 + (world_y - closest_y) ** 2 + (world_z - closest_z) ** 2)

                            if dist <= tolerance:
                                hit_count += 1
                                break

                    ball_heatmap[(i, j)] = hit_count
                    if hit_count > 0:
                        all_intensities.append(hit_count)

            self.heatmap_cache[ball_idx] = ball_heatmap

        # 最大強度を設定
        if all_intensities:
            self.heatmap_max_intensity = max(1, max(all_intensities))

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

    def draw_axis_3d(self):
        """座標軸を画面左下に描画（デザイン性のあるXYZラベル付き）"""
        glDisable(GL_LIGHTING)

        axis_origin = (-350, -200, -200)
        axis_length = 80

        # 軸の色（明るめ）
        x_color = (1.0, 0.3, 0.3)  # 赤
        y_color = (0.3, 1.0, 0.3)  # 緑
        z_color = (0.3, 0.6, 1.0)  # 青

        # 原点の小さな球
        glColor3f(0.8, 0.8, 0.8)
        glPushMatrix()
        glTranslatef(*axis_origin)
        quad = gluNewQuadric()
        gluSphere(quad, 5, 12, 12)
        gluDeleteQuadric(quad)
        glPopMatrix()

        # X軸（赤）- 矢印付き
        x_end = (axis_origin[0] + axis_length, axis_origin[1], axis_origin[2])
        self.draw_line_3d(axis_origin, x_end, x_color, 3)
        # 矢印の先端
        arrow_size = 8
        self.draw_line_3d(x_end, (x_end[0] - arrow_size, x_end[1] + arrow_size/2, x_end[2]), x_color, 2)
        self.draw_line_3d(x_end, (x_end[0] - arrow_size, x_end[1] - arrow_size/2, x_end[2]), x_color, 2)

        # Y軸（緑）- 矢印付き
        y_end = (axis_origin[0], axis_origin[1] + axis_length, axis_origin[2])
        self.draw_line_3d(axis_origin, y_end, y_color, 3)
        # 矢印の先端
        self.draw_line_3d(y_end, (y_end[0] + arrow_size/2, y_end[1] - arrow_size, y_end[2]), y_color, 2)
        self.draw_line_3d(y_end, (y_end[0] - arrow_size/2, y_end[1] - arrow_size, y_end[2]), y_color, 2)

        # Z軸（青）- 矢印付き
        z_end = (axis_origin[0], axis_origin[1], axis_origin[2] + axis_length)
        self.draw_line_3d(axis_origin, z_end, z_color, 3)
        # 矢印の先端
        self.draw_line_3d(z_end, (z_end[0], z_end[1] + arrow_size/2, z_end[2] - arrow_size), z_color, 2)
        self.draw_line_3d(z_end, (z_end[0], z_end[1] - arrow_size/2, z_end[2] - arrow_size), z_color, 2)

        # XYZラベルを描画（正しい向きで）
        s = 10  # ラベルのサイズ
        label_gap = 15

        # X ラベル（X軸の先端に、XZ平面上に描画）
        tx = axis_origin[0] + axis_length + label_gap
        ty = axis_origin[1]
        tz = axis_origin[2]
        # X の形を描画（斜めの2本線）
        self.draw_line_3d((tx, ty, tz - s), (tx, ty, tz + s), x_color, 2)
        self.draw_line_3d((tx + s, ty, tz - s), (tx - s, ty, tz + s), x_color, 2)

        # Y ラベル（Y軸の先端に、XY平面上に描画）
        tx = axis_origin[0]
        ty = axis_origin[1] + axis_length + label_gap
        tz = axis_origin[2]
        # Y の形を描画
        self.draw_line_3d((tx - s, ty + s, tz), (tx, ty, tz), y_color, 2)
        self.draw_line_3d((tx + s, ty + s, tz), (tx, ty, tz), y_color, 2)
        self.draw_line_3d((tx, ty, tz), (tx, ty - s, tz), y_color, 2)

        # Z ラベル（Z軸の先端に、YZ平面上に描画）
        tx = axis_origin[0]
        ty = axis_origin[1]
        tz = axis_origin[2] + axis_length + label_gap
        # Z の形を描画
        self.draw_line_3d((tx, ty + s, tz - s), (tx, ty + s, tz + s), z_color, 2)
        self.draw_line_3d((tx, ty + s, tz + s), (tx, ty - s, tz - s), z_color, 2)
        self.draw_line_3d((tx, ty - s, tz - s), (tx, ty - s, tz + s), z_color, 2)

        glEnable(GL_LIGHTING)

    def draw_water_plane_3d(self):
        """3D水面を描画（半透明、ゆらぎ対応）"""
        # 水面のY座標を計算（2D座標系を3D座標系に変換）
        water_y = -(self.engine.water_level - self.view_height / 2)

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        size = 500
        ripple_strength = self.engine.water_ripple_strength

        if ripple_strength > 0:
            # ゆらぎありの場合、メッシュで描画
            step = 25  # メッシュの細かさ
            freq = self.engine.water_ripple_frequency
            t = self.engine.water_ripple_time
            amplitude = ripple_strength * 15  # 高さの振幅

            glColor4f(0.2, 0.5, 0.8, 0.4)
            for i in range(-size, size, step):
                glBegin(GL_QUAD_STRIP)
                for j in range(-size, size + step, step):
                    for x in [i, i + step]:
                        # ゆらぎの高さを計算
                        ripple = (math.sin(x * freq + t) +
                                  0.5 * math.sin(x * freq * 2.3 + t * 1.7) +
                                  0.3 * math.sin(j * freq * 0.7 + t * 0.8) +
                                  math.sin(j * freq + t * 1.2) * 0.7)
                        y_offset = ripple * amplitude
                        glVertex3f(x, water_y + y_offset, j)
                glEnd()

            # グリッド線（ゆらぎあり）
            glColor4f(0.3, 0.6, 0.9, 0.5)
            glLineWidth(1.0)
            grid_step = 50
            glBegin(GL_LINES)
            for i in range(-10, 11):
                # X方向の線
                for j in range(-size, size, step):
                    x = i * grid_step
                    ripple1 = (math.sin(x * freq + t) + 0.5 * math.sin(x * freq * 2.3 + t * 1.7) +
                               0.3 * math.sin(j * freq * 0.7 + t * 0.8) + math.sin(j * freq + t * 1.2) * 0.7)
                    ripple2 = (math.sin(x * freq + t) + 0.5 * math.sin(x * freq * 2.3 + t * 1.7) +
                               0.3 * math.sin((j + step) * freq * 0.7 + t * 0.8) + math.sin((j + step) * freq + t * 1.2) * 0.7)
                    glVertex3f(x, water_y + ripple1 * amplitude, j)
                    glVertex3f(x, water_y + ripple2 * amplitude, j + step)
                # Z方向の線
                for j in range(-size, size, step):
                    z = i * grid_step
                    ripple1 = (math.sin(j * freq + t) + 0.5 * math.sin(j * freq * 2.3 + t * 1.7) +
                               0.3 * math.sin(z * freq * 0.7 + t * 0.8) + math.sin(z * freq + t * 1.2) * 0.7)
                    ripple2 = (math.sin((j + step) * freq + t) + 0.5 * math.sin((j + step) * freq * 2.3 + t * 1.7) +
                               0.3 * math.sin(z * freq * 0.7 + t * 0.8) + math.sin(z * freq + t * 1.2) * 0.7)
                    glVertex3f(j, water_y + ripple1 * amplitude, z)
                    glVertex3f(j + step, water_y + ripple2 * amplitude, z)
            glEnd()
        else:
            # ゆらぎなしの場合、フラットな平面
            glColor4f(0.2, 0.5, 0.8, 0.4)
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
                glVertex3f(i * step, water_y, -size)
                glVertex3f(i * step, water_y, size)
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
        for ball_idx, ball in enumerate(self.engine.balls):
            x_3d, y_3d, z_3d = ball['position']
            # ビュー座標系に変換
            x_3d_view = x_3d - self.view_width / 2
            y_3d_view = -(y_3d - self.view_height / 2)

            # ヒートマップモード時は表面の各点で色を変える
            if self.heatmap_mode:
                self.draw_sphere_heatmap_3d(ball_idx, ball['position'], ball['radius'], x_3d_view, y_3d_view, z_3d)
            else:
                self.draw_sphere_3d(x_3d_view, y_3d_view, z_3d, ball['radius'], (0.7, 0.8, 0.9), self.ball_rotation_angle)

        # 光源を描画（複数光源を中央配置）
        if self.show_light_source:
            light_x_2d, light_y_2d = self.light_position
            light_x_3d = light_x_2d - self.view_width / 2
            light_y_3d = -(light_y_2d - self.view_height / 2)

            # 光源の間隔をピクセルに変換
            light_z_spacing = self.light_spacing_mm * self.mm_to_pixel

            # 中心配置のオフセット計算
            total_light_width = (self.light_count - 1) * light_z_spacing
            start_light_z = total_light_width / 2  # +Z側から開始

            for i in range(self.light_count):
                # 中心がZ=0になるように配置
                light_z_3d = start_light_z - i * light_z_spacing
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

        # 座標軸を描画（画面左下の隅に配置）
        self.draw_axis_3d()

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

        # 光の強度を反映（diffuseとspecularを調整）
        intensity = self.light_intensity
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [intensity, intensity, intensity * 0.9, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [intensity, intensity, intensity * 0.95, 1])

        # 複数光源のグロー効果を描画（中央配置）
        if self.show_light_source:
            light_z_spacing = self.light_spacing_mm * self.mm_to_pixel
            total_light_width = (self.light_count - 1) * light_z_spacing
            start_light_z = total_light_width / 2  # +Z側から開始

            for i in range(self.light_count):
                light_z_3d = start_light_z - i * light_z_spacing
                self.draw_light_glow_3d(light_x_3d, light_y_3d, light_z_3d, self.light_angle, self.light_spread, self.light_intensity)

        # 最初の光源位置をOpenGLライトとして設定
        glLightfv(GL_LIGHT0, GL_POSITION, [light_x_3d, light_y_3d, 0, 1.0])

        # ヒートマップモード時は各球の強度を計算
        ball_intensities = {}
        max_intensity = 1
        if self.heatmap_mode:
            ball_intensities = self.calculate_ball_hit_intensity()
            if ball_intensities:
                max_intensity = max(ball_intensities.values()) if max(ball_intensities.values()) > 0 else 1

        # 球を描画（3D座標を使用）- 不透明なものを先に描画
        for ball_idx, ball in enumerate(self.engine.balls):
            x_3d, y_3d, z_3d = ball['position']
            # ビュー座標系に変換
            x_3d_view = x_3d - self.view_width / 2
            y_3d_view = -(y_3d - self.view_height / 2)

            # 光線が球に当たっているかを判定し、当たった光線の方向と数を記録
            # 3D空間での判定：光線の起点Z座標と球のZ座標が近いかどうかも考慮
            ball_hit = False
            hit_ray_dir = None
            hit_count = 0
            total_rays = len(self.engine.rays)
            ball_cx, ball_cy, ball_cz = ball['position'][0], ball['position'][1], ball['position'][2]
            ball_r = ball['radius']

            # 光源の間隔を取得
            light_z_spacing = self.light_spacing_mm * self.mm_to_pixel

            for ray in self.engine.rays:
                if len(ray.path) < 1:
                    continue

                # 光線の起点（光源位置）のZ座標を取得
                ray_origin_z = float(ray.path[0][2]) if len(ray.path[0]) > 2 else 0.0

                # 光線の起点Z座標と球のZ座標が近いかチェック（光源間隔の半分以内）
                z_tolerance = max(ball_r * 2, light_z_spacing / 2)
                if abs(ray_origin_z - ball_cz) > z_tolerance:
                    continue

                # 光線の経路をチェック
                ray_hits = False
                for i in range(len(ray.path) - 1):
                    p1 = ray.path[i]
                    p2 = ray.path[i + 1]
                    # 線分と球の交差判定（2D: x, y座標で判定）
                    p1x, p1y = float(p1[0]), float(p1[1])
                    p2x, p2y = float(p2[0]), float(p2[1])

                    dx = p2x - p1x
                    dy = p2y - p1y
                    line_len = math.sqrt(dx * dx + dy * dy)
                    if line_len < 0.001:
                        continue
                    # 球の中心から線分への最短距離を計算
                    t = max(0, min(1, ((ball_cx - p1x) * dx + (ball_cy - p1y) * dy) / (line_len * line_len)))
                    closest_x = p1x + t * dx
                    closest_y = p1y + t * dy
                    dist = math.sqrt((ball_cx - closest_x) ** 2 + (ball_cy - closest_y) ** 2)
                    if dist <= ball_r:
                        ray_hits = True
                        if not ball_hit:
                            ball_hit = True
                            # 最初にヒットした光線の方向を記録（正規化）
                            hit_ray_dir = (dx / line_len, dy / line_len)
                        break
                if ray_hits:
                    hit_count += 1

            if ball_hit and hit_ray_dir is not None:
                # 光線の進む方向から光源の方向を計算
                light_dir_x = hit_ray_dir[0]
                light_dir_y = hit_ray_dir[1]
                light_dir_z = 0.2
                glLightfv(GL_LIGHT0, GL_POSITION, [light_dir_x, light_dir_y, light_dir_z, 0.0])

                # ヒットした光線の割合に応じて照明強度を調整
                hit_ratio = hit_count / max(1, total_rays)
                adjusted_intensity = intensity * (0.3 + 0.7 * hit_ratio)  # 最低30%、最大100%
                glLightfv(GL_LIGHT0, GL_DIFFUSE, [adjusted_intensity, adjusted_intensity, adjusted_intensity * 0.9, 1])
            else:
                # 環境光のみ
                glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.1, 0.1, 0.1, 1])

            # ヒートマップモード時は表面の各点で色を変える、通常時は自然な色合い
            if self.heatmap_mode:
                self.draw_sphere_heatmap_3d(ball_idx, ball['position'], ball['radius'], x_3d_view, y_3d_view, z_3d)
            else:
                self.draw_sphere_3d(x_3d_view, y_3d_view, z_3d, ball['radius'], (0.85, 0.75, 0.7), self.ball_rotation_angle)

        # 座標軸を描画（画面左下の隅に配置）
        self.draw_axis_3d()

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

    def _set_ball_count(self, value: int):
        """球の個数を設定（スライダー用コールバック）"""
        self.ball_count = value
        self._rebuild_balls()

    def _set_ball_radius_mm(self, value: float):
        """球の半径を設定（mm単位、スライダー用コールバック）"""
        self.ball_radius_mm = round(value, 2)
        self._rebuild_balls()

    def _set_ball_spacing_mm(self, value: float):
        """球の間隔を設定（mm単位、スライダー用コールバック）"""
        self.ball_spacing_mm = round(value, 1)
        self._rebuild_balls()

    def _set_light_count(self, value: int):
        """光源の個数を設定（スライダー用コールバック）"""
        self.light_count = value
        self.update_simulation()

    def _set_light_spacing_mm(self, value: float):
        """光源の間隔を設定（mm単位、スライダー用コールバック）"""
        self.light_spacing_mm = round(value, 1)
        self.update_simulation()

    def _set_water_ripple(self, value: float):
        """水面ゆらぎ強度を設定（スライダー用コールバック）"""
        self.engine.water_ripple_strength = round(value, 2)
        self.update_simulation()

    def _set_ball_rotation_rpm(self, value: float):
        """球の回転速度を設定（rpm単位、スライダー用コールバック）"""
        self.ball_rotation_rpm = round(value, 1)

    def _rebuild_balls(self):
        """球を再構築（個数に応じてZ方向に配置）"""
        self.engine.balls.clear()

        ball_y = self.view_height * 0.7
        ball_x = self.view_width // 2
        # mm単位の半径をピクセルに変換
        ball_radius = self.ball_radius_mm * self.mm_to_pixel

        # Z方向の間隔（mm単位をピクセルに変換）
        z_spacing = self.ball_spacing_mm * self.mm_to_pixel

        # 中心配置：全体の幅を計算し、中央がZ=0になるようにオフセット
        total_width = (self.ball_count - 1) * z_spacing
        start_z = total_width / 2  # 最初の球のZ位置（+Z側）

        for i in range(self.ball_count):
            # 中心がZ=0になるように配置（+Zから-Zへ）
            ball_z = start_z - i * z_spacing
            self.engine.add_ball((ball_x, ball_y, ball_z), ball_radius)

        # 光線を再計算
        self.update_simulation()

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

    def calculate_ball_hit_intensity(self):
        """各球に当たった光線の数を計算してヒートマップ用の強度を返す"""
        ball_intensities = {}

        if not self.engine.balls:
            return ball_intensities

        for ball_idx, ball in enumerate(self.engine.balls):
            ball_cx, ball_cy, ball_cz = ball['position']
            ball_r = ball['radius']
            hit_count = 0

            for ray in self.engine.rays:
                if len(ray.path) < 2:
                    continue

                # 光線の経路をチェック（3D空間での判定）
                for i in range(len(ray.path) - 1):
                    p1 = ray.path[i]
                    p2 = ray.path[i + 1]
                    p1x, p1y = float(p1[0]), float(p1[1])
                    p2x, p2y = float(p2[0]), float(p2[1])
                    p1z = float(p1[2]) if len(p1) > 2 else 0.0
                    p2z = float(p2[2]) if len(p2) > 2 else 0.0

                    # 3D距離を計算
                    dx = p2x - p1x
                    dy = p2y - p1y
                    dz = p2z - p1z
                    line_len_sq = dx * dx + dy * dy + dz * dz
                    if line_len_sq < 0.001:
                        continue

                    # 球の中心から線分への最短距離を計算（3D）
                    t = max(0, min(1, ((ball_cx - p1x) * dx + (ball_cy - p1y) * dy + (ball_cz - p1z) * dz) / line_len_sq))
                    closest_x = p1x + t * dx
                    closest_y = p1y + t * dy
                    closest_z = p1z + t * dz
                    dist = math.sqrt((ball_cx - closest_x) ** 2 + (ball_cy - closest_y) ** 2 + (ball_cz - closest_z) ** 2)
                    if dist <= ball_r:
                        hit_count += 1
                        break

            ball_intensities[ball_idx] = hit_count

        return ball_intensities

    def get_heatmap_color(self, intensity: float, max_intensity: float) -> Tuple[float, float, float]:
        """強度からヒートマップ色を計算（青→シアン→緑→黄→赤）OpenGL用に0-1の範囲で返す"""
        if max_intensity == 0 or intensity == 0:
            return (0.0, 0.0, 1.0)  # 青（光が当たっていない）

        # 正規化 (0.0 ~ 1.0)
        normalized = min(1.0, intensity / max_intensity)

        if normalized < 0.25:
            # 青 → シアン
            ratio = normalized / 0.25
            return (0.0, ratio, 1.0)
        elif normalized < 0.5:
            # シアン → 緑
            ratio = (normalized - 0.25) / 0.25
            return (0.0, 1.0, 1.0 - ratio)
        elif normalized < 0.75:
            # 緑 → 黄
            ratio = (normalized - 0.5) / 0.25
            return (ratio, 1.0, 0.0)
        else:
            # 黄 → 赤
            ratio = (normalized - 0.75) / 0.25
            return (1.0, 1.0 - ratio, 0.0)

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

        # 複数光源を描画（独立した個数・間隔で中央配置）
        light_z_spacing = self.light_spacing_mm * self.mm_to_pixel * zoom
        halogen_width = 60 * zoom  # X方向の幅（横に長い）
        halogen_depth = 20 * zoom  # Y方向の奥行き

        # 中心配置のオフセット計算
        total_light_width = (self.light_count - 1) * light_z_spacing
        start_light_offset = total_light_width / 2  # +Z側から開始

        for i in range(self.light_count):
            # 中心がZ=0になるように配置（上面図では-Zが右側）
            light_z_offset = start_light_offset - i * light_z_spacing
            light_offset_x = -light_z_offset  # -Zが右側なので符号反転
            halogen_rect = pygame.Rect(
                int(top_light_x - halogen_width // 2 + light_offset_x),
                int(top_light_y - halogen_depth // 2),
                int(halogen_width),
                int(halogen_depth)
            )
            pygame.draw.rect(view_surface, (255, 255, 200), halogen_rect)
            pygame.draw.rect(view_surface, (200, 200, 0), halogen_rect, max(1, int(2 * zoom)))

        # 球を描画（横図のY座標を上面図のY座標に変換、Z座標でX位置をオフセット）
        for ball in self.engine.balls:
            pos_3d = ball['position']
            radius = ball['radius']
            # Z座標を上面図のX方向にオフセット（-Z方向が右側）
            top_x = int((self.view_width // 2) * zoom - pos_3d[2] * zoom)
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

    def draw_sidebar(self, surface: pygame.Surface = None, offset_x: int = 0, offset_y: int = 0):
        """サイドバーUIを描画（2Dモード用・3Dオーバーレイ用共通）"""
        if surface is None:
            surface = self.screen

        # UIパネルの背景
        panel_rect = pygame.Rect(offset_x, offset_y, self.ui_panel_width, self.height)
        pygame.draw.rect(surface, (240, 240, 245), panel_rect)
        pygame.draw.line(surface, (180, 180, 180),
                        (offset_x + self.ui_panel_width, offset_y),
                        (offset_x + self.ui_panel_width, offset_y + self.height), 2)

        y = offset_y + 20

        # タイトル
        title = self.title_font.render("パラメータ", True, self.COLOR_TEXT)
        surface.blit(title, (offset_x + 15, y))

        # タブグループを描画
        self.tab_group.x = offset_x + 10
        self.tab_group.y = offset_y + 50
        self.tab_group.draw(surface, self.small_font)

        # 現在のタブに属するスライダーのみ描画
        active_tab = self.tab_group.active_tab
        slider_count = 0
        slider_start_y = 105  # タブの下からの開始位置
        slider_spacing = 60   # スライダー間の間隔
        for slider in self.sliders:
            if slider.tab_index == active_tab:
                # スライダーの位置を動的に調整
                slider.rect.x = offset_x + 20
                slider.rect.y = offset_y + slider_start_y + slider_count * slider_spacing
                slider.draw(surface, self.small_font)
                slider_count += 1

        # スライダーの下に情報と操作説明を配置
        y = offset_y + slider_start_y + slider_count * slider_spacing + 25

        # 情報表示
        info_title = self.font.render("情報", True, self.COLOR_TEXT)
        surface.blit(info_title, (offset_x + 15, y))
        y += 22

        # 光源位置
        pos_text = self.small_font.render(f"光源: X={int(self.light_position[0])}, Y={int(self.light_position[1])}", True, (100, 100, 100))
        surface.blit(pos_text, (offset_x + 20, y))
        y += 18

        # 光線数
        text = self.small_font.render(f"光線数: {len(self.engine.rays)} 本", True, (100, 100, 100))
        surface.blit(text, (offset_x + 20, y))
        y += 25

        # 操作説明
        help_title = self.font.render("操作方法", True, self.COLOR_TEXT)
        surface.blit(help_title, (offset_x + 15, y))
        y += 22

        help_texts = [
            "左クリック: 光源移動(2D)",
            "ホイール: ズーム",
            "中クリック: 平行移動",
            "右クリック: 回転(3D)",
            "左右キー: 角度",
            "Q/E: 広がり",
            "↑↓: 水面",
            "1/2/3/4キー: ビュー切替",
            "H: ヒートマップ(3D)",
            "L: 光源表示(3D)",
            "R: リセット",
        ]

        for text in help_texts:
            help_surf = self.small_font.render(text, True, (100, 100, 100))
            surface.blit(help_surf, (offset_x + 20, y))
            y += 16

    def draw_ui(self):
        """UI要素を描画（2Dモード用）"""
        self.draw_sidebar()

    def handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            # タブのイベント処理を優先
            if self.tab_group.handle_event(event):
                continue

            # スライダーのイベント処理（現在のタブに属するスライダーのみ）
            slider_handled = False
            active_tab = self.tab_group.active_tab
            for slider in self.sliders:
                if slider.tab_index == active_tab and slider.handle_event(event):
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
                    self.slider_map['water_level'].value = int(self.engine.water_level)
                    self.update_simulation()
                elif event.key == pygame.K_DOWN:
                    # 水面を下げる
                    self.engine.water_level = min(self.view_height - 50, self.engine.water_level + 1)
                    self.slider_map['water_level'].value = int(self.engine.water_level)
                    self.update_simulation()
                elif event.key == pygame.K_LEFT:
                    # 光の角度を左に（0.1度ずつ）
                    self.light_angle -= np.pi / 1800  # 0.1度ずつ
                    self.slider_map['light_angle'].value = round(np.degrees(self.light_angle), 1)
                    self.update_simulation()
                elif event.key == pygame.K_RIGHT:
                    # 光の角度を右に（0.1度ずつ）
                    self.light_angle += np.pi / 1800  # 0.1度ずつ
                    self.slider_map['light_angle'].value = round(np.degrees(self.light_angle), 1)
                    self.update_simulation()
                elif event.key == pygame.K_q:
                    # 光の広がりを狭く
                    self.light_spread = max(0, self.light_spread - np.pi / 180)  # 1度ずつ
                    self.slider_map['light_spread'].value = int(np.degrees(self.light_spread))
                    self.update_simulation()
                elif event.key == pygame.K_e:
                    # 光の広がりを広く
                    self.light_spread = min(np.pi, self.light_spread + np.pi / 180)  # 1度ずつ
                    self.slider_map['light_spread'].value = int(np.degrees(self.light_spread))
                    self.update_simulation()
                elif event.key == pygame.K_n:
                    # 屈折率を下げる（0.01刻み）
                    new_index = max(1.00, self.engine.water_refractive_index - 0.01)
                    self.engine.water_refractive_index = round(new_index, 2)
                    self.slider_map['refractive_index'].value = self.engine.water_refractive_index
                    self.update_simulation()
                elif event.key == pygame.K_m:
                    # 屈折率を上げる（0.01刻み）
                    new_index = min(2.00, self.engine.water_refractive_index + 0.01)
                    self.engine.water_refractive_index = round(new_index, 2)
                    self.slider_map['refractive_index'].value = self.engine.water_refractive_index
                    self.update_simulation()
                elif event.key == pygame.K_p:
                    # プロファイルモード切替
                    self.profile_mode = not self.profile_mode
                elif event.key == pygame.K_x:
                    # プロファイルX軸スキャン（水平）
                    self.profile_scan_axis = 'X'
                elif event.key == pygame.K_y:
                    # プロファイルY軸スキャン（垂直）
                    self.profile_scan_axis = 'Y'
                elif event.key == pygame.K_1 or event.key == pygame.K_2:
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

                elif event.key == pygame.K_h:
                    # ヒートマップモードの切り替え（3Dモード時のみ有効）
                    if self.view_mode_3d or self.view_mode_natural_3d:
                        self.heatmap_mode = not self.heatmap_mode

                elif event.key == pygame.K_l:
                    # 光源表示の切り替え（3Dモード時のみ有効）
                    if self.view_mode_3d or self.view_mode_natural_3d:
                        self.show_light_source = not self.show_light_source

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左クリック
                    # 3Dモード時のUI操作
                    if self.view_mode_3d or self.view_mode_natural_3d:
                        mouse_pos = pygame.mouse.get_pos()
                        
                        # 視点切り替えボタンの判定
                        btn_width = 50
                        btn_height = 25
                        spacing_x = 5
                        spacing_y = 5
                        start_x = self.width - (btn_width * 2 + spacing_x) - 20
                        start_y = 60
                        
                        clicked_btn = False
                        for i, btn in enumerate(self.orientation_buttons):
                            row = i // 2
                            col = i % 2
                            x = start_x + col * (btn_width + spacing_x)
                            y = start_y + row * (btn_height + spacing_y)
                            rect = pygame.Rect(x, y, btn_width, btn_height)
                            
                            if rect.collidepoint(mouse_pos):
                                # 視点切り替え実行
                                target_angle = btn['angle']
                                self.camera_rotation = list(target_angle) # コピー
                                clicked_btn = True
                                self.camera_rotation = list(target_angle) # コピー
                                clicked_btn = True
                                break
                        
                        # 回転ボタン判定
                        if not clicked_btn:
                            rotation_start_row = 3
                            for i, btn in enumerate(self.rotation_buttons):
                                row = rotation_start_row
                                col = i % 2
                                x = start_x + col * (btn_width + spacing_x)
                                y = start_y + row * (btn_height + spacing_y)
                                rect = pygame.Rect(x, y, btn_width, btn_height)
                                
                                if rect.collidepoint(mouse_pos):
                                    self.camera_rotation[1] += btn['delta']
                                    clicked_btn = True
                                    break
                        
                                    clicked_btn = True
                                    break
                                    
                                    break
                                    
                        # Profile/Axisボタン判定
                        if not clicked_btn:
                            profile_row = 4
                            p_width = btn_width
                            
                            # Profile Button (Left)
                            bx1 = 10
                            by = 10 + profile_row * (btn_height + spacing_y)
                            rect1 = pygame.Rect(start_x + bx1 - 10, start_y + by, p_width, btn_height)
                            
                            if rect1.collidepoint(mouse_pos):
                                self.profile_mode = not self.profile_mode
                                clicked_btn = True
                                
                            # Axis Button (Right)
                            if not clicked_btn:
                                bx2 = 10 + btn_width + spacing_x
                                rect2 = pygame.Rect(start_x + bx2 - 10, start_y + by, p_width, btn_height)
                                
                                if rect2.collidepoint(mouse_pos):
                                    # Toggle Axis
                                    if self.profile_scan_axis == 'Y':
                                        self.profile_scan_axis = 'X'
                                        self.profile_pos = self.height // 2  # 画面中央にリセット
                                    else:
                                        self.profile_scan_axis = 'Y'
                                        self.profile_pos = self.width // 2  # 画面中央にリセット
                                    clicked_btn = True
                        
                        # プロファイルラインのドラッグ判定
                        if not clicked_btn and self.profile_mode:
                            sensitivity = 15  # クリック判定の許容範囲（ピクセル）
                            if self.profile_scan_axis == 'Y':
                                # 縦ライン (X座標で判定)
                                if abs(mouse_pos[0] - self.profile_pos) < sensitivity:
                                    self.dragging_profile_line = True
                                    clicked_btn = True
                            else:
                                # 横ライン (Y座標で判定)
                                if abs(mouse_pos[1] - self.profile_pos) < sensitivity:
                                    self.dragging_profile_line = True
                                    clicked_btn = True

                    # 2Dモード時のみ光源ドラッグ
                    if not self.view_mode_3d and not self.view_mode_natural_3d:
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
                    self.dragging_profile_line = False  # プロファイルラインのドラッグ終了
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
                elif self.dragging_profile_line:
                    # プロファイルラインのドラッグ中
                    mouse_pos = pygame.mouse.get_pos()
                    if self.profile_scan_axis == 'Y':
                        # 縦ライン (X座標を更新)
                        self.profile_pos = max(0, min(self.width - 1, mouse_pos[0]))
                    else:
                        # 横ライン (Y座標を更新)
                        self.profile_pos = max(0, min(self.height - 1, mouse_pos[1]))
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
        # 複数光源からの光線をすべてクリア
        self.engine.rays.clear()

        # 光源の間隔をピクセルに変換
        light_z_spacing = self.light_spacing_mm * self.mm_to_pixel

        # 中心配置：全体の幅を計算し、中央がZ=0になるようにオフセット
        total_light_width = (self.light_count - 1) * light_z_spacing
        start_light_z = total_light_width / 2  # 最初の光源のZ位置（+Z側）

        # 各光源から光線を生成
        for i in range(self.light_count):
            # 中心がZ=0になるように配置（+Zから-Zへ）
            light_z = start_light_z - i * light_z_spacing
            light_pos_3d = (self.light_position[0], self.light_position[1], light_z)

            # 3D光源を生成（光線数は光源数に応じて調整）
            rays_per_source_radial = max(5, 20 // self.light_count)
            rays_per_source_circular = max(4, 12 // self.light_count)

            self.engine.create_light_source_3d(
                light_pos_3d,
                num_rays_radial=rays_per_source_radial,
                num_rays_circular=rays_per_source_circular,
                spread_angle=self.light_spread,
                center_angle=self.light_angle
            )

        # 球の光強度を計算
        self.calculate_ball_intensity()

        # ヒートマップキャッシュを更新
        self.calculate_heatmap_cache()

    def draw_sidebar_overlay_3d(self):
        """3Dビューの上にサイドバーをOpenGLで直接描画"""
        # OpenGLの状態を保存
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # 2D正射影に切り替え
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # サイドバー背景
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.94, 0.94, 0.96, 0.95)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.ui_panel_width, 0)
        glVertex2f(self.ui_panel_width, self.height)
        glVertex2f(0, self.height)
        glEnd()

        # 境界線
        glColor4f(0.7, 0.7, 0.7, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(self.ui_panel_width, 0)
        glVertex2f(self.ui_panel_width, self.height)
        glEnd()

        glDisable(GL_BLEND)

        # 行列を元に戻す
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # テキストとスライダーをPygameで描画してOpenGLテクスチャとして転送
        # サイドバー用サーフェスを作成
        sidebar_surf = pygame.Surface((self.ui_panel_width, self.height), pygame.SRCALPHA)
        sidebar_surf.fill((240, 240, 245, 245))
        self.draw_sidebar(sidebar_surf, 0, 0)

        # PygameサーフェスをOpenGLで描画
        self._blit_pygame_surface_to_opengl(sidebar_surf, 0, 0)
        
        # 視点切り替えボタンを描画
        self.draw_orientation_buttons_overlay()

    def draw_profile_overlay(self):
        """輝度プロファイルを描画"""
        # マウス位置に基づいてスキャンラインを決定
        mouse_pos = pygame.mouse.get_pos()
        
        # マウス座標は左上原点だが、OpenGLは左下原点の場合がある
        # glReadPixelsは左下原点。Pygameは左上。
        # ここではPygame座標系(左上原点)で統一して考える。
        
        # スキャン位置の初期化（まだ設定されていない場合のみ）
        if self.profile_pos == 0:
            if self.profile_scan_axis == 'Y':
                self.profile_pos = self.width // 2
            else:
                self.profile_pos = self.height // 2
        
        # 画面キャプチャ
        # glReadPixels(x, y, width, height, format, type)
        # 現在のビューポート全体を取得
        # 注意: glReadPixelsは左下原点。
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # numpy配列に変換
        # shape: (height, width, 3)
        img_array = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        
        # 上下反転（OpenGL -> Pygame/Image座標系）
        img_array = np.flipud(img_array)
        
        # プロファイルデータの抽出
        if self.profile_scan_axis == 'Y':
            # Xを固定してY方向にスキャン（縦ライン）
            # self.profile_pos は X座標
            x = int(max(0, min(self.width - 1, self.profile_pos)))
            line_data = img_array[:, x, :] # (Height, 3)
            # 輝度計算 (簡易: 平均)
            intensity = np.mean(line_data, axis=1) # (Height,)
            axis_len = self.height
        else:
            # Yを固定してX方向にスキャン（横ライン）
            y = int(max(0, min(self.height - 1, self.profile_pos)))
            line_data = img_array[y, :, :] # (Width, 3)
            intensity = np.mean(line_data, axis=1) # (Width,)
            axis_len = self.width
            
        # 背景色を除去
        # スキャンライン上の最頻出色を背景色とみなす（動的判定）
        if len(line_data) > 0:
            # 高速化のため、line_dataそのものからユニークカウント
            colors, counts = np.unique(line_data.reshape(-1, 3), axis=0, return_counts=True)
            if len(counts) > 0:
                bg_color = colors[np.argmax(counts)]
            else:
                bg_color = np.array([25, 25, 38])
                
            # 背景色の許容範囲 (誤差20)
            diff = np.abs(line_data.astype(np.int16) - bg_color)
            is_bg = np.all(diff < 20, axis=1)
            intensity[is_bg] = 0
            
        # グラフ描画用Surface
        graph_height = 200 # 高さを増やす
        plot_x_start = 100 # 左マージンをさらに拡大 (数値が見えない問題対策)
        graph_width = self.width - (plot_x_start + 20) # 右マージン20
        plot_y_end = graph_height - 30 # 下マージン
        plot_top = 20 # 上マージン
        plot_h = plot_y_end - plot_top # 描画高さ
        
        s = pygame.Surface((self.width, graph_height), pygame.SRCALPHA)
        s.fill((0, 0, 0, 200)) # 背景を少し濃くする
        
        # グリッドとラベルの描画
        # Y軸 (0 - 255)
        # 一番上の線を255にする
        y_ticks = [0, 50, 100, 150, 200, 255]
        for val in y_ticks:
            py = plot_y_end - (val / 255.0) * plot_h
            # グリッド線
            pygame.draw.line(s, (80, 80, 80), (plot_x_start, py), (plot_x_start + graph_width, py), 1)
            # ラベル (文字色を白く、位置調整)
            label = self.small_font.render(str(val), True, (255, 255, 255))
            label_rect = label.get_rect(midright=(plot_x_start - 10, py))
            s.blit(label, label_rect)
            
        # X軸 (ピクセル位置)
        x_steps = 10
        for i in range(x_steps + 1):
            val = int((i / x_steps) * axis_len)
            px = plot_x_start + (i / x_steps) * graph_width
            # グリッド線
            pygame.draw.line(s, (80, 80, 80), (px, plot_y_end), (px, plot_y_end - plot_h), 1)
            # ラベル (間引いて表示)
            if i % 2 == 0:
                label = self.small_font.render(str(val), True, (200, 200, 200))
                label_rect = label.get_rect(midtop=(px, plot_y_end + 5))
                s.blit(label, label_rect)

        # 軸線
        pygame.draw.line(s, (200, 200, 200), (plot_x_start, plot_y_end), (plot_x_start + graph_width, plot_y_end), 2) # X軸
        pygame.draw.line(s, (200, 200, 200), (plot_x_start, plot_y_end), (plot_x_start, plot_y_end - plot_h), 2) # Y軸
        
        points = []
        for i in range(len(intensity)):
            # x座標: グラフ幅に合わせてスケーリング
            px = plot_x_start + (i / axis_len) * graph_width
            # y座標: 輝度に合わせて高さ計算（輝度が高いほど上）
            py = plot_y_end - (intensity[i] / 255.0) * plot_h
            points.append((px, py))
            
        if len(points) > 1:
            pygame.draw.lines(s, (255, 255, 0), False, points, 2)
            
        # ガイド線（現在のスキャン位置）
        # スキャンしている場所を示す線を描画するSurface
        guide_s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        if self.profile_scan_axis == 'Y':
            x = int(max(0, min(self.width - 1, self.profile_pos)))
            pygame.draw.line(guide_s, (255, 0, 0), (x, 0), (x, self.height), 1)
        else:
            y = int(max(0, min(self.height - 1, self.profile_pos)))
            pygame.draw.line(guide_s, (255, 0, 0), (0, y), (self.width, y), 1)
            
        # OpenGL描画
        # まずガイド線
        self._blit_pygame_surface_to_opengl(guide_s, 0, 0)
        # 次にグラフ（画面下部）
        self._blit_pygame_surface_to_opengl(s, 0, self.height - graph_height)

    def _blit_pygame_surface_to_opengl(self, surface: pygame.Surface, x: int, y: int):
        """PygameサーフェスをOpenGLで描画"""
        # サーフェスのピクセルデータを取得
        width = surface.get_width()
        height = surface.get_height()

        # RGBAデータを取得
        data = pygame.image.tostring(surface, 'RGBA', True)

        # OpenGLの状態を保存
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # 2D正射影
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # ブレンディングを有効化
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # ラスター位置を設定して描画
        glRasterPos2i(x, self.height - y - height)
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glDisable(GL_BLEND)

        # 行列を元に戻す
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def run(self):
        """メインループ"""
        self.update_simulation()

        # サイドバー用のオーバーレイサーフェスを作成
        self.sidebar_surface = pygame.Surface((self.ui_panel_width, self.height), pygame.SRCALPHA)

        while self.running:
            self.handle_events()

            # 水面ゆらぎのアニメーション更新
            if self.engine.water_ripple_strength > 0:
                self.engine.water_ripple_time += 0.05

            # 球の回転アニメーション更新（rpm to radians per frame at 60fps）
            if self.ball_rotation_rpm > 0:
                # 1rpm = 2π rad/min = 2π/60 rad/sec = 2π/3600 rad/frame (at 60fps)
                radians_per_frame = (2 * math.pi * self.ball_rotation_rpm) / (60 * 60)
                self.ball_rotation_angle += radians_per_frame
                # 2πを超えたらリセット
                if self.ball_rotation_angle > 2 * math.pi:
                    self.ball_rotation_angle -= 2 * math.pi

            # 描画
            if self.view_mode_3d:
                # 3Dモード（光線あり）
                self.draw_3d_view()
                if self.profile_mode:
                    self.draw_profile_overlay()
                self.draw_sidebar_overlay_3d()
                pygame.display.flip()
            elif self.view_mode_natural_3d:
                # 自然光3Dモード（光線なし）
                self.draw_3d_view_natural()
                if self.profile_mode:
                    self.draw_profile_overlay()
                self.draw_sidebar_overlay_3d()
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
