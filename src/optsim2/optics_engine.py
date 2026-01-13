"""
光学シミュレーションエンジン
光の屈折、反射、レイトレーシングを実装
"""
import numpy as np
from typing import List, Tuple, Optional
import math


class Ray:
    """光線クラス"""

    def __init__(self, origin: np.ndarray, direction: np.ndarray, intensity: float = 1.0):
        """
        Args:
            origin: 光線の始点 (x, y)
            direction: 光線の方向ベクトル (正規化される)
            intensity: 光線の強度 (0.0 - 1.0)
        """
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction = self.direction / np.linalg.norm(self.direction)  # 正規化
        self.intensity = intensity
        self.path = [self.origin.copy()]  # 光線の経路

    def propagate(self, distance: float):
        """光線を伝播させる"""
        new_point = self.origin + self.direction * distance
        self.origin = new_point
        self.path.append(new_point.copy())
        return new_point


class OpticsEngine:
    """光学シミュレーションエンジン"""

    # 屈折率
    N_AIR = 1.0  # 空気
    N_WATER = 1.33  # 水

    def __init__(self, width: int = 800, height: int = 600):
        """
        Args:
            width: シミュレーション空間の幅
            height: シミュレーション空間の高さ
        """
        self.width = width
        self.height = height
        self.water_level = height * 0.6  # 水面の位置（画面の60%の位置）
        self.rays = []
        self.balls = []

    def add_ball(self, position: Tuple[float, float], radius: float):
        """球を追加"""
        self.balls.append({
            'position': np.array(position, dtype=float),
            'radius': radius
        })

    def set_water_level(self, level: float):
        """水面の位置を設定"""
        self.water_level = level

    def refract(self, incident: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
        """
        スネルの法則を使用して屈折ベクトルを計算

        Args:
            incident: 入射ベクトル（正規化済み）
            normal: 界面の法線ベクトル（正規化済み）
            n1: 入射側の屈折率
            n2: 屈折側の屈折率

        Returns:
            屈折ベクトル、または全反射の場合はNone
        """
        # 入射角のコサイン
        cos_i = -np.dot(incident, normal)

        # 法線を正しい方向に向ける
        if cos_i < 0:
            cos_i = -cos_i
            normal = -normal
            n1, n2 = n2, n1

        # スネルの法則: n1 * sin(θ1) = n2 * sin(θ2)
        n = n1 / n2
        sin_t2 = n * n * (1.0 - cos_i * cos_i)

        # 全反射の判定
        if sin_t2 > 1.0:
            return None

        cos_t = math.sqrt(1.0 - sin_t2)
        refracted = n * incident + (n * cos_i - cos_t) * normal

        return refracted / np.linalg.norm(refracted)

    def reflect(self, incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        反射ベクトルを計算

        Args:
            incident: 入射ベクトル（正規化済み）
            normal: 界面の法線ベクトル（正規化済み）

        Returns:
            反射ベクトル
        """
        return incident - 2 * np.dot(incident, normal) * normal

    def intersect_sphere(self, ray: Ray, ball: dict) -> Optional[Tuple[float, np.ndarray]]:
        """
        光線と球の交点を計算

        Args:
            ray: 光線
            ball: 球の情報

        Returns:
            (距離, 交点の法線ベクトル) または None
        """
        oc = ray.origin - ball['position']
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - ball['radius'] ** 2
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:  # 数値誤差を避けるための閾値
            t = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if t < 0.001:
                return None

        intersection = ray.origin + t * ray.direction
        normal = (intersection - ball['position']) / ball['radius']

        return (t, normal)

    def intersect_water_surface(self, ray: Ray) -> Optional[float]:
        """
        光線と水面の交点を計算

        Args:
            ray: 光線

        Returns:
            交点までの距離、または None
        """
        # 水面はy = water_levelの水平面
        if abs(ray.direction[1]) < 0.001:  # 光線がほぼ水平
            return None

        t = (self.water_level - ray.origin[1]) / ray.direction[1]

        if t < 0.001:
            return None

        return t

    def trace_ray(self, ray: Ray, max_bounces: int = 5) -> Ray:
        """
        光線を追跡し、反射・屈折を計算

        Args:
            ray: 追跡する光線
            max_bounces: 最大反射・屈折回数

        Returns:
            経路が記録された光線
        """
        current_medium = self.N_AIR  # 初期は空気中

        for bounce in range(max_bounces):
            if ray.intensity < 0.01:  # 強度が弱くなったら終了
                break

            # 最も近い交点を探す
            min_distance = float('inf')
            hit_type = None
            hit_data = None

            # 水面との交差判定
            water_dist = self.intersect_water_surface(ray)
            if water_dist is not None and water_dist < min_distance:
                min_distance = water_dist
                hit_type = 'water'

            # 球との交差判定
            for ball in self.balls:
                result = self.intersect_sphere(ray, ball)
                if result is not None:
                    dist, normal = result
                    if dist < min_distance:
                        min_distance = dist
                        hit_type = 'sphere'
                        hit_data = normal

            # 交点がない場合、光線を十分遠くまで伝播
            if hit_type is None:
                ray.propagate(1000)
                break

            # 交点まで伝播
            ray.propagate(min_distance)

            # 水面での屈折
            if hit_type == 'water':
                normal = np.array([0, -1], dtype=float)  # 水面の法線（上向き）

                # 現在の媒質を判定
                if ray.origin[1] < self.water_level:
                    # 空気から水へ
                    n1, n2 = self.N_AIR, self.N_WATER
                    new_medium = self.N_WATER
                else:
                    # 水から空気へ
                    n1, n2 = self.N_WATER, self.N_AIR
                    new_medium = self.N_AIR

                refracted = self.refract(ray.direction, normal, n1, n2)

                if refracted is not None:
                    ray.direction = refracted
                    current_medium = new_medium
                else:
                    # 全反射
                    ray.direction = self.reflect(ray.direction, normal)

                ray.intensity *= 0.95  # わずかに減衰

            # 球での反射
            elif hit_type == 'sphere':
                ray.direction = self.reflect(ray.direction, hit_data)
                ray.intensity *= 0.8  # 反射で減衰

        return ray

    def create_light_source(self, position: Tuple[float, float], num_rays: int = 20, spread_angle: float = np.pi/3, center_angle: float = 0.0) -> List[Ray]:
        """
        点光源から放射される光線群を生成

        Args:
            position: 光源の位置
            num_rays: 光線の数
            spread_angle: 光の広がり角度（ラジアン）
            center_angle: 光の中心方向の角度（ラジアン、0は下向き、正の値で時計回り）

        Returns:
            光線のリスト
        """
        rays = []
        start_angle = center_angle - spread_angle / 2

        for i in range(num_rays):
            angle = start_angle + (spread_angle * i / (num_rays - 1))
            direction = np.array([math.sin(angle), math.cos(angle)])
            ray = Ray(position, direction)
            rays.append(self.trace_ray(ray))

        self.rays = rays
        return rays
