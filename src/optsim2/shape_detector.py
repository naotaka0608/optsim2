"""
手書き図形認識モジュール
OpenCVを使用して手書きの図形（カメラ、ハロゲン、水面、球など）を検出
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple


class ShapeDetector:
    """手書き図形を検出するクラス"""

    def __init__(self, image_path: str):
        """
        Args:
            image_path: 手書き画像のパス
        """
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.shapes = []

    def preprocess(self):
        """画像の前処理"""
        # ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        return binary

    def detect_circles(self, binary_image) -> List[Dict]:
        """円（球）を検出"""
        circles = []

        # ハフ円変換
        detected = cv2.HoughCircles(
            self.gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )

        if detected is not None:
            detected = np.uint16(np.around(detected))
            for circle in detected[0, :]:
                circles.append({
                    'type': 'circle',
                    'center': (int(circle[0]), int(circle[1])),
                    'radius': int(circle[2])
                })

        return circles

    def detect_rectangles(self, binary_image) -> List[Dict]:
        """矩形（カメラ、ハロゲンなど）を検出"""
        rectangles = []

        # 輪郭検出
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # 輪郭を近似
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 4頂点の多角形（矩形）を検出
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # 小さすぎる矩形は無視
                if w > 20 and h > 20:
                    rectangles.append({
                        'type': 'rectangle',
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'center': (x + w//2, y + h//2)
                    })

        return rectangles

    def detect_lines(self, binary_image) -> List[Dict]:
        """直線（光線、水面など）を検出"""
        lines = []

        # ハフ変換で直線検出
        detected = cv2.HoughLinesP(
            binary_image, rho=1, theta=np.pi/180,
            threshold=50, minLineLength=50, maxLineGap=10
        )

        if detected is not None:
            for line in detected:
                x1, y1, x2, y2 = line[0]
                lines.append({
                    'type': 'line',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'angle': np.arctan2(y2 - y1, x2 - x1)
                })

        return lines

    def detect_all_shapes(self) -> Dict:
        """全ての図形を検出"""
        binary = self.preprocess()

        circles = self.detect_circles(binary)
        rectangles = self.detect_rectangles(binary)
        lines = self.detect_lines(binary)

        return {
            'circles': circles,
            'rectangles': rectangles,
            'lines': lines
        }

    def classify_shapes(self, shapes: Dict) -> Dict:
        """検出した図形を分類（カメラ、ハロゲン、球、水面など）"""
        classified = {
            'cameras': [],
            'halogen': [],
            'balls': [],
            'water_surface': [],
            'light_rays': []
        }

        # 円は球として分類
        for circle in shapes['circles']:
            classified['balls'].append({
                'position': circle['center'],
                'radius': circle['radius']
            })

        # 矩形を上部にあるものはカメラ/ハロゲン、それ以外は他の用途
        image_height = self.image.shape[0]
        for rect in shapes['rectangles']:
            if rect['y'] < image_height * 0.3:
                # 上部にある矩形
                if rect['width'] > rect['height']:
                    classified['cameras'].append(rect)
                else:
                    classified['halogen'].append(rect)

        # 水平に近い直線は水面として分類
        for line in shapes['lines']:
            angle = abs(line['angle'])
            if angle < np.pi/6 or angle > 5*np.pi/6:  # ほぼ水平
                classified['water_surface'].append(line)
            else:
                # その他の斜めの線は光線
                classified['light_rays'].append(line)

        return classified
