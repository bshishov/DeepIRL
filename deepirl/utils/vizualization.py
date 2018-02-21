import cv2
import numpy as np
import enum


class ColorMap(enum.Enum):
    NONE = None
    HOT = cv2.COLORMAP_HOT
    JET = cv2.COLORMAP_JET


class TextAlignment(enum.Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class Rect(object):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def left_top_corner(self):
        return self.x, self.y

    @property
    def right_bottom_corner(self):
        return self.x + self.width, self.y + self.height

    @property
    def size(self):
        return self.width, self.height


class BaseDrawer(object):
    pass


class Drawer(BaseDrawer):
    def __init__(self, bounds: Rect):
        self.bounds = bounds

    def draw(self, region: np.ndarray):
        raise NotImplementedError


class FreeDrawer(BaseDrawer):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def draw(self, canvas: np.ndarray):
        raise NotImplementedError


class CompositeDrawer(Drawer):
    def __init__(self, bounds: Rect):
        super(CompositeDrawer, self).__init__(bounds)
        self._drawers = []

    def append(self, drawer: BaseDrawer):
        self._drawers.append(drawer)

    def draw(self, region: np.ndarray):
        for drawer in self._drawers:
            if isinstance(drawer, Drawer):
                region1 = region[drawer.bounds.top:drawer.bounds.bottom, drawer.bounds.left:drawer.bounds.right]
                drawer.draw(region1)
            if isinstance(drawer, FreeDrawer):
                drawer.draw(region)


class ImageDrawer(Drawer):
    def __init__(self, bounds: Rect, resize=True, color_map: ColorMap = ColorMap.NONE):
        super(self.__class__, self).__init__(bounds)
        self._img = None
        self.interpolation = cv2.INTER_NEAREST
        self.resize = resize
        self.overlay = False
        self.border = (200, 200, 200)
        self.color_map = color_map

    def draw(self, region: np.ndarray):
        if self._img is not None:
            if self.overlay:
                mask = np.max(self._img, axis=2) > 50
                region[mask] = self._img[mask]
            else:
                region[:] = self._img
        if self.border is not None:
            cv2.rectangle(region, (0, 0), (self.bounds.width - 1, self.bounds.height - 1), self.border)

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value: np.ndarray):
        self.set_value(value)

    def set_value(self, value: np.ndarray):
        if value is None:
            self._img = None
            return

        if len(value.shape) == 2:
            channels = 1
        else:
            channels = value.shape[2]

        if channels == 1:
            self._img = np.zeros(value.shape, dtype=np.uint8)
            cv2.normalize(value.astype(np.float32), self._img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if self.color_map is ColorMap.NONE:
                self._img = cv2.cvtColor(self._img, cv2.COLOR_GRAY2BGR)
            else:
                self._img = cv2.applyColorMap(self._img, self.color_map.value)
        else:
            self._img = value

        if self.resize:
            self._img = cv2.resize(self._img, self.bounds.size, interpolation=self.interpolation)


class PolicyDrawer(Drawer):
    def __init__(self, bounds: Rect, num_actions: int):
        super(self.__class__, self).__init__(bounds)
        self.num_actions = num_actions
        self.policy = np.random.random(num_actions)
        self.annotations = [str(x) for x in range(num_actions)]
        self.color = (128, 0, 0)
        self._block_width = int(self.bounds.width / num_actions)
        self.spacing = 2
        self.show_text = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, canvas: np.ndarray):
        for i in range(self.num_actions):
            start_x = self.x + self._block_width * i - self.spacing
            start_y = self.y + int((1 - self.policy[i]) * self.height)
            end_x = self.x + self._block_width * (i + 1)
            end_y = self.y + self.height
            cv2.rectangle(canvas, (start_x, start_y), (end_x, end_y), self.color, thickness=cv2.FILLED)

            if self.show_text:
                cv2.putText(canvas, self.annotations[i], (start_x + 10, self.y + 5), self.font, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, '{0:.2f}'.format(self.policy[i]), (start_x + 10, self.y + 40), self.font, 1.5,
                            (255, 255, 255), 2, cv2.LINE_AA)


class ShiftingBarchart(Drawer):
    def __init__(self, bounds: Rect, val_range: tuple):
        super(self.__class__, self).__init__(bounds)
        self.chart = np.zeros(shape=(self.size[1], self.size[0], 3), dtype=np.uint8)
        self.bar_width = 5
        self.positive_color = (128, 255, 128)
        self.negative_color = (128, 128, 255)
        self.val_range = val_range
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.last_value = 0.0
        self.title = 'Value'

    def val_to_y(self, val):
        v = (val - self.val_range[0]) / (self.val_range[1] - self.val_range[0])
        v = min(max(v, 0.0), 1.0)
        return int((1 - v) * self.size[1])

    def push(self, val: float):
        w, h = self.size

        # Shift
        self.chart[0:h, 0:w - self.bar_width] = self.chart[0:h, self.bar_width:w]

        val_y = self.val_to_y(val)
        zero_y = self.val_to_y(0)

        self.chart[0:h, w - self.bar_width:w] = (0, 0, 0)

        if val >= 0:
            self.chart[val_y:zero_y, w - self.bar_width:w] = self.positive_color
        else:
            self.chart[zero_y:val_y, w - self.bar_width:w] = self.negative_color

        cv2.line(self.chart, (0, zero_y), (w, zero_y), (255, 255, 255), 1)
        self.last_value = val

    def draw(self, canvas: np.ndarray):
        x, y = self.position
        w, h = self.size
        canvas[y:y + h, x:x + w, :] = self.chart
        if self.last_value >= 0.0:
            text_col = self.positive_color
        else:
            text_col = self.negative_color
        cv2.putText(canvas, '{0}: {1:+.2f}'.format(self.title, self.last_value), (x + 10, y + 40),
                    self.font, 1, text_col, 2, cv2.LINE_AA)


class StringDrawer(FreeDrawer):
    def __init__(self, x, y, text='', alignment:TextAlignment = TextAlignment.LEFT):
        super(self.__class__, self).__init__(x, y)
        self.text = text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = (255, 255, 255)
        self.size = 0.4
        self.thickness = 1
        self.line_height = 15
        self.alignment = alignment

    def draw(self, canvas: np.ndarray):
        for i, line in enumerate(self.text.split('\n')):
            y = self.y + i * self.line_height
            x = self.x
            if self.alignment is TextAlignment.RIGHT:
                size, _ = cv2.getTextSize(line, self.font, self.size, self.thickness)
                x -= size[0]
            if self.alignment is TextAlignment.CENTER:
                size, _ = cv2.getTextSize(line, self.font, self.size, self.thickness)
                x -= size[0] // 2
            cv2.putText(canvas, line, (x, int(y)), self.font, self.size, self.color, self.thickness, cv2.LINE_AA)


class HistogramDrawer(Drawer):
    def __init__(self, bounds: Rect, bins=100):
        super(HistogramDrawer, self).__init__(bounds)
        self.bins = bins
        self._img = np.zeros((bounds.height, bounds.width, 1), dtype=np.uint8)

    def set_value(self, value: np.ndarray):
        hist, bins = np.histogram(value, bins=self.bins, density=True)
        #hist, bins = np.histogram(value, bins=self.bins)
        min_val = value.min()
        max_val = value.max()
        mean_val = value.mean()
        sigma = value.std()

        img = np.zeros((self.bounds.height, len(hist), 3), dtype=np.uint8)
        hist = (np.clip(1 - np.array(hist), 0, 1) * self.bounds.height).astype(dtype=np.uint8)
        #hist = (np.clip(1 - np.array(hist) / (value.size / len(hist)), 0, 1) * self.bounds.height).astype(dtype=np.uint8)
        for i, val in enumerate(hist):
            pos = float(i / len(hist)) * (max_val - min_val) + min_val
            dist = np.abs(pos - mean_val)
            if dist < sigma:
                img[val:, i, ...] = 255
            elif dist < 3 * sigma:
                img[val:, i, ...] = 200
            elif dist < 5 * sigma:
                img[val:, i, ...] = 155
            else:
                img[val:, i, ...] = 100

        self._img = cv2.resize(img, self.bounds.size, interpolation=cv2.INTER_LINEAR)

    def draw(self, region: np.ndarray):
        region[:] = self._img


class ImgPlotDrawer(CompositeDrawer):
    def __init__(self, bounds: Rect, caption: str = '', color_map=ColorMap.NONE, resize=True):
        super(ImgPlotDrawer, self).__init__(bounds)
        self.caption_drawer = StringDrawer(3, 15, caption)
        self.img_drawer = ImageDrawer(Rect(0, 20, bounds.width, bounds.width), color_map=color_map, resize=resize)

        self._mean_val_drawer = StringDrawer(self.bounds.width // 2, self.img_drawer.bounds.bottom + 15,
                                             alignment=TextAlignment.CENTER)

        self.hist_drawer = HistogramDrawer(Rect(0, self.img_drawer.bounds.bottom + 20, bounds.width, 50))

        scale_height = 10
        self.scale_drawer = ImageDrawer(Rect(0, self.hist_drawer.bounds.bottom + 1, bounds.width, scale_height),
                                        color_map=color_map)
        scale = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (scale_height, 1))
        self.scale_drawer.img = scale

        self._min_val_drawer = StringDrawer(1, self.scale_drawer.bounds.bottom + 10)

        self._max_val_drawer = StringDrawer(self.bounds.width - 1, self.scale_drawer.bounds.bottom + 10,
                                            alignment=TextAlignment.RIGHT)

        self.append(self.caption_drawer)
        self.append(self.img_drawer)
        self.append(self.hist_drawer)
        self.append(self.scale_drawer)
        self.append(self._min_val_drawer)
        self.append(self._max_val_drawer)
        self.append(self._mean_val_drawer)
        self.mean_pos = 0

        self.bounds.height = self._max_val_drawer.y + 10

    def set_value(self, value: np.ndarray):
        self.img_drawer.img = value
        min_val = value.min()
        max_val = value.max()
        mean_val = value.mean()
        self._min_val_drawer.text = '{0:.3f}'.format(min_val)
        self._max_val_drawer.text = '{0:.3f}'.format(max_val)
        self._mean_val_drawer.text = '{0:.3f}'.format(mean_val)

        if max_val - min_val < 1e-6:
            self.mean_pos = self.bounds.width // 2
        else:
            self.mean_pos = int((mean_val - min_val) / (max_val - min_val) * self.bounds.width)
        self._mean_val_drawer.x = min(max(self.mean_pos, 20), self.bounds.width - 20)
        self.hist_drawer.set_value(value)

    def draw(self, region: np.ndarray):
        super(ImgPlotDrawer, self).draw(region)
        a1 = (self.mean_pos, self._mean_val_drawer.y + 5)
        b1 = (self.mean_pos, self.scale_drawer.bounds.bottom - 1)
        a2 = (self.mean_pos + 1, self._mean_val_drawer.y + 5)
        b2 = (self.mean_pos + 1, self.scale_drawer.bounds.bottom - 1)
        a3 = (self.mean_pos - 1, self._mean_val_drawer.y + 5)
        b3 = (self.mean_pos - 1, self.scale_drawer.bounds.bottom - 1)
        cv2.line(region, a1, b1, (255, 255, 255), thickness=1)
        cv2.line(region, a2, b2, (0, 0, 0), thickness=1)
        cv2.line(region, a3, b3, (0, 0, 0), thickness=1)


class Window(object):
    def __init__(self, width, height, window_name='main'):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.screen = np.zeros((height, width, 3), np.uint8)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)
        self.drawers = []

    def add_drawer(self, drawer: BaseDrawer):
        self.drawers.append(drawer)

    def remove_drawer(self, drawer: BaseDrawer):
        self.drawers.remove(drawer)

    def draw(self, clear=True):
        if clear:
            self.screen[:, :, :] = 0
        for drawer in self.drawers:
            if isinstance(drawer, Drawer):
                region = self.screen[drawer.bounds.top:drawer.bounds.bottom, drawer.bounds.left:drawer.bounds.right]
                drawer.draw(region)
            if isinstance(drawer, FreeDrawer):
                drawer.draw(self.screen)
        cv2.imshow(self.window_name, self.screen)
        cv2.waitKey(1)

    def __del__(self):
        cv2.destroyWindow(self.window_name)


if __name__ == '__main__':
    import time
    wnd = Window(840, 480)
    plot = ImgPlotDrawer(Rect(20, 50, 200, 330), caption='Test', color_map=ColorMap.HOT)
    plot2 = ImgPlotDrawer(Rect(240, 50, 200, 330), caption='Test 2')
    plot3 = ImgPlotDrawer(Rect(460, 50, 200, 330), caption='Test 3', color_map=ColorMap.JET)
    wnd.add_drawer(plot)
    wnd.add_drawer(plot2)
    wnd.add_drawer(plot3)
    t = 0

    while True:
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx ** 2 + xx * np.sin(t * 100) + yy ** 2 + 3 * np.cos(t * 100))
        plot.set_value(z)
        plot2.set_value(-z)
        plot3.set_value(z)
        wnd.draw()
        t += 0.001
        time.sleep(1.0 / 30)
